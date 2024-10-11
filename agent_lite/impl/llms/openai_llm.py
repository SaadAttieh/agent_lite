import asyncio
import base64
import json
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Union

import websockets
from langfuse.decorators import langfuse_context, observe
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseTool,
    Content,
    ImageContent,
    LLMResponse,
    LLMUsage,
    Message,
    RealtimeAudioBufferClearEvent,
    RealtimeAudioPayloadEvent,
    StreamingAssistantMessage,
    StreamingLLMResponse,
    SystemMessage,
    TextContent,
    ToolInvokation,
    ToolInvokationMessage,
    ToolResponseMessage,
    UserMessage,
)


@dataclass
class OpenAILLM(BaseLLM):
    api_key: str
    encourage_json_response: bool = False
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @observe(as_type="generation")
    async def run(self, messages: list[Message], tools: list[BaseTool]) -> LLMResponse:
        encoded_messages = OpenAILLM.encode_messages(messages)
        encoded_tools = OpenAILLM.encode_tools(tools)
        langfuse_context.update_current_observation(
            name=f"{self.__class__}.run",
            input=encoded_messages,
            model=self.model,
            metadata=dict(
                tools=encoded_tools or NOT_GIVEN,
                response_format=(
                    {"type": "json_object"}
                    if self.encourage_json_response
                    else NOT_GIVEN
                ),
            ),
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=encoded_messages,
            tools=encoded_tools or NOT_GIVEN,
            response_format=(
                {"type": "json_object"} if self.encourage_json_response else NOT_GIVEN
            ),
        )
        llm_usage = (
            LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            if response.usage
            else None
        )
        langfuse_context.update_current_observation(output=response)
        if response.usage:
            langfuse_context.update_current_observation(
                usage={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                }
            )

        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            assert assistant_message.content is not None
            return (AssistantMessage(content=assistant_message.content), llm_usage)
        return (
            ToolInvokationMessage(
                tools=[
                    OpenAILLM.parse_tool_call(tool_call)
                    for tool_call in assistant_message.tool_calls
                ]
            ),
            llm_usage,
        )

    async def run_stream(
        self, messages: list[Message], tools: list[BaseTool]
    ) -> StreamingLLMResponse:
        encoded_messages = OpenAILLM.encode_messages(messages)
        encoded_tools = OpenAILLM.encode_tools(tools)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=encoded_messages,
            tools=encoded_tools or NOT_GIVEN,
            response_format=(
                {"type": "json_object"} if self.encourage_json_response else NOT_GIVEN
            ),
            stream=True,
        )
        async for message in response:
            if message.choices[0].delta.tool_calls:
                return await self._parse_streamed_tool_invokation_message(
                    message, response
                )
            else:
                return await self._stream_assistant_response(message, response)
        return await self._stream_assistant_response(None, response)

    async def _parse_streamed_tool_invokation_message(
        self,
        first_message: ChatCompletionChunk,
        response: AsyncIterator[ChatCompletionChunk],
    ) -> ToolInvokationMessage:
        tool_calls: list[ToolInvokation] = []
        async for chunk in repair_iterator(first_message, response):
            if not chunk.choices[0].delta.tool_calls:
                continue
            for delta_tool_call in chunk.choices[0].delta.tool_calls:
                if delta_tool_call.index >= len(tool_calls):
                    tool_calls.extend(
                        ToolInvokation(id="", tool_name="", tool_params="")
                        for _ in range(delta_tool_call.index + 1 - len(tool_calls))
                    )
                if delta_tool_call.id:
                    tool_calls[delta_tool_call.index].id += delta_tool_call.id
                if delta_tool_call.function and delta_tool_call.function.name:
                    tool_calls[
                        delta_tool_call.index
                    ].tool_name += delta_tool_call.function.name
                if delta_tool_call.function and delta_tool_call.function.arguments:
                    tool_calls[
                        delta_tool_call.index
                    ].tool_params += delta_tool_call.function.arguments
        return ToolInvokationMessage(tools=tool_calls)

    async def _stream_assistant_response(
        self,
        first_message: ChatCompletionChunk | None,
        response: AsyncIterator[ChatCompletionChunk],
    ) -> StreamingAssistantMessage:
        async def stream_messages() -> AsyncIterator[str]:
            async for message in repair_iterator(first_message, response):
                if message.choices[0].delta.content:
                    yield message.choices[0].delta.content

        return StreamingAssistantMessage(internal_stream=stream_messages())

    async def run_realtime_voice(
        self,
        *,
        system_prompt: str | None,
        input_audio_format: str,
        output_audio_format: str,
        voice: str,
        temperature: float,
        tools: list[BaseTool],
        input_stream: AsyncIterator[RealtimeAudioPayloadEvent | ToolResponseMessage],
    ) -> AsyncIterator[
        RealtimeAudioPayloadEvent
        | ToolInvokationMessage
        | RealtimeAudioBufferClearEvent
    ]:
        encoded_tools = [
            dict(t["function"], **{"type": "function"})
            for t in OpenAILLM.encode_tools(tools)
        ]
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": input_audio_format,
                "output_audio_format": output_audio_format,
                "voice": voice,
                "instructions": system_prompt,
                "modalities": ["text", "audio"],
                "temperature": temperature,
                "tools": encoded_tools,
            },
        }
        print("Connecting to OpenAI Realtime API")
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            extra_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            await openai_ws.send(json.dumps(session_update))
            asyncio.create_task(
                OpenAILLM._stream_realtime_to_openai(input_stream, openai_ws)
            )
            async for response in OpenAILLM._read_openai_realtime_stream(openai_ws):
                yield response

    @staticmethod
    async def _read_openai_realtime_stream(
        openai_ws: websockets.WebSocketClientProtocol,
    ) -> AsyncIterator[
        RealtimeAudioPayloadEvent
        | ToolInvokationMessage
        | RealtimeAudioBufferClearEvent
    ]:
        async for openai_message in openai_ws:
            response = json.loads(openai_message)
            if response["type"] == "error":
                print(
                    f"Error from OpenAI Realtime API: {json.dumps(response, indent=2)}"
                )
            elif response["type"] == "input_audio_buffer.speech_started":
                yield RealtimeAudioBufferClearEvent()
            elif response["type"] == "response.function_call_arguments.done":
                call_id = response["call_id"]
                tool_params = response["arguments"]
                tool_name = response["name"]
                yield ToolInvokationMessage(
                    tools=[
                        ToolInvokation(
                            id=call_id, tool_name=tool_name, tool_params=tool_params
                        )
                    ]
                )
            elif response["type"] == "response.audio.delta" and response.get("delta"):
                try:
                    audio_payload = base64.b64encode(
                        base64.b64decode(response["delta"])
                    ).decode("utf-8")
                    yield RealtimeAudioPayloadEvent.from_base64(audio_payload)
                except Exception as e:
                    print(f"Error processing audio data: {e}")

    @staticmethod
    async def _stream_realtime_to_openai(
        message_stream: AsyncIterator[RealtimeAudioPayloadEvent | ToolResponseMessage],
        openai_ws: websockets.WebSocketClientProtocol,
    ) -> None:
        print("Streaming audio to OpenAI Realtime API")

        try:
            async for event in message_stream:
                if not openai_ws.open:
                    print("Found openai_ws closed.")
                    break
                if isinstance(event, ToolResponseMessage):
                    print("Sending tool response to OpenAI")
                    tool_response = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": event.tool_invokation_id,
                            "output": event.content,
                        },
                    }
                    await openai_ws.send(json.dumps(tool_response))
                    create_event = {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text", "audio"],
                        },
                    }
                    await openai_ws.send(json.dumps(create_event))
                    continue

                audio_append = {
                    "type": "input_audio_buffer.append",
                    "audio": event.as_base64(),
                }
                await openai_ws.send(json.dumps(audio_append))
        except Exception as e:
            print(f"Error streaming audio to OpenAI: {e}")
        print("Closing OpenAI Realtime API connection")
        if openai_ws.open:
            await openai_ws.close()

    @staticmethod
    def encode_messages(
        messages: list[Message],
    ) -> list[ChatCompletionMessageParam]:
        def encode_message(message: Message) -> ChatCompletionMessageParam:
            if isinstance(message, SystemMessage):
                return {
                    "role": "system",
                    "content": OpenAILLM.make_openai_system_message(message),
                }
            elif isinstance(message, UserMessage):
                return {
                    "role": "user",
                    "content": OpenAILLM.make_openai_message_content(message.content),
                }
            elif isinstance(message, AssistantMessage):
                return {"role": "assistant", "content": message.content}
            elif isinstance(message, ToolInvokationMessage):
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool.id,
                            "type": "function",
                            "function": {
                                "name": tool.tool_name,
                                "arguments": tool.tool_params,
                            },
                        }
                        for tool in message.tools
                    ],
                }
            elif isinstance(message, ToolResponseMessage):
                return {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_invokation_id,
                }
            else:
                raise ValueError(f"Invalid message type: {message.message_type}")

        return [encode_message(m) for m in messages]

    @staticmethod
    def encode_tools(tools: list[BaseTool]) -> list[ChatCompletionToolParam]:
        def encode_tool(tool: BaseTool) -> ChatCompletionToolParam:
            return {
                "type": "function",
                "function": {
                    "name": tool.get_name(),
                    "description": tool.get_description(),
                    "parameters": tool.get_args_schema().model_json_schema(),
                },
            }

        return [encode_tool(t) for t in tools]

    @staticmethod
    def parse_tool_call(
        tool_call: ChatCompletionMessageToolCall,
    ) -> ToolInvokation:
        return ToolInvokation(
            id=tool_call.id,
            tool_name=tool_call.function.name,
            tool_params=tool_call.function.arguments,
        )

    @staticmethod
    def make_openai_system_message(
        system_message: SystemMessage,
    ) -> Union[str, Iterable[Union[ChatCompletionContentPartTextParam]]]:
        content = system_message.content
        if isinstance(content, str):
            return content

        return [OpenAILLM.make_openai_text_block(c) for c in content]

    @staticmethod
    def make_openai_message_content(
        content: str | list[Content],
    ) -> Union[
        str,
        Iterable[
            Union[
                ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
            ]
        ],
    ]:
        if isinstance(content, str):
            return content
        return [
            (
                OpenAILLM.make_openai_text_block(c)
                if isinstance(c, TextContent)
                else OpenAILLM.make_openai_image_block(c)
            )
            for c in content
        ]

    @staticmethod
    def make_openai_text_block(
        inner_content: Content,
    ) -> ChatCompletionContentPartTextParam:
        if not isinstance(inner_content, TextContent):
            raise ValueError("Expected a text only content here.")
        return {"text": inner_content.text, "type": "text"}

    @staticmethod
    def make_openai_image_block(
        inner_content: Content,
    ) -> ChatCompletionContentPartImageParam:
        if not isinstance(inner_content, ImageContent):
            raise ValueError("Expected an image only content here.")
        assert inner_content.encoding_type() == "base64"
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{inner_content.file_type.value};{inner_content.encoding_type()},{inner_content.encoded_data}",
            },
        }


async def repair_iterator(
    first_item: ChatCompletionChunk | None, iterator: AsyncIterator[ChatCompletionChunk]
) -> AsyncIterator[ChatCompletionChunk]:
    if first_item:
        yield first_item
    async for item in iterator:
        yield item
