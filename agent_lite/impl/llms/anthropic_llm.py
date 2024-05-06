import json
from dataclasses import dataclass
from typing import AsyncIterator

import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types.beta.tools import (
    ToolParam,
    ToolsBetaMessageParam,
)

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseTool,
    LLMResponse,
    LLMUsage,
    Message,
    StreamingAssistantMessage,
    StreamingLLMResponse,
    SystemMessage,
    ToolInvokation,
    ToolInvokationMessage,
    ToolResponseMessage,
    UserMessage,
)

JSON_PROMPT = """You must output a valid JSON object.\n"""


@dataclass
class AnthropicLLM(BaseLLM):
    api_key: str
    encourage_json_response: bool = False
    remove_thinking_messages: bool = (
        True  # remove the <thinking> messages from the response
    )

    def __post_init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
        )

    async def run(self, messages: list[Message], tools: list[BaseTool]) -> LLMResponse:
        # system message
        system: str | None = None
        first_message_index = 0
        if isinstance(messages[0], SystemMessage):
            system = messages[0].content
            first_message_index = 1
        messages = messages[first_message_index:]

        # JSON mode
        if self.encourage_json_response:
            system = (system or "") + "\n\n" + JSON_PROMPT
            messages = messages + [AssistantMessage(content="{")]

        encoded_tools = AnthropicLLM.encode_tools(tools)

        # Messages:
        encoded_messages = AnthropicLLM.encode_messages(messages)

        response = await self.client.beta.tools.messages.create(
            max_tokens=4096,
            model=self.model,
            messages=encoded_messages,
            tools=encoded_tools,
            system=system or NOT_GIVEN,
        )
        llm_usage = (
            LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            if response.usage
            else None
        )
        message = response.content[-1]

        if message.type == "tool_use":
            return (
                ToolInvokationMessage(
                    tools=[
                        ToolInvokation(
                            id=message.id,
                            tool_name=message.name,
                            tool_params=json.dumps(message.input),
                        )
                    ]
                ),
                llm_usage,
            )
        elif self.encourage_json_response:
            response_text = message.text
            object_close = response_text.rfind("}")
            if object_close == -1:
                raise ValueError(
                    "The model did not respond with a valid JSON object whilst encourage_json_response is set to true. Anthropic LLM wrapper has not implemented a way to handle this case. Please disable the encourage_json_response option. Please file an issue."
                )
            return (
                AssistantMessage(content="{" + response_text[: object_close + 1]),
                llm_usage,
            )

        else:
            response_text = message.text
            if (
                self.remove_thinking_messages
                and "<thinking>" in response_text
                and "</thinking>" in response_text
            ):
                start_thinking = response_text.find("<thinking>")
                end_thinking = response_text.find("</thinking>") + len("</thinking>")
                response_text = (
                    response_text[:start_thinking] + response_text[end_thinking:]
                )
            if (
                self.remove_thinking_messages
                and "<answer>" in response_text
                and "</answer>" in response_text
            ):
                start_answer = response_text.find("<answer>") + len("<answer>")
                end_answer = response_text.find("</answer>")
                response_text = response_text[start_answer:end_answer]
            return (AssistantMessage(content=response_text), llm_usage)

    async def run_stream(
        self, messages: list[Message], tools: list[BaseTool]
    ) -> StreamingLLMResponse:
        # due to lack of support of tools, we temporarily use the old interface.
        from anthropic.types import (
            MessageParam,
        )

        if tools:
            raise ValueError(
                "Anthropic LLM does not support tool invokations in stream mode. This is a limitation of the Anthropic API. Anthropic state that this is being worked on:\nhttps://docs.anthropic.com/claude/docs/tool-use"
            )
        # system message
        system: str | None = None
        first_message_index = 0
        if isinstance(messages[0], SystemMessage):
            system = messages[0].content
            first_message_index = 1
        messages = messages[first_message_index:]

        # JSON mode
        if self.encourage_json_response:
            system = (system or "") + "\n\n" + JSON_PROMPT
            messages = messages + [AssistantMessage(content="{")]

        # Messages:
        encoded_messages_orig = AnthropicLLM.encode_messages(messages)
        encoded_messages: list[MessageParam] = []
        for message in encoded_messages_orig:
            encoded_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],  # type: ignore
                }
            )

        response = self.client.messages.stream(
            max_tokens=4096,
            model=self.model,
            messages=encoded_messages,
            system=system or NOT_GIVEN,
        )

        async def stream_response(
            response: anthropic.AsyncMessageStreamManager[anthropic.AsyncMessageStream],
        ) -> AsyncIterator[str]:
            async with response as stream:
                async for text in stream.text_stream:
                    yield text

        return StreamingAssistantMessage(internal_stream=stream_response(response))

    @staticmethod
    def encode_messages(
        messages: list[Message],
    ) -> list[ToolsBetaMessageParam]:
        def encode_message(message: Message) -> ToolsBetaMessageParam:
            if isinstance(message, SystemMessage):
                raise ValueError(
                    "System messages are not supported, Any initial system messages should have automatically been removed and moved to anthropic's api system parameter by this point."
                )
            elif isinstance(message, UserMessage):
                return {"role": "user", "content": message.content}
            elif isinstance(message, AssistantMessage):
                return {"role": "assistant", "content": message.content}
            elif isinstance(message, ToolInvokationMessage):
                return {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": t.id,
                            "name": t.tool_name,
                            "input": json.loads(t.tool_params),
                        }
                        for t in message.tools
                    ],
                }
            elif isinstance(message, ToolResponseMessage):
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_invokation_id,
                            "content": [{"type": "text", "text": message.content}],
                        }
                    ],
                }

            else:
                raise ValueError(f"Unsupported message type for this model: {message}")

        encoded_messages = [encode_message(m) for m in messages]
        encoded_messages = AnthropicLLM.group_encoded_messages_by_consecutive_role(
            encoded_messages
        )
        return encoded_messages

    @staticmethod
    def encode_tools(tools: list[BaseTool]) -> list[ToolParam]:
        def encode_tool(tool: BaseTool) -> ToolParam:
            schema = tool.get_args_schema().model_json_schema()

            return {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "input_schema": schema,  # type: ignore
            }

        return [encode_tool(t) for t in tools]

    @staticmethod
    def group_encoded_messages_by_consecutive_role(
        messages: list[ToolsBetaMessageParam],
    ) -> list[ToolsBetaMessageParam]:
        grouped_messages: list[ToolsBetaMessageParam] = []
        for message in messages:
            if (
                len(grouped_messages) == 0
                or grouped_messages[-1]["role"] != message["role"]
            ):
                grouped_messages.append(message)
            else:
                # if str convert to list
                if isinstance(grouped_messages[-1]["content"], str):
                    grouped_messages[-1]["content"] = [
                        {"type": "text", "text": grouped_messages[-1]["content"]}
                    ]
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]

                assert isinstance(grouped_messages[-1]["content"], list)
                assert isinstance(message["content"], list)

                grouped_messages[-1]["content"] += message["content"]
        return grouped_messages
