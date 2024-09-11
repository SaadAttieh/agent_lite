import json
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Union

import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.lib.streaming import AsyncPromptCachingBetaMessageStreamManager
from anthropic.types.beta.prompt_caching import (
    PromptCachingBetaImageBlockParam,
    PromptCachingBetaMessageParam,
    PromptCachingBetaTextBlockParam,
    PromptCachingBetaToolParam,
)
from langfuse.decorators import langfuse_context, observe

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseTool,
    Content,
    ImageContent,
    LLMResponse,
    LLMUsage,
    Message,
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

    @observe(as_type="generation")
    async def run(self, messages: list[Message], tools: list[BaseTool]) -> LLMResponse:
        # system message
        system: Union[
            str, Iterable[Union[PromptCachingBetaTextBlockParam]], None
        ] = None
        first_message_index = 0
        if isinstance(messages[0], SystemMessage):
            system = AnthropicLLM.make_anthropic_system_message(messages[0])
            first_message_index = 1
        messages = messages[first_message_index:]

        # JSON mode
        if self.encourage_json_response:
            messages = messages + [AssistantMessage(content="{")]

        encoded_tools = AnthropicLLM.encode_tools(tools)

        # Messages:
        encoded_messages = AnthropicLLM.encode_messages(messages)

        langfuse_context.update_current_observation(
            name=f"{self.__class__}.run",
            input=encoded_messages,
            model=self.model,
            metadata=dict(
                max_tokens=4096,
                tools=encoded_tools,
                system=system or NOT_GIVEN,
            ),
        )

        response = await self.client.beta.prompt_caching.messages.create(
            model=self.model,
            messages=encoded_messages,
            max_tokens=4096,
            tools=encoded_tools,
            system=system or NOT_GIVEN,
        )
        llm_usage = LLMUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
        )
        langfuse_context.update_current_observation(
            output=response,
            usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
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
        if tools:
            raise ValueError(
                "Anthropic LLM does not support tool invokations in stream mode. This is a limitation of the Anthropic API. Anthropic state that this is being worked on:\nhttps://docs.anthropic.com/claude/docs/tool-use"
            )
        # system message
        system: Union[
            str, Iterable[Union[PromptCachingBetaTextBlockParam]], None
        ] = None
        first_message_index = 0
        if isinstance(messages[0], SystemMessage):
            system = AnthropicLLM.make_anthropic_system_message(messages[0])
            first_message_index = 1
        messages = messages[first_message_index:]

        # JSON mode
        if self.encourage_json_response:
            messages = messages + [AssistantMessage(content="{")]

        # Messages:
        encoded_messages_orig = AnthropicLLM.encode_messages(messages)
        encoded_messages: list[PromptCachingBetaMessageParam] = []
        for message in encoded_messages_orig:
            encoded_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],  # type: ignore
                }
            )

        response = self.client.beta.prompt_caching.messages.stream(
            max_tokens=4096,
            model=self.model,
            messages=encoded_messages,
            system=system or NOT_GIVEN,
        )

        async def stream_response(
            response: AsyncPromptCachingBetaMessageStreamManager,
            encourage_json_response: bool,
        ) -> AsyncIterator[str]:
            if encourage_json_response:
                yield "{"
            async with response as stream:
                async for text in stream.text_stream:
                    yield text

        return StreamingAssistantMessage(
            internal_stream=stream_response(response, self.encourage_json_response)
        )

    @staticmethod
    def encode_messages(
        messages: list[Message],
    ) -> list[PromptCachingBetaMessageParam]:
        def encode_message(message: Message) -> PromptCachingBetaMessageParam:
            if isinstance(message, SystemMessage):
                raise ValueError(
                    "System messages are not supported, Any initial system messages should have automatically been removed and moved to anthropic's api system parameter by this point."
                )
            elif isinstance(message, UserMessage):
                return {
                    "role": "user",
                    "content": AnthropicLLM.make_anthropic_message_content(
                        message.content
                    ),
                }
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
    def encode_tools(tools: list[BaseTool]) -> list[PromptCachingBetaToolParam]:
        def encode_tool(tool: BaseTool) -> PromptCachingBetaToolParam:
            schema = tool.get_args_schema().model_json_schema()

            return {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "input_schema": schema,  # type: ignore
            }

        return [encode_tool(t) for t in tools]

    @staticmethod
    def group_encoded_messages_by_consecutive_role(
        messages: list[PromptCachingBetaMessageParam],
    ) -> list[PromptCachingBetaMessageParam]:
        grouped_messages: list[PromptCachingBetaMessageParam] = []
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

    @staticmethod
    def make_anthropic_system_message(
        system_message: SystemMessage,
    ) -> Union[str, Iterable[Union[PromptCachingBetaTextBlockParam]]]:
        content = system_message.content
        if isinstance(content, str):
            return content

        return [AnthropicLLM.make_anthropic_text_block(c) for c in content]

    @staticmethod
    def make_anthropic_message_content(
        content: str | list[Content],
    ) -> Union[
        str,
        Iterable[
            Union[PromptCachingBetaTextBlockParam, PromptCachingBetaImageBlockParam]
        ],
    ]:
        if isinstance(content, str):
            return content
        return [
            (
                AnthropicLLM.make_anthropic_text_block(c)
                if isinstance(c, TextContent)
                else AnthropicLLM.make_anthropic_image_block(c)
            )
            for c in content
        ]

    @staticmethod
    def make_anthropic_text_block(
        inner_content: Content,
    ) -> PromptCachingBetaTextBlockParam:
        if not isinstance(inner_content, TextContent):
            raise ValueError("Expected a text only content here.")
        result: PromptCachingBetaTextBlockParam = {
            "text": inner_content.text,
            "type": "text",
        }
        if inner_content.cached:
            result["cache_control"] = {"type": "ephemeral"}
        return result

    @staticmethod
    def make_anthropic_image_block(
        inner_content: Content,
    ) -> PromptCachingBetaImageBlockParam:
        if not isinstance(inner_content, ImageContent):
            raise ValueError("Expected an image only content here.")
        assert inner_content.encoding_type() == "base64"
        result: PromptCachingBetaImageBlockParam = {
            "type": "image",
            "source": {
                "data": inner_content.encoded_data,
                "media_type": inner_content.file_type.value,
                "type": "base64",
            },
        }
        if inner_content.cached:
            result["cache_control"] = {"type": "ephemeral"}
        return result
