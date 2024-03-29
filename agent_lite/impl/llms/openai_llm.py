from dataclasses import dataclass

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseTool,
    LLMResponse,
    LLMUsage,
    Message,
    SystemMessage,
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

    async def run(self, messages: list[Message], tools: list[BaseTool]) -> LLMResponse:
        encoded_messages = OpenAILLM.encode_messages(messages)
        encoded_tools = OpenAILLM.encode_tools(tools)
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

    @staticmethod
    def encode_messages(
        messages: list[Message],
    ) -> list[ChatCompletionMessageParam]:
        def encode_message(message: Message) -> ChatCompletionMessageParam:
            if isinstance(message, SystemMessage):
                return {"role": "system", "content": message.content}
            elif isinstance(message, UserMessage):
                return {"role": "user", "content": message.content}
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
