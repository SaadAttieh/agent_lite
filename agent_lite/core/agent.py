import json
import time
from typing import AsyncIterator, Protocol

from pydantic import BaseModel, Field

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseMemory,
    BaseTool,
    InMemoryChatHistory,
    LLMResponse,
    Message,
    StreamingAssistantMessage,
    StreamingToolDirectResponse,
    SystemMessage,
    ToolDirectResponse,
    ToolInvokationMessage,
    ToolResponseMessage,
    UnlimitedMemory,
    UserMessage,
)


class AgentRun(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    number_llm_invocations: int
    total_time: float
    llm_time: float
    input_message: str
    final_response: str
    final_message_chain: list[Message]

    def __str__(self) -> str:
        return f"""AgentRun:
    final_response: {self.final_response}
    model: {self.model}
    prompt_tokens: {self.prompt_tokens}
    completion_tokens: {self.completion_tokens}
    number_llm_invocations: {self.number_llm_invocations}
    total_time: {self.total_time}
    llm_time: {self.llm_time}
"""


class StreamingAgentRun(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model: str
    number_llm_invocations: int
    total_time_till_stream_start: float
    llm_time_till_stream_start: float
    input_message: str
    final_response_stream: AsyncIterator[str]
    final_message_chain: list[Message]

    def __str__(self) -> str:
        return f"""StreamingAgentRun:
    final_response: <streaming>
    model: {self.model}
    number_llm_invocations: {self.number_llm_invocations}
    total_time_till_stream_start: {self.total_time_till_stream_start}
    llm_time_till_stream_start: {self.llm_time_till_stream_start}
"""


class AgentRunIntermediate(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model: str
    prompt_tokens: int
    completion_tokens: int
    number_llm_invocations: int
    total_time: float
    llm_time: float
    input_message: str
    final_response: AssistantMessage | StreamingAssistantMessage
    final_message_chain: list[Message]


class LLMRunFunc(Protocol):
    async def __call__(
        self, messages: list[Message], tools: list[BaseTool]
    ) -> LLMResponse:
        ...


class Agent(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    system_prompt: str
    llm: BaseLLM
    memory: BaseMemory = Field(
        default_factory=lambda: UnlimitedMemory(chat_history=InMemoryChatHistory())
    )
    tools: list[BaseTool] = Field(default_factory=list)
    max_number_iterations: int | None = 30

    async def submit_message(self, input_message: str) -> AgentRun:
        agent_run_intermediate = await self._submit_message_helper(
            input_message, self.llm.run
        )
        await self.memory.add_message(UserMessage(content=input_message))
        if isinstance(agent_run_intermediate.final_response, StreamingAssistantMessage):
            final_response = (
                await agent_run_intermediate.final_response.consume_stream()
            )
        else:
            final_response = agent_run_intermediate.final_response.content

        await self.memory.add_message(AssistantMessage(content=final_response))

        agent_run = AgentRun(
            model=agent_run_intermediate.model,
            prompt_tokens=agent_run_intermediate.prompt_tokens,
            completion_tokens=agent_run_intermediate.completion_tokens,
            number_llm_invocations=agent_run_intermediate.number_llm_invocations,
            total_time=agent_run_intermediate.total_time,
            llm_time=agent_run_intermediate.llm_time,
            input_message=input_message,
            final_response=final_response,
            final_message_chain=agent_run_intermediate.final_message_chain,
        )
        print("Run complete")
        print(agent_run)
        return agent_run

    async def submit_message_and_stream_response(
        self,
        input_message: str,
    ) -> StreamingAgentRun:
        async def run_func(
            messages: list[Message], tools: list[BaseTool]
        ) -> LLMResponse:
            return await self.llm.run_stream(messages=messages, tools=tools), None

        agent_run_intermediate = await self._submit_message_helper(
            input_message, run_func
        )
        await self.memory.add_message(UserMessage(content=input_message))
        agent_run = StreamingAgentRun(
            model=agent_run_intermediate.model,
            number_llm_invocations=agent_run_intermediate.number_llm_invocations,
            total_time_till_stream_start=agent_run_intermediate.total_time,
            llm_time_till_stream_start=agent_run_intermediate.llm_time,
            input_message=input_message,
            final_response_stream=self.stream_then_save(
                agent_run_intermediate.final_response
            ),
            final_message_chain=agent_run_intermediate.final_message_chain,
        )
        print("Run complete")
        print(agent_run)
        return agent_run

    async def _submit_message_helper(
        self, input_message: str, run_func: LLMRunFunc
    ) -> AgentRunIntermediate:
        conversation_history = await self.memory.get_messages()
        messages = (
            [SystemMessage(content=self.system_prompt)]
            + conversation_history
            + [UserMessage(content=input_message)]
        )

        print("Starting run")
        prompt_tokens, completion_tokens = 0, 0
        total_start_time = time.time()
        cumulative_llm_time: float = 0
        number_iterations = 0
        while (
            not self.max_number_iterations
            or number_iterations < self.max_number_iterations
        ):
            number_iterations += 1
            llm_start_time = time.time()
            llm_response, llm_usage = await run_func(
                messages=messages,
                tools=self.tools,
            )
            messages.append(llm_response)

            t2 = time.time()
            cumulative_llm_time += t2 - llm_start_time
            total_time = t2 - total_start_time
            if llm_usage:
                prompt_tokens += llm_usage.prompt_tokens
                completion_tokens += llm_usage.completion_tokens

            if isinstance(llm_response, AssistantMessage):
                return AgentRunIntermediate(
                    model=self.llm.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    input_message=input_message,
                    final_response=llm_response,
                    final_message_chain=messages,
                    number_llm_invocations=number_iterations,
                    total_time=total_time,
                    llm_time=cumulative_llm_time,
                )
            elif not isinstance(llm_response, ToolInvokationMessage):
                raise Exception(
                    f"Unexpected response from LLM, LLM must return AssistantMessage, StreamingAssistantMessage  or ToolInvokationMessage. LLM response: {llm_response}"
                )
            for tool_call in llm_response.tools:
                print(
                    f"Invoking tool: {tool_call.tool_name} with params: {tool_call.tool_params}\n"
                )
                tool_response = await self.find_and_invoke_tool(
                    tool_call.tool_name, tool_call.tool_params
                )
                print(f"Tool response: {tool_response}\n")
                if isinstance(tool_response, ToolDirectResponse) or isinstance(
                    tool_response, StreamingToolDirectResponse
                ):
                    messages.append(tool_response)
                    return AgentRunIntermediate(
                        model=self.llm.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        input_message=input_message,
                        final_response=tool_response,
                        final_message_chain=messages,
                        number_llm_invocations=number_iterations,
                        total_time=total_time,
                        llm_time=cumulative_llm_time,
                    )
                else:
                    messages.append(
                        ToolResponseMessage(
                            tool_invokation_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            content=tool_response,
                        )
                    )

        raise Exception(f"Max iterations reached: {self.max_number_iterations}")

    async def stream_then_save(
        self, message: StreamingAssistantMessage | AssistantMessage
    ) -> AsyncIterator[str]:
        async def single_yield(value: str) -> AsyncIterator[str]:
            yield value

        if type(message) == AssistantMessage:
            message = StreamingAssistantMessage(
                internal_stream=single_yield(message.content)
            )

        assert isinstance(message, StreamingAssistantMessage)
        async for response in message.stream():
            yield response
        await self.memory.add_message(AssistantMessage(content=message.content))

    async def find_and_invoke_tool(
        self, tool_name: str, tool_params_as_json: str
    ) -> str | ToolDirectResponse | StreamingToolDirectResponse:
        tool_impl = next((t for t in self.tools if t.get_name() == tool_name), None)
        if not tool_impl:
            return f"Tool '{tool_name}' not found"
        try:
            args_schema = tool_impl.get_args_schema()
            parsed_params = args_schema.parse_raw(tool_params_as_json)
        except Exception as e:
            return f"Error parsing parameters for tool '{tool_name}': {e}"
        tool_response = await tool_impl._arun(parsed_params)
        if (
            isinstance(tool_response, StreamingToolDirectResponse)
            or isinstance(tool_response, ToolDirectResponse)
            or isinstance(tool_response, str)
        ):
            return tool_response
        return json.dumps(tool_response)
