import json
import time
from dataclasses import dataclass
from typing import AsyncIterator

from pydantic import BaseModel

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseMemory,
    BaseTool,
    Message,
    StreamingAssistantMessage,
    SystemMessage,
    ToolInvokationMessage,
    ToolResponseMessage,
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


@dataclass
class Agent:
    system_prompt: str
    llm: BaseLLM
    memory: BaseMemory
    tools: list[BaseTool]
    max_number_iterations: int | None = 30

    async def submit_message(self, input_message: str) -> AgentRun:
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
            llm_response, llm_usage = await self.llm.run(
                messages=messages,
                tools=self.tools,
            )
            cumulative_llm_time += time.time() - llm_start_time
            if llm_usage:
                prompt_tokens += llm_usage.prompt_tokens
                completion_tokens += llm_usage.completion_tokens
            messages.append(llm_response)
            if isinstance(llm_response, AssistantMessage):
                agent_run = AgentRun(
                    model=self.llm.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    input_message=input_message,
                    final_response=llm_response.content,
                    final_message_chain=messages,
                    number_llm_invocations=number_iterations,
                    total_time=time.time() - total_start_time,
                    llm_time=cumulative_llm_time,
                )
                print("Run completed")
                print(agent_run, end="\n\n")
                await self.memory.add_message(UserMessage(content=input_message))
                await self.memory.add_message(
                    AssistantMessage(content=agent_run.final_response)
                )

                return agent_run
            elif not isinstance(llm_response, ToolInvokationMessage):
                raise Exception(
                    f"Unexpected response from LLM, LLM must return AssistantMessage or ToolInvokationMessage. LLM response: {llm_response}"
                )
            for tool_call in llm_response.tools:
                print(
                    f"Invoking tool: {tool_call.tool_name} with params: {tool_call.tool_params}\n"
                )
                tool_response = await self.find_and_invoke_tool(
                    tool_call.tool_name, tool_call.tool_params
                )
                print(f"Tool response: {tool_response}\n")
                messages.append(
                    ToolResponseMessage(
                        tool_invokation_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        content=tool_response,
                    )
                )
        raise Exception(f"Max iterations reached: {self.max_number_iterations}")

    async def submit_message_and_stream_response(
        self, input_message: str
    ) -> StreamingAgentRun:
        conversation_history = await self.memory.get_messages()
        messages = (
            [SystemMessage(content=self.system_prompt)]
            + conversation_history
            + [UserMessage(content=input_message)]
        )

        print("Starting run")
        total_start_time = time.time()
        cumulative_llm_time: float = 0
        number_iterations = 0
        while (
            not self.max_number_iterations
            or number_iterations < self.max_number_iterations
        ):
            number_iterations += 1
            llm_start_time = time.time()
            llm_response = await self.llm.run_stream(
                messages=messages,
                tools=self.tools,
            )
            cumulative_llm_time += time.time() - llm_start_time
            messages.append(llm_response)
            if isinstance(llm_response, StreamingAssistantMessage):
                await self.memory.add_message(UserMessage(content=input_message))
                agent_run = StreamingAgentRun(
                    model=self.llm.model,
                    input_message=input_message,
                    final_response_stream=self.stream_then_save(llm_response),
                    final_message_chain=messages,
                    number_llm_invocations=number_iterations,
                    total_time_till_stream_start=time.time() - total_start_time,
                    llm_time_till_stream_start=cumulative_llm_time,
                )
                print("Run completed and response is being streamed")
                print(agent_run, end="\n\n")
                return agent_run
            elif not isinstance(llm_response, ToolInvokationMessage):
                raise Exception(
                    f"Unexpected response from LLM, LLM must return AssistantMessage or ToolInvokationMessage. LLM response: {llm_response}"
                )
            for tool_call in llm_response.tools:
                print(
                    f"Invoking tool: {tool_call.tool_name} with params: {tool_call.tool_params}\n"
                )
                tool_response = await self.find_and_invoke_tool(
                    tool_call.tool_name, tool_call.tool_params
                )
                print(f"Tool response: {tool_response}\n")
                messages.append(
                    ToolResponseMessage(
                        tool_invokation_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        content=tool_response,
                    )
                )
        raise Exception(f"Max iterations reached: {self.max_number_iterations}")

    async def stream_then_save(
        self, message: StreamingAssistantMessage
    ) -> AsyncIterator[str]:
        async for response in message.stream():
            yield response
        await self.memory.add_message(AssistantMessage(content=message.content))

    async def find_and_invoke_tool(
        self, tool_name: str, tool_params_as_json: str
    ) -> str:
        tool_impl = next((t for t in self.tools if t.get_name() == tool_name), None)
        if not tool_impl:
            return f"Tool '{tool_name}' not found"
        try:
            args_schema = tool_impl.get_args_schema()
            parsed_params = args_schema.parse_raw(tool_params_as_json)
        except Exception as e:
            return f"Error parsing parameters for tool '{tool_name}': {e}"
        tool_response = await tool_impl._arun(parsed_params)
        return (
            tool_response
            if isinstance(tool_response, str)
            else json.dumps(tool_response, indent=2)
        )
