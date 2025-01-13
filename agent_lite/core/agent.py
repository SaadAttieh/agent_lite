import asyncio
import json
import time
from typing import AsyncIterator, Protocol

from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

from agent_lite.async_deque import AsyncDeque
from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseMemory,
    BaseTool,
    Content,
    InMemoryChatHistory,
    LLMResponse,
    LLMUsage,
    Message,
    RealtimeAudioBufferClearEvent,
    RealtimeAudioPayloadEvent,
    StreamingAssistantMessage,
    StreamingToolDirectResponse,
    SystemMessage,
    TextContent,
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
    input_message: str | list[Content]
    final_response: str | None
    final_message_chain: list[Message]
    cache_read_tokens: int
    cache_write_tokens: int

    def __str__(self) -> str:
        return f"""AgentRun:
    final_response: {self.final_response}
    model: {self.model}
    prompt_tokens: {self.prompt_tokens}
    completion_tokens: {self.completion_tokens}
    number_llm_invocations: {self.number_llm_invocations}
    total_time: {self.total_time}
    llm_time: {self.llm_time}
    cache_read_tokens: {self.cache_read_tokens}
    cache_write_tokens: {self.cache_write_tokens}
"""


class StreamingAgentRun(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model: str
    number_llm_invocations: int
    total_time_till_stream_start: float
    llm_time_till_stream_start: float
    input_message: str | list[Content]
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
    input_message: str | list[Content]
    final_response: AssistantMessage | StreamingAssistantMessage | None
    final_message_chain: list[Message]
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMRunFunc(Protocol):
    async def __call__(
        self, messages: list[Message], tools: list[BaseTool]
    ) -> LLMResponse:
        ...


class Agent(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    system_prompt: str | list[Content] | None
    llm: BaseLLM
    memory: BaseMemory = Field(
        default_factory=lambda: UnlimitedMemory(chat_history=InMemoryChatHistory())
    )
    tools: list[BaseTool] = Field(default_factory=list)
    max_number_iterations: int | None = 30

    async def submit_message(self, input_message: str | list[Content]) -> AgentRun:
        # delegate to _submit_message only because the @observe decorator seems to be breaking the function type
        return await self._submit_message(input_message)

    @observe()
    async def _submit_message(self, input_message: str | list[Content]) -> AgentRun:
        langfuse_context.update_current_observation(name="agent_lite_submit_message", input=input_message)
        agent_run_intermediate = await self._submit_message_helper(
            input_message, self.llm.run
        )
        await self.memory.add_message(UserMessage(content=input_message))
        if isinstance(agent_run_intermediate.final_response, StreamingAssistantMessage):
            final_response: str | None = (
                await agent_run_intermediate.final_response.consume_stream()
            )
        else:
            final_response = (
                agent_run_intermediate.final_response.content
                if agent_run_intermediate.final_response
                else None
            )
        if final_response:
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
            cache_read_tokens=agent_run_intermediate.cache_read_tokens,
            cache_write_tokens=agent_run_intermediate.cache_write_tokens,
        )
        print("Run complete")
        print(agent_run)
        return agent_run

    async def submit_message_and_stream_response(
        self,
        input_message: str | list[Content],
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
        self, input_message: str | list[Content], run_func: LLMRunFunc
    ) -> AgentRunIntermediate:
        conversation_history = await self.memory.get_messages()
        messages: list[Message] = (
            ([SystemMessage(content=self.system_prompt)] if self.system_prompt else [])
            + conversation_history
            + [UserMessage(content=input_message)]
        )

        print("Starting run")
        total_llm_usage = LLMUsage()
        agent_run_start_time = time.time()
        cumulative_llm_time: float = 0
        number_iterations = 0
        while (
            not self.max_number_iterations
            or number_iterations < self.max_number_iterations
        ):
            number_iterations += 1
            llm_start_time = time.time()
            # use tenacity
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_fixed(1),
            ):
                with attempt:
                    llm_response, llm_usage = await run_func(
                        messages=messages,
                        tools=self.tools,
                    )
            t2 = time.time()
            cumulative_llm_time += t2 - llm_start_time
            total_time = t2 - agent_run_start_time

            if llm_usage:
                total_llm_usage = total_llm_usage + llm_usage

            if llm_response:
                messages.append(llm_response)

            if llm_response is None or isinstance(llm_response, AssistantMessage):
                return AgentRunIntermediate(
                    model=self.llm.model,
                    prompt_tokens=total_llm_usage.prompt_tokens,
                    completion_tokens=total_llm_usage.completion_tokens,
                    cache_read_tokens=total_llm_usage.cache_read_tokens,
                    cache_write_tokens=total_llm_usage.cache_write_tokens,
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
                        prompt_tokens=total_llm_usage.prompt_tokens,
                        completion_tokens=total_llm_usage.completion_tokens,
                        cache_read_tokens=total_llm_usage.cache_read_tokens,
                        cache_write_tokens=total_llm_usage.cache_write_tokens,
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
        self, message: StreamingAssistantMessage | AssistantMessage | None
    ) -> AsyncIterator[str]:
        async def single_yield(value: str | None) -> AsyncIterator[str]:
            if value is not None:
                yield value

        if message is None:
            return
        elif type(message) is AssistantMessage:
            message = StreamingAssistantMessage(
                internal_stream=single_yield(message.content)
            )

        assert isinstance(message, StreamingAssistantMessage)
        async for response in message.stream():
            yield response
        await self.memory.add_message(AssistantMessage(content=message.content))

    @observe()
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
        langfuse_context.update_current_observation(
            name=f"{tool_impl.__class__}._arun",
            input=parsed_params,
        )

        tool_response = await tool_impl._arun(parsed_params)
        langfuse_context.update_current_observation(output=tool_response)

        if (
            isinstance(tool_response, StreamingToolDirectResponse)
            or isinstance(tool_response, ToolDirectResponse)
            or isinstance(tool_response, str)
        ):
            return tool_response
        return json.dumps(tool_response)

    async def run_realtime(
        self,
        *,
        audio_input_stream: AsyncIterator[RealtimeAudioPayloadEvent],
        voice: str,
        input_audio_format: str,
        output_audio_format: str,
        temperature: float = 0.8,
    ) -> AsyncIterator[RealtimeAudioPayloadEvent | RealtimeAudioBufferClearEvent]:
        llm_message_queue: AsyncDeque[
            RealtimeAudioPayloadEvent | ToolResponseMessage
        ] = AsyncDeque(timeout=120)

        async def stream_audio_to_queue():
            try:
                async for audio_event in audio_input_stream:
                    await llm_message_queue.put_right(audio_event)
            finally:
                await llm_message_queue.terminate()

        if isinstance(self.system_prompt, str):
            system_prompt = self.system_prompt
        elif isinstance(self.system_prompt, list):
            assert all(isinstance(c, TextContent) for c in self.system_prompt)
            system_prompt = "\n".join(
                c.text for c in self.system_prompt if isinstance(c, TextContent)
            )
        else:
            system_prompt = None

        async def respond_to_tools(response: ToolInvokationMessage):
            for tool_call in response.tools:
                print(
                    f"Invoking tool: {tool_call.tool_name} with params: {tool_call.tool_params}\n"
                )
                tool_response = await self.find_and_invoke_tool(
                    tool_call.tool_name, tool_call.tool_params
                )
                print(f"Tool response: {tool_response}\n")
                await llm_message_queue.put_right(
                    ToolResponseMessage(
                        tool_invokation_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        content=json.dumps(tool_response),
                    )
                )

        asyncio.create_task(stream_audio_to_queue())
        async for response in self.llm.run_realtime_voice(
            system_prompt=system_prompt,
            input_audio_format=input_audio_format,
            output_audio_format=output_audio_format,
            voice=voice,
            temperature=temperature,
            tools=self.tools,
            input_stream=llm_message_queue,
        ):
            if isinstance(response, ToolInvokationMessage):
                asyncio.create_task(respond_to_tools(response))
                continue
            yield response
