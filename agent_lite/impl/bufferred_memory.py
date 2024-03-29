from dataclasses import dataclass

import tiktoken

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseMemory,
    InMemmoryChatHistory,
    Message,
    UnlimitedMemory,
    UserMessage,
)


@dataclass
class DropMessageStats:
    original_number_messages: int
    messages_to_drop: int
    original_number_tokens: int
    final_number_tokens: int


def _number_messages_to_drop(
    model: str, max_tokens: int, messages: list[str]
) -> DropMessageStats:
    encoding = tiktoken.encoding_for_model(model)
    number_tokens = sum(len(encoding.encode_ordinary(m)) for m in messages)
    print(f"number_tokens: {number_tokens}")
    if number_tokens <= max_tokens:
        return DropMessageStats(
            original_number_messages=len(messages),
            messages_to_drop=0,
            original_number_tokens=number_tokens,
            final_number_tokens=number_tokens,
        )
    new_number_tokens = number_tokens
    for i in range(len(messages)):
        new_number_tokens -= len(encoding.encode_ordinary(messages[i]))
        if new_number_tokens <= max_tokens:
            return DropMessageStats(
                original_number_messages=len(messages),
                messages_to_drop=i + 1,
                original_number_tokens=number_tokens,
                final_number_tokens=new_number_tokens,
            )
    raise ValueError("Buffer too small to fit messages")


@dataclass
class BufferedMemory(BaseMemory):
    model: str
    max_tokens: int

    async def add_message(self, message: Message) -> None:
        await self.chat_history.add_message(message)
        await self._maybe_trim()

    async def get_messages(self) -> list[Message]:
        await self._maybe_trim()
        return await self.chat_history.get_messages()

    async def _maybe_trim(self) -> bool:
        messages = await self.chat_history.get_messages()
        if len(messages) == 0:
            return False
        number_messages_to_drop = _number_messages_to_drop(
            self.model,
            self.max_tokens,
            [m.serialized_message_content() for m in messages],
        ).messages_to_drop
        if number_messages_to_drop > 0:
            await self.chat_history.delete_range(0, number_messages_to_drop)
            return True
        return False


@dataclass
class BufferedMemoryWithSummarizer(BaseMemory):
    model: str
    summarizer_llm: BaseLLM
    threshold_tokens: int
    target_max_tokens: int
    conversation_context: str

    def __post_init__(self) -> None:
        assert self.threshold_tokens >= self.target_max_tokens

    async def _summarized(self, messages: list[Message]) -> str:
        from agent_lite.core.agent import Agent

        agent = Agent(
            system_prompt="Your goal is to summariZe this conversation as concisely as possible, whilst still capturing the essence of the conversation.\nContext of the conversation:\n"
            + self.conversation_context,
            llm=self.summarizer_llm,
            memory=UnlimitedMemory(chat_history=InMemmoryChatHistory()),
            tools=[],
        )

        def messages_to_str(m: Message) -> str:
            if isinstance(m, UserMessage):
                return f"User: {m.content}"
            elif isinstance(m, AssistantMessage):
                return f"Assistant: {m.content}"
            else:
                raise ValueError(f"Does not summarize messages of type {type(m)}")

        conversation = "\n".join(messages_to_str(m) for m in messages)
        input = f"""Start of conversation to summarize:
-----------------
{conversation}
-----------------
End of conversation to summarize.
"""
        return (await agent.submit_message(input)).final_response

    async def _maybe_summarize(self) -> bool:
        messages = await self.chat_history.get_messages()
        if len(messages) == 0:
            return False
        drop_message_stats = _number_messages_to_drop(
            self.model,
            self.target_max_tokens,
            [m.serialized_message_content() for m in messages],
        )
        if drop_message_stats.messages_to_drop == 0:
            return False
        if drop_message_stats.original_number_tokens <= self.threshold_tokens:
            return False
        summarized = await self._summarized(
            messages[: drop_message_stats.messages_to_drop]
        )
        await self.chat_history.edit_message(
            0,
            UserMessage(
                content="Previous conversation summary:\n"
                + summarized
                + "\nRest of conversation follows:"
            ),
        )
        if drop_message_stats.messages_to_drop > 1:
            await self.chat_history.delete_range(1, drop_message_stats.messages_to_drop)
        return True

    async def add_message(self, message: Message) -> None:
        await self.chat_history.add_message(message)
        await self._maybe_summarize()

    async def get_messages(self) -> list[Message]:
        await self._maybe_summarize()
        return await self.chat_history.get_messages()
