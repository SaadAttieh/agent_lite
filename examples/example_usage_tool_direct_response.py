# import some core agent types
# now for our base tool imports
import asyncio
import os
from datetime import datetime
from typing import Type

from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from agent_lite.core import (
    BaseTool,
    InMemmoryChatHistory,
    StreamingToolDirectResponse,
    ToolDirectResponse,
    UnlimitedMemory,
)
from agent_lite.core.agent import Agent

# now we choose some specific implementations, in this case the OpenAILLM
# You also have choices for different memory strategies, and different storage strategies for chat history
from agent_lite.impl.llms.openai_llm import OpenAILLM

# ----------
# We will define a tool that forces the agent to respond directly with the result of the tool rather than processing it through the LLM
# tools can always return a ToolDirectResponse or a StreamingToolDirectResponse.
# These will always be coerced into the appropriate response for agent.submit_message(...) or agent.submit_message_and_stream_response(...)


class GetCurrentDateTimeToolArgs(BaseModel):
    timezone: str = Field(
        description="Timezone to get the current time. e.g. EUROPE/LONDON",
    )


class GetCurrentTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "Get the current time in the specified timezone"
    args_schema: Type[BaseModel] = GetCurrentDateTimeToolArgs
    description

    async def _arun(
        self, args: GetCurrentDateTimeToolArgs
    ) -> ToolDirectResponse | StreamingToolDirectResponse:
        now = datetime.now(ZoneInfo(args.timezone))
        # return day name along with date and time
        return ToolDirectResponse(content=now.strftime("%A, %d %B %Y %H:%M:%S"))


async def main():
    # choose a chat history
    chat_history = InMemmoryChatHistory()
    # Othe options:
    # from agent_lite.impl.postgres_chat_history import PostgresChatHistory

    # Choose a memory
    memory = UnlimitedMemory(chat_history=chat_history)
    # Other memory options include BufferedMemory  or BufferedMemoryWithSummarizer:
    # from agent_lite.impl.bufferred_memory import BufferedMemory, BufferedMemoryWithSummarizer

    # Choose an LLM:
    llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-turbo-preview")
    # Other options:
    # from agent_lite.impl.anthropic_llm import AnthropicLLM

    agent = Agent(
        system_prompt="You are a helpful agent.",
        memory=memory,
        tools=[GetCurrentTimeTool()],
        llm=llm,
    )

    response = await agent.submit_message("What is the time in Marbella?")
    print(response.final_response)


if __name__ == "__main__":
    asyncio.run(main())
