# import some core agent types
# now for our base tool imports
import asyncio
import io
import os
from datetime import datetime
from enum import Enum
from typing import Type

import httpx
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from agent_lite.core import (
    BaseTool,
    ImageContent,
    ImageFileType,
    InMemoryChatHistory,
    TextContent,
    UnlimitedMemory,
)
from agent_lite.core.agent import Agent

# now we choose some specific implementations, in this case the AnthropicLLM
# You also have choices for different memory strategies, and different storage strategies for chat history
from agent_lite.impl.llms.anthropic_llm import AnthropicLLM

# ----------
# We will define two tools:
# 1. GetCurrentTimeTool
# 2. GetBitcoinPriceTool


class GetCurrentDateTimeToolArgs(BaseModel):
    timezone: str = Field(
        description="Timezone to get the current time. e.g. EUROPE/LONDON",
    )


class GetCurrentTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "Get the current time in the specified timezone"
    args_schema: Type[BaseModel] = GetCurrentDateTimeToolArgs
    description

    async def _arun(self, args: GetCurrentDateTimeToolArgs) -> str:
        now = datetime.now(ZoneInfo(args.timezone))
        # return day name along with date and time
        return now.strftime("%A, %d %B %Y %H:%M:%S")


class Currency(Enum):
    USD = "United States Dollar"
    GBP = "British Pound Sterling"
    EUR = "Euro"


class GetBitcoinPriceToolArgs(BaseModel):
    currency: Currency = Field(description="Currency to get the price in. e.g. USD")


class GetBitcoinPriceTool(BaseTool):
    name: str = "get_bitcoin_price"
    description: str = "Get the current price of Bitcoin in the specified currency"
    args_schema: Type[BaseModel] = GetBitcoinPriceToolArgs

    async def _arun(self, args: GetBitcoinPriceToolArgs) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.coindesk.com/v1/bpi/currentprice.json"
            )
            data = response.json()
            return data["bpi"][args.currency.name]["rate"]


async def main():
    # choose a chat history
    chat_history = InMemoryChatHistory()
    # Othe options:
    # from agent_lite.impl.postgres_chat_history import PostgresChatHistory

    # Choose a memory
    memory = UnlimitedMemory(chat_history=chat_history)
    # Other memory options include BufferedMemory  or BufferedMemoryWithSummarizer:
    # from agent_lite.impl.bufferred_memory import BufferedMemory, BufferedMemoryWithSummarizer

    # Choose an LLM:
    llm = AnthropicLLM(
        api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-3-opus-20240229"
    )
    # Other options:
    # from agent_lite.impl.llms.openai_llm import OpenAILLM

    agent = Agent(
        system_prompt="You are a helpful agent.",
        memory=memory,
        tools=[GetCurrentTimeTool(), GetBitcoinPriceTool()],
        llm=llm,
    )
    image_file = await get_example_image_as_file()
    response = await agent.submit_message(
        [
            TextContent(
                text="First get me the time in Marbella, then get me the current price of Bitcoin in usd, then print the answers followed by a very short one line description of the following image:"
            ),
            ImageContent.from_file(image_file, ImageFileType.JPEG, cached=True),
        ]
    )
    print(response.final_response)
    # output
    # final_response: The time in Marbella is Monday, 26 August 2024 17:27:00.
    #
    # The current price of Bitcoin in USD is $63,304.895.
    #
    # A curious white cat with beautiful multicolored eyes.


async def get_example_image_as_file():
    print("Downloading example image...")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/June_odd-eyed-cat.jpg/320px-June_odd-eyed-cat.jpg"
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        response.raise_for_status()
        print("Downloaded example image.")
        return io.BytesIO(response.content)


if __name__ == "__main__":
    asyncio.run(main())
