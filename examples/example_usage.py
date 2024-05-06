# import some core agent types
# now for our base tool imports
import asyncio
import os
from datetime import datetime
from enum import Enum
from typing import Type

import httpx
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

from agent_lite.core import BaseTool, InMemoryChatHistory, UnlimitedMemory
from agent_lite.core.agent import Agent

# now we choose some specific implementations, in this case the OpenAILLM
# You also have choices for different memory strategies, and different storage strategies for chat history
from agent_lite.impl.llms.openai_llm import OpenAILLM

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
    llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-turbo")
    # Other options:
    # from agent_lite.impl.llms.anthropic_llm import AnthropicLLM

    agent = Agent(
        system_prompt="You are a helpful agent.",
        memory=memory,
        tools=[GetCurrentTimeTool(), GetBitcoinPriceTool()],
        llm=llm,
    )

    response = await agent.submit_message("What is the time in Marbella?")
    print(response.final_response)
    # output: The current time in Marbella is Friday, 29 March 2024, 12:56:54.
    # you can also print prompt_tokens, completion_tokens, number_llm_invocations, total_time, llm_time and more

    response = await agent.submit_message("What is the price of Bitcoin in GBP?")
    print(response.final_response)
    # output: The current price of Bitcoin in British Pound Sterling (GBP) is £55,628.76.


if __name__ == "__main__":
    asyncio.run(main())
