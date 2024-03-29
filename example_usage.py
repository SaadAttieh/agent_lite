#import some core agent types
from agent_lite.core import BaseTool, InMemmoryChatHistory, UnlimitedMemory
from agent_lite.core.agent import Agent
#now we choose some specific implementations, in this case the OpenAILLM
#You also have choices for different memory strategies, and different storage strategies for chat history
from agent_lite.impl.llms.openai_llm import OpenAILLM

#now for our base tool imports
import asyncio
import os
from datetime import datetime
from enum import Enum
from typing import Type

import httpx
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo


#----------
#We will define two tools:
#1. GetCurrentTimeTool
#2. GetBitcoinPriceTool

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
    agent = Agent(
        system_prompt="You are a helpful agent.",
        memory=UnlimitedMemory(chat_history=InMemmoryChatHistory()),
        tools=[GetCurrentTimeTool(), GetBitcoinPriceTool()],
        llm=OpenAILLM(
            api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-turbo-preview"
        ),
    )
    response = await agent.submit_message("What is the time in Marbella?")
    print(response.final_response)
    # you can also print prompt_tokens, completion_tokens, number_llm_invocations, total_time, llm_time and more
    response = await agent.submit_message("What is the price of Bitcoin in GBP?")
    print(response.final_response)


if __name__ == "__main__":
    asyncio.run(main())
