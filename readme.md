# agent_lite
me/agent_lite)
A lightweight and customizable library for building agentive behavior using Large Language Models (LLMs) in a type-safe and maintable way.

## Description

agent_lite is a minified version of the Langchain library, designed to provide a simple and easy-to-edit solution for building agentive behavior using LLMs. While Langchain is a full-featured and generic library, agent_lite focuses on a single core functionality: allowing users to build an agent that can use an LLM like GPT-4 to maintain a conversation while being able to optionally invoke a sequence of tools defined in Python to better achieve its task.

The library is intentionally kept small, making it easy to edit and customize according to your specific needs. If you require a full-featured library, maybe go use langchain. However, if you want agentive behavior in a simple, type-safe manner with a small and manageable codebase, agent_lite is the perfect choice.

## Usage

Here's an example of how to use agent_lite to build an agent with two tools: `GetCurrentTimeTool` and `GetBitcoinPriceTool`.

```python
from agent_lite.core import BaseTool, InMemmoryChatHistory, UnlimitedMemory
from agent_lite.core.agent import Agent
from agent_lite.impl.llms.openai_llm import OpenAILLM

import asyncio
import os
from datetime import datetime
from enum import Enum
from typing import Type

import httpx
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

class GetCurrentDateTimeToolArgs(BaseModel):
    timezone: str = Field(
        description="Timezone to get the current time. e.g. EUROPE/LONDON",
    )

class GetCurrentTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "Get the current time in the specified timezone"
    args_schema: Type[BaseModel] = GetCurrentDateTimeToolArgs

    async def _arun(self, args: GetCurrentDateTimeToolArgs) -> str:
        now = datetime.now(ZoneInfo(args.timezone))
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
    # output: The current time in Marbella is Friday, 29 March 2024, 12:56:54.
    
    response = await agent.submit_message("What is the price of Bitcoin in GBP?")
    print(response.final_response)
    # output: The current price of Bitcoin in British Pound Sterling (GBP) is £55,628.76.

if __name__ == "__main__":
    asyncio.run(main())
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/yourusername/agent_lite).