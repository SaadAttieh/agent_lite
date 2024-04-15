# agent_lite

A minified version of the Langchain library, designed to be small enough to easily read and edit but still provide a type-safe interface to building Agents on top of LLMs.

# Why:

Look, Langchain is full featured, its expansive, it has all the bells and whistles. And, most people should just use it.

But, it's huge, and learning how to provide the correct combination of keyword arguments to customise the behaviour consumes far too much of my time.

On the other end, the LLM APIs are quite low level, you need to parse tool invokations, track agent responses, tool outputs and user messages and most of this is done using untyped python dictionaries.

If you want control, a library that is small enough to have a quick read through and edit, but still retaining a typesafe binding from python functions to agent tools, agent_lite is a Decent enough compromise.

# Features:

- The library is split into:
- `core` (defining all the required types and interfaces) and
- `impl` where the differing implementations are provided (different LLMs, chat history storage, memory strategies, etc.).

## Core:

- Tool interface: build your tools as pydantic classes.
- Agent interface:,
  - execute an agent run: `load messages -> submit new input message -> optionall execute one or more tools -> produce final answer`.
  - Get run stats: number LLM invokations (round trips), prompt tokens, completion tokens, total time, time spent waiting for LLMs and more.
- LLM interface, easily add your own LLM implementation or your own strategy for converting messages and tool definitions into prompts.
- Chat history interface, define how chat history is stored.
- Memory interface, define how memory is managed, e.g. when the max number of tokens is exceeded.

## Impl:

- Leverage multiple LLMs, OpenAI and Anthropic currently supported, more to follow.
- Current LLMs are designed to work with finetuned function-calling models, these generally lead to better performance. gpt-3.5-turbo-0125, gpt-4-turbo-preview, claude-3 (haiku, sonnet and opus)
- BufferedMemory strategies provided, auto drop old messages, auto summarise old messages.
- Example postgres Chat History implementation provided

# Usage

## Mini example:
```python
    chat_history = InMemmoryChatHistory()
    memory = UnlimitedMemory(chat_history=chat_history)
    llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-turbo-preview")
    agent = Agent(
        system_prompt="You are a helpful agent.",
        memory=memory,
        tools=[]
    )
    response = await agent.submit_message("Hello")
    print(response.final_response)
```

## Complete example:

Here's an example of how to use agent_lite to build an agent with two tools: `GetCurrentTimeTool` and `GetBitcoinPriceTool`.

```python
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

from agent_lite.core import BaseTool, InMemmoryChatHistory, UnlimitedMemory
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
```

*Streaming responses handled*
```python
    response = await agent.submit_message_and_stream_response("What is the time in Marbella?")
    async for message in response.final_response_stream:
        print(message, end="")
    print()


*Tool direct responses*
```python
# We will define a tool that forces the agent to respond directly with the result of the tool rather than processing it through the LLM
# tools can always return a ToolDirectResponse or a StreamingToolDirectResponse.
# These will always be coerced into the appropriate response for agent.submit_message(...) or agent.submit_message_and_stream_response(...)
from agent_lite.core import (
    ...,
    StreamingToolDirectResponse,
    ToolDirectResponse,
    ...,
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


```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/SaadAttieh/agent_lite).
