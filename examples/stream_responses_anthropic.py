import asyncio
import os

from agent_lite.core.agent import Agent
from agent_lite.impl.llms.anthropic_llm import AnthropicLLM


async def main():
    # anthropic does not support tools and streaming together, so we will just try streaming
    # Choose an LLM:
    llm = AnthropicLLM(
        api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-3-opus-20240229"
    )
    agent = Agent(
        system_prompt="You are a helpful agent.",
        llm=llm,
    )

    response = await agent.submit_message_and_stream_response(
        "Tell me a story about a cat."
    )
    async for text in response.final_response_stream:
        print(text, end="")


if __name__ == "__main__":
    asyncio.run(main())
