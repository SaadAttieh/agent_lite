import asyncio
import os

from agent_lite.core.agent import Agent
from agent_lite.impl.llms.openai_llm import OpenAILLM


async def main():
    # Choose an LLM:
    llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-turbo")
    agent = Agent(
        system_prompt="You are a helpful agent.",
        llm=llm,
    )

    response = await agent.submit_message("Hi are you there?")
    print(response.final_response)


if __name__ == "__main__":
    asyncio.run(main())
