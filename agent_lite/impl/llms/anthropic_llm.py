import json
import xml.dom.minidom
from dataclasses import dataclass
from typing import Any

import anthropic
import xmltodict
from anthropic._types import NOT_GIVEN
from anthropic.types import MessageParam
from pydantic_xml import BaseXmlModel, element

from agent_lite.core import (
    AssistantMessage,
    BaseLLM,
    BaseTool,
    LLMResponse,
    LLMUsage,
    Message,
    SystemMessage,
    ToolInvokation,
    ToolInvokationMessage,
    ToolResponseMessage,
    UserMessage,
)

tool_prompt_prefix = """In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>


Here are the tools available:
"""

JSON_PROMPT = """You must output a valid JSON object.\n"""


@dataclass
class AnthropicLLM(BaseLLM):
    api_key: str
    encourage_json_response: bool = False

    def __post_init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
        )

    async def run(self, messages: list[Message], tools: list[BaseTool]) -> LLMResponse:
        if self.encourage_json_response and tools:
            raise ValueError(
                "Currently, the encourage_json_response option prompts Anthropic in a way that disallows the use of tools. This is a limitation of the current implementation of the encourage_json_response option. Please disable the encourage_json_response option if you want to use tools."
            )

        # system message
        system: str | None = None
        first_message_index = 0
        if isinstance(messages[0], SystemMessage):
            system = messages[0].content
            first_message_index = 1
        messages = messages[first_message_index:]

        # JSON mode
        if self.encourage_json_response:
            system = (system or "") + "\n\n" + JSON_PROMPT
            messages = messages + [AssistantMessage(content="{")]

        # Tools
        if tools:
            encoded_tools = AnthropicLLM.encode_tools(tools)
            system = (system or "") + "\n\n" + tool_prompt_prefix + encoded_tools

        # Messages:
        encoded_messages = AnthropicLLM.encode_messages(messages)

        response = await self.client.messages.create(
            max_tokens=4096,
            model=self.model,
            messages=encoded_messages,
            system=system or NOT_GIVEN,
        )
        llm_usage = (
            LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            if response.usage
            else None
        )
        response_text = response.content[-1].text
        print("Model raw response:\n" + response_text)
        if "<function_calls>" in response_text:
            return AnthropicLLM.parse_function_calls(response_text), llm_usage
        elif self.encourage_json_response:
            object_close = response_text.rfind("}")
            if object_close == -1:
                raise ValueError(
                    "The model did not respond with a valid JSON object whilst encourage_json_response is set to true. Anthropic LLM wrapper has not implemented a way to handle this case. Please disable the encourage_json_response option. Please file an issue."
                )
            return (
                AssistantMessage(content="{" + response_text[: object_close + 1]),
                llm_usage,
            )

        else:
            return (AssistantMessage(content=response_text), llm_usage)

    @staticmethod
    def parse_function_calls(response_text: str) -> ToolInvokationMessage:
        FUNCTION_CALLS_START_TAG, FUNCTION_CALLS_END_TAG = (
            "<function_calls>",
            "</function_calls>",
        )
        xml_str = response_text[
            response_text.find(FUNCTION_CALLS_START_TAG) : response_text.find(
                FUNCTION_CALLS_END_TAG
            )
            + len(FUNCTION_CALLS_END_TAG)
        ]
        parsed = xmltodict.parse(xml_str)
        invokations = parsed["function_calls"]["invoke"]
        if not isinstance(invokations, list):
            invokations = [invokations]
        return ToolInvokationMessage(
            raw_content=xml_str,
            tools=[
                ToolInvokation(
                    id="",
                    tool_name=invoke["tool_name"],
                    tool_params=json.dumps(invoke["parameters"]),
                )
                for invoke in invokations
            ],
        )

    @staticmethod
    def encode_messages(
        messages: list[Message],
    ) -> list[MessageParam]:
        def encode_message(
            message: Message | list[ToolResponseMessage],
        ) -> MessageParam:
            if isinstance(message, SystemMessage):
                raise ValueError(
                    "System messages are not supported, Any initial system messages should have automatically been removed and moved to anthropic's api system parameter by this point."
                )
            elif isinstance(message, UserMessage):
                return {"role": "user", "content": message.content}
            elif isinstance(message, AssistantMessage):
                return {"role": "assistant", "content": message.content}
            elif isinstance(message, ToolInvokationMessage):
                assert message.raw_content is not None
                return {"role": "assistant", "content": message.raw_content}
            elif (
                isinstance(message, list)
                and len(message) > 0
                and isinstance(message[0], ToolResponseMessage)
            ):
                response = AnthropicToolResponseContainer(
                    results=[
                        AnthropicToolResponse(
                            tool_name=m.tool_name,
                            stdout=m.content,
                        )
                        for m in message
                    ]
                )
                xml_str = response.to_xml()
                if isinstance(xml_str, bytes):
                    xml_str = xml_str.decode("utf-8")
                return {
                    "role": "user",
                    "content": AnthropicLLM.format_xml_str(xml_str),
                }
            else:
                raise ValueError(f"Unsupported message type for this model: {message}")

        grouped_messages = AnthropicLLM.group_consecutive_tool_response_messages(
            messages
        )
        return [encode_message(m) for m in grouped_messages]

    @staticmethod
    def encode_tools(
        tools: list[BaseTool],
    ) -> str:
        def rewrite_json_schema_param_type(json_schema_property: dict[str, Any]) -> str:
            if "anyOf" in json_schema_property:
                return " | ".join(
                    rewrite_json_schema_param_type(sub_property)
                    for sub_property in json_schema_property["anyOf"]
                )
            if json_schema_property["type"] == "string":
                return "str"
            elif json_schema_property["type"] == "number":
                return "float"
            elif json_schema_property["type"] == "integer":
                return "int"
            elif json_schema_property["type"] == "boolean":
                return "bool"
            elif json_schema_property["type"] == "array":
                return (
                    "list["
                    + rewrite_json_schema_param_type(json_schema_property["items"])
                    + "]"
                )
            elif json_schema_property["type"] == "null":
                return "null"
            else:
                raise ValueError(
                    f"Unsupported json schema type: {json_schema_property['type']}"
                )

        def encode_tool(tool: BaseTool) -> AnthropicXMLTool:
            params_json_schema = tool.get_args_schema().model_json_schema()
            params = [
                AnthropicXMLToolParam(
                    name=param_name,
                    type=rewrite_json_schema_param_type(param_properties),
                    description=(
                        "[required]"
                        if param_name in params_json_schema.get("required", [])
                        else "[optional]"
                    )
                    + " "
                    + param_properties["description"],
                )
                for param_name, param_properties in params_json_schema[
                    "properties"
                ].items()
            ]
            return AnthropicXMLTool(
                tool_name=tool.get_name(),
                description=tool.get_description(),
                parameters=params,
            )

        xml = AnthropicXMLToolContainer(tools=[encode_tool(t) for t in tools]).to_xml()
        if isinstance(xml, bytes):
            xml_str = xml.decode("utf-8")
        else:
            xml_str = xml
        return AnthropicLLM.format_xml_str(xml_str)

    @staticmethod
    def group_consecutive_tool_response_messages(
        messages: list[Message],
    ) -> list[Message | list[ToolResponseMessage]]:
        grouped_messages: list[Message | list[ToolResponseMessage]] = []
        current_group: list[ToolResponseMessage] = []
        for message in messages:
            if isinstance(message, ToolResponseMessage):
                current_group.append(message)
            else:
                if current_group:
                    grouped_messages.append(current_group)
                    current_group = []
                grouped_messages.append(message)
        if current_group:
            grouped_messages.append(current_group)
        return grouped_messages

    @staticmethod
    def format_xml_str(xml_str: str) -> str:
        dom = xml.dom.minidom.parseString(xml_str)
        xml_str = dom.toprettyxml()
        return xml_str[xml_str.find("\n") + 1 :]


class AnthropicXMLToolParam(BaseXmlModel):
    name: str = element()
    type: str = element()
    description: str = element()


class AnthropicXMLTool(BaseXmlModel):
    tool_name: str = element()
    description: str = element()
    parameters: list[AnthropicXMLToolParam] = element()


class AnthropicXMLToolContainer(BaseXmlModel, tag="tools"):
    tools: list[AnthropicXMLTool] = element(tag="tool_description")


class AnthropicToolResponse(BaseXmlModel):
    tool_name: str = element()
    stdout: str = element()


class AnthropicToolResponseContainer(BaseXmlModel, tag="function_results"):
    results: list[AnthropicToolResponse] = element(tag="result")
