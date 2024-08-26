import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import IO, Any, AsyncIterator, List, Type

from pydantic import BaseModel, Field


class Message(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def message_type(self) -> str:
        pass

    def serialized_message_content(self) -> str:
        return self.json()

    @classmethod
    def from_serialized(cls, message_type: str, serialized_content: str) -> "Message":
        all_message_types = [
            UserMessage,
            SystemMessage,
            AssistantMessage,
            ToolInvokationMessage,
            ToolResponseMessage,
        ]
        for message_type_class in all_message_types:
            if message_type_class.message_type() == message_type:  # type: ignore
                return message_type_class.model_validate_json(serialized_content)  # type: ignore
        raise ValueError(f"Unknown message type: {message_type}")


class ContentProperties(BaseModel):
    cached: bool = Field(False, exclude=True)


class TextContent(ContentProperties):
    text: str


class ImageFileType(str, Enum):
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"


class ImageContent(ContentProperties):
    encoded_data: str
    file_type: ImageFileType

    def encoding_type(self) -> str:
        return "base64"

    @staticmethod
    def from_file(
        file: IO[bytes], file_type: ImageFileType, cached: bool = False
    ) -> "ImageContent":
        data = file.read()
        encoded_data = base64.b64encode(data).decode("utf-8")
        return ImageContent(
            encoded_data=encoded_data, file_type=file_type, cached=cached
        )


def to_file(self, file: IO[bytes]) -> None:
    data = base64.b64decode(self.encoded_data)
    file.write(data)


Content = TextContent | ImageContent


class UserMessage(Message):
    content: str | list[Content]

    @classmethod
    def message_type(self) -> str:
        return "user"


class SystemMessage(Message):
    content: str | list[Content]

    @classmethod
    def message_type(self) -> str:
        return "system"


class AssistantMessage(Message):
    content: str

    @classmethod
    def message_type(self) -> str:
        return "assistant"


class ToolInvokation(BaseModel):
    id: str
    tool_name: str
    tool_params: str


class ToolInvokationMessage(Message):
    tools: list[ToolInvokation]
    raw_content: str | None = None

    @classmethod
    def message_type(self) -> str:
        return "tool_invokation"


class ToolResponseMessage(Message):
    tool_invokation_id: str
    tool_name: str
    content: str

    @classmethod
    def message_type(self) -> str:
        return "tool_response"


class StreamingAssistantMessage(AssistantMessage):
    internal_stream: AsyncIterator[str]
    content: str = ""

    async def stream(self) -> AsyncIterator[str]:
        async for chunk in self.internal_stream:
            self.content += chunk
            yield chunk

    async def consume_stream(self) -> str:
        async for _ in self.stream():
            pass
        return self.content


class ToolDirectResponse(AssistantMessage):
    pass


class StreamingToolDirectResponse(StreamingAssistantMessage):
    pass


class LLMUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


LLMResponse = tuple[
    AssistantMessage | ToolInvokationMessage | StreamingAssistantMessage,
    LLMUsage | None,
]

StreamingLLMResponse = ToolInvokationMessage | StreamingAssistantMessage


class NoArgs(BaseModel):
    pass


class BaseTool(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self) -> None:
        cls = self.__class__

        required_fields = {
            "name": (str, True),
            "description": (str, True),
            "args_schema": (BaseModel, False),
        }

        for field_name, (expected_type, required) in required_fields.items():
            if not hasattr(self, field_name):
                raise AttributeError(
                    f"'{cls.__name__}' is missing the required field '{field_name}'"
                )

            actual_value = getattr(self, field_name)
            if actual_value is None and required:
                raise AttributeError(
                    f"The field '{field_name}' in '{cls.__name__}' must not be None"
                )
            if (
                actual_value is not None
                and not isinstance(actual_value, expected_type)
                and not issubclass(actual_value, expected_type)
            ):
                raise TypeError(
                    f"The field '{field_name}' in '{cls.__name__}' must be of type '{expected_type.__name__}', "
                    f"got '{type(actual_value).__name__}' instead"
                )

    def get_name(self) -> str:
        return self.name  # type: ignore

    def get_description(self) -> str:
        return self.description  # type: ignore

    def get_args_schema(self) -> Type[BaseModel]:
        return self.args_schema or NoArgs  # type: ignore

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        pass


@dataclass
class BaseLLM(ABC):
    model: str

    @abstractmethod
    async def run(self, messages: list[Message], tools: list[BaseTool]) -> LLMResponse:
        pass

    # define abstract method run_stream with default to raise nontimplementederror
    async def run_stream(
        self, messages: list[Message], tools: list[BaseTool]
    ) -> StreamingLLMResponse:
        raise NotImplementedError()


class BaseChatHistory(ABC):
    @abstractmethod
    async def add_message(self, message: Message) -> None:
        pass

    async def delete_message(self, index: int) -> None:
        pass

    async def edit_message(self, index: int, message: Message) -> None:
        pass

    @abstractmethod
    async def get_messages(self) -> List[Message]:
        pass

    async def delete_range(self, start: int, end: int) -> None:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass


@dataclass
class BaseMemory(ABC):
    chat_history: BaseChatHistory

    def get_chat_history(self) -> BaseChatHistory:
        return self.chat_history

    @abstractmethod
    async def add_message(self, message: Message) -> None:
        pass

    @abstractmethod
    async def get_messages(self) -> list[Message]:
        pass


class InMemoryChatHistory(BaseChatHistory, BaseModel):
    messages: List[Message] = Field(default_factory=list)

    async def add_message(self, message: Message) -> None:
        self.messages.append(message)

    async def delete_message(self, index: int) -> None:
        self.messages.pop(index)

    async def edit_message(self, index: int, message: Message) -> None:
        self.messages[index] = message

    async def get_messages(self) -> List[Message]:
        return self.messages

    async def delete_range(self, start: int, end: int) -> None:
        self.messages = self.messages[:start] + self.messages[end:]

    async def clear(self) -> None:
        self.messages = []


class InMemmoryChatHistory(InMemoryChatHistory):
    pass


class UnlimitedMemory(BaseMemory):
    async def add_message(self, message: Message) -> None:
        await self.chat_history.add_message(message)

    async def get_messages(self) -> list[Message]:
        return await self.chat_history.get_messages()
