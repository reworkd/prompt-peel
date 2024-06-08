from typing import Literal, TypedDict, Union

Role = Union[Literal["system"], Literal["user"], Literal["assistant"]]


class ChatMessage(TypedDict):
    role: Role
    content: str
