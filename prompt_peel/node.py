from enum import Enum
from typing import Literal, TypedDict, Union

from prompt_peel.message import Role

NonChatNode = Union[str, "ScopeNode", "TopKNode", "EmptyNode"]
Node = Union[NonChatNode, "ChatNode", list[NonChatNode], list["ChatNode"]]


class NodeType(Enum):
    CHAT = "chat"
    SCOPE = "scope"
    TOP_K = "top_k"
    MIN_K = "min_k"
    EMPTY = "empty"


class NodeBase(TypedDict):
    priority: int


class ParentNode(NodeBase):
    children: list[NonChatNode]


class ChatNode(ParentNode):
    type: Literal[NodeType.CHAT]
    role: Role


class ScopeNode(ParentNode):
    type: Literal[NodeType.SCOPE]
    pass


class TopKNode(ParentNode):
    type: Literal[NodeType.TOP_K]
    top_k: int


class MinKNode(ParentNode):
    type: Literal[NodeType.MIN_K]
    min_k: int


class EmptyNode(NodeBase):
    type: Literal[NodeType.EMPTY]
    tokens: int


def is_type(node: Union[ChatNode, NonChatNode], *desired_types: NodeType) -> bool:
    """
    Similar to TypeScript, we introduce a type attribute to discern between node types
    We can't simply call isinstance as nodes are TypedDict and it isn't supported in python
    """
    if isinstance(node, str):
        return False
    return "type" in node and node["type"] in desired_types
