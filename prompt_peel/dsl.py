import sys
from enum import Enum
from typing import Union

from prompt_peel.exceptions import PriorityError
from prompt_peel.lib import Chain
from prompt_peel.node import (
    ChatNode,
    EmptyNode,
    MinKNode,
    NodeType,
    NonChatNode,
    ScopeNode,
    TopKNode,
)

"""
DSL for building prompt chains inspired by JSX patterns.
See README or test for examples on how to construct chains via this DSL.
"""


def system_message(*children: NonChatNode, priority: int = sys.maxsize) -> ChatNode:
    return {
        "type": NodeType.CHAT,
        "role": "system",
        "priority": priority,
        "children": with_validated_priority(priority, children),
    }


def user_message(*children: NonChatNode, priority: int = sys.maxsize) -> ChatNode:
    return {
        "type": NodeType.CHAT,
        "role": "user",
        "priority": priority,
        "children": with_validated_priority(priority, children),
    }


def assistant_message(*children: NonChatNode, priority: int = sys.maxsize) -> ChatNode:
    return {
        "type": NodeType.CHAT,
        "role": "assistant",
        "priority": priority,
        "children": with_validated_priority(priority, children),
    }


def scope(*children: NonChatNode, priority: int = sys.maxsize) -> ScopeNode:
    return {
        "type": NodeType.SCOPE,
        "priority": priority,
        "children": with_validated_priority(priority, children),
    }


def top_k(
    *children: NonChatNode, top_k_value: int, priority: int = sys.maxsize
) -> TopKNode:
    """
    Ensure there are a MAXIMUM of `top_k_value` children in the output
    Children will be sorted based on their priority
    """
    return {
        "type": NodeType.TOP_K,
        "priority": priority,
        "top_k": top_k_value,
        "children": with_validated_priority(priority, children),
    }


def min_k(
    *children: NonChatNode, min_k_value: int, priority: int = sys.maxsize
) -> MinKNode:
    """
    Ensure there are a MINIMUM of `min_k_value` children in the output
    Children will be sorted based on their priority
    """
    return {
        "type": NodeType.MIN_K,
        "priority": priority,
        "min_k": min_k_value,
        "children": with_validated_priority(priority, children),
    }


def empty(tokens: int, priority: int = sys.maxsize) -> EmptyNode:
    return {
        "type": NodeType.EMPTY,
        "priority": priority,
        "tokens": tokens,
    }


MessageBuilder = Union[ChatNode, EmptyNode]


def with_validated_priority(
    parent_priority: int, children: tuple[NonChatNode, ...]
) -> list[NonChatNode]:
    # It is a strange pattern to have children with higher priorities than the parent
    # If the child has a default priority, we will set it to the priority of the parent
    for child in children:
        if not isinstance(child, str) and child["priority"] == sys.maxsize:
            child["priority"] = parent_priority

    # If the child has a hardcoded priority, throw an exception
    if any(
        not isinstance(child, str) and child["priority"] > parent_priority
        for child in children
    ):
        raise PriorityError("Children cannot have higher priority than parent")

    # Otherwise, we are good to go
    return list(children)


def peel(*prompt_element_builder: MessageBuilder) -> Chain:
    return Chain(list(prompt_element_builder))


class TokenSpace(Enum):
    """
    Approximations for safety
    """

    GPT_4 = 8_000
    GPT_4_32_K = 32_000
    GPT_4_TURBO = 120_000

    EIGHT_K = 8_000
    TWELVE_K = 12_000
    THIRTY_FOUR_K = 34_000  # Some leeway for empty space
    SIXTEEN_K = 16_000

    CLAUDE_3 = 200_000
