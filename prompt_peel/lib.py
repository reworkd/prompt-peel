import sys
import textwrap
from functools import reduce
from typing import List, Set, Union

from prompt_peel.exceptions import (
    InsufficientChildrenError,
    PriorityError,
    UnknownNodeError,
)
from prompt_peel.message import ChatMessage
from prompt_peel.node import ChatNode, EmptyNode, Node, NodeType, NonChatNode, is_type
from prompt_peel.token_counter import Cl100kBaseTokenCounter, TokenCounter

"""
The core logic of the library.
Note we use `# type: ignore` to ignore attribute type errors for TypedDicts. In these cases, we can be certain
    that the attribute exists and is of the correct type (assuming you are using the DSL correctly to generate chains).
"""


class Chain:
    def __init__(
        self,
        prompt_elements: list[Union[ChatNode, EmptyNode]],
        token_counter: TokenCounter = Cl100kBaseTokenCounter(),
    ):
        self.prompt_elements: list[ChatNode] = [
            element  # type: ignore
            for element in prompt_elements
            if is_type(element, NodeType.CHAT)
        ]
        self.empty_parent_elements: list[EmptyNode] = [
            element  # type: ignore
            for element in prompt_elements
            if is_type(element, NodeType.EMPTY)
        ]
        self.token_counter = token_counter

    def render(self, token_space: int = sys.maxsize) -> list[ChatMessage]:
        # 1. Iterate through all prompt elements and build a sorted list of priorities.
        #    These become the candidate priorities that we can binary search through.
        priorities = self.get_priorities()

        # 2. Search through the list of priorities and find the smallest priority that satisfies constraint
        #    We are assuming all context is useful and we want to stuff as much context as possible
        optimal_priority = self.get_optimal_priority(priorities, token_space)

        # 3. Return materialized prompt chain with the optimal priority in a format the OpenAI API understands
        return self.render_priority(optimal_priority)

    def get_priorities(self) -> Set[int]:
        return reduce(
            lambda x, y: x.union(y),
            [self._get_priorities(self.prompt_elements)],
            set(),
        )

    def get_optimal_priority(self, priorities: Set[int], token_space: int) -> int:
        if len(priorities) == 0:
            return 0

        # Start with the lowest priority value and work our way up until we find a priority that fits
        required_token_space = 0
        for priority in sorted(list(priorities)):
            rendered_prompt = self.render_priority(priority)

            prompt_token_count = self.token_counter.count_prompt(rendered_prompt)
            empty_token_count = self._get_empty_tokens(
                self.prompt_elements, priority
            ) + sum([element["tokens"] for element in self.empty_parent_elements])
            required_token_space = prompt_token_count + empty_token_count
            if required_token_space <= token_space:
                return priority

        raise PriorityError(
            f"The minimum required token space is {required_token_space}"
            f" which cannot satisfy the constraint of {token_space} tokens."
            f" Please increase token space or reduce prompt size. Prompt:\n\n"
            f"{self.prompt_elements}."
        )

    def render_priority(self, priority: int) -> list[ChatMessage]:
        return [
            {
                "role": element["role"],
                "content": textwrap.dedent(
                    self._get_content(element, priority)
                ).strip(),  # Strip to emulate JSX formatting
            }
            for element in self.prompt_elements
            if is_type(element, NodeType.CHAT)
        ]

    def _get_priorities(self, child_node: Node) -> Set[int]:
        """DFS on a Node and return a set of all priorities in tree"""
        if isinstance(child_node, List):
            res: Set[int] = set()
            for node in child_node:
                res = res.union(self._get_priorities(node))
            return res

        if isinstance(child_node, str):
            return set()

        if is_type(
            child_node, NodeType.CHAT, NodeType.SCOPE, NodeType.TOP_K, NodeType.MIN_K
        ):
            return {child_node["priority"]}.union(
                self._get_priorities(child_node["children"])  # type: ignore
            )

        if is_type(child_node, NodeType.EMPTY):
            return {child_node["priority"]}  # type: ignore

        raise UnknownNodeError(
            f"Unknown child node type {type(child_node)} - {child_node}"
        )

    def _get_empty_tokens(self, child_node: Node, min_priority: int) -> int:
        """DFS on a Node. Filter lower priorities and count empty tokens in `Empty` nodes"""
        if isinstance(child_node, List):
            return sum(
                [self._get_empty_tokens(node, min_priority) for node in child_node]
            )

        if isinstance(child_node, str) or child_node["priority"] < min_priority:
            return 0

        if is_type(
            child_node, NodeType.CHAT, NodeType.SCOPE, NodeType.TOP_K, NodeType.MIN_K
        ):
            children: list[NonChatNode] = child_node["children"]  # type: ignore
            return sum(
                [self._get_empty_tokens(node, min_priority) for node in children]
            )

        if is_type(child_node, NodeType.EMPTY):
            return child_node["tokens"]  # type: ignore

        raise UnknownNodeError(
            f"Unknown child node type {type(child_node)} - {child_node}"
        )

    def _get_content(
        self, child_node: Union[Node, list[Node]], min_priority: int
    ) -> str:
        """DFS on a Node. Filter lower priorities and return contents as a string"""
        if isinstance(child_node, List):
            return "".join(
                [self._get_content(child, min_priority) for child in child_node]
            )

        if isinstance(child_node, str):
            return child_node

        if child_node["priority"] < min_priority:
            return ""

        if is_type(child_node, NodeType.CHAT, NodeType.SCOPE):
            return self._get_content(child_node["children"], min_priority)  # type: ignore

        if is_type(child_node, NodeType.TOP_K):
            sorted_children = sort_by_priority(
                child_node["children"],  # type: ignore
                child_node["priority"],  # type: ignore
            )  # type: ignore
            return self._get_content(
                sorted_children[: child_node["top_k"]],  # type: ignore
                min_priority,
            )

        if is_type(child_node, NodeType.MIN_K):
            # Also handle when child doesn't have priority key. It should use the current element priority
            min_k = child_node["min_k"]  # type: ignore
            priority = child_node["priority"]  # type: ignore
            sorted_children = sort_by_priority(child_node["children"], priority)  # type: ignore
            filtered_children = [
                child
                for child in sorted_children
                if get_priority(child, priority) >= min_priority
            ]

            if len(filtered_children) < min_k:
                raise InsufficientChildrenError(
                    f"MinK node has {len(filtered_children)} valid children with priority greater than {min_priority}"
                    f" but requires at least {child_node['min_k']}."  # type: ignore
                )

            return self._get_content(sorted_children, min_priority)

        if is_type(child_node, NodeType.EMPTY):  # type: ignore
            return ""

        raise UnknownNodeError(
            f"Unknown child node type {type(child_node)} - {child_node}"
        )


def sort_by_priority(children: list[Node], parent_priority: int) -> list[Node]:
    return sorted(
        children,
        key=lambda x: get_priority(x, parent_priority),
        reverse=True,
    )


def get_priority(node: Node, parent_priority: int) -> int:
    return node["priority"] if "priority" in node else parent_priority  # type: ignore
