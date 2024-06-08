from typing import Callable

from tests.utils import parameterized_messages

from prompt_peel.dsl import peel
from prompt_peel.message import Role
from prompt_peel.node import ChatNode

"""
Tests that cover more complex prompting use cases.
Focus is on building up composite prompt components from simple prompt components akin to React/JSX.
"""


@parameterized_messages
def test_message_mapping(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        *[message_function("Hello world!") for _ in range(5)],
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "Hello world!",
        }
        for _ in range(5)
    ]
    assert actual == expected
