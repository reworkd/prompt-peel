from typing import Callable

from tests.utils import parameterized_messages

from prompt_peel.dsl import empty, peel
from prompt_peel.message import Role
from prompt_peel.node import ChatNode


@parameterized_messages
def test_empty_adds_nothing(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            empty(10),
            empty(2),
            "Begin-",
            empty(100),
            empty(12),
            "-End",
            empty(1000),
            empty(10),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "Begin--End",
        }
    ]
    assert actual == expected
