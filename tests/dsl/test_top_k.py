from typing import Callable

from tests.utils import parameterized_messages

from prompt_peel.dsl import peel, scope, top_k
from prompt_peel.message import Role
from prompt_peel.node import ChatNode


@parameterized_messages
def test_empty_top_k(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k(top_k_value=10),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "",
        }
    ]
    assert actual == expected


@parameterized_messages
def test_single_child(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k("1", top_k_value=5),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "1",
        }
    ]
    assert actual == expected


@parameterized_messages
def test_0_top_k(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k("1", top_k_value=0),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "",
        }
    ]
    assert actual == expected


@parameterized_messages
def test_multiple_children(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k(*[str(i + 1) for i in range(10)], top_k_value=5),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": str("12345"),
        }
    ]
    assert actual == expected


@parameterized_messages
def test_top_k_sorting(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k(
                scope("1", priority=0),
                scope("2", priority=39),
                scope("3", priority=999),
                scope("4", priority=10),
                scope("5", priority=500),
                top_k_value=4,
            ),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "3524",
        }
    ]
    assert actual == expected


@parameterized_messages
def test_correct_priority_used(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k(
                scope("\n1 ", priority=10),
                scope("\n2 ", priority=10),
                scope("\n3 ", priority=200),
                scope("\n4 ", priority=20),
                scope("\n5 ", priority=100),
                scope("\n6 ", priority=300),
                scope("\n7 ", priority=200),
                top_k_value=2,
            ),
        )
    ).render(4)
    expected = [
        {
            "role": expected_role,
            "content": "6 \n3",
        }
    ]
    assert actual == expected
