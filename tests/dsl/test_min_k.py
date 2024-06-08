from typing import Callable

import pytest
from tests.utils import parameterized_messages

from prompt_peel.dsl import min_k, peel, scope
from prompt_peel.exceptions import InsufficientChildrenError
from prompt_peel.message import Role
from prompt_peel.node import ChatNode


@parameterized_messages
def test_empty_min_k(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            min_k(min_k_value=0),
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
            min_k("1", min_k_value=1),
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
def test_multiple_min_children(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            min_k(*[str(i + 1) for i in range(5)], min_k_value=3),
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
def test_missing_children(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    with pytest.raises(InsufficientChildrenError):
        peel(
            message_function(
                min_k(
                    "1",
                    "2",
                    min_k_value=3,
                ),
            )
        ).render()


@parameterized_messages
def test_insufficient_priority_children(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    with pytest.raises(InsufficientChildrenError):
        peel(
            message_function(
                min_k(
                    scope("0 0", priority=30),
                    scope("1 1", priority=20),
                    scope("2 2", priority=10),
                    scope("3 3", priority=0),
                    scope("4 4", priority=0),
                    min_k_value=2,
                ),
            )
        ).render(3)


@parameterized_messages
def test_min_k_sorting(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            min_k(
                scope("1", priority=0),
                scope("2", priority=39),
                scope("3", priority=999),
                scope("4", priority=10),
                scope("5", priority=500),
                min_k_value=3,
            ),
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "35241",
        }
    ]
    assert actual == expected


@parameterized_messages
def test_correct_priority_used(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            min_k(
                scope("\n1 ", priority=10),
                scope("\n2 ", priority=10),
                scope("\n3 ", priority=200),
                scope("\n4 ", priority=20),
                scope("\n5 ", priority=100),
                scope("\n6 ", priority=300),
                scope("\n7 ", priority=200),
                scope("\n8 ", priority=400),
                min_k_value=2,
            ),
        )
    ).render(4)
    expected = [
        {
            "role": expected_role,
            "content": "8 \n6",
        }
    ]
    assert actual == expected
