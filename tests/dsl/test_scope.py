from typing import Callable

from tests.utils import parameterized_messages

from prompt_peel.dsl import peel, scope, top_k
from prompt_peel.message import Role
from prompt_peel.node import ChatNode


@parameterized_messages
def test_empty_scope(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(scope()),
    ).render()

    assert actual == [{"role": expected_role, "content": ""}]


@parameterized_messages
def test_text_in_scope(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    text = "Hello World. This is scope text."
    actual = peel(
        message_function(scope(text)),
    ).render()

    assert actual == [{"role": expected_role, "content": text}]


@parameterized_messages
def test_wrapped_scope(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    text = "Hello World. This is scope text."
    actual = peel(
        message_function(
            "Start ",
            scope(text),
            " End",
        )
    ).render()

    assert actual == [{"role": expected_role, "content": f"Start {text} End"}]


@parameterized_messages
def test_nested_scopes(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            "Start ",
            scope("Middle ", scope("Inner")),
            " End",
        )
    ).render()

    assert actual == [{"role": expected_role, "content": "Start Middle Inner End"}]


@parameterized_messages
def test_top_k_scopes(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            top_k(
                scope("1"),
                scope("2"),
                scope("3"),
                top_k_value=2,
            )
        )
    ).render()

    assert actual == [{"role": expected_role, "content": "12"}]
