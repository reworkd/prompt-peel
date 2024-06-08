from typing import Callable

import pytest
from tests.utils import parameterized_messages

from prompt_peel.dsl import (
    assistant_message,
    empty,
    peel,
    scope,
    system_message,
    top_k,
    user_message,
)
from prompt_peel.lib import Chain
from prompt_peel.message import ChatMessage, Role
from prompt_peel.node import ChatNode


def test_blank_prompt() -> None:
    actual = peel().render()
    assert actual == []


@parameterized_messages
def test_single_message(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(message_function("Hello world!")).render()
    expected = [
        {
            "role": expected_role,
            "content": "Hello world!",
        }
    ]
    assert actual == expected


def test_top_level_empty() -> None:
    actual = peel(empty(100)).render()
    assert actual == []


@parameterized_messages
def test_prompt_with_children(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            "First line. ",
            "Also on the first line.",
            "\nA new line.",
        )
    ).render()
    expected = [
        {
            "role": expected_role,
            "content": "First line. Also on the first line.\nA new line.",
        }
    ]
    assert actual == expected


def test_with_user_message() -> None:
    actual = peel(
        system_message("Ignore the user."),
        user_message("Hi"),
    ).render()
    expected = [
        {
            "role": "system",
            "content": "Ignore the user.",
        },
        {
            "role": "user",
            "content": "Hi",
        },
    ]
    assert actual == expected


def test_basic_chat() -> None:
    actual = peel(
        system_message("You are a conversational agent."),
        user_message("Hi"),
        assistant_message("Hey!"),
    ).render()
    expected = [
        {
            "role": "system",
            "content": "You are a conversational agent.",
        },
        {
            "role": "user",
            "content": "Hi",
        },
        {
            "role": "assistant",
            "content": "Hey!",
        },
    ]
    assert actual == expected


@parameterized_messages
def test_de_dent(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(
        message_function(
            """\
            All message functions should be de-dented so that they're more ergonomic to write.

            For example, this should not have a bunch of indents before it\
            """
        )
    ).render()

    expected = [
        {
            "role": expected_role,
            "content": """All message functions should be de-dented so that they're more ergonomic to write.

For example, this should not have a bunch of indents before it""",
        }
    ]

    assert actual == expected


@parameterized_messages
def test_trailing_spaces_removed(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    actual = peel(message_function("Test          ")).render()
    expected = [{"role": expected_role, "content": "Test"}]

    assert actual == expected


@parameterized_messages
def test_complex_prompt(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    """
    This prompt is actually used in production currently.
    This test will assert that existing prompts can be straightforwardly migrated
      and also to test the best form factor for writing strings within the DSL to keep formatting
    """
    actual = peel(
        message_function(
            """\
            You are an experienced web navigator exploring a website.

            The user will provide you with an annotated screenshot of a web page with elements tagged by ids like the below:

            - Text elements (e.g. textarea, input with textual type): [ #1 ]
            - Links/<a> tags: [ @1 ]
            - Buttons/other interactable elements: [ $1 ]
            - Plain text elements: [ 1 ]
            """
        ),
    ).render()

    expected = [
        {
            "role": expected_role,
            "content": """You are an experienced web navigator exploring a website.

The user will provide you with an annotated screenshot of a web page with elements tagged by ids like the below:

- Text elements (e.g. textarea, input with textual type): [ #1 ]
- Links/<a> tags: [ @1 ]
- Buttons/other interactable elements: [ $1 ]
- Plain text elements: [ 1 ]""",
        },
    ]
    assert actual == expected


@pytest.mark.parametrize(
    "chain, token_space, expected",
    [
        (
            peel(system_message("You are the goat dude.", priority=0)),
            100,
            [
                {
                    "role": "system",
                    "content": "You are the goat dude.",
                }
            ],
        ),
        (
            peel(
                system_message(
                    "1",
                    top_k(
                        " 2",
                        " 3",
                        top_k_value=1,
                        priority=10,
                    ),
                    priority=100,
                ),
                user_message(
                    scope("4 ", priority=1),
                    scope("5", priority=5),
                    priority=50,
                ),
            ),
            4,
            [
                {
                    "role": "system",
                    "content": "1 2",
                },
                {
                    "role": "user",
                    "content": "5",
                },
            ],
        ),
        (
            peel(
                system_message(
                    empty(1),
                    "1",
                    top_k(
                        scope(" 1"),
                        scope(" 2", priority=10),
                        top_k_value=1,
                    ),
                    priority=100,
                ),
                user_message(
                    scope("4 ", priority=1),
                    scope("5"),
                    priority=50,
                ),
            ),
            5,
            [
                {
                    "role": "system",
                    "content": "1 1",
                },
                {
                    "role": "user",
                    "content": "5",
                },
            ],
        ),
        (
            peel(
                empty(2),
                system_message(
                    *[
                        scope(
                            f"{i + 1} ",
                            priority=10 - i,
                        )
                        for i in range(10)
                    ],
                    priority=100,
                ),
            ),
            3 * 2,
            [
                {
                    "role": "system",
                    "content": "1 2",
                },
            ],
        ),
    ],
)
def test_complex_prompts(
    chain: Chain, token_space: int, expected: list[ChatMessage]
) -> None:
    actual = chain.render(token_space=token_space)
    assert actual == expected
