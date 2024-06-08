from typing import Any, Callable, Tuple

import pytest

from prompt_peel.dsl import assistant_message, system_message, user_message
from prompt_peel.message import Role


def message_dsl_and_role() -> list[Tuple[Callable[..., Any], Role]]:
    return [
        (system_message, "system"),
        (user_message, "user"),
        (assistant_message, "assistant"),
    ]


def parameterized_messages(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Custom decorator that allows us to parametrize tests with the available OpenAI roles as DSL functions.
    You must annotate your test function with the following:
        message_function: Callable[..., ChatNode], expected_role: Role
    """
    return pytest.mark.parametrize(
        "message_function, expected_role", message_dsl_and_role()
    )(func)
