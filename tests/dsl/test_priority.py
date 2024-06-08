import sys
from typing import Callable, Set

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
from prompt_peel.exceptions import PriorityError
from prompt_peel.lib import Chain
from prompt_peel.message import Role
from prompt_peel.node import ChatNode


@pytest.mark.parametrize(
    "chain, priorities",
    [
        (peel(), set()),
        (peel(system_message()), {sys.maxsize}),  # Default value
        (peel(system_message(priority=1)), {1}),  # Override
        (
            # Multiple chat nodes
            peel(
                system_message(priority=1),
                user_message(priority=2),
                assistant_message(),
            ),
            {1, 2, sys.maxsize},
        ),
        (
            # Inner nodes
            peel(
                system_message(
                    scope(priority=1),
                    priority=2,
                )
            ),
            {1, 2},
        ),
        (
            # Duplicates
            peel(
                system_message(
                    scope(priority=1),
                    priority=2,
                ),
                system_message(priority=1),
                system_message(
                    top_k(scope(priority=1), priority=1, top_k_value=1), priority=2
                ),
                system_message(priority=2),
            ),
            {1, 2},
        ),
        (
            # Complex example. Each message will be prefixed with it's index
            # So all priorities in the first message start with a 1 (like 11, 12, 13)
            peel(
                system_message(
                    scope(
                        scope(
                            top_k(
                                "Hello",
                                "World",
                                top_k_value=3,
                                priority=11,
                            ),
                            priority=12,
                        ),
                        priority=13,
                    ),
                    top_k(
                        scope(
                            empty(tokens=1, priority=14),
                            priority=15,
                        ),
                        top_k_value=0,
                        priority=16,
                    ),
                    empty(tokens=1, priority=17),
                    priority=18,
                ),
                user_message(
                    scope(priority=21),
                    top_k(top_k_value=0, priority=22),
                    empty(tokens=1, priority=23),
                    priority=24,
                ),
                assistant_message(
                    scope(priority=31),
                    top_k(top_k_value=0, priority=32),
                    empty(tokens=1, priority=33),
                    priority=34,
                ),
            ),
            {11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 31, 32, 33, 34},
        ),
    ],
)
def test_get_priorities(chain: Chain, priorities: Set[int]) -> None:
    assert chain.get_priorities() == priorities


@pytest.mark.parametrize(
    "chain, token_space, optimal_priority",
    [
        (peel(), 100, 0),
        (peel(system_message()), 0, sys.maxsize),  # Default value
        (peel(system_message(priority=1)), 0, 1),  # Override
        (
            # Prune for lowest priority
            peel(
                system_message("1", priority=1),
                user_message("1", priority=2),
                assistant_message(),
            ),
            1,
            2,
        ),
    ],
)
def test_get_optimal_priority(
    chain: Chain, token_space: int, optimal_priority: int
) -> None:
    assert (
        chain.get_optimal_priority(chain.get_priorities(), token_space)
        == optimal_priority
    )


@pytest.mark.parametrize(
    "chain, token_space",
    [
        (
            peel(
                system_message("1", priority=1),
            ),
            0,
        ),
        (
            peel(
                system_message("1", priority=1),
                user_message("1", priority=1),
                assistant_message("1a", priority=1),
            ),
            3,
        ),
        (
            peel(
                system_message(
                    "1",
                    empty(10),
                    priority=1,
                ),
            ),
            1,
        ),
    ],
)
def test_error_if_prompt_exceeds_space(chain: Chain, token_space: int) -> None:
    with pytest.raises(PriorityError):
        chain.render(token_space=token_space)


def test_error_if_child_has_higher_priority_than_parent() -> None:
    with pytest.raises(PriorityError):
        system_message(
            "1",
            user_message("1", priority=2),
            priority=1,
        )


@parameterized_messages
def test_child_defaults_to_parent_priority_if_not_given(
    message_function: Callable[..., ChatNode], expected_role: Role
) -> None:
    chain = peel(
        message_function(scope("test"), priority=1),
    )

    assert chain.get_priorities() == {1}
