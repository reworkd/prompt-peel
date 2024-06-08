import pytest

from prompt_peel.format import dedent


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("", ""),
        ("""""", ""),
        ("""Test""", "Test"),
        (
            """
            Test
            """,
            "Test",
        ),
        (
            """
            Hello
            World
            """,
            "Hello\nWorld",
        ),
        ("No leading spaces", "No leading spaces"),
        (
            "    Mixed\n  whitespace\n    Indentation",
            "Mixed\nwhitespace\n  Indentation",
        ),
        ("\tTab\tIndented\n\tLine", "Tab\tIndented\nLine"),
        ("    Only one line with spaces     ", "Only one line with spaces"),
        (
            "\n\n\n   Leading and trailing newlines   \n\n",
            "Leading and trailing newlines",
        ),
    ],
)
def test_dedent(input_string, expected_output):
    assert dedent(input_string) == expected_output
