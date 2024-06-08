import pytest
import tiktoken

from prompt_peel.token_counter import Cl100kBaseTokenCounter

encoding = tiktoken.get_encoding("cl100k_base")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        (" ", 1),
        ("Hello my name is asim", 6),
        (
            """This is a really long paragraph. Some say it's way too long innit. !@#$%^&*())

Well what does it mean to be too long even? Right now we're just at a mere 46 tokens. You could pass this off as two paragraphs but is it really? What is a paragraph really? Sometimes it's just a sentence. That feels too short. Other times it runs on and on and you wonder whether it will ever end or if the writer simply forgot that paragraphs exist. 

Oh well. I guess it's long enough now""",
            114,
        ),
    ],
)
def test_count(text: str, expected: int) -> None:
    assert Cl100kBaseTokenCounter().count(text) == expected
