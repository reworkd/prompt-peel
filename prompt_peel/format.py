import textwrap


def dedent(text: str) -> str:
    return textwrap.dedent(text).strip()
