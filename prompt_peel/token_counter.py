from abc import ABC, abstractmethod

from tiktoken import get_encoding

from prompt_peel.message import ChatMessage


class TokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        pass

    def count_prompt(self, prompt: list[ChatMessage]) -> int:
        return sum([self.count(message["content"]) for message in prompt])


class Cl100kBaseTokenCounter(TokenCounter):
    def __init__(self) -> None:
        self.encoding = get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> list[int]:
        return self.encoding.encode(text)
