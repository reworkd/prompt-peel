class PromptError(Exception):
    """
    Base class for all exceptions raised by this module.
    """

    pass


class PriorityError(PromptError):
    """
    Raised when an invalid priority is encountered.
    """

    pass


class InsufficientChildrenError(PromptError):
    pass


class InvalidPromptError(PromptError):
    pass


class UnknownNodeError(PromptError):
    """
    Raised when an unknown node is encountered during traversal.
    """

    pass
