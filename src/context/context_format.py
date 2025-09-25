from enum import Enum

class ContextFormat(Enum):
    """Available context formatting options."""
    XML = "xml"
    MINIMAL = "minimal"
    TEXT_ONLY = "text_only"