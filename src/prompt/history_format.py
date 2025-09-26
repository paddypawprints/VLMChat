from enum import Enum

class HistoryFormat(Enum):
    """
    Available conversation history formatting options.

    Defines the different ways conversation history can be formatted
    for inclusion in prompts sent to the language model.
    """
    XML = "xml"        # XML-structured format with tags
    MINIMAL = "minimal"  # Condensed format with abbreviations
