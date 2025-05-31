"""
schema.py
"""

from typing import Literal, List

from pydantic import BaseModel, Field, field_validator


class DialogueItem(BaseModel):
    speaker: str
    text: str


class ShortDialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str = ""
    name_of_guest: str
    dialogue: List[DialogueItem] = Field(
        ..., description="A list of dialogue items, typically between 11 to 17 items"
    )


class MediumDialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str = ""
    name_of_guest: str
    dialogue: List[DialogueItem] = Field(
        ..., description="A list of dialogue items, typically between 19 to 29 items"
    )
