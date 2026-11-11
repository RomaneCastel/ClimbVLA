"""Shared data format for every ClimbVLA stage.

Coordinate convention, used by every model in this file:
    - Wall plane, in meters.
    - Origin at the bottom-left of the filmed wall area.
    - x points right, y points up (not down like image pixels).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class HoldType(StrEnum):
    """Kind of hold. Filled in by the hold classifier."""

    CRIMP = "crimp"
    SLOPER = "sloper"
    JUG = "jug"
    PINCH = "pinch"
    VOLUME = "volume"
    FOOTHOLD = "foothold"


class Hold(BaseModel):
    """One hold on the wall."""

    # Reject unknown fields, so a typo like "raduis" fails loudly
    # instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Unique within a route, e.g. 'h03'.")
    x: float = Field(description="Center, meters from the left edge.")
    y: float = Field(description="Center, meters above the bottom edge.")
    radius: float = Field(
        gt=0,
        le=1.0,
        description="Contact radius in meters: a hand or foot closer than this is 'on' the hold.",
    )
    color: str | None = Field(default=None, description="Route color, e.g. 'yellow'.")
    hold_type: HoldType | None = Field(
        default=None, description="Unknown (None) until the classifier has run."
    )
