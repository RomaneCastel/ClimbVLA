"""Shared data format for every ClimbVLA stage.

Coordinate convention, used by every model in this file:
    - Wall plane, in meters.
    - Origin at the bottom-left of the filmed wall area.
    - x points right, y points up (not down like image pixels).
"""

from enum import StrEnum

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class Route(BaseModel):
    """One boulder problem: its holds, where it starts and where it ends."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="e.g. 'gym-wall2-yellow-01'.")
    color: str | None = Field(default=None, description="Color of the route's holds.")
    grade: str | None = Field(default=None, description="As given by the gym, e.g. 'V3' or '6a'.")
    wall_angle_deg: float = Field(
        default=0.0,
        ge=-20,
        le=70,
        description="0 = vertical, positive = overhanging, negative = slab.",
    )
    holds: list[Hold] = Field(min_length=1)
    start_hold_ids: list[str] = Field(
        min_length=1, max_length=4, description="Holds touched at the start."
    )
    top_hold_id: str = Field(description="The finish hold.")

    @model_validator(mode="after")
    def check_hold_references(self) -> Self:
        """Checks that need several fields at once, run after the field checks."""
        ids = [h.id for h in self.holds]

        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate hold ids: {duplicates}")

        unknown = [i for i in [*self.start_hold_ids, self.top_hold_id] if i not in ids]
        if unknown:
            raise ValueError(f"start/top hold ids not in holds: {unknown}")

        if self.top_hold_id in self.start_hold_ids:
            raise ValueError("the top hold cannot also be a start hold")

        return self

    def hold(self, hold_id: str) -> Hold:
        """Look up a hold by id."""
        for h in self.holds:
            if h.id == hold_id:
                return h
        raise KeyError(hold_id)
