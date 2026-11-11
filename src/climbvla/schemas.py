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


class KeypointName(StrEnum):
    """The 17 COCO body keypoints that pose models such as ViTPose output.

    'left' and 'right' are always the climber's own left and right.
    """

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


# Below this confidence a keypoint is treated as not visible. Pose models
# always output all 17 joints and guess the hidden ones (e.g. the nose of a
# climber seen from behind) with low confidence. Every stage uses this one
# threshold, through PoseFrame.is_visible, instead of picking its own.
MIN_KEYPOINT_CONFIDENCE = 0.5


class Keypoint(BaseModel):
    """One body joint, in wall-plane meters."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How sure the pose model is. 1.0 for simulated poses.",
    )


class PoseFrame(BaseModel):
    """The climber's skeleton in one video frame (or one simulation step)."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0, description="Position of the frame in the video.")
    t: float = Field(ge=0.0, description="Time in seconds from the start of the video.")
    keypoints: dict[KeypointName, Keypoint] = Field(
        description="All 17 keypoints. Hidden joints are kept, with low confidence."
    )

    @model_validator(mode="after")
    def check_all_keypoints_present(self) -> Self:
        missing = [name.value for name in KeypointName if name not in self.keypoints]
        if missing:
            raise ValueError(f"missing keypoints: {missing}")
        return self

    def __getitem__(self, name: KeypointName | str) -> Keypoint:
        """Shortcut: frame['left_wrist'] instead of frame.keypoints[...]."""
        return self.keypoints[KeypointName(name)]

    def is_visible(
        self, name: KeypointName | str, min_confidence: float = MIN_KEYPOINT_CONFIDENCE
    ) -> bool:
        """True if the pose model is confident enough about this joint to use it."""
        return self[name].confidence >= min_confidence
