"""Shared data format for every ClimbVLA stage.

Coordinate convention, used by every model in this file:
    - Wall plane, in meters.
    - Origin at the bottom-left of the filmed wall area.
    - x points right, y points up (not down like image pixels).
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

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


class BodyPart(StrEnum):
    """Body parts that can be in contact with a hold. 'left' is the climber's left.

    Knees are for knee bars and knee locks. Contact detection can report them;
    the planner and the simulation start with hands and feet only.
    """

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"

    @property
    def keypoint(self) -> KeypointName:
        """The pose keypoint used to track this body part."""
        return {
            BodyPart.LEFT_HAND: KeypointName.LEFT_WRIST,
            BodyPart.RIGHT_HAND: KeypointName.RIGHT_WRIST,
            BodyPart.LEFT_FOOT: KeypointName.LEFT_ANKLE,
            BodyPart.RIGHT_FOOT: KeypointName.RIGHT_ANKLE,
            BodyPart.LEFT_KNEE: KeypointName.LEFT_KNEE,
            BodyPart.RIGHT_KNEE: KeypointName.RIGHT_KNEE,
        }[self]

    @property
    def is_hand(self) -> bool:
        return self in (BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND)

    @property
    def is_foot(self) -> bool:
        return self in (BodyPart.LEFT_FOOT, BodyPart.RIGHT_FOOT)

    @property
    def is_knee(self) -> bool:
        return self in (BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE)


# ---------- What a body part can touch ----------
#
# A contact is on a hold, on the bare wall (a smear, a palm, a knee scum) or
# on the ground. Each case is its own small class with only the fields that
# make sense for it. The "kind" field tells pydantic which class to build when
# reading JSON: {"kind": "wall", "x": 1.3, "y": 0.5} becomes a WallTarget.


class HoldTarget(BaseModel):
    """On a hold of the route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hold"] = "hold"
    hold_id: str = Field(min_length=1)


class WallTarget(BaseModel):
    """On the bare wall, not a hold: a smear, a palm, a knee scum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["wall"] = "wall"
    x: float = Field(description="Meters, same wall-plane frame as holds.")
    y: float


class GroundTarget(BaseModel):
    """On the ground or the mat: before the start, or after dropping off."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ground"] = "ground"


Target = Annotated[HoldTarget | WallTarget | GroundTarget, Field(discriminator="kind")]


class ContactEvent(BaseModel):
    """One body part touching one target, from the moment it lands until it leaves.

    Whether a HoldTarget's hold_id exists in the route is checked by
    Trajectory, which knows the route; a ContactEvent alone does not.
    """

    model_config = ConfigDict(extra="forbid")

    body_part: BodyPart
    target: Target
    t_start: float = Field(ge=0.0, description="Seconds: when the body part lands.")
    t_end: float | None = Field(
        default=None,
        description="Seconds: when it leaves. None if still there when the video ends.",
    )

    @model_validator(mode="after")
    def check_times(self) -> Self:
        if self.t_end is not None and self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be after t_start ({self.t_start})")
        return self

    @property
    def duration(self) -> float | None:
        """Seconds in contact, or None if the contact never ended."""
        return None if self.t_end is None else self.t_end - self.t_start

    def is_active(self, t: float) -> bool:
        """True if the body part is in contact at time t."""
        return self.t_start <= t and (self.t_end is None or t < self.t_end)


class Move(BaseModel):
    """One body part going from one target to another.

    It starts when the body part leaves from_target and ends when it lands on
    to_target. Moves that happen together (a dyno, where both hands leave at
    once) are several Moves with overlapping times.

    What the other body parts are touching during the move is not stored here:
    it can be computed from the ContactEvents, and storing it twice would let
    the two copies disagree.
    """

    model_config = ConfigDict(extra="forbid")

    body_part: BodyPart
    from_target: Target = Field(description="Where it leaves from (GroundTarget at the start).")
    to_target: Target = Field(description="Where it lands.")
    t_start: float = Field(ge=0.0, description="Seconds: when it leaves from_target.")
    t_end: float = Field(description="Seconds: when it lands on to_target.")

    @model_validator(mode="after")
    def check_move(self) -> Self:
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be after t_start ({self.t_start})")
        if self.from_target == self.to_target:
            raise ValueError(f"a move must change target, got {self.to_target} twice")
        return self

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start
