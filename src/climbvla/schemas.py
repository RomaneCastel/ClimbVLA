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


class TrajectorySource(StrEnum):
    USER = "user"  # tracked from the user's video
    REFERENCE = "reference"  # produced by the planner and the simulation


# Two times closer than this are the same instant (float rounding).
TIME_TOLERANCE = 1e-6


class Trajectory(BaseModel):
    """One whole climb of one route: the skeleton over time, contacts and moves.

    The same model holds the user's climb and the simulated reference, so the
    scorer can compare them directly.

    The route is not copied in here, only its id: it lives in its own file.
    Checks that need the route are in check_against_route().
    """

    model_config = ConfigDict(extra="forbid")

    route_id: str = Field(min_length=1)
    source: TrajectorySource
    frames: list[PoseFrame] = Field(min_length=1)
    contacts: list[ContactEvent] = Field(default_factory=list)
    moves: list[Move] = Field(default_factory=list)

    # ----- checks that need only this trajectory -----

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        self._check_frames_in_order()
        self._check_times_inside_climb()
        self._check_no_overlapping_contacts()
        self._check_moves_link_contacts()
        return self

    def _check_frames_in_order(self) -> None:
        for prev, cur in zip(self.frames, self.frames[1:]):
            if cur.frame_index <= prev.frame_index or cur.t <= prev.t:
                raise ValueError(
                    f"frames must be in increasing order: frame {cur.frame_index} "
                    f"(t={cur.t}) comes after frame {prev.frame_index} (t={prev.t})"
                )

    def _check_times_inside_climb(self) -> None:
        first, last = self.start_time - TIME_TOLERANCE, self.end_time + TIME_TOLERANCE
        events = [*self.contacts, *self.moves]
        for e in events:
            for t in (e.t_start, e.t_end):
                if t is not None and not first <= t <= last:
                    raise ValueError(
                        f"time {t} is outside the climb [{self.start_time}, {self.end_time}]: {e}"
                    )

    def _check_no_overlapping_contacts(self) -> None:
        """A body part cannot be in two places at once."""
        for part in BodyPart:
            mine = sorted(self.contacts_of(part), key=lambda c: c.t_start)
            for prev, cur in zip(mine, mine[1:]):
                if prev.t_end is None or prev.t_end > cur.t_start + TIME_TOLERANCE:
                    raise ValueError(
                        f"{part} has overlapping contacts: {prev.target} and {cur.target}"
                    )

    def _check_moves_link_contacts(self) -> None:
        """Each move leaves a contact that ends at t_start and lands on one starting at t_end."""
        for move in self.moves:
            leaves = self._find_contact(move.body_part, move.from_target, end=move.t_start)
            lands = self._find_contact(move.body_part, move.to_target, start=move.t_end)
            if leaves is None:
                raise ValueError(f"no contact ends at t={move.t_start} for move {move}")
            if lands is None:
                raise ValueError(f"no contact starts at t={move.t_end} for move {move}")

    def _find_contact(
        self, part: BodyPart, target: Target, start: float | None = None, end: float | None = None
    ) -> ContactEvent | None:
        for c in self.contacts_of(part):
            if c.target != target:
                continue
            if start is not None and abs(c.t_start - start) > TIME_TOLERANCE:
                continue
            if end is not None and (c.t_end is None or abs(c.t_end - end) > TIME_TOLERANCE):
                continue
            return c
        return None

    # ----- checks that need the route -----

    def check_against_route(self, route: Route) -> None:
        """Raise ValueError if this trajectory does not fit the given route."""
        if route.id != self.route_id:
            raise ValueError(f"trajectory is for route {self.route_id!r}, not {route.id!r}")
        known = {h.id for h in route.holds}
        targets = [c.target for c in self.contacts]
        targets += [t for m in self.moves for t in (m.from_target, m.to_target)]
        unknown = sorted({t.hold_id for t in targets if isinstance(t, HoldTarget)} - known)
        if unknown:
            raise ValueError(f"hold ids not in route {route.id!r}: {unknown}")

    # ----- helpers -----

    @property
    def start_time(self) -> float:
        return self.frames[0].t

    @property
    def end_time(self) -> float:
        return self.frames[-1].t

    def contacts_of(self, part: BodyPart) -> list[ContactEvent]:
        return [c for c in self.contacts if c.body_part == part]

    def contacts_at(self, t: float) -> list[ContactEvent]:
        """Everything touching at time t: the supports during a move, for example."""
        return [c for c in self.contacts if c.is_active(t)]


# ---------- Scoring result ----------
#
# The metrics are still being designed, so they are stored
# as name -> value dictionaries for now, e.g. {"hesitation_s": 1.2}. Once the
# list settles, the important ones can become typed fields.

Metrics = dict[str, float]


class MoveAudit(BaseModel):
    """The comparison for one of the user's moves."""

    model_config = ConfigDict(extra="forbid")

    user_move: int = Field(ge=0, description="Index into the user trajectory's moves.")
    reference_move: int | None = Field(
        ge=0,
        description="Index of the matching reference move, or None if the reference has none.",
    )
    score: float = Field(ge=0.0, le=100.0, description="This move's score, 0 to 100.")
    user_metrics: Metrics = Field(default_factory=dict)
    reference_metrics: Metrics = Field(default_factory=dict)
    feedback: str | None = Field(
        default=None, description="Written coaching for this move, added in Step 10."
    )


class Audit(BaseModel):
    """The result of comparing a user climb with the reference climb of the same route."""

    model_config = ConfigDict(extra="forbid")

    route_id: str = Field(min_length=1)
    beta_score: float = Field(ge=0.0, le=100.0)
    components: dict[str, float] = Field(
        default_factory=dict,
        description="The parts the Beta Score is mixed from, each 0 to 1, "
        "e.g. {'sequence_agreement': 0.8, 'motion_similarity': 0.6}.",
    )
    climb_metrics: Metrics = Field(
        default_factory=dict, description="Whole-climb metrics of the user's climb."
    )
    moves: list[MoveAudit] = Field(default_factory=list)
    missed_reference_moves: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=list,
        description="Indices of reference moves that no user move matched.",
    )

    @model_validator(mode="after")
    def check_audit(self) -> Self:
        bad = {k: v for k, v in self.components.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(f"components must be between 0 and 1: {bad}")

        user = [m.user_move for m in self.moves]
        if len(user) != len(set(user)):
            raise ValueError("a user move is audited twice")

        matched = [m.reference_move for m in self.moves if m.reference_move is not None]
        if len(matched) != len(set(matched)):
            raise ValueError("a reference move is matched to two user moves")
        if set(matched) & set(self.missed_reference_moves):
            raise ValueError("a reference move is both matched and missed")
        return self

    def check_against(self, user: Trajectory, reference: Trajectory) -> None:
        """Raise ValueError if the move indices do not fit the two trajectories."""
        if user.source != TrajectorySource.USER or reference.source != TrajectorySource.REFERENCE:
            raise ValueError("expected a user trajectory and a reference trajectory, in that order")
        for traj in (user, reference):
            if traj.route_id != self.route_id:
                raise ValueError(f"audit is for route {self.route_id!r}, not {traj.route_id!r}")

        if any(m.user_move >= len(user.moves) for m in self.moves):
            raise ValueError(f"user move index out of range (user has {len(user.moves)} moves)")
        ref_indices = [m.reference_move for m in self.moves if m.reference_move is not None]
        ref_indices += self.missed_reference_moves
        if any(i >= len(reference.moves) for i in ref_indices):
            raise ValueError(
                f"reference move index out of range (reference has {len(reference.moves)} moves)"
            )
