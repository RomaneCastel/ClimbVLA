"""Hand-made labels of where the wrists and ankles really are (Step 2.8).

To measure a pose model, a person clicks the true positions on some frames:
the "ground truth". Only wrists and ankles are labeled, because they decide
contacts with holds.

Each point is one of:
    - visible: seen in the image, clicked where it is;
    - hidden: behind the body or out of view, clicked where the person
      thinks it is (from the arm or leg that is visible, the frames around);
    - not labeled (None): the person can't tell at all.

Hidden points are kept apart in the results: they are exactly the case the
models get wrong, and the labels themselves are less precise there.
"""

from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from climbvla.schemas import KeypointName
from climbvla.track import PoseTrack

K = KeypointName

# The points a person labels, in the order the tool asks for them.
LABELED_POINTS = [K.LEFT_WRIST, K.RIGHT_WRIST, K.LEFT_ANKLE, K.RIGHT_ANKLE]


class Visibility(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"


class PointLabel(BaseModel):
    """One point clicked by hand, in pixels (y pointing down)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    visibility: Visibility


class FrameLabel(BaseModel):
    """The labels of one frame. A point missing from `points` is not labeled yet."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    t: float = Field(ge=0)
    points: dict[KeypointName, PointLabel | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_points(self) -> Self:
        unexpected = [name for name in self.points if name not in LABELED_POINTS]
        if unexpected:
            raise ValueError(f"only wrists and ankles are labeled, not {unexpected}")
        return self

    @property
    def done(self) -> bool:
        """True once every point has an answer (a position, or "can't tell")."""
        return all(name in self.points for name in LABELED_POINTS)


class LabelSet(BaseModel):
    """All the labels for one video."""

    model_config = ConfigDict(extra="forbid")

    video: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[FrameLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def check_frames(self) -> Self:
        indexes = [f.frame_index for f in self.frames]
        if len(set(indexes)) != len(indexes):
            raise ValueError("a frame is labeled twice")
        outside = [
            f.frame_index
            for f in self.frames
            for p in f.points.values()
            if p is not None and (p.x > self.width or p.y > self.height)
        ]
        if outside:
            raise ValueError(f"points outside the {self.width}x{self.height} image on frames {outside}")
        return self

    @property
    def n_done(self) -> int:
        return sum(f.done for f in self.frames)


def choose_frames(track: PoseTrack, every: float = 0.5) -> list[int]:
    """Rows of the track to label: one every `every` seconds, where a person was found.

    Evenly spaced in time, so easy and hard moments are both included as they
    come in the climb, rather than picked by hand (which would bias the result).
    """
    rows: list[int] = []
    next_t = float(track.t[0])
    found = track.found
    for row, t in enumerate(track.t):
        if t >= next_t - 1e-9 and found[row]:
            rows.append(row)
            next_t = float(t) + every
    return rows


def empty_label_set(track: PoseTrack, every: float = 0.5) -> LabelSet:
    """A label set with the chosen frames and no points yet."""
    return LabelSet(
        video=track.video,
        width=track.width,
        height=track.height,
        frames=[
            FrameLabel(frame_index=int(track.frame_index[r]), t=float(track.t[r]))
            for r in choose_frames(track, every)
        ],
    )


def label_arrays(labels: LabelSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The labels as arrays, for the evaluation.

    Returns (frame_index (F,), xy (F, 4, 2) NaN where not labeled,
    hidden (F, 4) bool), in LABELED_POINTS order.
    """
    frame_index = np.array([f.frame_index for f in labels.frames])
    xy = np.full((len(labels.frames), len(LABELED_POINTS), 2), np.nan)
    hidden = np.zeros((len(labels.frames), len(LABELED_POINTS)), dtype=bool)
    for i, frame in enumerate(labels.frames):
        for j, name in enumerate(LABELED_POINTS):
            point = frame.points.get(name)
            if point is not None:
                xy[i, j] = (point.x, point.y)
                hidden[i, j] = point.visibility == Visibility.HIDDEN
    return frame_index, xy, hidden
