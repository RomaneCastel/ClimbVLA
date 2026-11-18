"""Cleaning the raw pose before contacts are detected (Step 2.7).

Contact detection asks "is this hand near a hold AND nearly still?". The raw
pose is bad at the "still" part:
    - points under the confidence threshold are guesses;
    - left and right are sometimes swapped for a few frames;
    - every point jitters by a few pixels from frame to frame, and a speed is a
      difference between frames, so a 3 px jitter at 30 fps looks like 90 px/s.

The operations, in this order:
    1. hide the guesses (NaN below the confidence threshold);
    2. fix left/right swaps, before anything blends left and right together;
    3. drop isolated points: seen for only 1 or 2 frames, they are guesses too;
    4. fill short gaps with straight lines, unless the point would have had to
       move impossibly fast across the gap (long gaps stay unknown);
    5. smooth with a Savitzky-Golay filter.

Filling and smoothing assume the known points are right, give or take some
jitter. A first version without step 3 and the speed limit made the overlay
look worse: it drew lines between right and wrong positions of a hidden wrist,
and the smoothing spread each wrong point over 0.3 s.

It takes milliseconds, so nothing is saved: clean() runs on the saved PoseTrack
whenever a later step needs it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from climbvla.pose import COCO_ORDER
from climbvla.schemas import MIN_KEYPOINT_CONFIDENCE, TIME_TOLERANCE, KeypointName
from climbvla.track import PoseTrack

K = KeypointName
N_KEYPOINTS = len(COCO_ORDER)

# Left/right pairs that can be swapped, as indexes in COCO_ORDER.
LEFT_RIGHT_PAIRS = [
    (COCO_ORDER.index(left), COCO_ORDER.index(right))
    for left, right in [
        (K.LEFT_EYE, K.RIGHT_EYE),
        (K.LEFT_EAR, K.RIGHT_EAR),
        (K.LEFT_SHOULDER, K.RIGHT_SHOULDER),
        (K.LEFT_ELBOW, K.RIGHT_ELBOW),
        (K.LEFT_WRIST, K.RIGHT_WRIST),
        (K.LEFT_HIP, K.RIGHT_HIP),
        (K.LEFT_KNEE, K.RIGHT_KNEE),
        (K.LEFT_ANKLE, K.RIGHT_ANKLE),
    ]
]

# Defaults, chosen on the first real video (RobertClimbs.mp4, 30 fps):
# 35 of the right wrist's 44 gaps last 0.3 s or less, the longest 1.07 s.
# A hand can do a whole move in 0.5 s, so longer gaps are not guessed.
MAX_GAP = 0.3  # seconds
# Between 7 and 13.5 s of that video, the hidden right wrist was "seen" in
# stretches of 1 to 5 frames, alternating between two places 1.3 to 1.5 torso
# lengths apart: 6 to 13 torso lengths per second, which no hand does. Real
# jumps across short gaps were 0.02 to 0.24 torso lengths.
MIN_STRETCH = 3  # frames: shorter known stretches are dropped as guesses
MAX_SPEED = 3.0  # torso lengths per second (about 1.5 m/s): faster = can't fill
SWAP_RATIO = 0.5  # swap when swapping is less than half the cost of keeping
MAX_REFERENCE_AGE = 0.5  # seconds: older than this, a past frame says too little
SMOOTH_WINDOW = 0.3  # seconds (9 frames at 30 fps)
SMOOTH_ORDER = 2  # degree of the polynomial fitted inside each window


@dataclass(frozen=True, eq=False)
class CleanPose:
    """The pose after cleaning, in pixels. One row per frame of the PoseTrack."""

    t: np.ndarray  # (N,) seconds
    xy: np.ndarray  # (N, 17, 2) pixels; NaN where the position is unknown
    confidence: np.ndarray  # (N, 17) as the model gave it; NaN where nobody was found
    filled: np.ndarray  # (N, 17) bool: True where the position was interpolated
    swapped: np.ndarray  # (N, 17) bool: True where left and right were swapped back

    def __post_init__(self) -> None:
        n = len(self.t)
        shapes = {
            "xy": (self.xy, (n, N_KEYPOINTS, 2)),
            "confidence": (self.confidence, (n, N_KEYPOINTS)),
            "filled": (self.filled, (n, N_KEYPOINTS)),
            "swapped": (self.swapped, (n, N_KEYPOINTS)),
        }
        for name, (array, shape) in shapes.items():
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")

    def __len__(self) -> int:
        return len(self.t)

    @property
    def known(self) -> np.ndarray:
        """(N, 17) bool: True where the position is known (seen or filled)."""
        return ~np.isnan(self.xy[..., 0])


# ---------- 1. hide the guesses ----------


def hide_guesses(
    keypoints: np.ndarray, min_confidence: float = MIN_KEYPOINT_CONFIDENCE
) -> np.ndarray:
    """(N, 17, 3) raw keypoints -> (N, 17, 2) positions, NaN below min_confidence.

    Unknown is better than wrong: later steps can handle "unknown", but they
    can't tell a wrong position from a right one.
    """
    xy = keypoints[..., :2].copy()
    xy[~(keypoints[..., 2] >= min_confidence)] = np.nan  # NaN confidence counts as low
    return xy


# ---------- 2. left/right swaps ----------


def fix_swaps(
    t: np.ndarray,
    xy: np.ndarray,
    ratio: float = SWAP_RATIO,
    max_reference_age: float = MAX_REFERENCE_AGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap left and right back where the model swapped them.

    For each pair (e.g. the wrists), frame by frame, compared with the last
    frame where both points were known:
        keep = |left - previous left| + |right - previous right|
        swap = |left - previous right| + |right - previous left|
    If swap < ratio * keep, the labels were swapped. When the two points are
    close (arms crossing), both costs are similar and nothing is swapped.

    Returns (corrected xy, (N, 17) bool mask of the swapped points).
    """
    xy = xy.copy()
    swapped = np.zeros(xy.shape[:2], dtype=bool)
    for left, right in LEFT_RIGHT_PAIRS:
        reference: tuple[float, np.ndarray, np.ndarray] | None = None  # (t, left, right)
        for i in range(len(t)):
            a, b = xy[i, left], xy[i, right]
            if np.isnan(a).any() or np.isnan(b).any():
                continue  # can't compare with one point unknown
            if reference is not None and t[i] - reference[0] <= max_reference_age:
                _, prev_a, prev_b = reference
                keep = np.linalg.norm(a - prev_a) + np.linalg.norm(b - prev_b)
                swap = np.linalg.norm(a - prev_b) + np.linalg.norm(b - prev_a)
                if swap < ratio * keep:
                    xy[i, left], xy[i, right] = b.copy(), a.copy()
                    swapped[i, [left, right]] = True
            reference = (float(t[i]), xy[i, left].copy(), xy[i, right].copy())
    return xy, swapped


# ---------- 3. short gaps ----------


def _unknown(values: np.ndarray) -> np.ndarray:
    """(N,) bool: True where a series of points (N,) or (N, D) is unknown."""
    unknown = np.isnan(values)
    return unknown if unknown.ndim == 1 else unknown.any(axis=1)


def nan_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of unknown values; end is excluded.

    values: (N,) numbers, or (N, D) points (a point is unknown if any of its
    coordinates is NaN).
    """
    edges = np.flatnonzero(np.diff(np.r_[0, _unknown(values).astype(int), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def body_size(xy: np.ndarray) -> float:
    """The climber's torso length in pixels: median distance from the middle of
    the shoulders to the middle of the hips. NaN if never seen.

    Used as the unit for distances, so the rules work whether the climber is
    near the camera or far from it.
    """
    i = {name: j for j, name in enumerate(COCO_ORDER)}
    shoulders = (xy[:, i[K.LEFT_SHOULDER]] + xy[:, i[K.RIGHT_SHOULDER]]) / 2
    hips = (xy[:, i[K.LEFT_HIP]] + xy[:, i[K.RIGHT_HIP]]) / 2
    lengths = np.linalg.norm(shoulders - hips, axis=1)
    return float(np.nanmedian(lengths)) if not np.isnan(lengths).all() else float("nan")


def drop_isolated(values: np.ndarray, min_frames: int = MIN_STRETCH) -> np.ndarray:
    """Make unknown every stretch of known values shorter than min_frames.

    A point seen for one or two frames between unknown frames is usually a
    guess that happened to pass the confidence threshold (seen on a real video:
    a hidden wrist placed on the knee, then on the chest, a frame each).
    """
    values = values.copy()
    known = ~_unknown(values)
    edges = np.flatnonzero(np.diff(np.r_[0, known.astype(int), 0]))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_frames:
            values[start:end] = np.nan
    return values


def fill_short_gaps(
    t: np.ndarray,
    values: np.ndarray,
    max_gap: float = MAX_GAP,
    max_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill each short run of unknown values with a straight line between its two neighbors.

    values: (N,) numbers or (N, D) points. A gap's length runs from the last
    known point before it to the first one after it. Stays unknown:
    - a gap longer than max_gap;
    - a gap at the very start or end (a line needs both ends);
    - with max_speed (units per second): a gap the point could not have
      crossed, because it would have had to move faster than that. The two
      ends then disagree, so at least one is wrong, and a line between them
      would draw a limb sweeping across the body.
    Uses the times, not the frame indexes.

    Returns (filled values, (N,) bool mask of the filled points).
    """
    values = values.copy()
    filled = np.zeros(len(values), dtype=bool)
    for start, end in nan_runs(values):
        if start == 0 or end == len(values):
            continue
        before, after = values[start - 1], values[end]
        duration = t[end] - t[start - 1]
        # TIME_TOLERANCE: 0.4 - 0.1 is 0.30000000000000004 in floating point
        if duration > max_gap + TIME_TOLERANCE:
            continue
        if max_speed is not None and np.linalg.norm(after - before) > max_speed * duration:
            continue
        share = (t[start:end] - t[start - 1]) / duration  # 0 at `before`, 1 at `after`
        if values.ndim == 1:
            values[start:end] = before + share * (after - before)
        else:
            values[start:end] = before + share[:, None] * (after - before)
        filled[start:end] = True
    return values, filled


# ---------- 4. smoothing ----------


def smooth(
    t: np.ndarray,
    values: np.ndarray,
    window: float = SMOOTH_WINDOW,
    order: int = SMOOTH_ORDER,
) -> np.ndarray:
    """Savitzky-Golay smoothing of one series, piece by piece between NaN.

    In a window sliding along time, it fits a small polynomial to the points
    and keeps its value at the center. Unlike a plain average, it keeps the
    shape of quick moves while removing the jitter. It assumes evenly spaced
    frames, which phone videos are, closely enough.
    """
    values = values.copy()
    if len(t) < 2:
        return values
    dt = float(np.median(np.diff(t)))
    frames = int(round(window / dt)) | 1  # an odd number of frames
    known = ~np.isnan(values)
    edges = np.flatnonzero(np.diff(np.r_[0, known.astype(int), 0]))
    for start, end in zip(edges[::2], edges[1::2]):
        length = end - start
        size = min(frames, length if length % 2 else length - 1)
        if size > order:  # too short to fit the polynomial: left as it is
            values[start:end] = savgol_filter(values[start:end], size, order)
    return values


# ---------- all together ----------


def clean(
    track: PoseTrack,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
    max_gap: float = MAX_GAP,
    swap_ratio: float = SWAP_RATIO,
    smooth_window: float = SMOOTH_WINDOW,
    min_stretch: int = MIN_STRETCH,
    max_speed: float = MAX_SPEED,
) -> CleanPose:
    """The cleaning operations, in order, on every keypoint.

    max_speed is in torso lengths per second (see body_size).
    """
    t = track.t
    xy = hide_guesses(track.keypoints, min_confidence)
    xy, swapped = fix_swaps(t, xy, swap_ratio)
    torso = body_size(xy)
    speed_limit = max_speed * torso if not np.isnan(torso) else None  # pixels per second

    filled = np.zeros(xy.shape[:2], dtype=bool)
    for j in range(N_KEYPOINTS):
        points = drop_isolated(xy[:, j], min_stretch)
        points, filled[:, j] = fill_short_gaps(t, points, max_gap, speed_limit)
        for c in range(2):  # smoothing works on one coordinate at a time
            points[:, c] = smooth(t, points[:, c], smooth_window)
        xy[:, j] = points
    swapped &= ~np.isnan(xy[..., 0])  # a swapped point may have been dropped since

    return CleanPose(
        t=t, xy=xy, confidence=track.keypoints[..., 2], filled=filled, swapped=swapped
    )
