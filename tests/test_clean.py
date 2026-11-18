"""Tests for cleaning the pose: hiding guesses, swaps, gaps, smoothing."""

import numpy as np
import pytest

from climbvla.clean import (
    CleanPose,
    body_size,
    clean,
    drop_isolated,
    fill_short_gaps,
    fix_swaps,
    hide_guesses,
    nan_runs,
    smooth,
)
from climbvla.pose import COCO_ORDER
from climbvla.schemas import KeypointName
from climbvla.track import PoseTrack

FPS = 30
NAN = np.nan
LW = COCO_ORDER.index(KeypointName.LEFT_WRIST)
RW = COCO_ORDER.index(KeypointName.RIGHT_WRIST)


def times(n: int) -> np.ndarray:
    return np.arange(n) / FPS


# ---------- 1. hiding guesses ----------


def test_hide_guesses():
    keypoints = np.ones((2, 17, 3))
    keypoints[0, LW] = (10, 20, 0.9)
    keypoints[0, RW] = (30, 40, 0.2)
    keypoints[1] = NAN  # nobody found on this frame
    xy = hide_guesses(keypoints)
    assert xy.shape == (2, 17, 2)
    assert tuple(xy[0, LW]) == (10, 20)
    assert np.isnan(xy[0, RW]).all()
    assert np.isnan(xy[1]).all()
    assert keypoints[0, RW, 0] == 30  # the input is not changed


# ---------- 3. gaps (tested before swaps: they are simpler) ----------


def test_nan_runs():
    assert nan_runs(np.array([1, NAN, NAN, 4, NAN])) == [(1, 3), (4, 5)]
    assert nan_runs(np.array([1.0, 2.0])) == []


def test_no_gap_nothing_changes():
    values = np.array([0.0, 1, 2, 3])
    out, filled = fill_short_gaps(times(4), values)
    assert list(out) == [0, 1, 2, 3]
    assert not filled.any()


def test_short_gap_filled_with_a_straight_line():
    values = np.array([0.0, 1, NAN, NAN, 4, 5])
    out, filled = fill_short_gaps(times(6), values)
    assert out == pytest.approx([0, 1, 2, 3, 4, 5])
    assert list(filled) == [False, False, True, True, False, False]
    assert np.isnan(values[2])  # the input is not changed


def test_long_gap_stays_unknown():
    values = np.array([0.0] + [NAN] * 12 + [13.0])  # 13 frames = 0.43 s > 0.3 s
    out, filled = fill_short_gaps(times(14), values)
    assert np.isnan(out[1:13]).all()
    assert not filled.any()


def test_gap_at_the_start_or_end_stays_unknown():
    out, filled = fill_short_gaps(times(5), np.array([NAN, 1, 2, 3, NAN]))
    assert np.isnan(out[0]) and np.isnan(out[4])
    assert not filled.any()


def test_gaps_use_times_not_indexes():
    t = np.array([0.0, 0.1, 0.2, 0.4])
    out, _ = fill_short_gaps(t, np.array([0.0, 1, NAN, 4]))
    assert out[2] == pytest.approx(2.0)  # 0.2 s is a third of the way from 0.1 s to 0.4 s


# ---------- isolated points and impossible jumps ----------


def test_isolated_points_are_dropped():
    values = np.array([NAN, 5.0, NAN, NAN, 6, 7, NAN, 1, 2, 3, NAN])
    out = drop_isolated(values, min_frames=3)
    assert np.isnan(out[:7]).all()  # the 1-frame and 2-frame stretches
    assert list(out[7:10]) == [1, 2, 3]  # 3 frames: kept


def test_isolated_points_work_on_2d_points():
    points = np.array([[1.0, 1], [NAN, NAN], [2, 2], [3, 3], [4, 4]])
    out = drop_isolated(points, min_frames=2)
    assert np.isnan(out[0]).all() and not np.isnan(out[2:]).any()


def test_fill_2d_points():
    points = np.array([[0.0, 0], [NAN, NAN], [2, 4]])
    out, filled = fill_short_gaps(times(3), points)
    assert tuple(out[1]) == (1, 2)
    assert list(filled) == [False, True, False]


def test_impossible_jump_is_not_filled():
    # 300 px in 0.1 s = 3000 px/s: faster than the limit of 600 px/s
    points = np.array([[100.0, 800], [NAN, NAN], [NAN, NAN], [100, 1100]])
    out, filled = fill_short_gaps(times(4), points, max_speed=600)
    assert np.isnan(out[1:3]).all() and not filled.any()
    # a slow move across the same gap is filled
    points[3] = (100, 830)  # 30 px in 0.1 s = 300 px/s
    out, filled = fill_short_gaps(times(4), points, max_speed=600)
    assert filled[1:3].all()


def test_body_size_is_the_torso_length():
    xy = np.full((2, 17, 2), NAN)
    for name, point in [("left_shoulder", (90, 100)), ("right_shoulder", (110, 100)),
                        ("left_hip", (90, 300)), ("right_hip", (110, 300))]:
        xy[:, COCO_ORDER.index(KeypointName(name))] = point
    assert body_size(xy) == pytest.approx(200)
    assert np.isnan(body_size(np.full((2, 17, 2), NAN)))


# ---------- 2. swaps ----------


def two_wrists(n: int) -> np.ndarray:
    """Left wrist still at (100, 100), right wrist still at (300, 100)."""
    xy = np.full((n, 17, 2), NAN)
    xy[:, LW] = (100, 100)
    xy[:, RW] = (300, 100)
    return xy


def test_swap_is_fixed():
    xy = two_wrists(6)
    xy[3, LW], xy[3, RW] = (300, 100), (100, 100)  # the model swaps them on frame 3
    out, swapped = fix_swaps(times(6), xy)
    assert (out[:, LW] == (100, 100)).all()
    assert (out[:, RW] == (300, 100)).all()
    assert list(np.flatnonzero(swapped[:, LW])) == [3]


def test_swap_lasting_several_frames_is_fixed():
    xy = two_wrists(8)
    xy[2:5, [LW, RW]] = xy[2:5, [RW, LW]]  # swapped on frames 2, 3 and 4
    out, swapped = fix_swaps(times(8), xy)
    assert (out[:, LW] == (100, 100)).all()
    assert list(np.flatnonzero(swapped[:, LW])) == [2, 3, 4]


def test_hands_crossing_are_not_swapped():
    # the hands move toward each other and cross: at every frame, each hand is
    # closer to its own previous position than to the other hand's
    n = 9
    xy = np.full((n, 17, 2), NAN)
    xy[:, LW, 0] = np.linspace(100, 300, n)
    xy[:, RW, 0] = np.linspace(300, 100, n)
    xy[:, [LW, RW], 1] = 100
    out, swapped = fix_swaps(times(n), xy)
    assert not swapped.any()
    np.testing.assert_array_equal(out, xy)


def test_no_swap_decided_with_one_point_unknown():
    xy = two_wrists(4)
    xy[2, RW] = NAN
    xy[2, LW] = (300, 100)  # where the right wrist was: but the right one is unknown
    _, swapped = fix_swaps(times(4), xy)
    assert not swapped.any()


def test_old_reference_is_not_used():
    xy = two_wrists(4)
    xy[3, [LW, RW]] = xy[3, [RW, LW]]
    t = np.array([0.0, 0.03, 0.07, 2.0])  # 2 s since the last frame: too old to compare
    _, swapped = fix_swaps(t, xy)
    assert not swapped.any()


# ---------- 4. smoothing ----------


def test_smoothing_removes_jitter():
    rng = np.random.default_rng(0)
    n = 60
    still = 100 + rng.normal(0, 3, n)  # a still hand, with 3 px of jitter
    out = smooth(times(n), still)
    speed_before = np.abs(np.diff(still)).mean() * FPS
    speed_after = np.abs(np.diff(out)).mean() * FPS
    assert speed_after < speed_before / 2


def test_smoothing_keeps_a_straight_move():
    line = np.linspace(0, 300, 30)  # a steady move: nothing to remove
    assert smooth(times(30), line) == pytest.approx(line)


def test_smoothing_works_around_gaps():
    values = np.r_[np.full(20, 5.0), [NAN] * 5, np.full(20, 9.0)]
    out = smooth(times(45), values)
    assert np.isnan(out[20:25]).all()
    assert out[:20] == pytest.approx(5.0) and out[25:] == pytest.approx(9.0)


def test_smoothing_leaves_very_short_pieces():
    values = np.array([NAN, 3.0, 7.0, NAN])
    assert list(smooth(times(4), values)[1:3]) == [3, 7]


# ---------- all together ----------


def make_track(n: int = 30) -> PoseTrack:
    keypoints = np.zeros((n, 17, 3))
    keypoints[..., 0] = 100 + 10 * np.arange(17)  # every joint still, each at its own x
    keypoints[..., 1] = 200
    keypoints[..., 2] = 0.9
    keypoints[10:13, RW, 2] = 0.1  # right wrist hidden for 3 frames (0.1 s)
    keypoints[5, [LW, RW], :2] = keypoints[5, [RW, LW], :2]  # wrists swapped on frame 5
    return PoseTrack(
        video="climb.mp4",
        width=640,
        height=480,
        models="fake",
        frame_index=np.arange(n),
        t=times(n),
        boxes=np.ones((n, 5)),
        keypoints=keypoints,
    )


def test_clean_runs_the_cleaning_steps():
    track = make_track()
    pose = clean(track)
    assert isinstance(pose, CleanPose) and len(pose) == 30
    assert pose.known.all()  # the 3-frame gap was filled
    assert list(np.flatnonzero(pose.filled[:, RW])) == [10, 11, 12]
    assert list(np.flatnonzero(pose.swapped[:, LW])) == [5]
    # every joint stays where it was, swap and gap included
    assert pose.xy[:, LW, 0] == pytest.approx(100 + 10 * LW)
    assert pose.xy[:, RW, 0] == pytest.approx(100 + 10 * RW)
    # the raw track is not changed
    assert track.keypoints[5, LW, 0] == 100 + 10 * RW


def test_clean_does_not_draw_a_hidden_wrist_sweeping_across_the_body():
    # Seen on a real video: a hidden wrist "seen" a few frames at a time,
    # alternating between two places far apart. Nothing should be filled
    # between them, and the 1- and 2-frame stretches are dropped.
    track = make_track(40)
    k = track.keypoints
    k[:, RW, 2] = 0.1  # hidden...
    for start, length, y in [(10, 1, 800), (13, 2, 1100), (17, 4, 800), (24, 4, 1100)]:
        k[start : start + length, RW] = (100 + 10 * RW, y, 0.9)  # ...except a few frames
    pose = clean(track)
    assert not pose.filled[:, RW].any()  # the jumps are too fast to fill
    assert pose.known[:, RW].sum() == 8  # the two 4-frame stretches remain
    assert np.isnan(pose.xy[10:16, RW]).all()  # the 1- and 2-frame ones are gone


def test_clean_pose_checks_shapes():
    with pytest.raises(ValueError, match="xy has shape"):
        CleanPose(
            t=times(3),
            xy=np.zeros((3, 16, 2)),
            confidence=np.zeros((3, 17)),
            filled=np.zeros((3, 17), bool),
            swapped=np.zeros((3, 17), bool),
        )


# ---------- drawing the cleaned pose ----------


def test_draw_clean_pose_skips_unknown_points():
    from climbvla.draw import draw_clean_pose

    n = 3
    xy = np.full((n, 17, 2), NAN)
    xy[:, LW] = (50, 60)  # only the left wrist is known
    pose = CleanPose(
        t=times(n),
        xy=xy,
        confidence=np.full((n, 17), 0.9),
        filled=np.zeros((n, 17), bool),
        swapped=np.zeros((n, 17), bool),
    )
    out = draw_clean_pose(np.zeros((200, 200, 3), np.uint8), pose, row=0)
    assert out[60, 50].any()  # the known point is drawn
    assert out[:, 100:].sum() == 0  # nothing else: no line or dot from a NaN point
