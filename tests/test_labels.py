"""Tests for the hand labels and the labeling tool's logic (no window)."""

import numpy as np
import pytest
from pydantic import ValidationError

from climbvla.label_tool import LabelSession, crop_box
from climbvla.labels import (
    LABELED_POINTS,
    FrameLabel,
    LabelSet,
    PointLabel,
    Visibility,
    choose_frames,
    empty_label_set,
    label_arrays,
)
from climbvla.schemas import KeypointName as K
from climbvla.track import PoseTrack

FPS = 30


def make_track(n: int = 90, nobody: tuple[int, ...] = ()) -> PoseTrack:
    boxes = np.tile([100.0, 50, 200, 400, 0.9], (n, 1))
    keypoints = np.ones((n, 17, 3))
    for r in nobody:
        boxes[r] = np.nan
        keypoints[r] = np.nan
    return PoseTrack(
        video="climb.mp4",
        width=640,
        height=480,
        models="fake",
        frame_index=np.arange(n),
        t=np.arange(n) / FPS,
        boxes=boxes,
        keypoints=keypoints,
    )


# ---------- choosing frames ----------


def test_one_frame_every_half_second():
    assert choose_frames(make_track(90)) == [0, 15, 30, 45, 60, 75]


def test_frames_without_a_person_are_skipped():
    # nobody on frame 15: the next frame with a person is taken instead
    assert choose_frames(make_track(90, nobody=(15,)))[:3] == [0, 16, 31]


def test_empty_label_set():
    labels = empty_label_set(make_track(90))
    assert [f.frame_index for f in labels.frames] == [0, 15, 30, 45, 60, 75]
    assert labels.n_done == 0
    assert (labels.width, labels.height) == (640, 480)


# ---------- the format ----------


def test_only_wrists_and_ankles():
    with pytest.raises(ValidationError, match="only wrists and ankles"):
        FrameLabel(frame_index=0, t=0, points={K.NOSE: None})


def test_a_frame_is_done_when_every_point_has_an_answer():
    frame = FrameLabel(frame_index=0, t=0)
    for name in LABELED_POINTS[:3]:
        frame.points[name] = PointLabel(x=1, y=1, visibility=Visibility.VISIBLE)
    assert not frame.done
    frame.points[K.RIGHT_ANKLE] = None  # "can't tell" is an answer
    assert frame.done


def test_label_set_rules():
    frame = FrameLabel(frame_index=3, t=0.1)
    with pytest.raises(ValidationError, match="labeled twice"):
        LabelSet(video="v.mp4", width=640, height=480, frames=[frame, frame])
    far = FrameLabel(frame_index=3, t=0.1, points={K.LEFT_WRIST: PointLabel(x=900, y=1, visibility="visible")})
    with pytest.raises(ValidationError, match="outside"):
        LabelSet(video="v.mp4", width=640, height=480, frames=[far])


def test_round_trip_and_arrays():
    labels = empty_label_set(make_track(30))
    labels.frames[0].points = {
        K.LEFT_WRIST: PointLabel(x=10, y=20, visibility=Visibility.VISIBLE),
        K.RIGHT_WRIST: PointLabel(x=30, y=40, visibility=Visibility.HIDDEN),
        K.LEFT_ANKLE: None,
    }
    again = LabelSet.model_validate_json(labels.model_dump_json())
    assert again == labels
    frame_index, xy, hidden = label_arrays(again)
    assert list(frame_index) == [0, 15]
    assert tuple(xy[0, 0]) == (10, 20) and tuple(xy[0, 1]) == (30, 40)
    assert np.isnan(xy[0, 2]).all() and np.isnan(xy[0, 3]).all() and np.isnan(xy[1]).all()
    assert list(hidden[0]) == [False, True, False, False]


# ---------- the labeling session ----------


def test_points_are_asked_in_order():
    session = LabelSession(empty_label_set(make_track(30)))
    assert session.asking == K.LEFT_WRIST
    session.click(10, 20)
    assert session.asking == K.RIGHT_WRIST
    session.toggle_hidden()
    session.click(30, 40)
    session.skip()  # left ankle: can't tell
    session.click(50, 60)
    assert session.asking is None and session.frame.done
    points = session.frame.points
    assert points[K.LEFT_WRIST].visibility == Visibility.VISIBLE
    assert points[K.RIGHT_WRIST].visibility == Visibility.HIDDEN
    assert points[K.LEFT_ANKLE] is None
    assert points[K.RIGHT_ANKLE].visibility == Visibility.VISIBLE  # hidden mode lasts one click
    assert "all 4 points done" in session.click(1, 1)


def test_undo_removes_the_last_answer():
    session = LabelSession(empty_label_set(make_track(30)))
    session.click(10, 20)
    session.skip()
    session.undo()
    assert list(session.frame.points) == [K.LEFT_WRIST]
    assert session.asking == K.RIGHT_WRIST


def test_clicks_are_kept_inside_the_image():
    session = LabelSession(empty_label_set(make_track(30)))
    session.click(-5, 1000)
    point = session.frame.points[K.LEFT_WRIST]
    assert (point.x, point.y) == (0, 480)


def test_moving_between_frames_and_resuming():
    labels = empty_label_set(make_track(90))
    session = LabelSession(labels)
    for _ in range(4):
        session.click(1, 1)
    assert session.go(+1) and session.current == 1
    assert not session.go(-2)  # nothing before the first frame
    # a new session continues at the first frame not done
    assert LabelSession(labels).current == 1


def test_crop_box_adds_a_margin_inside_the_image():
    assert crop_box(np.array([100.0, 50, 200, 400]), 640, 480) == (40, 0, 360, 480)
