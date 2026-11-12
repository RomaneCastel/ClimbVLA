"""Tests for running pose estimation on a whole video.

A fake estimator stands in for the neural networks, so these run in a second.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from climbvla.pose import COCO_ORDER, PersonBox, PixelPose
from climbvla.schemas import KeypointName
from climbvla.track import PoseTrack, track_video

FPS = 10
N_FRAMES = 12
WIDTH, HEIGHT = 64, 48
LEFT_WRIST = COCO_ORDER.index(KeypointName.LEFT_WRIST)


@pytest.fixture
def video(tmp_path) -> Path:
    """A tiny 10 fps black video."""
    path = tmp_path / "climb.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    writer.release()
    return path


def fake_pose(k: int) -> PixelPose:
    """The pose the fake estimator returns on its k-th call."""
    keypoints = {name: (k + j, 2.0 * j, 0.9) for j, name in enumerate(COCO_ORDER)}
    return PixelPose(box=PersonBox(x=k, y=1, w=20, h=30, score=0.8), keypoints=keypoints)


class FakeEstimator:
    """Same estimate() method as PoseEstimator, without any neural network."""

    def __init__(self, nobody_on: tuple[int, ...] = ()):
        self.calls = 0
        self.nobody_on = nobody_on  # calls on which it finds nobody
        self.previous_boxes: list[PersonBox | None] = []  # what each call was given

    def estimate(
        self, image_bgr: np.ndarray, previous: PersonBox | None = None
    ) -> PixelPose | None:
        assert image_bgr.shape == (HEIGHT, WIDTH, 3)
        self.previous_boxes.append(previous)
        k = self.calls
        self.calls += 1
        return None if k in self.nobody_on else fake_pose(k)


def make_track(n: int = 3, **changes) -> PoseTrack:
    """A small valid PoseTrack built by hand; `changes` replace some fields."""
    fields = dict(
        video="climb.mp4",
        width=WIDTH,
        height=HEIGHT,
        models="fake",
        frame_index=np.arange(n),
        t=np.arange(n) / FPS,
        boxes=np.ones((n, 5)),
        keypoints=np.ones((n, 17, 3)),
    )
    fields.update(changes)
    return PoseTrack(**fields)


# ---------- tracking a video ----------


def test_track_every_frame(video):
    track = track_video(video, FakeEstimator())
    assert len(track) == N_FRAMES
    assert list(track.frame_index) == list(range(N_FRAMES))
    assert track.t == pytest.approx([i / FPS for i in range(N_FRAMES)])
    assert (track.video, track.width, track.height) == ("climb.mp4", WIDTH, HEIGHT)
    assert track.found.all()
    assert tuple(track.keypoints[3, LEFT_WRIST]) == (3 + LEFT_WRIST, 2.0 * LEFT_WRIST, 0.9)
    assert tuple(track.boxes[3]) == (3, 1, 20, 30, 0.8)


def test_frames_without_person_are_kept_as_nan(video):
    track = track_video(video, FakeEstimator(nobody_on=(2, 5)))
    assert len(track) == N_FRAMES  # the frames are still there...
    assert list(np.flatnonzero(~track.found)) == [2, 5]
    assert np.isnan(track.keypoints[2]).all()  # ...but empty
    assert track.pose_at(2) is None
    assert track.pose_at(3) == fake_pose(3)


def test_last_found_box_is_passed_on(video):
    estimator = FakeEstimator(nobody_on=(2, 3))
    track_video(video, estimator, end=0.5)
    boxes = estimator.previous_boxes
    assert boxes[0] is None  # first frame: nothing to follow yet
    assert boxes[1] == fake_pose(0).box
    assert boxes[2] == fake_pose(1).box
    # nobody on calls 2 and 3: keep following the last box where the climber was
    assert boxes[3] == fake_pose(1).box
    assert boxes[4] == fake_pose(1).box
    assert boxes[5] == fake_pose(4).box


def test_models_only_run_on_kept_frames(video):
    estimator = FakeEstimator()
    track = track_video(video, estimator, start=0.3, end=0.8, step=2)
    assert list(track.frame_index) == [4, 6, 8]
    assert estimator.calls == 3


def test_no_frames_in_range(video):
    with pytest.raises(ValueError, match="no frames"):
        track_video(video, FakeEstimator(), start=100)


def test_visible_ratio(video):
    track = track_video(video, FakeEstimator(nobody_on=(0, 1, 2)))
    assert track.visible_ratio(KeypointName.LEFT_WRIST) == pytest.approx(9 / 12)


# ---------- saving and loading ----------


def test_save_and_load_round_trip(video, tmp_path):
    track = track_video(video, FakeEstimator(nobody_on=(4,)))
    path = tmp_path / "climb_pose.npz"
    track.save(path)
    loaded = PoseTrack.load(path)

    assert (loaded.video, loaded.width, loaded.height, loaded.models) == (
        track.video,
        track.width,
        track.height,
        track.models,
    )
    for name in ("frame_index", "t", "boxes", "keypoints"):
        # assert_array_equal counts NaN == NaN as equal, unlike ==
        np.testing.assert_array_equal(getattr(loaded, name), getattr(track, name))


def test_load_rejects_another_format_version(tmp_path):
    path = tmp_path / "old.npz"
    make_track().save(path)
    with np.load(path) as data:
        arrays = dict(data)
    arrays["meta"] = np.array(str(arrays["meta"]).replace('"format_version": 1', '"format_version": 99'))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="format version 99"):
        PoseTrack.load(path)


# ---------- rules ----------


def test_valid_track():
    assert len(make_track()) == 3


def test_wrong_shape_rejected():
    with pytest.raises(ValueError, match="keypoints has shape"):
        make_track(keypoints=np.ones((3, 16, 3)))


def test_empty_track_rejected():
    with pytest.raises(ValueError, match="at least one frame"):
        make_track(n=0)


def test_times_must_increase():
    with pytest.raises(ValueError, match="increase"):
        make_track(t=np.array([0.0, 0.2, 0.1]))


def test_box_without_keypoints_rejected():
    keypoints = np.ones((3, 17, 3))
    keypoints[1, LEFT_WRIST] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        make_track(keypoints=keypoints)


def test_keypoints_without_box_rejected():
    boxes = np.ones((3, 5))
    boxes[1] = np.nan
    with pytest.raises(ValueError, match="without a person"):
        make_track(boxes=boxes)
