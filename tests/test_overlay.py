"""Tests for the overlay video. No neural network: the pose comes from a fake estimator."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from climbvla.overlay import draw_frame, output_fps, write_overlay
from climbvla.pose import COCO_ORDER, PersonBox, PixelPose
from climbvla.schemas import KeypointName
from climbvla.track import PoseTrack, track_video
from climbvla.video import read_info

FPS = 10
N_FRAMES = 12
WIDTH, HEIGHT = 160, 120


@pytest.fixture
def video(tmp_path) -> Path:
    """A tiny 10 fps black video."""
    path = tmp_path / "climb.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    writer.release()
    return path


# The same confident pose on every frame, spread over the middle of the image.
POSE = PixelPose(
    box=PersonBox(x=40, y=10, w=80, h=100, score=0.9),
    keypoints={name: (50.0 + 4 * j, 20.0 + 5 * j, 0.9) for j, name in enumerate(COCO_ORDER)},
)


class FakeEstimator:
    def __init__(self, nobody_on: tuple[int, ...] = ()):
        self.calls = 0
        self.nobody_on = nobody_on

    def estimate(self, image_bgr, previous=None) -> PixelPose | None:
        k = self.calls
        self.calls += 1
        return None if k in self.nobody_on else POSE


def test_one_output_frame_per_tracked_frame(video, tmp_path):
    track = track_video(video, FakeEstimator())
    out = tmp_path / "overlay.mp4"
    assert write_overlay(video, track, out) == N_FRAMES
    info = read_info(out)
    assert (info.width, info.height, info.frame_count) == (WIDTH, HEIGHT, N_FRAMES)
    assert info.fps == pytest.approx(FPS)


def test_skipped_frames_slow_the_frame_rate(video, tmp_path):
    track = track_video(video, FakeEstimator(), step=2)  # frames 0, 2, 4...
    out = tmp_path / "overlay.mp4"
    assert write_overlay(video, track, out) == N_FRAMES // 2
    assert read_info(out).fps == pytest.approx(FPS / 2)  # still plays at real speed


def test_output_fps():
    track = PoseTrack(
        video="climb.mp4",
        width=WIDTH,
        height=HEIGHT,
        models="fake",
        frame_index=np.array([0, 3, 6]),
        t=np.array([0.0, 0.1, 0.2]),
        boxes=np.ones((3, 5)),
        keypoints=np.ones((3, 17, 3)),
    )
    assert output_fps(track, 30.0) == pytest.approx(10.0)


def test_scale_shrinks_the_output(video, tmp_path):
    track = track_video(video, FakeEstimator())
    out = tmp_path / "overlay.mp4"
    write_overlay(video, track, out, scale=0.5)
    info = read_info(out)
    assert (info.width, info.height) == (WIDTH // 2, HEIGHT // 2)


def test_skeleton_drawn_only_where_a_climber_was_found(video):
    track = track_video(video, FakeEstimator(nobody_on=(1,)))
    black = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    x, y, _ = POSE.keypoints[KeypointName.LEFT_HIP]
    point = (int(y), int(x))  # images are indexed [row, column] = [y, x]

    with_pose = draw_frame(black, track, row=0)
    assert with_pose[point].any()

    without_pose = draw_frame(black, track, row=1)
    assert not without_pose[point].any()
    red = (without_pose[..., 2] > 200) & (without_pose[..., 0] < 50)
    assert red.any()  # the red "no climber found" text


def test_wrong_pose_file_rejected(video, tmp_path):
    track = track_video(video, FakeEstimator())
    other = tmp_path / "other.mp4"
    writer = cv2.VideoWriter(str(other), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="wrong pose file"):
        write_overlay(other, track, tmp_path / "overlay.mp4")


def test_overlay_of_the_cleaned_pose(video, tmp_path):
    from climbvla.clean import clean

    track = track_video(video, FakeEstimator(nobody_on=(3,)))
    out = tmp_path / "clean.mp4"
    assert write_overlay(video, track, out, cleaned=clean(track)) == N_FRAMES
    assert read_info(out).frame_count == N_FRAMES
