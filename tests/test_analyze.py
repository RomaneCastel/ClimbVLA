"""Tests for the one-command pipeline (step 2.12), on a tiny video with a fake
pose estimator and a fake clicking window: no neural network, no window."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from climbvla.analyze import CURRENT_MODELS, Files, analyze, main, pose_out_of_date
from climbvla.holds import HoldClicker
from climbvla.pose import COCO_ORDER, PersonBox, PixelPose
from climbvla.schemas import PixelHold, VideoContacts, VideoMetrics, VideoRoute
from climbvla.track import PoseTrack

FPS = 10
N_FRAMES = 12
WIDTH, HEIGHT = 160, 120

POSE = PixelPose(
    box=PersonBox(x=40, y=10, w=80, h=100, score=0.9),
    keypoints={name: (50.0 + 4 * j, 20.0 + 5 * j, 0.9) for j, name in enumerate(COCO_ORDER)},
)


class FakeEstimator:
    models = CURRENT_MODELS

    def __init__(self):
        self.calls = 0

    def estimate(self, image_bgr, previous=None) -> PixelPose:
        self.calls += 1
        return POSE


class Counter:
    """make_estimator that counts how many times the models were "loaded"."""

    def __init__(self):
        self.made = 0

    def __call__(self) -> FakeEstimator:
        self.made += 1
        return FakeEstimator()


def fake_click(image: np.ndarray) -> HoldClicker:
    """What a user does in the window: two holds, the first is the start, the second the top."""
    clicker = HoldClicker(width=image.shape[1], height=image.shape[0])
    for x, y in ((60, 60), (100, 30)):
        clicker.press(x, y)
        clicker.release(x + 10, y)
    clicker.next_stage()
    clicker.release(60, 60)  # start hold: a click on it
    clicker.next_stage()
    clicker.release(100, 30)  # top hold
    clicker.next_stage()
    assert clicker.stage == "done"
    return clicker


@pytest.fixture
def video(tmp_path) -> Path:
    """A tiny 10 fps black video."""
    path = tmp_path / "climb.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    writer.release()
    return path


def write_route(video: Path, width: int = WIDTH, height: int = HEIGHT) -> None:
    route = VideoRoute(
        route_id="r",
        video=video.name,
        width=width,
        height=height,
        frame_t=0.0,
        holds=[PixelHold(id="h1", x=60, y=60, radius=10), PixelHold(id="h2", x=100, y=30, radius=10)],
        start_hold_ids=["h1"],
        top_hold_id="h2",
    )
    Files.of(video).route.write_text(route.model_dump_json())


def quiet(_: str) -> None:
    pass


def test_first_run_makes_every_file_and_the_second_reuses_the_pose(video):
    write_route(video)
    make = Counter()
    result = analyze(video, make_estimator=make, click=None, log=quiet)
    files = Files.of(video)
    for path in (files.pose, files.contacts, files.timeline, files.metrics, files.overlay):
        assert path.exists(), path
    assert make.made == 1
    assert VideoContacts.model_validate_json(files.contacts.read_text()) == result.contacts
    assert VideoMetrics.model_validate_json(files.metrics.read_text()) == result.metrics

    again = analyze(video, make_estimator=make, click=None, log=quiet)
    assert make.made == 1  # the pose was reused
    assert files.pose not in again.written
    assert files.metrics in again.written  # the fast steps always run again


def test_pose_made_again_when_the_video_is_newer(video):
    write_route(video)
    make = Counter()
    analyze(video, make_estimator=make, click=None, overlay=False, log=quiet)
    files = Files.of(video)
    later = files.pose.stat().st_mtime + 10
    os.utime(video, (later, later))
    assert pose_out_of_date(files) == "the video is newer than the pose file"
    analyze(video, make_estimator=make, click=None, overlay=False, log=quiet)
    assert make.made == 2


def test_pose_made_again_when_made_by_other_models(video):
    files = Files.of(video)
    assert pose_out_of_date(files) == "no pose file yet"
    write_route(video)
    analyze(video, make_estimator=Counter(), click=None, overlay=False, log=quiet)
    assert pose_out_of_date(files) is None
    reason = pose_out_of_date(files, models="detector + vitpose-b")
    assert reason is not None and "vitpose-b" in reason


def test_redo_pose_forces_the_models(video):
    write_route(video)
    make = Counter()
    analyze(video, make_estimator=make, click=None, overlay=False, log=quiet)
    analyze(video, make_estimator=make, click=None, overlay=False, redo_pose=True, log=quiet)
    assert make.made == 2


def test_missing_holds_without_clicking_is_a_clear_error(video):
    with pytest.raises(FileNotFoundError, match="climbvla.holds"):
        analyze(video, make_estimator=Counter(), click=None, overlay=False, log=quiet)


def test_missing_holds_open_the_clicking_window(video):
    result = analyze(video, make_estimator=Counter(), click=fake_click, route_id="gym-1", overlay=False, log=quiet)
    files = Files.of(video)
    route = VideoRoute.model_validate_json(files.route.read_text())
    assert route.route_id == "gym-1"
    assert [h.id for h in route.holds] == ["h1", "h2"]
    assert files.route in result.written and files.route.with_suffix(".png").exists()


def test_holds_from_another_video_are_refused(video):
    write_route(video, width=1080, height=1920)
    with pytest.raises(ValueError, match="holds were clicked on a 1080x1920"):
        analyze(video, make_estimator=Counter(), click=None, overlay=False, log=quiet)


def test_pose_from_another_video_is_refused(video):
    write_route(video)
    analyze(video, make_estimator=Counter(), click=None, overlay=False, log=quiet)
    files = Files.of(video)
    track = PoseTrack.load(files.pose)
    other = PoseTrack(
        video="other.mp4", width=640, height=480, models=track.models, frame_index=track.frame_index,
        t=track.t, boxes=track.boxes, keypoints=track.keypoints,
    )  # fmt: skip
    other.save(files.pose)
    with pytest.raises(ValueError, match="pose was made on a 640x480"):
        analyze(video, make_estimator=Counter(), click=None, overlay=False, log=quiet)


def test_no_video_skips_the_overlay(video):
    write_route(video)
    analyze(video, make_estimator=Counter(), click=None, overlay=False, log=quiet)
    assert not Files.of(video).overlay.exists()


def test_command_line_reports_a_missing_video_without_a_traceback(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path / "nope.mp4", click=None, log=quiet)
    with pytest.raises(SystemExit, match="error: .*holds"):
        video = tmp_path / "v.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        writer.write(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        writer.release()
        PoseTrack(
            video="v.mp4", width=WIDTH, height=HEIGHT, models=CURRENT_MODELS,
            frame_index=np.array([0]), t=np.array([0.0]), boxes=np.full((1, 5), np.nan),
            keypoints=np.full((1, len(COCO_ORDER), 3), np.nan),
        ).save(Files.of(video).pose)  # fmt: skip
        main([str(video), "--no-click", "--no-video"])
