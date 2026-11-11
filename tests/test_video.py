"""Tests for video reading. Run with: uv run pytest"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from climbvla.video import iter_frames, read_info

FPS = 10
N_FRAMES = 12
WIDTH, HEIGHT = 64, 48


@pytest.fixture
def video(tmp_path) -> Path:
    """A tiny 10 fps test video: frame i has a white stripe at x = 5*i."""
    path = tmp_path / "stripes.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for i in range(N_FRAMES):
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        image[:, 5 * i : 5 * i + 5] = 255
        writer.write(image)
    writer.release()
    return path


def stripe_position(image: np.ndarray) -> int:
    """x of the white stripe's center, to know which frame we are looking at."""
    return int(round(np.nonzero(image[HEIGHT // 2, :, 0] > 128)[0].mean()))


def test_read_info(video):
    info = read_info(video)
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.fps == pytest.approx(FPS)
    assert info.frame_count == N_FRAMES
    assert info.duration == pytest.approx(N_FRAMES / FPS)


def test_all_frames_in_order_with_timestamps(video):
    frames = list(iter_frames(video))
    assert [f.index for f in frames] == list(range(N_FRAMES))
    assert [f.t for f in frames] == pytest.approx([i / FPS for i in range(N_FRAMES)])
    assert frames[0].image.shape == (HEIGHT, WIDTH, 3)


def test_frames_are_the_right_images(video):
    for f in iter_frames(video):
        assert stripe_position(f.image) == pytest.approx(5 * f.index + 2, abs=1)


def test_start_end_and_step(video):
    frames = list(iter_frames(video, start=0.3, end=0.8, step=2))
    # frames 3..8 are inside [0.3 s, 0.8 s]; step=2 keeps the even indices
    assert [f.index for f in frames] == [4, 6, 8]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_info("does-not-exist.mp4")


def test_not_a_video(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("this is not a video")
    with pytest.raises(ValueError):
        read_info(path)


def test_step_must_be_positive(video):
    with pytest.raises(ValueError):
        next(iter_frames(video, step=0))
