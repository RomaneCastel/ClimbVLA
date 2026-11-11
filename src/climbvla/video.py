"""Reading climbing videos frame by frame.

Every frame comes with its own timestamp, read from the video file, instead
of being computed as index / fps. Phones often record at a variable frame
rate, so index / fps can drift by a fraction of a second over a climb.

Try it on a video:
    uv run python -m climbvla.video path/to/climb.mp4
"""

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """What the video file says about itself."""

    path: Path
    width: int  # pixels, after rotation (portrait phone videos are taller than wide)
    height: int
    fps: float  # nominal; the real timing comes from each frame's timestamp
    frame_count: int  # an estimate from the file header; can be off by a few frames

    @property
    def duration(self) -> float:
        """Approximate length in seconds, from the header."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    """One decoded video frame."""

    index: int  # 0, 1, 2... in the order frames are decoded
    t: float  # seconds from the start of the video, from the file's timestamps
    image: np.ndarray  # height x width x 3, in OpenCV's RGB color order


def _open(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot read {path} as a video")
    return cap


def read_info(path: str | Path) -> VideoInfo:
    cap = _open(path)
    try:
        # Size from a real decoded frame: phone videos store a rotation flag,
        # and OpenCV applies it to the frames but the header may not show it.
        # just reading the first frame is enough to get the size.  
        ok, first = cap.read()
        if not ok:
            raise ValueError(f"{path} has no readable frames")
        height, width = first.shape[:2]
        return VideoInfo(
            path=Path(path),
            width=width,
            height=height,
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path, start: float = 0.0, end: float | None = None, step: int = 1
) -> Iterator[Frame]:
    """Yield the frames between start and end seconds, keeping one in every `step`.

    Frames are read one after another from the beginning. That is slower than
    jumping straight to `start`, but jumping in compressed video lands on the
    nearest keyframe, not the exact frame asked for.
    """
    if step < 1:
        raise ValueError("step must be at least 1")
    cap = _open(path)
    try:
        index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                return  # end of the video
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # timestamp of the frame just read
            if end is not None and t > end:
                return
            if t >= start and index % step == 0:
                yield Frame(index=index, t=t, image=image)
            index += 1
    finally:
        cap.release()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m climbvla.video VIDEO")
    info = read_info(sys.argv[1])
    print(f"{info.path.name}: {info.width}x{info.height}, {info.fps:.2f} fps (nominal)")
    print(f"header says {info.frame_count} frames, about {info.duration:.2f} s")

    # Decode everything once to get the real numbers.
    times = [f.t for f in iter_frames(info.path)]
    gaps = np.diff(times)
    print(f"decoded {len(times)} frames, from {times[0]:.3f} s to {times[-1]:.3f} s")
    if len(gaps):
        print(
            f"time between frames: mean {gaps.mean() * 1000:.1f} ms, "
            f"min {gaps.min() * 1000:.1f} ms, max {gaps.max() * 1000:.1f} ms"
        )


if __name__ == "__main__":
    main()
