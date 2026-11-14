"""Overlay video: the saved pose drawn on every frame, to check it by eye.

This reads the pose file written by climbvla.track, so it takes seconds,
not minutes: the models are not run again. Every later step (cleaning,
contacts, moves) will be checked with an overlay like this one.

Try it:
    uv run python -m climbvla.overlay climb.mp4
    -> reads climb_pose.npz, writes climb_overlay.mp4 next to the video
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from climbvla.draw import draw_pose, draw_text_lines
from climbvla.track import PoseTrack
from climbvla.video import iter_frames, read_info

TEXT_COLOR = (255, 255, 255)
NOBODY_COLOR = (0, 0, 255)  # red


def output_fps(track: PoseTrack, video_fps: float) -> float:
    """Frame rate that keeps the overlay at real speed.

    If only one frame in every `step` was tracked, the overlay has `step`
    times fewer frames, so it must play `step` times slower per second.
    """
    if len(track) < 2:
        return video_fps
    step = float(np.median(np.diff(track.frame_index)))
    return video_fps / step


def draw_frame(image: np.ndarray, track: PoseTrack, row: int) -> np.ndarray:
    """One overlay frame: the skeleton (if any) and the time in the corner."""
    pose = track.pose_at(row)
    out = draw_pose(image, pose) if pose is not None else image.copy()

    lines = [(f"t = {track.t[row]:.2f} s   frame {track.frame_index[row]}", TEXT_COLOR)]
    if pose is None:
        lines.append(("no climber found", NOBODY_COLOR))
    draw_text_lines(out, lines)
    return out


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    # H.264 ('avc1') plays in QuickTime and browsers. 'mp4v' is a fallback
    # that every OpenCV build can write, but fewer players can read.
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"cannot write a video to {path}")


def write_overlay(
    video: str | Path, track: PoseTrack, out: str | Path, scale: float = 1.0
) -> int:
    """Write the overlay video for the tracked frames. Returns the number of frames written.

    `scale` shrinks the output (0.5 = half width and height) for smaller files.
    """
    info = read_info(video)
    if (info.width, info.height) != (track.width, track.height):
        raise ValueError(
            f"{video} is {info.width}x{info.height} but the pose was tracked on a "
            f"{track.width}x{track.height} video ({track.video}): wrong pose file?"
        )
    if not 0 < scale <= 1:
        raise ValueError("scale must be between 0 and 1")

    width, height = round(info.width * scale), round(info.height * scale)
    row_of = {int(index): row for row, index in enumerate(track.frame_index)}
    first, last = float(track.t[0]), float(track.t[-1])

    writer = _open_writer(Path(out), output_fps(track, info.fps), width, height)
    written = 0
    try:
        for frame in iter_frames(video, start=first, end=last):
            row = row_of.get(frame.index)
            if row is None:
                continue  # this frame was skipped by the tracking (step > 1)
            image = draw_frame(frame.image, track, row)
            if scale != 1:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            written += 1
    finally:
        writer.release()
    return written


# ---------- command line ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the saved pose on every frame.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--pose", type=Path, help="pose file (default: <video>_pose.npz)")
    parser.add_argument("--out", type=Path, help="output video (default: <video>_overlay.mp4)")
    parser.add_argument("--scale", type=float, default=1.0, help="e.g. 0.5 for half size")
    args = parser.parse_args()

    pose_path = args.pose or args.video.with_name(f"{args.video.stem}_pose.npz")
    track = PoseTrack.load(pose_path)
    out = args.out or args.video.with_name(f"{args.video.stem}_overlay.mp4")
    written = write_overlay(args.video, track, out, scale=args.scale)
    print(f"wrote {out}: {written} frames")


if __name__ == "__main__":
    main()
