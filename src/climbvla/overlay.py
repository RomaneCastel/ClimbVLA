"""Overlay video: the saved pose drawn on every frame, to check it by eye.

This reads the pose file written by climbvla.track, so it takes seconds,
not minutes: the models are not run again. Every later step (cleaning,
contacts, moves) will be checked with an overlay like this one.

Try it:
    uv run python -m climbvla.overlay climb.mp4
    -> reads climb_pose.npz, writes climb_overlay.mp4 next to the video
    uv run python -m climbvla.overlay climb.mp4 --clean
    -> the cleaned pose instead (step 2.7), in climb_clean.mp4
    uv run python -m climbvla.overlay climb.mp4 --contacts
    -> the cleaned pose, the holds and the contacts (step 2.9), in climb_contacts.mp4
    uv run python -m climbvla.overlay climb.mp4 --metrics
    -> the same plus the center of mass and the elbow angles
       (step 2.11), in climb_metrics.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from climbvla.clean import CleanPose, clean
from climbvla.contacts import PART_LABELS, PARTS, contact_label
from climbvla.draw import (
    draw_clean_pose,
    draw_com,
    draw_contacts,
    draw_holds,
    draw_pose,
    draw_text_lines,
)
from climbvla.metrics import HANDS, Series, series
from climbvla.moves import move_at, regrip_at
from climbvla.schemas import BodyPart, PixelContact, VideoContacts, VideoRoute
from climbvla.track import PoseTrack
from climbvla.video import iter_frames, read_info


def part_status(contacts: VideoContacts, on: PixelContact | None, part: BodyPart, t: float) -> str:
    """What one body part is doing at time t: 'h4', 'h5 -> h4' (a move), 'regrip h4', or '-'."""
    if on is not None:
        return contact_label(on)
    move = move_at(contacts, part, t)
    if move is not None:
        return f"{move.from_target.label} -> {move.to_target.label}"
    regrip = regrip_at(contacts, part, t)
    if regrip is not None:
        return f"regrip {regrip.target.label}"
    return "-"


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


def draw_frame(
    image: np.ndarray,
    track: PoseTrack,
    row: int,
    cleaned: CleanPose | None = None,
    route: VideoRoute | None = None,
    contacts: VideoContacts | None = None,
    metrics: Series | None = None,
) -> np.ndarray:
    """One overlay frame: the skeleton (if any) and the time in the corner.

    With `cleaned`, the cleaned pose is drawn instead of the raw one.
    With `route` and `contacts` (step 2.9), the holds and the contacts active
    on this frame are drawn too, and listed in the corner.
    With `metrics` (step 2.11, needs `contacts`), the center of mass is drawn
    and the elbow angles listed.
    """
    if route is not None:
        image = draw_holds(image, route.holds, route.start_hold_ids, route.top_hold_id)
    pose = track.pose_at(row)
    if pose is None:
        out = image.copy()
    elif cleaned is not None:
        out = draw_clean_pose(image, cleaned, row)
    else:
        out = draw_pose(image, pose)
    t = float(track.t[row])
    if contacts is not None:
        out = draw_contacts(out, contacts.contacts, t)
    if metrics is not None and contacts is not None:
        out = draw_com(out, (float(metrics.com[row, 0]), float(metrics.com[row, 1])))

    lines = [(f"t = {t:.2f} s   frame {track.frame_index[row]}", TEXT_COLOR)]
    if contacts is not None:
        on = {c.body_part: c for c in contacts.contacts if c.t_start <= t < c.t_end}
        for part in PARTS:
            lines.append((f"{PART_LABELS[part]} {part_status(contacts, on.get(part), part, t)}", TEXT_COLOR))
        if metrics is not None:
            angles = "  ".join(
                f"{PART_LABELS[h][0]} {metrics.elbow[h][row]:.0f}" if not np.isnan(metrics.elbow[h][row])
                else f"{PART_LABELS[h][0]} ?"
                for h in HANDS
            )
            lines.append((f"elbows {angles}", TEXT_COLOR))
    elif cleaned is not None:
        lines.append(("cleaned pose", TEXT_COLOR))
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
    video: str | Path,
    track: PoseTrack,
    out: str | Path,
    scale: float = 1.0,
    cleaned: CleanPose | None = None,
    route: VideoRoute | None = None,
    contacts: VideoContacts | None = None,
    metrics: Series | None = None,
) -> int:
    """Write the overlay video for the tracked frames. Returns the number of frames written.

    `scale` shrinks the output (0.5 = half width and height) for smaller files.
    With `cleaned` (from climbvla.clean.clean(track)), the cleaned pose is drawn.
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
            image = draw_frame(frame.image, track, row, cleaned, route, contacts, metrics)
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
    parser.add_argument(
        "--clean", action="store_true", help="draw the cleaned pose (default out: <video>_clean.mp4)"
    )
    parser.add_argument(
        "--contacts",
        action="store_true",
        help="also draw the holds and contacts (step 2.9; implies --clean; "
        "reads <video>_route.json and <video>_contacts.json; default out: <video>_contacts.mp4)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="also draw the center of mass and elbow angles (step 2.11; "
        "implies --contacts; default out: <video>_metrics.mp4)",
    )
    args = parser.parse_args()
    args.contacts = args.contacts or args.metrics

    pose_path = args.pose or args.video.with_name(f"{args.video.stem}_pose.npz")
    track = PoseTrack.load(pose_path)
    route = contacts = None
    if args.contacts:
        route = VideoRoute.model_validate_json(
            args.video.with_name(f"{args.video.stem}_route.json").read_text()
        )
        contacts = VideoContacts.model_validate_json(
            args.video.with_name(f"{args.video.stem}_contacts.json").read_text()
        )
    cleaned = clean(track) if args.clean or args.contacts else None
    metrics = series(cleaned, contacts) if args.metrics else None
    suffix = (
        "_metrics.mp4" if args.metrics
        else "_contacts.mp4" if args.contacts
        else "_clean.mp4" if args.clean
        else "_overlay.mp4"
    )
    out = args.out or args.video.with_name(f"{args.video.stem}{suffix}")
    written = write_overlay(
        args.video, track, out, scale=args.scale, cleaned=cleaned, route=route, contacts=contacts,
        metrics=metrics,
    )
    print(f"wrote {out}: {written} frames")


if __name__ == "__main__":
    main()
