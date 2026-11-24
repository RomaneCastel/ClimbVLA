"""One command from a climbing video to moves, metrics and an overlay (Step 2.12).

    uv run climbvla climb.mp4
    (or: uv run python -m climbvla.analyze climb.mp4)

It runs the steps of the pipeline in order, next to the video:

    1. pose      climb_pose.npz       the slow step (about 2 minutes): reused if
                                      it is up to date, else the models run
    2. holds     climb_route.json     clicked by hand: reused if it exists, else
                                      the clicking window opens
    3. contacts  climb_contacts.json  contacts, moves and regrips (+ .png timeline)
    4. metrics   climb_metrics.json
    5. overlay   climb_metrics.mp4    the video with everything drawn, to check

Steps 3 to 5 take seconds, so they always run again: they can never be left
stale after a change in the code or the holds. Only the pose is kept between
runs. A saved pose is out of date, and made again, when the video is newer
than it, or when it was made by other models than the current ones (the
project changed its keypoint model in step 2.8).

Before going on, it checks that the pose, the holds and the video have the
same image size: files from another video would otherwise give contacts that
look plausible and are wrong.
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from climbvla.clean import clean
from climbvla.contacts import detect_contacts, draw_timeline
from climbvla.holds import click_holds, click_route
from climbvla.metrics import compute_metrics, print_tables
from climbvla.moves import find_moves
from climbvla.overlay import write_overlay
from climbvla.pose import DETECTOR_MODEL, KEYPOINT_MODEL
from climbvla.schemas import VideoContacts, VideoMetrics, VideoRoute
from climbvla.track import Estimator, PoseTrack, track_video
from climbvla.video import read_info

CURRENT_MODELS = f"{DETECTOR_MODEL} + {KEYPOINT_MODEL}"


@dataclass(frozen=True)
class Files:
    """Where each step reads and writes, next to the video."""

    video: Path
    pose: Path
    route: Path
    contacts: Path
    timeline: Path
    metrics: Path
    overlay: Path

    @classmethod
    def of(cls, video: Path) -> "Files":
        def side(suffix: str) -> Path:
            return video.with_name(f"{video.stem}{suffix}")

        return cls(
            video=video,
            pose=side("_pose.npz"),
            route=side("_route.json"),
            contacts=side("_contacts.json"),
            timeline=side("_contacts.png"),
            metrics=side("_metrics.json"),
            overlay=side("_metrics.mp4"),
        )


def pose_out_of_date(files: Files, models: str = CURRENT_MODELS) -> str | None:
    """Why the saved pose must be made again, or None if it can be reused."""
    if not files.pose.exists():
        return "no pose file yet"
    if files.pose.stat().st_mtime < files.video.stat().st_mtime:
        return "the video is newer than the pose file"
    saved = PoseTrack.load(files.pose).models
    if saved != models:
        return f"the pose was made by {saved}, the current models are {models}"
    return None


def check_same_video(files: Files, track: PoseTrack, route: VideoRoute) -> None:
    """Raise ValueError if the pose or the holds come from another video."""
    info = read_info(files.video)
    size = (info.width, info.height)
    problems = []
    if (track.width, track.height) != size:
        problems.append(f"the pose was made on a {track.width}x{track.height} video ({track.video})")
    if (route.width, route.height) != size:
        problems.append(f"the holds were clicked on a {route.width}x{route.height} video ({route.video})")
    if problems:
        raise ValueError(f"{files.video.name} is {size[0]}x{size[1]}, but " + "; ".join(problems))


@dataclass(frozen=True)
class Result:
    contacts: VideoContacts
    metrics: VideoMetrics
    written: list[Path]


def default_estimator() -> Estimator:
    # loads the neural networks: imported only when the pose must be made
    from climbvla.pose import PoseEstimator

    return PoseEstimator()


def analyze(
    video: Path,
    *,
    route_id: str | None = None,
    click_t: float = 0.0,
    redo_pose: bool = False,
    overlay: bool = True,
    scale: float = 0.5,
    make_estimator: Callable[[], Estimator] = default_estimator,
    click=click_holds,
    log: Callable[[str], None] = print,
) -> Result:
    """Run the whole pipeline on one video. See the module docstring.

    `click=None` means: never open the clicking window, fail if the holds are
    missing. `make_estimator` and `click` are replaced by fakes in the tests.
    """
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    files = Files.of(video)
    written: list[Path] = []

    # 1. pose
    reason = "asked to redo it" if redo_pose else pose_out_of_date(files)
    if reason is None:
        track = PoseTrack.load(files.pose)
        log(f"1. pose: reusing {files.pose.name}")
    else:
        log(f"1. pose: running the models ({reason}); about 2 minutes for a 30 s video")
        estimator = make_estimator()
        began = time.monotonic()
        track = track_video(video, estimator, models=getattr(estimator, "models", CURRENT_MODELS), verbose=True)
        track.save(files.pose)
        written.append(files.pose)
        log(f"   {len(track)} frames in {time.monotonic() - began:.0f} s, climber found on {track.found.mean():.0%}")

    # 2. holds
    if files.route.exists():
        route = VideoRoute.model_validate_json(files.route.read_text())
        log(f"2. holds: reusing {files.route.name} ({len(route.holds)} holds)")
    elif click is None:
        raise FileNotFoundError(
            f"{files.route.name} is missing: click the holds first "
            f"(uv run python -m climbvla.holds {video} --route-id ROUTE)"
        )
    else:
        log(f"2. holds: {files.route.name} is missing, opening the clicking window")
        route = click_route(video, route_id or video.stem, click_t, out=files.route, click=click)
        if route is None:
            raise SystemExit("quit the clicking window: nothing more done")
        written += [files.route, files.route.with_suffix(".png")]
    check_same_video(files, track, route)

    # 3. contacts and moves
    pose = clean(track)
    contacts = find_moves(detect_contacts(pose, route, track=track))
    files.contacts.write_text(contacts.model_dump_json(indent=2))
    cv2.imwrite(str(files.timeline), draw_timeline(contacts, float(pose.t[-1])))
    written += [files.contacts, files.timeline]
    log(
        f"3. contacts: {len(contacts.contacts)} contacts, {len(contacts.moves)} moves, "
        f"{len(contacts.regrips)} regrip{'s' * (len(contacts.regrips) != 1)}"
    )

    # 4. metrics
    metrics, series = compute_metrics(pose, contacts)
    files.metrics.write_text(metrics.model_dump_json(indent=2))
    written.append(files.metrics)
    log("4. metrics: done")

    # 5. overlay
    if overlay:
        frames = write_overlay(
            video, track, files.overlay, scale=scale, cleaned=pose, route=route,
            contacts=contacts, metrics=series,
        )  # fmt: skip
        written.append(files.overlay)
        log(f"5. overlay: {frames} frames")
    return Result(contacts=contacts, metrics=metrics, written=written)


# ---------- command line ----------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="climbvla",
        description="From a climbing video to moves, metrics and an overlay video.",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--route-id", help="if the holds must be clicked (default: the video's name)")
    parser.add_argument("--click-t", type=float, default=0.0, help="frame to click the holds on, in seconds")
    parser.add_argument("--no-click", action="store_true", help="fail instead of opening the clicking window")
    parser.add_argument("--redo-pose", action="store_true", help="run the pose models even if the pose is saved")
    parser.add_argument("--no-video", action="store_true", help="skip the overlay video (the slowest of steps 3-5)")
    parser.add_argument("--scale", type=float, default=0.5, help="overlay size, e.g. 1 for full size")
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.video,
            route_id=args.route_id,
            click_t=args.click_t,
            redo_pose=args.redo_pose,
            overlay=not args.no_video,
            scale=args.scale,
            click=None if args.no_click else click_holds,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print()
    print_tables(result.contacts, result.metrics)
    print("\nwrote:")
    for path in result.written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
