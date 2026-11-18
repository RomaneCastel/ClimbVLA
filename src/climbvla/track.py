"""Running pose estimation on a whole video and saving the result.

Pose estimation takes about 0.2 s per frame, so a 30 s video at 30 fps takes
a few minutes. It is run once, and the result is saved to a file that every
later step reads instead of rerunning the models.

What is saved is the RAW output of the models: pixels, y pointing down,
confidences as given, nothing cleaned or converted. Cleaning and conversion
to meters come later, and they can change without rerunning the models.

Try it:
    uv run python -m climbvla.track climb.mp4
    -> writes climb_pose.npz next to the video
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from climbvla.pose import (
    COCO_ORDER,
    DETECTOR_MODEL,
    KEYPOINT_MODEL,
    PersonBox,
    PixelPose,
    keypoints_from_arrays,
)
from climbvla.schemas import MIN_KEYPOINT_CONFIDENCE, KeypointName
from climbvla.video import iter_frames, read_info

# Written into every file. If the file layout changes one day, old files are
# rejected with a clear message instead of being misread.
FORMAT_VERSION = 1

N_KEYPOINTS = len(COCO_ORDER)


class Estimator(Protocol):
    """Anything with an estimate() method like PoseEstimator's.

    track_video() only needs that one method. The tests pass a fake estimator
    with the same method, so they run in a second without the neural networks.
    """

    def estimate(
        self, image_bgr: np.ndarray, previous: PersonBox | None = None
    ) -> PixelPose | None: ...


@dataclass(frozen=True, eq=False)  # eq=False: comparing arrays with == is ambiguous
class PoseTrack:
    """The climber's raw pose on every processed frame of a video, in pixels.

    Stored as arrays with one row per frame, rather than as a list of
    objects, because the next steps work along time: smoothing the left
    wrist means taking keypoints[:, i] for all frames at once.

    Frames where nobody was found are kept, filled with NaN,
    so row i is always the i-th processed frame and times stay regular.
    """

    video: str  # the video's file name, for information
    width: int  # image size in pixels
    height: int
    models: str  # which models produced the pose
    frame_index: np.ndarray  # (N,) int: index of the frame in the video
    t: np.ndarray  # (N,) float: seconds from the start of the video
    boxes: np.ndarray  # (N, 5): x, y, w, h, score. NaN if nobody was found
    keypoints: np.ndarray  # (N, 17, 3): x, y, confidence in COCO_ORDER. NaN if nobody

    def __post_init__(self) -> None:
        n = len(self.t)
        if n == 0:
            raise ValueError("a pose track needs at least one frame")
        expected = {
            "frame_index": (self.frame_index, (n,)),
            "t": (self.t, (n,)),
            "boxes": (self.boxes, (n, 5)),
            "keypoints": (self.keypoints, (n, N_KEYPOINTS, 3)),
        }
        for name, (array, shape) in expected.items():
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("frame times must increase")
        # A frame either has a box and all its keypoints, or neither.
        found = self.found
        if np.isnan(self.boxes[found]).any() or np.isnan(self.keypoints[found]).any():
            raise ValueError("a frame with a person has missing values")
        if not np.isnan(self.keypoints[~found]).all():
            raise ValueError("a frame without a person has keypoints")

    def __len__(self) -> int:
        return len(self.t)

    @property
    def found(self) -> np.ndarray:
        """(N,) bool: True on the frames where a person was found."""
        return ~np.isnan(self.boxes[:, 0])

    def pose_at(self, i: int) -> PixelPose | None:
        """Row i as a PixelPose (what draw_pose takes), or None if nobody was found."""
        if not self.found[i]:
            return None
        x, y, w, h, score = (float(v) for v in self.boxes[i])
        return PixelPose(
            box=PersonBox(x=x, y=y, w=w, h=h, score=score),
            keypoints=keypoints_from_arrays(self.keypoints[i, :, :2], self.keypoints[i, :, 2]),
        )

    def visible_ratio(self, name: KeypointName) -> float:
        """Share of all frames where this keypoint was seen with enough confidence."""
        confidence = self.keypoints[:, COCO_ORDER.index(name), 2]
        return float(np.mean(confidence >= MIN_KEYPOINT_CONFIDENCE))  # NaN >= x is False

    # ---------- saving and loading ----------

    def save(self, path: str | Path) -> None:
        """Write to a .npz file: numpy's format for several arrays in one file."""
        meta = {
            "format_version": FORMAT_VERSION,
            "video": self.video,
            "width": self.width,
            "height": self.height,
            "models": self.models,
        }
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),  # the text fields, as one JSON string
            frame_index=self.frame_index,
            t=self.t,
            boxes=self.boxes,
            keypoints=self.keypoints,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseTrack":
        # allow_pickle=False: a .npz file can contain "pickled" Python objects,
        # and unpickling can run arbitrary code. We only store plain arrays.
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"{path} has format version {meta.get('format_version')}, "
                    f"this code reads version {FORMAT_VERSION}: rerun the tracking"
                )
            return cls(
                video=meta["video"],
                width=meta["width"],
                height=meta["height"],
                models=meta["models"],
                frame_index=data["frame_index"],
                t=data["t"],
                boxes=data["boxes"],
                keypoints=data["keypoints"],
            )


def track_video(
    path: str | Path,
    estimator: Estimator,
    start: float = 0.0,
    end: float | None = None,
    step: int = 1,
    models: str = f"{DETECTOR_MODEL} + {KEYPOINT_MODEL}",
    verbose: bool = False,
) -> PoseTrack:
    """Run the estimator on the frames between start and end, one in every `step`.

    The climber's box from the last frame where they were found is passed to
    the estimator, so it follows the same person through the video.
    Limit: if the climber is lost and comes back far from that box (after a
    long gap), they are not found again. The "person found" share shows it.
    """
    info = read_info(path)
    frame_index: list[int] = []
    times: list[float] = []
    boxes: list[list[float]] = []
    keypoints: list[np.ndarray] = []

    last_box: PersonBox | None = None  # the climber's box, the last time they were found
    began = time.monotonic()
    for frame in iter_frames(path, start=start, end=end, step=step):
        pose = estimator.estimate(frame.image, previous=last_box)
        frame_index.append(frame.index)
        times.append(frame.t)
        if pose is None:
            boxes.append([np.nan] * 5)
            keypoints.append(np.full((N_KEYPOINTS, 3), np.nan))
        else:
            last_box = b = pose.box
            boxes.append([b.x, b.y, b.w, b.h, b.score])
            keypoints.append(np.array([pose.keypoints[name] for name in COCO_ORDER]))

        if verbose and len(times) % 30 == 0:
            rate = len(times) / (time.monotonic() - began)
            print(f"  {frame.t:6.1f} s of {info.duration:.1f} s  ({rate:.1f} frames/s)")

    if not times:
        raise ValueError(f"{path} has no frames between {start} s and {end} s")
    return PoseTrack(
        video=info.path.name,
        width=info.width,
        height=info.height,
        models=models,
        frame_index=np.array(frame_index, dtype=int),
        t=np.array(times, dtype=float),
        boxes=np.array(boxes, dtype=float),
        keypoints=np.array(keypoints, dtype=float),
    )


# ---------- command line ----------


def main() -> None:
    from climbvla.pose import PoseEstimator

    parser = argparse.ArgumentParser(description="Run pose estimation on a whole video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, help="output file (default: <video>_pose.npz)")
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, help="seconds")
    parser.add_argument("--step", type=int, default=1, help="keep one frame in every STEP")
    args = parser.parse_args()

    estimator = PoseEstimator()
    print(f"running on {estimator.device}")
    track = track_video(
        args.video,
        estimator,
        start=args.start,
        end=args.end,
        step=args.step,
        models=estimator.models,
        verbose=True,
    )
    out = args.out or args.video.with_name(f"{args.video.stem}_pose.npz")
    track.save(out)

    print(f"wrote {out}: {len(track)} frames, person found on {track.found.mean():.0%}")
    print("seen with enough confidence, over all frames:")
    for name in (
        KeypointName.LEFT_WRIST,
        KeypointName.RIGHT_WRIST,
        KeypointName.LEFT_ANKLE,
        KeypointName.RIGHT_ANKLE,
        KeypointName.LEFT_KNEE,
        KeypointName.RIGHT_KNEE,
    ):
        print(f"  {name:12} {track.visible_ratio(name):.0%}")


if __name__ == "__main__":
    main()
