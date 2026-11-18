"""Comparing pose models against hand labels (Step 2.8).

Three keypoint models are run on the labeled frames, all given the SAME person
box (the one found in step 2.3), so only the keypoint models are compared:
    - vitpose-b:   ViTPose base, the model used so far;
    - vitpose-h:   ViTPose+ huge, the same family, about 7x bigger;
    - rtmpose-x:   RTMPose-x, another family, trained on 7 datasets.

Each model's wrists and ankles are compared with the clicked positions:
    - error: distance to the label, in torso lengths (body units, as in 2.7);
    - PCK@0.2 ("percentage of correct keypoints"): the share of points within
      0.2 torso lengths of the label (about 43 px on the first video);
    - for the confidence threshold: how many wrong points it keeps and how
      many right points it throws away.
Visible and hidden points are reported apart: hidden ones are the problem.

Try it:
    uv run python -m climbvla.pose_eval climb.mp4 --predict-only   # no labels needed
    uv run python -m climbvla.pose_eval climb.mp4                  # once labeled
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from climbvla.clean import body_size, hide_guesses
from climbvla.labels import LABELED_POINTS, LabelSet, label_arrays
from climbvla.keypoint_models import MODEL_NAMES, KeypointModel, load_model
from climbvla.pose import COCO_ORDER
from climbvla.track import PoseTrack
from climbvla.video import iter_frames

LABELED_INDEX = [COCO_ORDER.index(name) for name in LABELED_POINTS]
PCK_DISTANCES = (0.1, 0.2)  # torso lengths
CORRECT_WITHIN = 0.2  # torso lengths: "right" for the threshold table
THRESHOLDS = (0.5, 0.6, 0.7)


# ---------- running them ----------


def predict(
    video: Path, track: PoseTrack, frame_indexes: list[int], model: KeypointModel
) -> np.ndarray:
    """(F, 17, 3) keypoints on the given frames, inside the climber's box from the track."""
    row_of = {int(i): r for r, i in enumerate(track.frame_index)}
    wanted = set(frame_indexes)
    found: dict[int, np.ndarray] = {}
    for frame in iter_frames(video):
        if frame.index in wanted:
            found[frame.index] = model.keypoints(frame.image, track.boxes[row_of[frame.index]])
    return np.stack([found[i] for i in frame_indexes])


def prediction_path(video: Path, name: str) -> Path:
    return video.with_name(f"{video.stem}_pred_{name}.npz")


def load_or_predict(video: Path, track: PoseTrack, frame_indexes: list[int], name: str) -> np.ndarray:
    """Predictions are saved next to the video, so each model runs only once."""
    path = prediction_path(video, name)
    if path.exists():
        with np.load(path) as data:
            if list(data["frame_index"]) == list(frame_indexes):
                return data["keypoints"]
    model = load_model(name)
    keypoints = predict(video, track, frame_indexes, model)
    np.savez_compressed(path, frame_index=np.array(frame_indexes), keypoints=keypoints)
    return keypoints


# ---------- scoring ----------


@dataclass(frozen=True)
class GroupScore:
    """Scores of one model on one group of labeled points (e.g. the hidden ones)."""

    n: int
    pck: dict[float, float]  # distance (torso lengths) -> share within it
    median_error: float  # torso lengths


@dataclass(frozen=True)
class ThresholdCount:
    """What a confidence threshold does to the labeled points."""

    threshold: float
    kept_right: int
    kept_wrong: int  # confident but wrong: the dangerous ones
    dropped_right: int  # real points thrown away
    dropped_wrong: int


def errors(labels_xy: np.ndarray, predicted: np.ndarray, torso: float) -> np.ndarray:
    """(F, 4) distance from each labeled point to the model's, in torso lengths.

    labels_xy: (F, 4, 2), NaN where not labeled. predicted: (F, 17, 3).
    """
    points = predicted[:, LABELED_INDEX, :2]
    return np.linalg.norm(points - labels_xy, axis=-1) / torso


def score_group(err: np.ndarray) -> GroupScore:
    err = err[~np.isnan(err)]
    if len(err) == 0:
        return GroupScore(n=0, pck={d: float("nan") for d in PCK_DISTANCES}, median_error=float("nan"))
    return GroupScore(
        n=len(err),
        pck={d: float(np.mean(err <= d)) for d in PCK_DISTANCES},
        median_error=float(np.median(err)),
    )


def threshold_counts(err: np.ndarray, scores: np.ndarray) -> list[ThresholdCount]:
    """err, scores: (F, 4); only labeled points (err not NaN) count."""
    labeled = ~np.isnan(err)
    right = labeled & (err <= CORRECT_WITHIN)
    wrong = labeled & ~right
    counts = []
    for t in THRESHOLDS:
        kept = scores >= t
        counts.append(
            ThresholdCount(
                threshold=t,
                kept_right=int((kept & right).sum()),
                kept_wrong=int((kept & wrong).sum()),
                dropped_right=int((~kept & right).sum()),
                dropped_wrong=int((~kept & wrong).sum()),
            )
        )
    return counts


@dataclass(frozen=True)
class ModelReport:
    name: str
    all: GroupScore
    visible: GroupScore
    hidden: GroupScore
    per_point: dict[str, GroupScore]
    thresholds: list[ThresholdCount]


def evaluate(
    name: str, labels_xy: np.ndarray, hidden: np.ndarray, predicted: np.ndarray, torso: float
) -> ModelReport:
    err = errors(labels_xy, predicted, torso)
    scores = predicted[:, LABELED_INDEX, 2]
    return ModelReport(
        name=name,
        all=score_group(err),
        visible=score_group(np.where(hidden, np.nan, err)),
        hidden=score_group(np.where(hidden, err, np.nan)),
        per_point={n.value: score_group(err[:, j]) for j, n in enumerate(LABELED_POINTS)},
        thresholds=threshold_counts(err, scores),
    )


def format_report(reports: list[ModelReport], torso: float) -> str:
    """The results as Markdown tables."""
    lines = [
        f"Torso length (the distance unit): {torso:.0f} px. "
        f"PCK@d = share of points within d torso lengths of the label.",
        "",
        "| Model | Points | PCK@0.1 | PCK@0.2 | Median error | Visible PCK@0.2 | Hidden PCK@0.2 (n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.all.n} | {r.all.pck[0.1]:.0%} | {r.all.pck[0.2]:.0%} | "
            f"{r.all.median_error:.2f} | {r.visible.pck[0.2]:.0%} | "
            f"{r.hidden.pck[0.2]:.0%} ({r.hidden.n}) |"
        )
    lines += ["", "| Model | " + " | ".join(n.value for n in LABELED_POINTS) + " |",
              "|---|" + "---|" * len(LABELED_POINTS)]
    for r in reports:
        cells = [f"{r.per_point[n.value].pck[0.2]:.0%}" for n in LABELED_POINTS]
        lines.append(f"| {r.name} (PCK@0.2) | " + " | ".join(cells) + " |")
    lines += ["", "Confidence threshold (right = within 0.2 torso lengths):", "",
              "| Model | Threshold | Kept right | Kept wrong | Dropped right | Dropped wrong |",
              "|---|---|---|---|---|---|"]
    for r in reports:
        for c in r.thresholds:
            lines.append(
                f"| {r.name} | {c.threshold} | {c.kept_right} | {c.kept_wrong} | "
                f"{c.dropped_right} | {c.dropped_wrong} |"
            )
    return "\n".join(lines)


# ---------- command line ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pose models against hand labels.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--predict-only", action="store_true", help="run the models, no scoring")
    args = parser.parse_args()

    track = PoseTrack.load(args.video.with_name(f"{args.video.stem}_pose.npz"))
    labels_path = args.video.with_name(f"{args.video.stem}_labels.json")
    if labels_path.exists():
        labels = LabelSet.model_validate_json(labels_path.read_text())
    else:
        from climbvla.labels import empty_label_set

        labels = empty_label_set(track)
    frame_indexes = [f.frame_index for f in labels.frames]

    predictions = {}
    for name in args.models:
        print(f"{name}: predicting {len(frame_indexes)} frames (saved, so only once)")
        predictions[name] = load_or_predict(args.video, track, frame_indexes, name)
    if args.predict_only:
        return

    done = [i for i, f in enumerate(labels.frames) if f.done]
    print(f"scoring on {len(done)} labeled frames of {len(labels.frames)}")
    _, labels_xy, hidden = label_arrays(labels)
    torso = body_size(hide_guesses(track.keypoints))
    reports = [
        evaluate(name, labels_xy[done], hidden[done], predictions[name][done], torso)
        for name in args.models
    ]
    report = format_report(reports, torso)
    out = args.video.with_name(f"{args.video.stem}_pose_eval.md")
    out.write_text(report + "\n")
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
