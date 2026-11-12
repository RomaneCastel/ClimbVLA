"""Finding the climber's skeleton in an image.

ViTPose is a "top-down" pose model, so it works in two stages:
    1. a person detector (RT-DETR) finds a box around each person;
    2. ViTPose looks inside the climber's box and finds the 17 COCO keypoints.

Everything here is in PIXELS, with y pointing DOWN, as in the image.
Converting to wall-plane meters is a later step.

Try it:
    uv run python -m climbvla.pose photo.jpg
    uv run python -m climbvla.pose climb.mp4 --t 3.5
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

from climbvla.schemas import KeypointName

DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL = "usyd-community/vitpose-base-simple"

# The 17 keypoints in the order ViTPose outputs them (the COCO order).
# KeypointName was written in the same order.
COCO_ORDER = list(KeypointName)


@dataclass(frozen=True)
class PersonBox:
    """A detected person, in pixels: top-left corner, width and height."""

    x: float
    y: float
    w: float
    h: float
    score: float  # detector confidence, 0 to 1


@dataclass(frozen=True)
class PixelPose:
    """The climber's 17 keypoints in one image, in pixels."""

    box: PersonBox
    keypoints: dict[KeypointName, tuple[float, float, float]]  # name -> (x, y, confidence)


def pick_climber(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, person_label: int
) -> PersonBox | None:
    """Choose the climber among the detections: the most confident person.

    boxes are (x1, y1, x2, y2) corners. Returns None if no person was found.
    With one climber filmed up close, the most confident person is almost
    always the climber; people in the background are smaller and less sure.
    """
    is_person = labels == person_label
    if not is_person.any():
        return None
    best = np.flatnonzero(is_person)[np.argmax(scores[is_person])]
    x1, y1, x2, y2 = (float(v) for v in boxes[best])
    return PersonBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, score=float(scores[best]))


def check_keypoint_order(id2label: dict[int, str]) -> None:
    """Make sure the model outputs keypoints in COCO_ORDER.

    The model names them like 'L_Wrist'; we name them 'left_wrist'. A model
    with another order would silently swap joints, so fail loudly instead.
    """
    def ours(label: str) -> str:
        label = label.lower()
        for short, long in (("l_", "left_"), ("r_", "right_")):
            if label.startswith(short):
                return long + label[len(short) :]
        return label

    model_names = [ours(id2label[i]) for i in range(len(id2label))]
    if model_names != [k.value for k in COCO_ORDER]:
        raise ValueError(f"unexpected keypoint order from the pose model: {model_names}")


def keypoints_from_arrays(
    points: np.ndarray, scores: np.ndarray
) -> dict[KeypointName, tuple[float, float, float]]:
    """Turn the model's 17x2 points and 17 scores into {name: (x, y, confidence)}.

    ViTPose's scores are heatmap peaks, not true probabilities, and can come
    out slightly above 1 (1.01 was seen on a real video). They are clipped to
    0..1 so they fit the Keypoint.confidence rule.
    """
    scores = np.clip(scores, 0.0, 1.0)
    return {
        name: (float(points[i, 0]), float(points[i, 1]), float(scores[i]))
        for i, name in enumerate(COCO_ORDER)
    }


def default_device() -> str:
    """The Mac's GPU (Apple 'mps') if available, else the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PoseEstimator:
    """Loads both models once; then call estimate() on as many images as needed."""

    def __init__(self, device: str | None = None, detection_threshold: float = 0.3):
        self.device = device or default_device()
        self.detection_threshold = detection_threshold

        # First use downloads the weights (~170 MB in total) into the
        # Hugging Face cache (~/.cache/huggingface); later uses read them from there.
        self.det_processor = AutoImageProcessor.from_pretrained(DETECTOR_MODEL)
        self.detector = RTDetrForObjectDetection.from_pretrained(DETECTOR_MODEL).to(self.device).eval()
        self.person_label = self.detector.config.label2id["person"]

        self.pose_processor = AutoImageProcessor.from_pretrained(POSE_MODEL)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(self.device).eval()
        check_keypoint_order(self.pose_model.config.id2label)

    @torch.no_grad()  # inference only: no gradients, less memory, faster
    def detect_climber(self, image: Image.Image) -> PersonBox | None:
        inputs = self.det_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        result = self.det_processor.post_process_object_detection(
            outputs,
            threshold=self.detection_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        return pick_climber(
            result["boxes"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            result["labels"].cpu().numpy(),
            self.person_label,
        )

    @torch.no_grad()
    def estimate(self, image_bgr: np.ndarray) -> PixelPose | None:
        """Find the climber's skeleton in an OpenCV (BGR) image. None if no person."""
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        box = self.detect_climber(image)
        if box is None:
            return None

        boxes = [[[box.x, box.y, box.w, box.h]]]  # one image, one box, COCO (x, y, w, h)
        inputs = self.pose_processor(image, boxes=boxes, return_tensors="pt").to(self.device)
        outputs = self.pose_model(**inputs)
        result = self.pose_processor.post_process_pose_estimation(outputs, boxes=boxes)[0][0]

        keypoints = keypoints_from_arrays(
            result["keypoints"].cpu().numpy(), result["scores"].cpu().numpy()
        )
        return PixelPose(box=box, keypoints=keypoints)


# ---------- command line ----------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: Path, t: float) -> np.ndarray:
    """An image file as is, or the frame of a video at time t."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot read {path} as an image")
        return image
    from climbvla.video import iter_frames

    frame = next(iter_frames(path, start=t), None)
    if frame is None:
        raise ValueError(f"{path} has no frame at or after {t} s")
    return frame.image


def main() -> None:
    from climbvla.draw import draw_pose

    parser = argparse.ArgumentParser(description="Draw the climber's skeleton on one image.")
    parser.add_argument("input", type=Path, help="an image, or a video")
    parser.add_argument("--t", type=float, default=0.0, help="for a video: time in seconds")
    parser.add_argument("--out", type=Path, help="output image (default: <input>_pose.png)")
    args = parser.parse_args()

    image = load_image(args.input, args.t)
    estimator = PoseEstimator()
    print(f"running on {estimator.device}")
    pose = estimator.estimate(image)
    if pose is None:
        print("no person found")
        return

    print(f"person box score {pose.box.score:.2f}")
    for name, (x, y, conf) in pose.keypoints.items():
        print(f"  {name:15} x={x:7.1f}  y={y:7.1f}  confidence={conf:.2f}")

    out = args.out or args.input.with_name(f"{args.input.stem}_pose.png")
    cv2.imwrite(str(out), draw_pose(image, pose))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
