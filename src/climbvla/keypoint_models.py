"""The keypoint models: they find the 17 COCO keypoints inside a person box.

All three were compared on hand labels in step 2.8 (see pose_eval.py):
    - rtmpose-x:   RTMPose-x, run with onnxruntime on the CPU: the one
                   the pipeline uses since 2.8. As accurate as the biggest
                   ViTPose, 3 to 12 times faster, and its confidence drops when
                   it is wrong, so a threshold turns its mistakes into
                   "unknown" rather than wrong positions;
    - vitpose-b:   ViTPose base, the model used until 2.8;
    - vitpose-h:   ViTPose+ huge, the same family, about 7x bigger.
Each model is given the same person box, found by the detector in pose.py.
"""

import urllib.request
import zipfile
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from climbvla.pose import check_keypoint_order, default_device

class KeypointModel(Protocol):
    """A model that finds the 17 COCO keypoints inside a given person box."""

    name: str

    def keypoints(self, image_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
        """box: (x, y, w, h) in pixels. Returns (17, 3): x, y, score."""
        ...




class ViTPoseModel:
    """ViTPose (base) or ViTPose+ (huge), from Hugging Face."""

    def __init__(self, name: str, repo: str, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, VitPoseForPoseEstimation

        self.name = name
        self.device = device or default_device()
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = VitPoseForPoseEstimation.from_pretrained(repo).to(self.device).eval()
        check_keypoint_order(self.model.config.id2label)
        # ViTPose+ mixes experts trained on several datasets: index 0 is COCO.
        self.dataset_index = torch.tensor([0], device=self.device) if "plus" in repo else None

    def keypoints(self, image_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        boxes = [[[float(v) for v in box[:4]]]]
        with torch.no_grad():
            inputs = self.processor(image, boxes=boxes, return_tensors="pt").to(self.device)
            if self.dataset_index is not None:
                inputs["dataset_index"] = self.dataset_index
            outputs = self.model(**inputs)
        result = self.processor.post_process_pose_estimation(outputs, boxes=boxes)[0][0]
        points = result["keypoints"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        return np.column_stack([points, scores]).astype(float)


RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
)
RTMPOSE_DIR = Path.home() / ".cache" / "climbvla" / "rtmpose-x"


def rtmpose_file() -> Path:
    """The RTMPose-x model file, downloaded once (184 MB) into ~/.cache/climbvla."""
    found = sorted(RTMPOSE_DIR.glob("**/end2end.onnx"))
    if found:
        return found[0]
    RTMPOSE_DIR.mkdir(parents=True, exist_ok=True)
    archive = RTMPOSE_DIR / "rtmpose-x.zip"
    print(f"downloading RTMPose-x (184 MB) to {RTMPOSE_DIR}")
    urllib.request.urlretrieve(RTMPOSE_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(RTMPOSE_DIR)
    return sorted(RTMPOSE_DIR.glob("**/end2end.onnx"))[0]


def rtmpose_affine(box: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, np.ndarray]:
    """The transform from the image to the model's input, as mmpose does it.

    The box is enlarged by 1.25 and widened or heightened to the input's
    shape, then scaled (no rotation). Returns (2x3 matrix, scale, box center).
    """
    x, y, w, h = box[:4]
    center = np.array([x + w / 2, y + h / 2])
    box_w, box_h = w * 1.25, h * 1.25
    if box_w / box_h > width / height:
        box_h = box_w * height / width
    else:
        box_w = box_h * width / height
    scale = width / box_w  # = height / box_h
    matrix = np.array(
        [[scale, 0, width / 2 - scale * center[0]], [0, scale, height / 2 - scale * center[1]]]
    )
    return matrix, scale, center


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, split: float = 2.0
) -> np.ndarray:
    """RTMPose's output -> (K, 3) x, y in input pixels, and score.

    SimCC predicts, for each keypoint, one score per possible x position and
    one per y position (at `split` times the input resolution); the position
    is the best of each, and the score the lower of the two best scores.
    """
    x = np.argmax(simcc_x, axis=1) / split
    y = np.argmax(simcc_y, axis=1) / split
    score = np.minimum(simcc_x.max(axis=1), simcc_y.max(axis=1))
    return np.column_stack([x, y, score])


class RTMPoseModel:
    """RTMPose-x, run with onnxruntime from the published ONNX file."""

    WIDTH, HEIGHT = 288, 384
    MEAN = np.array([123.675, 116.28, 103.53])  # RGB
    STD = np.array([58.395, 57.12, 57.375])

    def __init__(self, name: str = "rtmpose-x"):
        import onnxruntime as ort

        self.name = name
        # The CPU, on purpose: CoreML is faster (0.02 s per frame instead of
        # 0.07 s) but computes at lower precision; on 12 frames its points moved
        # by up to 3 px and its scores by up to 0.05, enough to change which
        # points pass the 0.5 threshold that step 2.8 measured on the CPU.
        self.session = ort.InferenceSession(str(rtmpose_file()), providers=["CPUExecutionProvider"])

    def keypoints(self, image_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
        matrix, scale, center = rtmpose_affine(box, self.WIDTH, self.HEIGHT)
        crop = cv2.warpAffine(image_bgr, matrix, (self.WIDTH, self.HEIGHT), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        simcc_x, simcc_y = self.session.run(None, {"input": tensor})
        points = decode_simcc(simcc_x[0], simcc_y[0])
        # back from the input to the image
        points[:, 0] = (points[:, 0] - self.WIDTH / 2) / scale + center[0]
        points[:, 1] = (points[:, 1] - self.HEIGHT / 2) / scale + center[1]
        return points


def load_model(name: str) -> KeypointModel:
    if name == "vitpose-b":
        return ViTPoseModel(name, "usyd-community/vitpose-base-simple")
    if name == "vitpose-h":
        return ViTPoseModel(name, "usyd-community/vitpose-plus-huge")
    if name == "rtmpose-x":
        return RTMPoseModel(name)
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ["vitpose-b", "vitpose-h", "rtmpose-x"]
