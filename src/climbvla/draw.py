"""Drawing skeletons on images, to check results by eye."""

import cv2
import numpy as np

from climbvla.pose import PixelPose
from climbvla.schemas import MIN_KEYPOINT_CONFIDENCE, KeypointName

K = KeypointName

# Which keypoints are joined by a line ("bones"), as in the usual COCO drawing.
SKELETON = [
    (K.LEFT_SHOULDER, K.RIGHT_SHOULDER),
    (K.LEFT_HIP, K.RIGHT_HIP),
    (K.LEFT_SHOULDER, K.LEFT_HIP),
    (K.RIGHT_SHOULDER, K.RIGHT_HIP),
    (K.LEFT_SHOULDER, K.LEFT_ELBOW),
    (K.LEFT_ELBOW, K.LEFT_WRIST),
    (K.RIGHT_SHOULDER, K.RIGHT_ELBOW),
    (K.RIGHT_ELBOW, K.RIGHT_WRIST),
    (K.LEFT_HIP, K.LEFT_KNEE),
    (K.LEFT_KNEE, K.LEFT_ANKLE),
    (K.RIGHT_HIP, K.RIGHT_KNEE),
    (K.RIGHT_KNEE, K.RIGHT_ANKLE),
    (K.NOSE, K.LEFT_EYE),
    (K.NOSE, K.RIGHT_EYE),
    (K.LEFT_EYE, K.LEFT_EAR),
    (K.RIGHT_EYE, K.RIGHT_EAR),
]

# Colors are BGR in OpenCV.
LEFT_COLOR = (255, 160, 0)  # blue-ish: the climber's left side
RIGHT_COLOR = (0, 140, 255)  # orange: the climber's right side
CENTER_COLOR = (200, 200, 200)
LOW_CONFIDENCE_COLOR = (0, 0, 255)  # red: below the visibility threshold
BOX_COLOR = (0, 255, 0)


def _side_color(name: KeypointName) -> tuple[int, int, int]:
    if name.startswith("left"):
        return LEFT_COLOR
    if name.startswith("right"):
        return RIGHT_COLOR
    return CENTER_COLOR


def draw_pose(image: np.ndarray, pose: PixelPose) -> np.ndarray:
    """Return a copy of the image with the box and skeleton drawn on it.

    Left side blue, right side orange, so left/right swaps are easy to spot.
    Keypoints below MIN_KEYPOINT_CONFIDENCE are drawn as small red circles,
    and the bones touching them are not drawn.
    """
    out = image.copy()
    size = max(image.shape[:2])
    thickness = max(2, size // 400)
    radius = max(3, size // 250)

    b = pose.box
    cv2.rectangle(out, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), BOX_COLOR, thickness)

    def point(name):
        x, y, _ = pose.keypoints[name]
        return int(round(x)), int(round(y))

    def visible(name):
        return pose.keypoints[name][2] >= MIN_KEYPOINT_CONFIDENCE

    for a, b_ in SKELETON:
        if visible(a) and visible(b_):
            color = _side_color(a) if _side_color(a) == _side_color(b_) else CENTER_COLOR
            cv2.line(out, point(a), point(b_), color, thickness, cv2.LINE_AA)

    for name in pose.keypoints:
        if visible(name):
            cv2.circle(out, point(name), radius, _side_color(name), -1, cv2.LINE_AA)
        else:
            cv2.circle(out, point(name), radius, LOW_CONFIDENCE_COLOR, 1, cv2.LINE_AA)
    return out
