"""Drawing skeletons, holds and text on images, to check results by eye."""

import cv2
import numpy as np

from climbvla.clean import CleanPose
from climbvla.pose import COCO_ORDER, PixelPose
from climbvla.schemas import MIN_KEYPOINT_CONFIDENCE, KeypointName, PixelContact, PixelHold

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


SWAPPED_COLOR = (255, 255, 255)  # white ring: left and right were swapped back here


def draw_clean_pose(image: np.ndarray, pose: CleanPose, row: int) -> np.ndarray:
    """Return a copy of the image with row `row` of a cleaned pose drawn on it.

    Seen points are filled dots, interpolated points hollow circles, points
    swapped back get a white ring. Unknown points (NaN) are not drawn at all,
    nor the bones touching them.
    """
    out = image.copy()
    size = max(image.shape[:2])
    thickness = max(2, size // 400)
    radius = max(3, size // 250)
    index = {name: j for j, name in enumerate(COCO_ORDER)}

    def known(name):
        return not np.isnan(pose.xy[row, index[name]]).any()

    def point(name):
        x, y = pose.xy[row, index[name]]
        return int(round(x)), int(round(y))

    for a, b in SKELETON:
        if known(a) and known(b):
            color = _side_color(a) if _side_color(a) == _side_color(b) else CENTER_COLOR
            cv2.line(out, point(a), point(b), color, thickness, cv2.LINE_AA)

    for name, j in index.items():
        if not known(name):
            continue
        filled = pose.filled[row, j]
        cv2.circle(out, point(name), radius, _side_color(name), thickness if filled else -1, cv2.LINE_AA)
        if pose.swapped[row, j]:
            cv2.circle(out, point(name), radius * 2, SWAPPED_COLOR, thickness, cv2.LINE_AA)
    return out


# ---------- text ----------


def _text_scale(image: np.ndarray) -> float:
    # readable on large videos; below 0.4 OpenCV's font is unreadable
    return max(0.4, max(image.shape[:2]) / 1200)


def _put_text(
    image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int]
) -> None:
    thickness = max(1, round(2 * scale))
    # a black outline under the text keeps it readable on any background
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_text_lines(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Write (text, color) lines in the top-left corner, IN PLACE."""
    scale = _text_scale(image)
    margin = round(20 * scale)
    line = round(45 * scale)
    for i, (text, color) in enumerate(lines):
        _put_text(image, text, (margin, margin + line * (i + 1)), scale, color)


# ---------- holds ----------

HOLD_COLOR = (0, 255, 255)  # yellow
START_COLOR = (0, 200, 0)  # green
TOP_COLOR = (255, 0, 255)  # magenta


def draw_holds(
    image: np.ndarray,
    holds: list[PixelHold],
    start_hold_ids: list[str] | tuple[str, ...] = (),
    top_hold_id: str | None = None,
) -> np.ndarray:
    """Return a copy of the image with each hold's circle and id.

    Start holds are green, the top hold magenta, the others yellow.
    """
    out = image.copy()
    scale = _text_scale(image)
    thickness = max(2, round(2 * scale))
    for h in holds:
        if h.id == top_hold_id:
            color = TOP_COLOR
        elif h.id in start_hold_ids:
            color = START_COLOR
        else:
            color = HOLD_COLOR
        center = (round(h.x), round(h.y))
        cv2.circle(out, center, round(h.radius), color, thickness, cv2.LINE_AA)
        label_origin = (round(h.x + h.radius * 0.7), round(h.y - h.radius * 0.7))
        _put_text(out, h.id, label_origin, scale * 0.8, color)
    return out


# ---------- contacts ----------

CONTACT_COLORS = {"hold": (0, 220, 0), "wall": (200, 200, 200), "ground": (40, 90, 160)}  # BGR
PART_SHORT = {"left_hand": "LH", "right_hand": "RH", "left_foot": "LF", "right_foot": "RF"}


def draw_contacts(image: np.ndarray, contacts: list[PixelContact], t: float) -> np.ndarray:
    """Return a copy of the image with the contacts active at time t.

    Each one is a thick ring at the median contact point, labeled e.g. "LH h4".
    """
    out = image.copy()
    scale = _text_scale(image)
    radius = round(28 * scale)
    for c in contacts:
        if not c.t_start <= t < c.t_end:
            continue
        center = (round(c.x), round(c.y))
        cv2.circle(out, center, radius, CONTACT_COLORS[c.kind], max(3, round(5 * scale)), cv2.LINE_AA)
        target = c.hold_id if c.kind == "hold" else c.kind
        label = f"{PART_SHORT.get(c.body_part.value, c.body_part.value)} {target}"
        _put_text(out, label, (center[0] + radius, center[1] + radius), scale * 0.8, CONTACT_COLORS[c.kind])
    return out


# ---------- center of mass (step 2.11) ----------

COM_COLOR = (255, 255, 255)


def draw_com(image: np.ndarray, com: tuple[float, float]) -> np.ndarray:
    """Return a copy of the image with the center of mass as a white dot
    (unchanged if the COM is unknown)."""
    out = image.copy()
    if np.isnan(com[0]):
        return out
    scale = _text_scale(image)
    center, radius = (round(com[0]), round(com[1])), round(12 * scale)
    cv2.circle(out, center, radius + 3, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(out, center, radius, COM_COLOR, -1, cv2.LINE_AA)
    return out
