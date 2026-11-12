"""Tests for pose estimation and drawing.

The parts that need the neural networks are slow (model loading takes about a
minute), so they only run when asked:  uv run pytest -m slow
"""

import numpy as np
import pytest

from climbvla.draw import draw_pose
from climbvla.pose import (
    COCO_ORDER,
    PersonBox,
    PixelPose,
    box_overlap,
    check_keypoint_order,
    keypoints_from_arrays,
    pick_climber,
)
from climbvla.schemas import KeypointName

PERSON, DOG = 0, 16

# ---------- choosing the climber ----------


def test_pick_climber_takes_most_confident_person():
    boxes = np.array([[0, 0, 10, 20], [50, 60, 150, 360], [5, 5, 8, 9]], dtype=float)
    scores = np.array([0.4, 0.9, 0.95])
    labels = np.array([PERSON, PERSON, DOG])  # the dog is more confident but is not a person
    box = pick_climber(boxes, scores, labels, PERSON)
    assert (box.x, box.y, box.w, box.h) == (50, 60, 100, 300)  # corners -> x, y, width, height
    assert box.score == pytest.approx(0.9)


def test_pick_climber_without_person():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    assert pick_climber(boxes, np.array([0.9]), np.array([DOG]), PERSON) is None
    assert pick_climber(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), PERSON) is None


def box(x, y, w, h, score=0.9) -> PersonBox:
    return PersonBox(x=x, y=y, w=w, h=h, score=score)


def test_box_overlap():
    assert box_overlap(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_overlap(box(0, 0, 10, 10), box(20, 0, 10, 10)) == 0.0  # apart
    assert box_overlap(box(0, 0, 10, 10), box(10, 0, 10, 10)) == 0.0  # touching edges
    # half of each box is shared: 50 shared / 150 in total
    assert box_overlap(box(0, 0, 10, 10), box(5, 0, 10, 10)) == pytest.approx(1 / 3)


# Seen on a real video: the climber high on the wall, a bystander on the floor
# who is more confident on this frame.
CLIMBER_THEN = box(100, 100, 200, 400)
SCENE = np.array([[105, 110, 305, 500], [400, 900, 460, 1100]], dtype=float)  # corners
SCENE_SCORES = np.array([0.85, 0.95])


def test_pick_climber_follows_the_previous_box():
    labels = np.array([PERSON, PERSON])
    # without a previous box, the bystander wins (most confident)...
    assert pick_climber(SCENE, SCENE_SCORES, labels, PERSON).x == 400
    # ...with the climber's previous box, the climber is kept
    picked = pick_climber(SCENE, SCENE_SCORES, labels, PERSON, previous=CLIMBER_THEN)
    assert (picked.x, picked.score) == (105, pytest.approx(0.85))


def test_pick_climber_none_when_only_others_are_seen():
    # only the bystander is detected: better no pose than the wrong person's
    bystander_only = SCENE[1:]
    assert (
        pick_climber(bystander_only, SCENE_SCORES[1:], np.array([PERSON]), PERSON, CLIMBER_THEN)
        is None
    )


# ---------- keypoint order ----------

VITPOSE_LABELS = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow", 9: "L_Wrist", 10: "R_Wrist",
    11: "L_Hip", 12: "R_Hip", 13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle",
}  # copied from the vitpose-base-simple config


def test_vitpose_order_is_accepted():
    check_keypoint_order(VITPOSE_LABELS)


def test_swapped_order_is_rejected():
    swapped = dict(VITPOSE_LABELS)
    swapped[9], swapped[10] = swapped[10], swapped[9]  # wrists swapped
    with pytest.raises(ValueError, match="unexpected keypoint order"):
        check_keypoint_order(swapped)


def test_keypoints_from_arrays_names_and_clips():
    points = np.arange(34, dtype=float).reshape(17, 2)
    scores = np.full(17, 0.8)
    scores[9] = 1.01  # seen on a real video
    scores[0] = -0.02
    kp = keypoints_from_arrays(points, scores)
    assert list(kp) == COCO_ORDER
    assert kp[KeypointName.LEFT_WRIST] == (18.0, 19.0, 1.0)
    assert kp[KeypointName.NOSE][2] == 0.0


# ---------- drawing ----------


def make_pose(confidence: float = 0.9) -> PixelPose:
    keypoints = {name: (100.0 + 10 * i, 50.0 + 15 * i, confidence) for i, name in enumerate(COCO_ORDER)}
    return PixelPose(box=PersonBox(x=80, y=30, w=200, h=300, score=0.9), keypoints=keypoints)


def test_draw_pose_draws_on_a_copy():
    image = np.zeros((400, 400, 3), np.uint8)
    out = draw_pose(image, make_pose())
    assert out.shape == image.shape
    assert out.any()  # something was drawn
    assert not image.any()  # the original is untouched


def test_low_confidence_points_are_red_and_unjoined():
    pose = make_pose(confidence=0.1)
    out = draw_pose(np.zeros((400, 400, 3), np.uint8), pose)
    x, y, _ = pose.keypoints[KeypointName.LEFT_WRIST]
    # a red outline is drawn around the point...
    patch = out[int(y) - 6 : int(y) + 7, int(x) - 6 : int(x) + 7]
    assert (patch[..., 2] > 200).any()
    # ...but no bone leaves it: halfway to the elbow, the image is still black
    ex, ey, _ = pose.keypoints[KeypointName.LEFT_ELBOW]
    assert not out[int((y + ey) / 2), int((x + ex) / 2)].any()


# ---------- the real models (slow) ----------


@pytest.mark.slow
def test_models_find_nobody_in_a_black_image():
    from climbvla.pose import PoseEstimator

    assert PoseEstimator().estimate(np.zeros((480, 640, 3), np.uint8)) is None
