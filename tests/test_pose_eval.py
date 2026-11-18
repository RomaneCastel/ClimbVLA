"""Tests for comparing pose models with hand labels. No model is loaded."""

import numpy as np
import pytest

from climbvla.labels import LABELED_POINTS
from climbvla.pose import COCO_ORDER
from climbvla.keypoint_models import decode_simcc, rtmpose_affine
from climbvla.pose_eval import (
    errors,
    evaluate,
    format_report,
    score_group,
    threshold_counts,
)

NAN = np.nan
TORSO = 100.0


def predictions(points: list[tuple[float, float, float]]) -> np.ndarray:
    """(1, 17, 3) with the 4 labeled points set, in LABELED_POINTS order."""
    p = np.zeros((1, 17, 3))
    for name, point in zip(LABELED_POINTS, points):
        p[0, COCO_ORDER.index(name)] = point
    return p


def test_errors_are_in_torso_lengths():
    labels = np.array([[[0.0, 0], [100, 100], [NAN, NAN], [50, 50]]])
    pred = predictions([(10, 0, 0.9), (100, 160, 0.9), (0, 0, 0.9), (50, 50, 0.9)])
    err = errors(labels, pred, TORSO)
    assert err[0, 0] == pytest.approx(0.1)
    assert err[0, 1] == pytest.approx(0.6)
    assert np.isnan(err[0, 2])  # not labeled: not scored
    assert err[0, 3] == 0


def test_score_group():
    score = score_group(np.array([0.05, 0.15, 0.5, NAN]))
    assert score.n == 3
    assert score.pck[0.1] == pytest.approx(1 / 3)
    assert score.pck[0.2] == pytest.approx(2 / 3)
    assert score.median_error == pytest.approx(0.15)
    assert score_group(np.array([NAN])).n == 0


def test_threshold_counts():
    err = np.array([[0.05, 0.5, 0.1, 0.9]])  # right, wrong, right, wrong
    scores = np.array([[0.9, 0.65, 0.55, 0.3]])
    at = {c.threshold: c for c in threshold_counts(err, scores)}
    # 0.5 keeps the first three: two right, one wrong (the confident 0.65 one)
    assert (at[0.5].kept_right, at[0.5].kept_wrong, at[0.5].dropped_right, at[0.5].dropped_wrong) == (2, 1, 0, 1)
    # 0.7 drops the confident wrong one, but also a right one
    assert (at[0.7].kept_right, at[0.7].kept_wrong, at[0.7].dropped_right, at[0.7].dropped_wrong) == (1, 0, 1, 2)


def test_visible_and_hidden_are_scored_apart():
    labels = np.array([[[0.0, 0], [100, 100], [0, 0], [0, 0]]])
    hidden = np.array([[False, True, False, False]])
    pred = predictions([(0, 0, 0.9), (100, 400, 0.6), (0, 0, 0.9), (0, 0, 0.9)])
    report = evaluate("fake", labels, hidden, pred, TORSO)
    assert report.visible.n == 3 and report.visible.pck[0.2] == 1
    assert report.hidden.n == 1 and report.hidden.pck[0.2] == 0
    assert report.per_point["right_wrist"].pck[0.2] == 0
    assert "| fake |" in format_report([report], TORSO)


def test_decode_simcc():
    simcc_x = np.zeros((2, 20))
    simcc_y = np.zeros((2, 30))
    simcc_x[0, 10], simcc_y[0, 6] = 0.9, 0.7  # keypoint 0 at (5, 3), score 0.7
    simcc_x[1, 4], simcc_y[1, 20] = 0.4, 0.8  # keypoint 1 at (2, 10), score 0.4
    points = decode_simcc(simcc_x, simcc_y)
    assert points[0] == pytest.approx([5, 3, 0.7])
    assert points[1] == pytest.approx([2, 10, 0.4])


def test_rtmpose_affine_maps_the_box_center_to_the_input_center():
    box = np.array([100.0, 200, 100, 300])  # x, y, w, h: center (150, 350)
    matrix, scale, center = rtmpose_affine(box, 288, 384)
    assert tuple(center) == (150, 350)
    mapped = matrix @ np.array([150, 350, 1])
    assert mapped == pytest.approx([144, 192])  # the middle of a 288x384 input
    # the enlarged box (1.25x the height here) fills the input's height
    assert scale == pytest.approx(384 / (300 * 1.25))
