"""Tests for the shared data format. Run with: uv run pytest"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from climbvla.schemas import Hold, HoldType, KeypointName, PoseFrame, Route

EXAMPLES = Path(__file__).parent.parent / "examples"


# ---------- Hold ----------


def test_hold_round_trip():
    hold = Hold(id="h01", x=1.2, y=0.85, radius=0.08, hold_type="jug")
    assert Hold.model_validate_json(hold.model_dump_json()) == hold
    assert hold.hold_type is HoldType.JUG


@pytest.mark.parametrize(
    "bad",
    [
        dict(id="h", x=1, y=1, radius=-0.05),  # negative radius
        dict(id="h", x=1, y=1, radius=0.05, hold_type="banana"),  # unknown type
        dict(id="h", x=1, y=1, raduis=0.05),  # typo in field name
        dict(id="", x=1, y=1, radius=0.05),  # empty id
    ],
)
def test_hold_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        Hold(**bad)


# ---------- Route ----------


@pytest.fixture
def route() -> Route:
    """The hand-written example route, loaded from disk."""
    return Route.model_validate_json((EXAMPLES / "route_example.json").read_text())


def test_example_route_loads(route):
    assert len(route.holds) == 6
    assert route.hold("h4").y == pytest.approx(3.1)


def test_route_round_trip(route):
    assert Route.model_validate_json(route.model_dump_json()) == route


def test_unknown_hold_lookup_raises(route):
    with pytest.raises(KeyError):
        route.hold("nope")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["holds"].append(dict(d["holds"][0])), "duplicate hold ids"),
        (lambda d: d.update(top_hold_id="h9"), "not in holds"),
        (lambda d: d.update(start_hold_ids=["h1", "h4"]), "cannot also be a start hold"),
        (lambda d: d["holds"][0].update(radius=0), "greater than 0"),
    ],
)
def test_route_rejects_bad_input(route, change, message):
    data = route.model_dump()
    change(data)
    with pytest.raises(ValidationError, match=message):
        Route.model_validate(data)


# ---------- PoseFrame ----------


def make_pose_data(**overrides) -> dict:
    """Raw data for a valid PoseFrame: every keypoint at (1.0, 1.5), confidence 0.9."""
    data = {
        "frame_index": 12,
        "t": 0.4,
        "keypoints": {name.value: {"x": 1.0, "y": 1.5, "confidence": 0.9} for name in KeypointName},
    }
    data.update(overrides)
    return data


def test_pose_frame_round_trip():
    frame = PoseFrame.model_validate(make_pose_data())
    assert PoseFrame.model_validate_json(frame.model_dump_json()) == frame


def test_pose_frame_lookup_by_name():
    frame = PoseFrame.model_validate(make_pose_data())
    assert frame["left_wrist"] is frame.keypoints[KeypointName.LEFT_WRIST]


def test_keypoint_confidence_defaults_to_one():
    data = make_pose_data()
    data["keypoints"]["nose"] = {"x": 1.0, "y": 1.8}
    assert PoseFrame.model_validate(data)["nose"].confidence == 1.0


def _with_keypoint(name, value):
    data = make_pose_data()
    data["keypoints"][name] = value
    return data


def _without_keypoint(name):
    data = make_pose_data()
    del data["keypoints"][name]
    return data


@pytest.mark.parametrize(
    "data, message",
    [
        (make_pose_data(t=-0.1), "greater than or equal to 0"),
        (make_pose_data(frame_index=-1), "greater than or equal to 0"),
        (_with_keypoint("nose", {"x": 1, "y": 1, "confidence": 1.5}), "less than or equal to 1"),
        (_with_keypoint("left_tail", {"x": 1, "y": 1}), "Input should be"),
        (_without_keypoint("right_ankle"), "missing keypoints: \\['right_ankle'\\]"),
    ],
)
def test_pose_frame_rejects_bad_input(data, message):
    with pytest.raises(ValidationError, match=message):
        PoseFrame.model_validate(data)
