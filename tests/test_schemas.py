"""Tests for the shared data format. Run with: uv run pytest"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from climbvla.schemas import (
    BodyPart,
    ContactEvent,
    Hold,
    HoldType,
    KeypointName,
    Move,
    PoseFrame,
    Route,
)

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


def test_is_visible_uses_confidence_threshold():
    # Climber seen from behind: the nose is a low-confidence guess.
    data = _with_keypoint("nose", {"x": 1.0, "y": 1.8, "confidence": 0.2})
    frame = PoseFrame.model_validate(data)

    assert not frame.is_visible("nose")
    assert frame.is_visible("left_wrist")  # confidence 0.9 from make_pose_data
    assert frame.is_visible("nose", min_confidence=0.1)  # a stage may relax the rule


# ---------- BodyPart and ContactEvent ----------


def test_body_part_maps_to_keypoint():
    assert BodyPart.LEFT_HAND.keypoint is KeypointName.LEFT_WRIST
    assert BodyPart.RIGHT_FOOT.keypoint is KeypointName.RIGHT_ANKLE
    assert BodyPart.LEFT_KNEE.keypoint is KeypointName.LEFT_KNEE


def test_each_body_part_is_exactly_one_kind():
    for part in BodyPart:
        assert [part.is_hand, part.is_foot, part.is_knee].count(True) == 1


def test_knee_contact():
    # knee bar on a volume
    contact = ContactEvent(body_part="right_knee", hold_id="v1", t_start=4.0, t_end=6.5)
    assert contact.body_part.is_knee


def test_contact_round_trip_and_duration():
    contact = ContactEvent(body_part="left_hand", hold_id="h2", t_start=3.4, t_end=5.0)
    assert ContactEvent.model_validate_json(contact.model_dump_json()) == contact
    assert contact.duration == pytest.approx(1.6)


def test_open_contact_has_no_end():
    # e.g. the hands on the top hold when the video stops
    contact = ContactEvent(body_part="right_hand", hold_id="h4", t_start=12.0)
    assert contact.t_end is None and contact.duration is None
    assert contact.is_active(99.0)


def test_is_active_includes_start_excludes_end():
    contact = ContactEvent(body_part="left_foot", hold_id="f1", t_start=1.0, t_end=2.0)
    assert not contact.is_active(0.9)
    assert contact.is_active(1.0)
    assert contact.is_active(1.5)
    assert not contact.is_active(2.0)  # at t_end the body part has left


@pytest.mark.parametrize(
    "bad, message",
    [
        (dict(body_part="left_hand", hold_id="h1", t_start=2.0, t_end=1.0), "must be after"),
        (dict(body_part="left_hand", hold_id="h1", t_start=2.0, t_end=2.0), "must be after"),
        (dict(body_part="left_elbow", hold_id="h1", t_start=0.0), "Input should be"),
        (dict(body_part="left_hand", hold_id="", t_start=0.0), "at least 1 character"),
        (dict(body_part="left_hand", hold_id="h1", t_start=-1.0), "greater than or equal to 0"),
    ],
)
def test_contact_rejects_bad_input(bad, message):
    with pytest.raises(ValidationError, match=message):
        ContactEvent(**bad)


# ---------- Move ----------


def test_move_round_trip_and_duration():
    move = Move(body_part="right_hand", from_hold_id="h1", to_hold_id="h2", t_start=3.0, t_end=3.8)
    assert Move.model_validate_json(move.model_dump_json()) == move
    assert move.duration == pytest.approx(0.8)


def test_first_move_comes_from_the_ground():
    move = Move(body_part="left_foot", from_hold_id=None, to_hold_id="f1", t_start=0.0, t_end=0.5)
    assert move.from_hold_id is None


def test_from_hold_must_be_given_explicitly():
    # No default on purpose: forgetting it is an error, not a silent "from the ground".
    with pytest.raises(ValidationError, match="from_hold_id"):
        Move(body_part="left_hand", to_hold_id="h1", t_start=0.0, t_end=1.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(t_end=2.0), "must be after"),  # ends before it starts
        (dict(to_hold_id="h1"), "must change hold"),  # same hold
        (dict(body_part="left_elbow"), "Input should be"),
        (dict(to_hold_id=""), "at least 1 character"),
    ],
)
def test_move_rejects_bad_input(overrides, message):
    data = dict(body_part="left_hand", from_hold_id="h1", to_hold_id="h2", t_start=2.5, t_end=3.0)
    data.update(overrides)
    with pytest.raises(ValidationError, match=message):
        Move(**data)
