"""Tests for the shared data format. Run with: uv run pytest"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from climbvla.schemas import (
    Audit,
    BodyPart,
    ContactEvent,
    GroundTarget,
    Hold,
    HoldTarget,
    HoldType,
    KeypointName,
    Move,
    PoseFrame,
    Route,
    Trajectory,
    WallTarget,
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


# ---------- BodyPart ----------


def test_body_part_maps_to_keypoint():
    assert BodyPart.LEFT_HAND.keypoint is KeypointName.LEFT_WRIST
    assert BodyPart.RIGHT_FOOT.keypoint is KeypointName.RIGHT_ANKLE
    assert BodyPart.LEFT_KNEE.keypoint is KeypointName.LEFT_KNEE


def test_each_body_part_is_exactly_one_kind():
    for part in BodyPart:
        assert [part.is_hand, part.is_foot, part.is_knee].count(True) == 1


# ---------- Targets ----------


def on(hold_id: str) -> dict:
    """JSON-style target for a hold, to keep the tests short."""
    return {"kind": "hold", "hold_id": hold_id}


SMEAR = {"kind": "wall", "x": 1.3, "y": 0.5}
GROUND = {"kind": "ground"}


@pytest.mark.parametrize(
    "data, expected_class",
    [(on("h1"), HoldTarget), (SMEAR, WallTarget), (GROUND, GroundTarget)],
)
def test_kind_picks_the_target_class(data, expected_class):
    contact = ContactEvent(body_part="left_foot", target=data, t_start=0.0)
    assert type(contact.target) is expected_class


@pytest.mark.parametrize(
    "bad_target, message",
    [
        ({"kind": "wall", "x": 1.0, "y": 1.0, "hold_id": "h1"}, "Extra inputs"),  # smear with a hold
        ({"kind": "hold"}, "hold_id"),  # hold without an id
        ({"kind": "hold", "hold_id": ""}, "at least 1 character"),
        ({"kind": "ceiling"}, "does not match any of the expected tags"),
        ({"hold_id": "h1"}, "Unable to extract tag"),  # kind missing
    ],
)
def test_bad_targets_are_rejected(bad_target, message):
    with pytest.raises(ValidationError, match=message):
        ContactEvent(body_part="left_foot", target=bad_target, t_start=0.0)


# ---------- ContactEvent ----------


def test_contact_round_trip_and_duration():
    contact = ContactEvent(body_part="left_hand", target=on("h2"), t_start=3.4, t_end=5.0)
    assert ContactEvent.model_validate_json(contact.model_dump_json()) == contact
    assert contact.duration == pytest.approx(1.6)


def test_smear_contact_round_trip():
    contact = ContactEvent(body_part="right_foot", target=SMEAR, t_start=2.0, t_end=3.0)
    back = ContactEvent.model_validate_json(contact.model_dump_json())
    assert back == contact
    assert isinstance(back.target, WallTarget) and back.target.x == pytest.approx(1.3)


def test_knee_contact():
    # knee bar on a volume
    contact = ContactEvent(body_part="right_knee", target=on("v1"), t_start=4.0, t_end=6.5)
    assert contact.body_part.is_knee


def test_open_contact_has_no_end():
    # e.g. the hands on the top hold when the video stops
    contact = ContactEvent(body_part="right_hand", target=on("h4"), t_start=12.0)
    assert contact.t_end is None and contact.duration is None
    assert contact.is_active(99.0)


def test_is_active_includes_start_excludes_end():
    contact = ContactEvent(body_part="left_foot", target=on("f1"), t_start=1.0, t_end=2.0)
    assert not contact.is_active(0.9)
    assert contact.is_active(1.0)
    assert contact.is_active(1.5)
    assert not contact.is_active(2.0)  # at t_end the body part has left


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(t_end=1.0), "must be after"),
        (dict(t_end=2.0), "must be after"),
        (dict(body_part="left_elbow"), "Input should be"),
        (dict(t_start=-1.0), "greater than or equal to 0"),
    ],
)
def test_contact_rejects_bad_input(overrides, message):
    data = dict(body_part="left_hand", target=on("h1"), t_start=2.0)
    data.update(overrides)
    with pytest.raises(ValidationError, match=message):
        ContactEvent(**data)


# ---------- Move ----------


def test_move_round_trip_and_duration():
    move = Move(body_part="right_hand", from_target=on("h1"), to_target=on("h2"), t_start=3.0, t_end=3.8)
    assert Move.model_validate_json(move.model_dump_json()) == move
    assert move.duration == pytest.approx(0.8)


def test_first_move_comes_from_the_ground():
    move = Move(body_part="left_foot", from_target=GROUND, to_target=on("f1"), t_start=0.0, t_end=0.5)
    assert isinstance(move.from_target, GroundTarget)


def test_foot_moves_from_smear_to_hold():
    move = Move(body_part="left_foot", from_target=SMEAR, to_target=on("f2"), t_start=4.0, t_end=4.6)
    assert isinstance(move.from_target, WallTarget)


def test_smear_to_another_smear_is_a_move():
    higher = {"kind": "wall", "x": 1.3, "y": 0.9}
    Move(body_part="right_foot", from_target=SMEAR, to_target=higher, t_start=1.0, t_end=1.4)


def test_from_target_must_be_given():
    with pytest.raises(ValidationError, match="from_target"):
        Move(body_part="left_hand", to_target=on("h1"), t_start=0.0, t_end=1.0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(t_end=2.0), "must be after"),  # ends before it starts
        (dict(to_target=on("h1")), "must change target"),  # same hold
        (dict(from_target=SMEAR, to_target=dict(SMEAR)), "must change target"),  # same wall point
        (dict(from_target=GROUND, to_target=GROUND), "must change target"),
        (dict(body_part="left_elbow"), "Input should be"),
    ],
)
def test_move_rejects_bad_input(overrides, message):
    data = dict(body_part="left_hand", from_target=on("h1"), to_target=on("h2"), t_start=2.5, t_end=3.0)
    data.update(overrides)
    with pytest.raises(ValidationError, match=message):
        Move(**data)


# ---------- Trajectory ----------


def make_trajectory_data() -> dict:
    """A tiny valid climb of the example route, 0 to 2 s.

    Right hand on h1 from the start. Left foot stands on the ground, then
    moves up to foothold f1 between 0.5 s and 1.0 s.
    """
    return {
        "id": "romane-try1",
        "route_id": "example-yellow-01",
        "source": "video",
        "frames": [make_pose_data(frame_index=i, t=i * 0.5) for i in range(5)],
        "contacts": [
            {"body_part": "right_hand", "target": on("h1"), "t_start": 0.0},
            {"body_part": "left_foot", "target": GROUND, "t_start": 0.0, "t_end": 0.5},
            {"body_part": "left_foot", "target": on("f1"), "t_start": 1.0},
        ],
        "moves": [
            {"body_part": "left_foot", "from_target": GROUND, "to_target": on("f1"),
             "t_start": 0.5, "t_end": 1.0},
        ],
    }


def test_trajectory_round_trip():
    traj = Trajectory.model_validate(make_trajectory_data())
    assert Trajectory.model_validate_json(traj.model_dump_json()) == traj
    assert (traj.start_time, traj.end_time) == (0.0, 2.0)


def test_trajectory_who_and_when():
    data = make_trajectory_data()
    data.update(climber="Romane", climber_height_m=1.68, recorded_at="2026-11-11T18:30:00")
    traj = Trajectory.model_validate(data)
    assert traj.recorded_at.hour == 18  # the text was turned into a real datetime
    assert Trajectory.model_validate_json(traj.model_dump_json()) == traj


def test_who_and_when_are_optional():
    traj = Trajectory.model_validate(make_trajectory_data())
    assert traj.climber is None and traj.climber_height_m is None and traj.recorded_at is None


def test_contacts_at_gives_supports_during_a_move():
    traj = Trajectory.model_validate(make_trajectory_data())
    # at 0.7 s the left foot is in the air: only the right hand touches
    assert [c.body_part for c in traj.contacts_at(0.7)] == [BodyPart.RIGHT_HAND]
    assert len(traj.contacts_at(1.5)) == 2


def test_trajectory_fits_example_route(route):
    Trajectory.model_validate(make_trajectory_data()).check_against_route(route)


def test_trajectory_with_unknown_hold_does_not_fit_route(route):
    data = make_trajectory_data()
    data["contacts"][0]["target"] = on("h99")
    traj = Trajectory.model_validate(data)  # fine on its own...
    with pytest.raises(ValueError, match="h99"):  # ...but not on this route
        traj.check_against_route(route)


def test_trajectory_for_another_route_does_not_fit(route):
    data = make_trajectory_data()
    data["route_id"] = "other-route"
    with pytest.raises(ValueError, match="not 'example-yellow-01'"):
        Trajectory.model_validate(data).check_against_route(route)


def _frames_out_of_order(d):
    d["frames"][2], d["frames"][3] = d["frames"][3], d["frames"][2]


def _contact_after_end(d):
    d["contacts"][0]["t_start"] = 5.0


def _overlapping_contacts(d):
    d["contacts"][1]["t_end"] = 1.2  # still on the ground when it lands on f1 at 1.0


def _move_without_landing(d):
    d["contacts"][2]["t_start"] = 1.5  # move lands at 1.0 but contact starts at 1.5


def _move_without_leaving(d):
    d["moves"][0]["from_target"] = on("f2")  # it never stood on f2


@pytest.mark.parametrize(
    "change, message",
    [
        (_frames_out_of_order, "increasing order"),
        (_contact_after_end, "outside the climb"),
        (_overlapping_contacts, "overlapping contacts"),
        (_move_without_landing, "no contact starts"),
        (_move_without_leaving, "no contact ends"),
        (lambda d: d.update(frames=[]), "at least 1 item"),
        (lambda d: d.update(source="dream"), "Input should be"),
        (lambda d: d.update(source="user"), "Input should be"),  # old name, now "video"
        (lambda d: d.update(climber_height_m=17.0), "less than or equal to 2.5"),  # cm by mistake
        (lambda d: d.update(recorded_at="yesterday"), "datetime"),
        (lambda d: d.pop("id"), "id"),
    ],
)
def test_trajectory_rejects_inconsistent_data(change, message):
    data = make_trajectory_data()
    change(data)
    with pytest.raises(ValidationError, match=message):
        Trajectory.model_validate(data)


# ---------- Audit ----------


def make_audit_data() -> dict:
    """Audit of a 2-move user climb against a 3-move reference.

    User move 0 matches reference move 0. User move 1 has no match.
    Reference moves 1 and 2 were missed.
    """
    return {
        "route_id": "example-yellow-01",
        "user_trajectory_id": "romane-try1",
        "reference_trajectory_id": "sim-1",
        "beta_score": 62.5,
        "components": {"sequence_agreement": 0.5, "motion_similarity": 0.75},
        "climb_metrics": {"straight_arm_ratio": 0.4, "com_path_ratio": 1.3},
        "moves": [
            {"user_move": 0, "reference_move": 0, "score": 80.0,
             "user_metrics": {"hesitation_s": 1.2}, "reference_metrics": {"hesitation_s": 0.0}},
            {"user_move": 1, "reference_move": None, "score": 40.0},
        ],
        "missed_reference_moves": [1, 2],
    }


def make_solo_audit_data() -> dict:
    """Audit of the same 2-move climb with no reference: metrics only."""
    return {
        "route_id": "example-yellow-01",
        "user_trajectory_id": "romane-try1",
        "climb_metrics": {"straight_arm_ratio": 0.4},
        "moves": [
            {"user_move": 0, "reference_move": None, "user_metrics": {"hesitation_s": 1.2}},
            {"user_move": 1, "reference_move": None},
        ],
    }


def make_moves_trajectory(id: str, source: str, n_moves: int) -> Trajectory:
    """A trajectory whose right hand climbs h1 -> h2 -> h3 ... one move per second."""
    holds = [f"h{i + 1}" for i in range(n_moves + 1)]
    contacts, moves = [], []
    for i, hold in enumerate(holds):
        end = i + 0.5 if i < n_moves else None
        contacts.append({"body_part": "right_hand", "target": on(hold), "t_start": float(i), "t_end": end})
    for i in range(n_moves):
        moves.append({"body_part": "right_hand", "from_target": on(holds[i]),
                      "to_target": on(holds[i + 1]), "t_start": i + 0.5, "t_end": float(i + 1)})
    return Trajectory.model_validate({
        "id": id,
        "route_id": "example-yellow-01",
        "source": source,
        "frames": [make_pose_data(frame_index=i, t=float(i)) for i in range(n_moves + 1)],
        "contacts": contacts,
        "moves": moves,
    })


def test_audit_round_trip():
    audit = Audit.model_validate(make_audit_data())
    assert Audit.model_validate_json(audit.model_dump_json()) == audit
    assert audit.has_reference
    assert audit.moves[1].feedback is None  # coaching text comes later


def test_audit_against_simulation():
    audit = Audit.model_validate(make_audit_data())
    user = make_moves_trajectory("romane-try1", "video", 2)
    audit.check_against(user, make_moves_trajectory("sim-1", "simulation", 3))


def test_audit_against_a_friends_climb():
    # The reference is another real climb: nothing special is needed.
    data = make_audit_data()
    data["reference_trajectory_id"] = "alex-send"
    audit = Audit.model_validate(data)
    user = make_moves_trajectory("romane-try1", "video", 2)
    audit.check_against(user, make_moves_trajectory("alex-send", "video", 3))


def test_audit_without_reference():
    audit = Audit.model_validate(make_solo_audit_data())
    assert not audit.has_reference and audit.beta_score is None
    audit.check_against(make_moves_trajectory("romane-try1", "video", 2))


@pytest.mark.parametrize(
    "user_id, user_moves, ref_id, ref_moves, message",
    [
        ("romane-try1", 1, "sim-1", 3, "user move index out of range"),  # audit mentions user move 1
        ("romane-try1", 2, "sim-1", 2, "reference move index out of range"),  # mentions ref move 2
        ("romane-try2", 2, "sim-1", 3, "audit expects 'romane-try1'"),  # another attempt
        ("romane-try1", 2, "sim-2", 3, "audit expects 'sim-1'"),  # another reference
    ],
)
def test_audit_does_not_fit_wrong_trajectories(user_id, user_moves, ref_id, ref_moves, message):
    audit = Audit.model_validate(make_audit_data())
    user = make_moves_trajectory(user_id, "video", user_moves)
    with pytest.raises(ValueError, match=message):
        audit.check_against(user, make_moves_trajectory(ref_id, "simulation", ref_moves))


def test_audit_with_reference_needs_the_reference():
    audit = Audit.model_validate(make_audit_data())
    with pytest.raises(ValueError, match="reference trajectory is None"):
        audit.check_against(make_moves_trajectory("romane-try1", "video", 2))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(beta_score=120.0), "less than or equal to 100"),
        (lambda d: d.update(beta_score=None), "needs a beta_score"),
        (lambda d: d["components"].update(balance=1.5), "between 0 and 1"),
        (lambda d: d["moves"][1].update(user_move=0), "audited twice"),
        (lambda d: d["moves"][1].update(reference_move=0), "matched to two user moves"),
        (lambda d: d.update(missed_reference_moves=[0, 1, 2]), "both matched and missed"),
        (lambda d: d.update(missed_reference_moves=[-1]), "greater than or equal to 0"),
        (lambda d: d["moves"][0].pop("reference_move"), "reference_move"),  # must be explicit
        (lambda d: d.update(reference_trajectory_id="romane-try1"), "its own reference"),
    ],
)
def test_audit_rejects_bad_input(change, message):
    data = make_audit_data()
    change(data)
    with pytest.raises(ValidationError, match=message):
        Audit.model_validate(data)


@pytest.mark.parametrize(
    "change, leftover",
    [
        (lambda d: d.update(beta_score=50.0), "beta_score"),
        (lambda d: d.update(components={"sequence_agreement": 0.5}), "components"),
        (lambda d: d.update(missed_reference_moves=[0]), "missed_reference_moves"),
        (lambda d: d["moves"][0].update(score=70.0), "move 0"),
        (lambda d: d["moves"][1].update(reference_move=0), "move 1"),
    ],
)
def test_audit_without_reference_rejects_comparison_fields(change, leftover):
    data = make_solo_audit_data()
    change(data)
    with pytest.raises(ValidationError, match=leftover):
        Audit.model_validate(data)


# ---------- Example files (Step 1 "done" criterion) ----------


@pytest.fixture
def example_climb() -> Trajectory:
    return Trajectory.model_validate_json((EXAMPLES / "trajectory_example.json").read_text())


def test_example_climb_fits_example_route(example_climb, route):
    example_climb.check_against_route(route)


def test_example_climb_round_trip(example_climb):
    assert Trajectory.model_validate_json(example_climb.model_dump_json()) == example_climb


def test_example_climb_content(example_climb):
    assert len(example_climb.frames) == 15
    assert len(example_climb.moves) == 7
    # it ends with both hands matched on the top hold
    on_top = [c.body_part for c in example_climb.contacts_at(7.0) if c.target == HoldTarget(hold_id="h4")]
    assert sorted(on_top) == [BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND]
    # the right foot finishes on a smear
    assert any(isinstance(c.target, WallTarget) for c in example_climb.contacts_of(BodyPart.RIGHT_FOOT))
    # seen from behind: the face is not visible, the wrists are
    assert not example_climb.frames[0].is_visible("nose")
    assert example_climb.frames[0].is_visible("left_wrist")


def test_example_climb_wrists_are_on_their_holds(example_climb, route):
    """While a hand is on a hold, its wrist keypoint is within the hold's radius."""
    for frame in example_climb.frames:
        for c in example_climb.contacts_at(frame.t):
            if c.body_part.is_hand and isinstance(c.target, HoldTarget):
                hold, wrist = route.hold(c.target.hold_id), frame[c.body_part.keypoint]
                assert ((wrist.x - hold.x) ** 2 + (wrist.y - hold.y) ** 2) ** 0.5 <= hold.radius


def test_example_file_is_up_to_date():
    """The JSON file matches what the script would generate now."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_example", EXAMPLES / "make_trajectory_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.to_readable_json(module.build()) + "\n"
    assert (EXAMPLES / "trajectory_example.json").read_text() == expected
