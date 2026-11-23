"""Tests for moves from contacts (step 2.10), on made-up contacts with known answers."""

import pytest
from pydantic import ValidationError

from climbvla.moves import MoveRules, find_moves, move_at, regrip_at, same_target
from climbvla.overlay import part_status
from climbvla.schemas import (
    BodyPart,
    PixelContact,
    PixelMove,
    PixelRegrip,
    PixelTarget,
    VideoContacts,
)

TORSO = 100.0  # pixels
LH, RH, LF = BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND, BodyPart.LEFT_FOOT


def on(part, kind, t0, t1, hold=None, x=0.0, y=0.0) -> PixelContact:
    return PixelContact(body_part=part, kind=kind, hold_id=hold, x=x, y=y, t_start=t0, t_end=t1)


def video(*contacts: PixelContact) -> VideoContacts:
    return VideoContacts(video="v.mp4", route_id="r", torso_px=TORSO, contacts=list(contacts))


def hold(hold_id: str, x=0.0, y=0.0) -> PixelTarget:
    return PixelTarget(kind="hold", hold_id=hold_id, x=x, y=y)


# ---------- same target ----------


def test_same_hold_is_same_target_and_other_hold_is_not():
    assert same_target(on(LH, "hold", 0, 1, "h1"), on(LH, "hold", 2, 3, "h1"), TORSO)
    assert not same_target(on(LH, "hold", 0, 1, "h1"), on(LH, "hold", 2, 3, "h2"), TORSO)


def test_ground_is_always_the_same_target():
    assert same_target(on(LF, "ground", 0, 1, x=0), on(LF, "ground", 2, 3, x=300), TORSO)


def test_wall_spots_are_the_same_target_only_when_close():
    a = on(LF, "wall", 0, 1, x=0, y=0)
    assert same_target(a, on(LF, "wall", 2, 3, x=30, y=40), TORSO)  # 0.5 torso lengths
    assert not same_target(a, on(LF, "wall", 2, 3, x=60, y=80), TORSO)  # 1.0
    assert same_target(a, on(LF, "wall", 2, 3, x=60, y=80), TORSO, MoveRules(same_wall=1.0))


def test_different_kinds_are_different_targets():
    assert not same_target(on(LF, "wall", 0, 1), on(LF, "hold", 2, 3, "h1"), TORSO)
    assert not same_target(on(LF, "ground", 0, 1), on(LF, "wall", 2, 3), TORSO)


# ---------- find_moves ----------


def test_two_holds_make_one_move_from_leaving_to_landing():
    result = find_moves(video(on(LH, "hold", 0, 2, "h1", 10, 20), on(LH, "hold", 2.5, 5, "h2", 30, 40)))
    assert result.regrips == []
    [m] = result.moves
    assert (m.body_part, m.t_start, m.t_end) == (LH, 2, 2.5)
    assert m.from_target == hold("h1", 10, 20)
    assert m.to_target == hold("h2", 30, 40)
    assert m.duration == pytest.approx(0.5)


def test_coming_back_to_the_same_hold_is_a_regrip_not_a_move():
    result = find_moves(video(on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.3, 5, "h1")))
    assert result.moves == []
    [r] = result.regrips
    assert (r.body_part, r.target.label, r.t_off, r.t_on) == (LH, "h1", 2, 2.3)


def test_a_long_time_off_the_same_hold_is_still_a_regrip():
    """Chalking up for 3 s and coming back: no progress, so no move."""
    result = find_moves(video(on(RH, "hold", 0, 2, "h1"), on(RH, "hold", 5, 6, "h1")))
    assert result.moves == []
    assert result.regrips[0].duration == pytest.approx(3)


def test_a_regrip_then_a_move_leaves_from_the_second_contact():
    result = find_moves(
        video(on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.2, 4, "h1"), on(LH, "hold", 4.5, 6, "h2"))
    )
    assert [(r.t_off, r.t_on) for r in result.regrips] == [(2, 2.2)]
    [m] = result.moves
    assert (m.from_target.label, m.to_target.label, m.t_start, m.t_end) == ("h1", "h2", 4, 4.5)


def test_ground_to_wall_to_far_wall_to_close_wall():
    result = find_moves(
        video(
            on(LF, "ground", 0, 1, x=0, y=500),
            on(LF, "wall", 1.5, 2, x=0, y=300),
            on(LF, "wall", 2.5, 3, x=0, y=150),  # 1.5 torso lengths higher: a move
            on(LF, "wall", 3.2, 4, x=0, y=120),  # 0.3 away: re-placed, a regrip
        )
    )
    assert [(m.from_target.label, m.to_target.label) for m in result.moves] == [
        ("ground", "wall"),
        ("wall", "wall"),
    ]
    assert [(r.t_off, r.t_on) for r in result.regrips] == [(3, 3.2)]


def test_a_move_can_last_zero_seconds():
    """Sliding off a hold onto the wall right below: lands on the frame it leaves."""
    [m] = find_moves(video(on(LF, "hold", 0, 2, "h3"), on(LF, "wall", 2, 3))).moves
    assert m.duration == 0


def test_a_dyno_is_two_overlapping_moves():
    result = find_moves(
        video(
            on(LH, "hold", 0, 2, "h1"),
            on(RH, "hold", 0, 2.1, "h2"),
            on(LH, "hold", 2.6, 4, "h3"),
            on(RH, "hold", 2.5, 4, "h3"),
        )
    )
    assert [(m.body_part, m.t_start) for m in result.moves] == [(LH, 2), (RH, 2.1)]


def test_one_contact_or_none_gives_no_move():
    assert find_moves(video(on(LH, "hold", 0, 2, "h1"))).moves == []
    assert find_moves(video()).moves == []


def test_find_moves_leaves_the_contacts_untouched():
    original = video(on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.3, 5, "h1"))
    result = find_moves(original)
    assert result.contacts == original.contacts
    assert original.moves == [] and original.regrips == []


def test_moves_are_sorted_by_start_time_across_body_parts():
    result = find_moves(
        video(
            on(LH, "hold", 0, 3, "h1"),
            on(LH, "hold", 3.5, 5, "h2"),
            on(RH, "hold", 0, 1, "h3"),
            on(RH, "hold", 1.5, 5, "h4"),
        )
    )
    assert [m.t_start for m in result.moves] == [1, 3]


def test_result_round_trips_through_json():
    result = find_moves(
        video(on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.3, 4, "h1"), on(LH, "hold", 4.5, 6, "h2"))
    )
    assert VideoContacts.model_validate_json(result.model_dump_json()) == result


# ---------- what a body part is doing at time t ----------


def test_move_at_and_regrip_at():
    result = find_moves(
        video(on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.2, 4, "h1"), on(LH, "hold", 4.5, 6, "h2"))
    )
    assert move_at(result, LH, 4.2).to_target.label == "h2"
    assert move_at(result, LH, 3) is None
    assert move_at(result, RH, 4.2) is None
    assert regrip_at(result, LH, 2.1).target.label == "h1"
    assert part_status(result, None, LH, 4.2) == "h1 -> h2"
    assert part_status(result, None, LH, 2.1) == "regrip h1"
    assert part_status(result, None, LH, 7) == "-"


# ---------- the models' own checks ----------


def test_old_contacts_files_without_moves_still_load():
    data = '{"video": "v.mp4", "route_id": "r", "torso_px": 100, "contacts": []}'
    assert VideoContacts.model_validate_json(data).moves == []


def test_a_move_must_change_hold():
    with pytest.raises(ValidationError, match="change target"):
        PixelMove(body_part=LH, from_target=hold("h1"), to_target=hold("h1"), t_start=1, t_end=2)


def test_a_move_cannot_end_before_it_starts():
    with pytest.raises(ValidationError, match="before"):
        PixelMove(body_part=LH, from_target=hold("h1"), to_target=hold("h2"), t_start=2, t_end=1)


def test_a_target_needs_a_hold_id_exactly_when_it_is_a_hold():
    with pytest.raises(ValidationError, match="hold_id"):
        PixelTarget(kind="hold", x=0, y=0)
    with pytest.raises(ValidationError, match="hold_id"):
        PixelTarget(kind="wall", hold_id="h1", x=0, y=0)


def test_a_move_must_link_two_contacts():
    contacts = [on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.5, 4, "h2")]
    good = PixelMove(body_part=LH, from_target=hold("h1"), to_target=hold("h2"), t_start=2, t_end=2.5)
    VideoContacts(video="v", route_id="r", torso_px=TORSO, contacts=contacts, moves=[good])
    late = good.model_copy(update={"t_start": 1.5})
    with pytest.raises(ValidationError, match="ends at"):
        VideoContacts(video="v", route_id="r", torso_px=TORSO, contacts=contacts, moves=[late])
    wrong_hold = good.model_copy(update={"to_target": hold("h7")})
    with pytest.raises(ValidationError, match="starts at"):
        VideoContacts(video="v", route_id="r", torso_px=TORSO, contacts=contacts, moves=[wrong_hold])


def test_a_regrip_must_link_two_contacts():
    contacts = [on(LH, "hold", 0, 2, "h1"), on(LH, "hold", 2.5, 4, "h1")]
    regrip = PixelRegrip(body_part=LH, target=hold("h1"), t_off=2, t_on=3)
    with pytest.raises(ValidationError, match="links no contacts"):
        VideoContacts(video="v", route_id="r", torso_px=TORSO, contacts=contacts, regrips=[regrip])
