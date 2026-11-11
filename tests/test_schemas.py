"""Tests for the shared data format. Run with: uv run pytest"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from climbvla.schemas import Hold, HoldType, Route

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
