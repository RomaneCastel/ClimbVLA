"""Tests for the hold-clicking tool and the VideoRoute format.

The window itself isn't tested (it needs a person to click); everything it
does goes through HoldClicker, which is.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from climbvla.draw import START_COLOR, TOP_COLOR, draw_holds
from climbvla.holds import HoldClicker
from climbvla.schemas import PixelHold, VideoRoute

WIDTH, HEIGHT = 1000, 2000  # portrait, like a phone video


def drag(clicker: HoldClicker, center, edge) -> str | None:
    clicker.press(*center)
    return clicker.release(*edge)


def click(clicker: HoldClicker, point) -> str | None:
    return drag(clicker, point, point)


def clicker_with_holds() -> HoldClicker:
    """Three holds: h1 and h2 low, h3 high."""
    clicker = HoldClicker(width=WIDTH, height=HEIGHT)
    drag(clicker, (300, 1500), (340, 1500))  # radius 40
    drag(clicker, (600, 1400), (600, 1430))  # radius 30
    drag(clicker, (500, 300), (550, 300))  # radius 50
    return clicker


# ---------- stage 1: holds ----------


def test_drag_gives_center_and_radius():
    clicker = clicker_with_holds()
    assert [h.id for h in clicker.holds] == ["h1", "h2", "h3"]
    h1 = clicker.holds[0]
    assert (h1.x, h1.y, h1.radius) == (300, 1500, pytest.approx(40))


def test_plain_click_gets_the_default_size():
    clicker = HoldClicker(width=WIDTH, height=HEIGHT)
    click(clicker, (100, 100))
    assert clicker.holds[0].radius == pytest.approx(0.015 * HEIGHT)  # 30 px


def test_clicks_outside_the_image_are_kept_inside():
    clicker = HoldClicker(width=WIDTH, height=HEIGHT)
    click(clicker, (-20, HEIGHT + 50))
    assert (clicker.holds[0].x, clicker.holds[0].y) == (0, HEIGHT)


def test_undo_removes_the_last_hold():
    clicker = clicker_with_holds()
    clicker.undo()
    assert [h.id for h in clicker.holds] == ["h1", "h2"]
    click(clicker, (100, 100))
    assert clicker.holds[-1].id == "h3"  # the id is reused, no gap


def test_need_two_holds_to_go_on():
    clicker = HoldClicker(width=WIDTH, height=HEIGHT)
    click(clicker, (100, 100))
    assert "at least 2" in clicker.next_stage()
    assert clicker.stage == "holds"


# ---------- stage 2 and 3: start and top ----------


def test_hold_at_finds_the_clicked_hold():
    clicker = clicker_with_holds()
    assert clicker.hold_at(310, 1490).id == "h1"
    assert clicker.hold_at(800, 800) is None


def test_full_session():
    clicker = clicker_with_holds()
    assert clicker.next_stage() is None and clicker.stage == "start"
    assert click(clicker, (300, 1500)) is None  # h1
    assert click(clicker, (600, 1400)) is None  # h2
    assert clicker.start_hold_ids == ["h1", "h2"]

    assert clicker.next_stage() is None and clicker.stage == "top"
    assert click(clicker, (500, 310)) is None  # h3
    assert clicker.next_stage() is None and clicker.stage == "done"

    route = clicker.to_video_route("gym-yellow-01", "climb.mp4", 0.5, color="yellow")
    assert route.start_hold_ids == ["h1", "h2"]
    assert route.top_hold_id == "h3"
    assert (route.width, route.height, route.frame_t) == (WIDTH, HEIGHT, 0.5)


def test_clicking_a_start_hold_again_unselects_it():
    clicker = clicker_with_holds()
    clicker.next_stage()
    click(clicker, (300, 1500))
    click(clicker, (300, 1500))
    assert clicker.start_hold_ids == []
    assert "at least 1" in clicker.next_stage()


def test_click_next_to_any_hold():
    clicker = clicker_with_holds()
    clicker.next_stage()
    assert click(clicker, (800, 800)) == "no hold there"


def test_at_most_four_start_holds():
    clicker = HoldClicker(width=WIDTH, height=HEIGHT)
    for i in range(5):
        drag(clicker, (100 + 150 * i, 1800), (130 + 150 * i, 1800))
    clicker.next_stage()
    for i in range(4):
        click(clicker, (100 + 150 * i, 1800))
    assert click(clicker, (700, 1800)) == "at most 4 start holds"
    assert len(clicker.start_hold_ids) == 4


def test_top_cannot_be_a_start_hold():
    clicker = clicker_with_holds()
    clicker.next_stage()
    click(clicker, (300, 1500))
    clicker.next_stage()
    assert "cannot be a start" in click(clicker, (300, 1500))
    assert clicker.top_hold_id is None
    assert clicker.next_stage() == "click the top hold"


def test_render_draws_on_a_copy():
    clicker = clicker_with_holds()
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    out = clicker.render(image, message="hello")
    assert out.shape == image.shape and out.any()
    assert not image.any()


# ---------- drawing ----------


def test_start_and_top_holds_have_their_own_colors():
    holds = [
        PixelHold(id="a", x=100, y=100, radius=20),
        PixelHold(id="b", x=300, y=100, radius=20),
    ]
    out = draw_holds(np.zeros((400, 400, 3), np.uint8), holds, ["a"], "b")
    # the circle's leftmost point, on each hold
    assert tuple(out[100, 80]) == START_COLOR
    assert tuple(out[100, 280]) == TOP_COLOR


# ---------- the VideoRoute format ----------


def make_video_route(**changes) -> VideoRoute:
    data = dict(
        route_id="gym-yellow-01",
        video="climb.mp4",
        width=WIDTH,
        height=HEIGHT,
        frame_t=0.0,
        holds=[
            {"id": "h1", "x": 300, "y": 1500, "radius": 40},
            {"id": "h2", "x": 500, "y": 300, "radius": 50},
        ],
        start_hold_ids=["h1"],
        top_hold_id="h2",
    )
    data.update(changes)
    return VideoRoute.model_validate(data)


def test_video_route_round_trip():
    route = make_video_route(color="yellow", grade="6a")
    assert VideoRoute.model_validate_json(route.model_dump_json()) == route
    assert route.hold("h2").y == 300


def test_video_route_shares_the_hold_id_rules():
    with pytest.raises(ValidationError, match="not in holds"):
        make_video_route(top_hold_id="h9")
    with pytest.raises(ValidationError, match="cannot also be a start"):
        make_video_route(start_hold_ids=["h2"])


def test_video_route_hold_outside_the_image():
    with pytest.raises(ValidationError, match="outside"):
        make_video_route(width=200)  # h1 is at x = 300


def test_pixel_hold_needs_a_positive_radius():
    with pytest.raises(ValidationError):
        PixelHold(id="h1", x=1, y=1, radius=0)
