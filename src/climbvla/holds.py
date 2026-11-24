"""Clicking a route's holds on a video frame (Step 2.6).

A stand-in for the hold detector of step 3: you show the holds by hand,
in pixels, on one frame of the video. The camera must not move, so the
positions are valid on every frame.

Try it:
    uv run python -m climbvla.holds climb.mp4 --route-id gym-yellow-01 --t 0.5
    -> writes climb_route.json (and climb_route.png, to check) next to the video

In the window, three stages, Enter to go to the next one:
    1. holds: drag from a hold's center to its edge (a plain click = default size)
    2. start: click the 1 to 4 start holds
    3. top:   click the top hold
    u = undo, q or Esc = quit without saving
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from climbvla.draw import draw_holds, draw_text_lines
from climbvla.schemas import PixelHold, VideoRoute
from climbvla.video import iter_frames

# Two short lines per stage, so they fit on a portrait video.
INSTRUCTIONS = {
    "holds": ("1/3 HOLDS: drag center -> edge", "u = undo, Enter = next"),
    "start": ("2/3 START: click 1-4 start holds", "u = undo, Enter = next"),
    "top": ("3/3 TOP: click the top hold", "Enter = save"),
}
WHITE = (255, 255, 255)
MESSAGE_COLOR = (0, 0, 255)  # red

# A drag shorter than this (in image pixels) counts as a plain click.
MIN_DRAG = 5.0
# Size of a hold given by a plain click, as a share of the image's longest side.
DEFAULT_RADIUS_SHARE = 0.015

# The window must fit on a laptop screen; the image is shown smaller if needed.
MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT = 1400, 900


@dataclass
class HoldClicker:
    """Everything the clicking tool knows, apart from the window itself.

    Kept separate from the window code so it can be tested without clicking.
    All coordinates are image pixels.
    """

    width: int
    height: int
    holds: list[PixelHold] = field(default_factory=list)
    start_hold_ids: list[str] = field(default_factory=list)
    top_hold_id: str | None = None
    stage: str = "holds"  # then "start", "top", "done"
    pressed_at: tuple[float, float] | None = None  # where the mouse button went down

    @property
    def default_radius(self) -> float:
        return DEFAULT_RADIUS_SHARE * max(self.width, self.height)

    def _inside(self, x: float, y: float) -> tuple[float, float]:
        return min(max(x, 0.0), float(self.width)), min(max(y, 0.0), float(self.height))

    def press(self, x: float, y: float) -> None:
        if self.stage == "holds":
            self.pressed_at = self._inside(x, y)

    def release(self, x: float, y: float) -> str | None:
        """Mouse button released at (x, y). Returns a message to show, if any."""
        x, y = self._inside(x, y)
        if self.stage == "holds":
            if self.pressed_at is None:
                return None
            cx, cy = self.pressed_at
            self.pressed_at = None
            radius = math.dist((cx, cy), (x, y))
            if radius < MIN_DRAG:
                radius = self.default_radius
            hold_id = f"h{len(self.holds) + 1}"
            self.holds.append(PixelHold(id=hold_id, x=cx, y=cy, radius=radius))
            return None

        hold = self.hold_at(x, y)
        if hold is None:
            return "no hold there"
        if self.stage == "start":
            if hold.id in self.start_hold_ids:
                self.start_hold_ids.remove(hold.id)  # clicking again unselects
            elif len(self.start_hold_ids) == 4:
                return "at most 4 start holds"
            else:
                self.start_hold_ids.append(hold.id)
        elif self.stage == "top":
            if hold.id in self.start_hold_ids:
                return "the top hold cannot be a start hold"
            self.top_hold_id = hold.id
        return None

    def hold_at(self, x: float, y: float) -> PixelHold | None:
        """The hold whose center is nearest to (x, y), if (x, y) is inside its circle."""
        inside = [h for h in self.holds if math.dist((h.x, h.y), (x, y)) <= h.radius]
        return min(inside, key=lambda h: math.dist((h.x, h.y), (x, y)), default=None)

    def undo(self) -> None:
        if self.stage == "holds" and self.holds:
            self.holds.pop()
        elif self.stage == "start" and self.start_hold_ids:
            self.start_hold_ids.pop()
        elif self.stage == "top":
            self.top_hold_id = None

    def next_stage(self) -> str | None:
        """Go to the next stage. Returns why not, if the current one isn't finished."""
        if self.stage == "holds":
            if len(self.holds) < 2:
                return "click at least 2 holds (a start and a top)"
            self.stage = "start"
        elif self.stage == "start":
            if not self.start_hold_ids:
                return "click at least 1 start hold"
            self.stage = "top"
        elif self.stage == "top":
            if self.top_hold_id is None:
                return "click the top hold"
            self.stage = "done"
        return None

    def to_video_route(
        self,
        route_id: str,
        video: str,
        frame_t: float,
        color: str | None = None,
        grade: str | None = None,
    ) -> VideoRoute:
        if self.top_hold_id is None:
            raise ValueError("the top hold has not been chosen yet")
        return VideoRoute(
            route_id=route_id,
            video=video,
            width=self.width,
            height=self.height,
            frame_t=frame_t,
            color=color,
            grade=grade,
            holds=self.holds,
            start_hold_ids=self.start_hold_ids,
            top_hold_id=self.top_hold_id,
        )

    def render(
        self,
        image: np.ndarray,
        drag_to: tuple[float, float] | None = None,
        message: str | None = None,
    ) -> np.ndarray:
        """The image with the holds so far, the hold being dragged, and the instructions."""
        out = draw_holds(image, self.holds, self.start_hold_ids, self.top_hold_id)
        if self.pressed_at is not None and drag_to is not None:
            center = (round(self.pressed_at[0]), round(self.pressed_at[1]))
            radius = round(math.dist(self.pressed_at, drag_to))
            cv2.circle(out, center, max(radius, 1), WHITE, 2, cv2.LINE_AA)
        lines = [(text, WHITE) for text in INSTRUCTIONS.get(self.stage, ())]
        if message:
            lines.append((message, MESSAGE_COLOR))
        draw_text_lines(out, lines)
        return out


def click_holds(image: np.ndarray) -> HoldClicker | None:
    """Open a window to click the holds. Returns None if the user quit."""
    height, width = image.shape[:2]
    clicker = HoldClicker(width=width, height=height)
    scale = min(1.0, MAX_WINDOW_WIDTH / width, MAX_WINDOW_HEIGHT / height)
    window_size = (round(width * scale), round(height * scale))
    ui: dict = {"drag_to": None, "message": None}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        ix, iy = x / scale, y / scale  # window pixels -> image pixels
        if event == cv2.EVENT_LBUTTONDOWN:
            clicker.press(ix, iy)
            ui["message"] = None
        elif event == cv2.EVENT_MOUSEMOVE and clicker.pressed_at is not None:
            ui["drag_to"] = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP:
            ui["message"] = clicker.release(ix, iy)
            ui["drag_to"] = None

    window = "ClimbVLA - click the holds"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            shown = clicker.render(image, ui["drag_to"], ui["message"])
            cv2.imshow(window, cv2.resize(shown, window_size, interpolation=cv2.INTER_AREA))
            key = cv2.waitKey(30) & 0xFF  # also lets the window process clicks
            if key in (13, 10):  # Enter
                ui["message"] = clicker.next_stage()
                if clicker.stage == "done":
                    return clicker
            elif key == ord("u"):
                clicker.undo()
            elif key in (ord("q"), 27):  # q or Esc
                return None
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None  # the window was closed
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macOS needs one more event loop turn to really close it


def click_route(
    video: Path,
    route_id: str,
    t: float = 0.0,
    color: str | None = None,
    grade: str | None = None,
    out: Path | None = None,
    click=click_holds,
) -> VideoRoute | None:
    """Click the holds on the video's frame at `t` seconds and save the route.

    Writes <video>_route.json (or `out`) and a .png next to it with the holds
    drawn, to check. Returns None, saving nothing, if the user quit.
    `click` is the clicking window; tests pass a fake one.
    """
    frame = next(iter_frames(video, start=t), None)
    if frame is None:
        raise ValueError(f"{video} has no frame at or after {t} s")
    clicker = click(frame.image)
    if clicker is None:
        return None
    route = clicker.to_video_route(route_id, video.name, frame.t, color=color, grade=grade)
    out = out or video.with_name(f"{video.stem}_route.json")
    out.write_text(route.model_dump_json(indent=2))
    cv2.imwrite(
        str(out.with_suffix(".png")),
        draw_holds(frame.image, route.holds, route.start_hold_ids, route.top_hold_id),
    )
    return route


# ---------- command line ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Click a route's holds on a video frame.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--route-id", required=True, help="e.g. gym-yellow-01")
    parser.add_argument("--t", type=float, default=0.0, help="frame to click on, in seconds")
    parser.add_argument("--color", help="the route's hold color")
    parser.add_argument("--grade", help="e.g. V3 or 6a")
    parser.add_argument("--out", type=Path, help="output (default: <video>_route.json)")
    args = parser.parse_args()

    print(__doc__.split("In the window,")[1])
    out = args.out or args.video.with_name(f"{args.video.stem}_route.json")
    try:
        route = click_route(args.video, args.route_id, args.t, args.color, args.grade, out)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if route is None:
        print("quit: nothing saved")
        return
    print(f"wrote {out} ({len(route.holds)} holds) and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
