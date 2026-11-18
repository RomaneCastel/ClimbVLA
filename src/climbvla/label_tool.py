"""Clicking the true wrists and ankles on frames of a video (Step 2.8).

The frames come from the pose file (one every 0.5 s by default). Each frame is
shown zoomed on the climber, using the box found in step 2.3. The model's own
skeleton is NOT shown, so it can't influence where you click.

Try it:
    uv run python -m climbvla.label_tool climb.mp4
    -> writes climb_labels.json next to the video; run it again to continue

For each frame, click in order: left wrist, right wrist, left ankle, right
ankle. "Left" is the climber's OWN left.
    h      = the next point is hidden: click where you think it is
    s      = can't tell at all: skip this point
    u      = undo the last point
    Enter  = next frame        p = previous frame
    q/Esc  = save and quit
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from climbvla.draw import LEFT_COLOR, RIGHT_COLOR, draw_text_lines
from climbvla.labels import LABELED_POINTS, LabelSet, PointLabel, Visibility, empty_label_set
from climbvla.schemas import KeypointName
from climbvla.track import PoseTrack
from climbvla.video import iter_frames

MARGIN = 0.3  # around the climber's box, as a share of its size
MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT = 1400, 900
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def describe(name: KeypointName) -> str:
    """'left wrist' -> "LEFT wrist" for the instructions."""
    side, part = name.value.split("_")
    return f"{side.upper()} {part}"


def crop_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the climber's box (x, y, w, h) plus a margin, inside the image."""
    x, y, w, h = box[:4]
    x0, y0 = max(0, int(x - MARGIN * w)), max(0, int(y - MARGIN * h))
    x1, y1 = min(width, int(x + w + MARGIN * w)), min(height, int(y + h + MARGIN * h))
    return x0, y0, x1, y1


@dataclass
class LabelSession:
    """The labeling tool's state, apart from the window, so it can be tested.

    All coordinates are full-image pixels.
    """

    labels: LabelSet
    current: int = 0  # index in labels.frames
    hidden_next: bool = False  # the next click is a hidden point

    def __post_init__(self) -> None:
        undone = [i for i, f in enumerate(self.labels.frames) if not f.done]
        self.current = undone[0] if undone else 0

    @property
    def frame(self):
        return self.labels.frames[self.current]

    @property
    def asking(self) -> KeypointName | None:
        """The point to click next on this frame, or None if all are answered."""
        return next((name for name in LABELED_POINTS if name not in self.frame.points), None)

    def click(self, x: float, y: float) -> str | None:
        name = self.asking
        if name is None:
            return "all 4 points done: Enter for the next frame, u to undo"
        visibility = Visibility.HIDDEN if self.hidden_next else Visibility.VISIBLE
        x = min(max(x, 0.0), float(self.labels.width))
        y = min(max(y, 0.0), float(self.labels.height))
        self.frame.points[name] = PointLabel(x=x, y=y, visibility=visibility)
        self.hidden_next = False
        return None

    def skip(self) -> None:
        name = self.asking
        if name is not None:
            self.frame.points[name] = None
            self.hidden_next = False

    def toggle_hidden(self) -> None:
        self.hidden_next = not self.hidden_next

    def undo(self) -> None:
        answered = [name for name in LABELED_POINTS if name in self.frame.points]
        if answered:
            del self.frame.points[answered[-1]]
        self.hidden_next = False

    def go(self, step: int) -> bool:
        """Move to another frame; False at either end."""
        target = self.current + step
        if 0 <= target < len(self.labels.frames):
            self.current = target
            self.hidden_next = False
            return True
        return False

    def instructions(self) -> list[tuple[str, tuple[int, int, int]]]:
        n = len(self.labels.frames)
        lines = [(f"frame {self.current + 1}/{n}  t={self.frame.t:.2f}s  done {self.labels.n_done}/{n}", WHITE)]
        name = self.asking
        if name is None:
            lines.append(("all done: Enter = next frame, u = undo", YELLOW))
        else:
            mode = "  [HIDDEN: click where you think it is]" if self.hidden_next else ""
            lines.append((f"click the climber's {describe(name)}{mode}", YELLOW))
        lines.append(("h hidden  s can't tell  u undo  Enter next  p prev  q quit", WHITE))
        return lines


def draw_points(image: np.ndarray, session: LabelSession, offset: tuple[int, int], scale: float) -> None:
    """Draw the points already clicked on this frame, IN PLACE on the shown image."""
    for name, point in session.frame.points.items():
        if point is None:
            continue
        x = int(round((point.x - offset[0]) * scale))
        y = int(round((point.y - offset[1]) * scale))
        color = LEFT_COLOR if name.startswith("left") else RIGHT_COLOR
        filled = point.visibility == Visibility.VISIBLE
        cv2.circle(image, (x, y), 7, color, -1 if filled else 2, cv2.LINE_AA)
        cv2.putText(image, describe(name), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run(video: Path, track: PoseTrack, labels: LabelSet, out: Path) -> LabelSet:
    """Open the window and label until done or quit. Saves after every frame."""
    session = LabelSession(labels)
    row_of = {int(i): r for r, i in enumerate(track.frame_index)}
    wanted = {f.frame_index for f in labels.frames}
    images = {f.index: f.image for f in iter_frames(video) if f.index in wanted}
    message: dict = {"text": None}
    view: dict = {}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONUP and view:
            (x0, y0), scale = view["offset"], view["scale"]
            message["text"] = session.click(x0 + x / scale, y0 + y / scale)

    def save() -> None:
        out.write_text(LabelSet.model_validate(labels.model_dump()).model_dump_json(indent=2))

    window = "ClimbVLA - label wrists and ankles"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            frame = session.frame
            image = images[frame.frame_index]
            x0, y0, x1, y1 = crop_box(track.boxes[row_of[frame.frame_index]], labels.width, labels.height)
            crop = image[y0:y1, x0:x1]
            scale = min(MAX_WINDOW_WIDTH / crop.shape[1], MAX_WINDOW_HEIGHT / crop.shape[0])
            shown = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)))
            view.update(offset=(x0, y0), scale=scale)
            draw_points(shown, session, (x0, y0), scale)
            lines = session.instructions()
            if message["text"]:
                lines.append((message["text"], RED))
            draw_text_lines(shown, lines)
            cv2.imshow(window, shown)

            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                save()
                message["text"] = None if session.go(+1) else "last frame: q to save and quit"
            elif key == ord("p"):
                session.go(-1)
            elif key == ord("h"):
                session.toggle_hidden()
            elif key == ord("s"):
                session.skip()
            elif key == ord("u"):
                session.undo()
            elif key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                save()
                return labels
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Click the true wrists and ankles.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=float, default=0.5, help="seconds between labeled frames")
    args = parser.parse_args()

    track = PoseTrack.load(args.video.with_name(f"{args.video.stem}_pose.npz"))
    out = args.video.with_name(f"{args.video.stem}_labels.json")
    if out.exists():
        labels = LabelSet.model_validate_json(out.read_text())
        print(f"continuing {out}: {labels.n_done}/{len(labels.frames)} frames done")
    else:
        labels = empty_label_set(track, args.every)
        print(f"{len(labels.frames)} frames to label, one every {args.every} s")
    print(__doc__.split("For each frame,")[1])
    labels = run(args.video, track, labels, out)
    print(f"saved {out}: {labels.n_done}/{len(labels.frames)} frames done")


if __name__ == "__main__":
    main()
