"""Build examples/trajectory_example.json, a hand-designed climb of the example route.

The contacts and moves below are written by hand. The 17 keypoints of each
frame are then computed from them with a very simple body model, because
typing 15 frames x 17 joints by hand would be error-prone.

The climb (route example-yellow-01, see route_example.json):
    0.0 s   standing on the ground, both hands matched on the start hold h1
    0.5 s   left foot steps up to f1
    1.0 s   right foot steps up to f2
    2.0 s   right hand goes to h2
    3.0 s   left hand goes to h3
    4.0 s   right foot leaves f2 and smears on the wall
    5.0 s   right hand goes to the top hold h4
    6.0 s   left hand matches on h4; the climb ends at 7.0 s

Run with: uv run python examples/make_trajectory_example.py
"""

import json
import re
from pathlib import Path

from climbvla.schemas import (
    BodyPart,
    ContactEvent,
    GroundTarget,
    HoldTarget,
    KeypointName,
    Move,
    PoseFrame,
    Route,
    Target,
    Trajectory,
    WallTarget,
)

HERE = Path(__file__).parent
ROUTE = Route.model_validate_json((HERE / "route_example.json").read_text())

FRAME_STEP = 0.5  # seconds between frames: 2 fps keeps the file readable
END_TIME = 7.0

# Where each foot stands on the ground before leaving it (x, y in meters).
GROUND_SPOT = {BodyPart.LEFT_FOOT: (0.85, 0.0), BodyPart.RIGHT_FOOT: (1.35, 0.0)}

LH, RH, LF, RF = BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND, BodyPart.LEFT_FOOT, BodyPart.RIGHT_FOOT
GROUND = GroundTarget()
SMEAR = WallTarget(x=1.35, y=0.95)


def on(hold_id: str) -> HoldTarget:
    return HoldTarget(hold_id=hold_id)


# ---------- the climb, by hand ----------

MOVES = [
    Move(body_part=LF, from_target=GROUND, to_target=on("f1"), t_start=0.5, t_end=1.0),
    Move(body_part=RF, from_target=GROUND, to_target=on("f2"), t_start=1.0, t_end=1.5),
    Move(body_part=RH, from_target=on("h1"), to_target=on("h2"), t_start=2.0, t_end=2.5),
    Move(body_part=LH, from_target=on("h1"), to_target=on("h3"), t_start=3.0, t_end=3.5),
    Move(body_part=RF, from_target=on("f2"), to_target=SMEAR, t_start=4.0, t_end=4.5),
    Move(body_part=RH, from_target=on("h2"), to_target=on("h4"), t_start=5.0, t_end=5.5),
    Move(body_part=LH, from_target=on("h3"), to_target=on("h4"), t_start=6.0, t_end=6.5),
]

# Contacts follow from the moves: each body part sits on a target from the
# end of one move to the start of the next.
START_TARGET = {LH: on("h1"), RH: on("h1"), LF: GROUND, RF: GROUND}


def contacts_from_moves() -> list[ContactEvent]:
    contacts = []
    for part, target in START_TARGET.items():
        t_start = 0.0
        for move in (m for m in MOVES if m.body_part == part):
            contacts.append(
                ContactEvent(body_part=part, target=target, t_start=t_start, t_end=move.t_start)
            )
            target, t_start = move.to_target, move.t_end
        contacts.append(ContactEvent(body_part=part, target=target, t_start=t_start))  # still there
    return sorted(contacts, key=lambda c: (c.t_start, c.body_part))


# ---------- a very simple body model for the keypoints ----------


def position(target: Target, part: BodyPart) -> tuple[float, float]:
    if isinstance(target, HoldTarget):
        hold = ROUTE.hold(target.hold_id)
        return hold.x, hold.y
    if isinstance(target, WallTarget):
        return target.x, target.y
    return GROUND_SPOT[part]


def extremity(part: BodyPart, t: float, contacts: list[ContactEvent]) -> tuple[float, float]:
    """Where a hand or foot is at time t: on its target, or on the way between two."""
    for c in contacts:
        if c.body_part == part and c.is_active(t):
            return position(c.target, part)
    for m in MOVES:
        if m.body_part == part and m.t_start <= t < m.t_end:
            (x0, y0), (x1, y1) = position(m.from_target, part), position(m.to_target, part)
            a = (t - m.t_start) / m.duration
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    raise ValueError(f"{part} is nowhere at t={t}")


def midpoint(p, q, push_x=0.0):
    return (p[0] + q[0]) / 2 + push_x, (p[1] + q[1]) / 2


def pose_at(t: float, contacts: list[ContactEvent]) -> dict[KeypointName, tuple[float, float, float]]:
    lw, rw = extremity(LH, t, contacts), extremity(RH, t, contacts)
    la, ra = extremity(LF, t, contacts), extremity(RF, t, contacts)

    # Pelvis between hands and feet, a bit closer to the feet.
    cx = (lw[0] + rw[0] + la[0] + ra[0]) / 4
    cy = 0.6 * (la[1] + ra[1]) / 2 + 0.4 * (lw[1] + rw[1]) / 2
    lhip, rhip = (cx - 0.12, cy), (cx + 0.12, cy)
    lsho, rsho = (cx - 0.18, cy + 0.50), (cx + 0.18, cy + 0.50)
    head = (cx, cy + 0.72)

    k = KeypointName
    # (x, y, confidence). Seen from behind: the face is a low-confidence guess.
    return {
        k.NOSE: (*head, 0.2),
        k.LEFT_EYE: (head[0] - 0.03, head[1] + 0.03, 0.2),
        k.RIGHT_EYE: (head[0] + 0.03, head[1] + 0.03, 0.2),
        k.LEFT_EAR: (head[0] - 0.07, head[1], 0.6),
        k.RIGHT_EAR: (head[0] + 0.07, head[1], 0.6),
        k.LEFT_SHOULDER: (*lsho, 0.95),
        k.RIGHT_SHOULDER: (*rsho, 0.95),
        k.LEFT_ELBOW: (*midpoint(lsho, lw, push_x=-0.08), 0.9),
        k.RIGHT_ELBOW: (*midpoint(rsho, rw, push_x=+0.08), 0.9),
        k.LEFT_WRIST: (*lw, 0.9),
        k.RIGHT_WRIST: (*rw, 0.9),
        k.LEFT_HIP: (*lhip, 0.9),
        k.RIGHT_HIP: (*rhip, 0.9),
        k.LEFT_KNEE: (*midpoint(lhip, la, push_x=-0.10), 0.85),
        k.RIGHT_KNEE: (*midpoint(rhip, ra, push_x=+0.10), 0.85),
        k.LEFT_ANKLE: (*la, 0.85),
        k.RIGHT_ANKLE: (*ra, 0.85),
    }


def build() -> Trajectory:
    contacts = contacts_from_moves()
    n_frames = round(END_TIME / FRAME_STEP) + 1
    frames = []
    for i in range(n_frames):
        t = i * FRAME_STEP
        keypoints = {
            name: {"x": round(x, 3), "y": round(y, 3), "confidence": conf}
            for name, (x, y, conf) in pose_at(t, contacts).items()
        }
        frames.append(PoseFrame(frame_index=i, t=t, keypoints=keypoints))

    traj = Trajectory(
        id="example-climb-01",
        route_id=ROUTE.id,
        source="video",
        climber="example climber",
        climber_height_m=1.68,
        recorded_at="2026-11-11T18:30:00",
        frames=frames,
        contacts=contacts,
        moves=MOVES,
    )
    traj.check_against_route(ROUTE)
    return traj


def to_readable_json(traj: Trajectory) -> str:
    """Indented JSON, but with each keypoint and each target on one line."""
    text = json.dumps(traj.model_dump(mode="json"), indent=2)
    return re.sub(r"\{\n\s+([^{}\[\]]*?)\n\s+\}", lambda m: "{" + re.sub(r"\n\s+", " ", m[1]) + "}", text)


if __name__ == "__main__":
    out = HERE / "trajectory_example.json"
    out.write_text(to_readable_json(build()) + "\n")
    print(f"wrote {out}")
