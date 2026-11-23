"""Contact detection, in pixels (Step 2.9).

A hand or foot is "on" something when it is close to it AND nearly still,
for long enough. Each body part goes through the video frame by frame, either
free or in contact:

    free -> in contact: close to a target and slow, for at least min_contact;
    in contact -> free: too far from the target or too fast, for at least
                        min_release.

Hysteresis: the rule to stay is looser than the rule to land (exit margins
are wider than enter margins, the exit speed higher than the enter speed).
With a single threshold, a hand resting right at the edge of a hold would
flicker on/off/on with the pose's jitter and make dozens of fake moves; with
two, it has to clearly leave before the contact ends. A thermostat works the
same way: heat on below 19 °C, off above 21 °C, not on and off around 20 °C.

What can be touched:
    - a hold of the route (hands and feet): the hold's clicked circle;
    - the ground (feet): a foot as low in the image as the feet were in the
      first half second, when the climber is standing before the start;
    - the wall (feet only): a foot that is still but not on a route hold. On
      a real video that is a smear OR a hold of another route, which the
      route's clicked holds can't tell apart. Hands get no wall contacts: a
      still hand off the holds is more often chalking or shaking out.

Where the contact point is:
    - hand: the pose has no fingers, so the point is pushed past the wrist,
      along the forearm, to the middle of the hand (hand_reach);
    - foot: the ankle. The toes, which touch the hold, are about 0.4 torso
      lengths away in a direction the 17 keypoints don't give, so feet get
      wider margins than hands instead.

A hidden body part (unknown position) keeps its contact: a hand that goes
behind the body while on a hold is assumed to stay there. If it reappears
somewhere else, the contact ends right after the last frame it was seen on
the target. hidden_s in each contact says how long it was a guess.

Weak evidence: cleaning hides every point under the confidence threshold
(0.5), because one such point is often wrong. But on the first real video,
Robert's right hand held h5 for 6 s (frames 211-399) in plain sight, and the
pose model saw it only with confidence 0.2-0.46: each point 0.24-0.66 torso
lengths from the hand-labeled wrist, yet all scattered AROUND h5. A wrong
guess jumps around (2.7); a hand on a hold stays put. So where the confident
point is unknown, the median of the low-confidence points over half a second
is used, for hold contacts only and with wider margins. Contacts only ask
"which hold?", a coarser question than "where exactly?".

Distances are in torso lengths and speeds in torso lengths per second (see
clean.body_size), so the rules work whatever the climber's size in the image.

Try it:
    uv run python -m climbvla.contacts climb.mp4
    -> reads climb_pose.npz and climb_route.json,
       writes climb_contacts.json (with the moves of step 2.10, see moves.py)
       and climb_contacts.png (a timeline)
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from climbvla.clean import CleanPose, body_size, clean
from climbvla.moves import find_moves
from climbvla.pose import COCO_ORDER
from climbvla.schemas import (
    TIME_TOLERANCE,
    BodyPart,
    KeypointName,
    PixelContact,
    PixelHold,
    VideoContacts,
    VideoRoute,
)
from climbvla.track import PoseTrack

INDEX = {name: j for j, name in enumerate(COCO_ORDER)}
PARTS = [BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND, BodyPart.LEFT_FOOT, BodyPart.RIGHT_FOOT]
ELBOW = {
    BodyPart.LEFT_HAND: KeypointName.LEFT_ELBOW,
    BodyPart.RIGHT_HAND: KeypointName.RIGHT_ELBOW,
}


@dataclass(frozen=True)
class ContactRules:
    """The thresholds. Defaults chosen on the first real video (RobertClimbs.mp4).

    On that video, a hand on a hold moved 0.05 to 0.6 torso lengths per
    second (the wrist turns as the body moves), and 1.9 to 8 during a move.
    The left wrist on h4 sat 0.15 to 0.34 torso lengths INSIDE the hold's
    circle; the left ankle on h2 sat 0.1 to 0.36 OUTSIDE it, because the
    toes, not the ankle, are on the hold.
    """

    hand_reach: float = 0.4  # middle of the hand, past the wrist, in forearm lengths
    hand_enter: float = 0.10  # torso lengths beyond the hold's circle
    hand_exit: float = 0.25
    foot_enter: float = 0.35
    foot_exit: float = 0.50
    still_enter: float = 0.6  # torso lengths per second
    still_exit: float = 1.2
    min_contact: float = 0.2  # seconds close and still before it counts
    min_release: float = 0.1  # seconds away before it ends
    ground_margin: float = 0.3  # torso lengths above the ground level still count
    ground_window: float = 0.5  # seconds at the start used to find the ground
    wall_exit: float = 0.25  # torso lengths a foot can slide on the wall
    # Weak evidence (low-confidence points), used where the confident point is unknown:
    weak_confidence: float = 0.2  # below this, a point is ignored completely
    weak_window: float = 0.5  # seconds: median over this window, half of it at least seen
    # On the h5 case the medians sat up to 0.36 torso lengths outside the
    # circle while the hand-labeled wrist was at most 0.17 outside: the median
    # of scattered guesses is off by about 0.2, hence a wide exit margin.
    weak_enter: float = 0.15  # torso lengths beyond the hold's circle
    weak_exit: float = 0.45


# ---------- where each body part touches, and how fast it moves ----------


def contact_points(pose: CleanPose, part: BodyPart, hand_reach: float) -> np.ndarray:
    """(N, 2) pixels: the point of this body part that touches things."""
    point = pose.xy[:, INDEX[part.keypoint]].copy()
    if part.is_hand:
        elbow = pose.xy[:, INDEX[ELBOW[part]]]
        has_elbow = ~np.isnan(elbow[:, 0])
        point[has_elbow] += hand_reach * (point[has_elbow] - elbow[has_elbow])
    return point


def rolling_median(t: np.ndarray, points: np.ndarray, window: float) -> np.ndarray:
    """(N, 2): median of the known points within window/2 seconds of each frame.

    NaN where fewer than half of the window's frames are known: a few scattered
    points are not enough to say where the body part is.
    """
    out = np.full_like(points, np.nan)
    known = ~np.isnan(points[:, 0])
    half = window / 2 + TIME_TOLERANCE
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    for i in range(len(t)):
        inside = known[lo[i] : hi[i]]
        if inside.sum() * 2 >= hi[i] - lo[i]:
            out[i] = np.median(points[lo[i] : hi[i]][inside], axis=0)
    return out


def weak_points(
    track: PoseTrack, pose: CleanPose, part: BodyPart, rules: ContactRules
) -> np.ndarray:
    """(N, 2) pixels: the contact point from the low-confidence points (see the
    module docstring), NaN where there are too few of them."""

    def median_of(name: KeypointName) -> np.ndarray:
        raw = track.keypoints[:, INDEX[name]]
        usable = raw[:, 2] >= rules.weak_confidence  # NaN confidence is not usable
        points = np.where(usable[:, None], raw[:, :2], np.nan)
        return rolling_median(pose.t, points, rules.weak_window)

    # No push toward the fingers here: the elbow is usually just as unsure
    # (confidence 0.3-0.58 in the h5 case), and a push in a wrong direction
    # moved the point onto the neighboring hold h4. The wrist alone landed on h5.
    return median_of(part.keypoint)


def speeds(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N,) pixels per second, from the frame before and the frame after.

    The average of the two one-frame speeds where both are known, the one
    that is known otherwise, NaN where the point itself is unknown.
    """
    n = len(t)
    step = np.full(n - 1, np.nan) if n > 1 else np.empty(0)
    if n > 1:
        step = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(t)
    before = np.r_[np.nan, step]  # speed from frame i-1 to i
    after = np.r_[step, np.nan]  # speed from frame i to i+1
    both = np.stack([before, after])
    count = (~np.isnan(both)).sum(axis=0)
    total = np.nansum(both, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def ground_level(
    pose: CleanPose, holds: list[PixelHold], window: float
) -> float | None:
    """Pixels: how low the lowest foot is at the start of the video, or None.

    Assumes the video starts with the climber standing. If the lowest foot in
    the first `window` seconds is not below every hold, the climber was
    already on the wall, and the ground is left unknown.
    """
    first = pose.t <= pose.t[0] + window + TIME_TOLERANCE
    ankles = pose.xy[first][:, [INDEX[KeypointName.LEFT_ANKLE], INDEX[KeypointName.RIGHT_ANKLE]], 1]
    lowest = np.nanmax(ankles, axis=1) if not np.isnan(ankles).all() else np.array([np.nan])
    lowest = lowest[~np.isnan(lowest)]
    if len(lowest) == 0:
        return None
    level = float(np.median(lowest))
    if holds and level <= max(h.y + h.radius for h in holds):
        return None
    return level


# ---------- one body part, frame by frame ----------


@dataclass
class _Open:
    """A contact that has started and not ended yet."""

    kind: str
    hold: int | None  # index in the holds list
    start: int  # frame indexes
    last_seen: int
    anchor: np.ndarray  # where it landed (used for the wall)
    leaving_since: int | None = None
    seen: list[int] = field(default_factory=list)


def detect_part(
    t: np.ndarray,
    points: np.ndarray,
    speed: np.ndarray,
    part: BodyPart,
    holds: list[PixelHold],
    torso: float,
    ground_y: float | None,
    rules: ContactRules = ContactRules(),
    weak: np.ndarray | None = None,
) -> list[PixelContact]:
    """The contacts of one body part. points and speed in pixels (per second).

    weak: (N,) bool, True where the point comes from low-confidence points
    (see weak_points). Those frames can only land on or stay on a hold, with
    the weak margins; for a wall or ground contact they count as hidden.
    """
    n = len(t)
    weak = np.zeros(n, dtype=bool) if weak is None else weak
    hand = part.is_hand
    enter_gap = (rules.hand_enter if hand else rules.foot_enter) * torso
    exit_gap = (rules.hand_exit if hand else rules.foot_exit) * torso
    weak_enter, weak_exit = rules.weak_enter * torso, rules.weak_exit * torso
    still_enter, still_exit = rules.still_enter * torso, rules.still_exit * torso
    ground_margin = rules.ground_margin * torso
    wall_exit = rules.wall_exit * torso

    if holds:
        centers = np.array([[h.x, h.y] for h in holds])
        radii = np.array([h.radius for h in holds])
        gaps = np.linalg.norm(points[:, None, :] - centers[None], axis=2) - radii  # (N, holds)
    else:
        gaps = np.full((n, 0), np.nan)
    known = ~np.isnan(points[:, 0])

    def target_at(i: int) -> tuple[str, int | None] | None:
        """What frame i would land on (the stricter, "enter" rules)."""
        if not known[i]:
            return None
        # NaN speed is not "still". Weak frames skip the speed test (see
        # still_on): a median over half a second is already a stillness test.
        if not weak[i] and not speed[i] <= still_enter:
            return None
        if gaps.shape[1] and np.nanmin(gaps[i]) <= (weak_enter if weak[i] else enter_gap):
            return "hold", int(np.nanargmin(gaps[i]))
        if hand or weak[i]:
            return None
        if ground_y is not None and points[i, 1] >= ground_y - ground_margin:
            return "ground", None
        return "wall", None

    def still_on(c: _Open, i: int) -> bool:
        """Is frame i still on c's target (the looser, "exit" rules)?"""
        if weak[i] and c.kind != "hold":
            return True  # weak evidence says nothing about the wall or the ground
        if weak[i]:
            # Distance only: a moving median jumps whenever the points inside
            # its window change (fake speeds up to 8 torso lengths per second
            # on the h5 case), so its speed says little.
            return gaps[i, c.hold] <= weak_exit
        if speed[i] > still_exit:  # NaN compares False: no speed is no reason to leave
            return False
        if c.kind == "hold":
            return gaps[i, c.hold] <= (weak_exit if weak[i] else exit_gap)
        if c.kind == "ground":
            return points[i, 1] >= ground_y - 2 * ground_margin
        return np.linalg.norm(points[i] - c.anchor) <= wall_exit

    def close(c: _Open) -> PixelContact:
        # It leaves right after the last frame it was seen on the target
        # (or the video ends while it is still there).
        end_t = float(t[min(c.last_seen + 1, n - 1)])
        span = slice(c.start, c.last_seen + 1)
        hidden = int((~known[span] | weak[span]).sum())  # not seen confidently
        dt = float(np.median(np.diff(t))) if n > 1 else 0.0
        sure = [i for i in c.seen if not weak[i]]
        where = np.median(points[sure or c.seen], axis=0)
        return PixelContact(
            body_part=part,
            kind=c.kind,
            hold_id=holds[c.hold].id if c.hold is not None else None,
            x=float(where[0]),
            y=float(where[1]),
            t_start=float(t[c.start]),
            t_end=end_t,
            hidden_s=hidden * dt,
        )

    contacts: list[PixelContact] = []
    current: _Open | None = None
    candidate: tuple[tuple[str, int | None], int] | None = None  # (target, first frame)

    for i in range(n):
        if current is None:
            target = target_at(i)
            if target is None:
                candidate = None
                continue
            if (
                candidate is None
                or candidate[0] != target
                or (target[0] == "wall" and np.linalg.norm(points[i] - points[candidate[1]]) > wall_exit)
            ):
                candidate = (target, i)
            start = candidate[1]
            if t[i] - t[start] >= rules.min_contact - TIME_TOLERANCE:
                kind, hold = candidate[0]
                current = _Open(kind, hold, start, i, points[start].copy(), seen=list(range(start, i + 1)))
                candidate = None
            continue

        if not known[i]:
            continue  # hidden: assumed to stay where it was
        if still_on(current, i):
            current.last_seen = i
            current.leaving_since = None
            current.seen.append(i)
            continue
        if current.leaving_since is None:
            current.leaving_since = i
        came_back_elsewhere = current.last_seen < i - 1 and current.leaving_since == i
        if came_back_elsewhere or t[i] - t[current.leaving_since] >= rules.min_release - TIME_TOLERANCE:
            contacts.append(close(current))
            current = None
            candidate = None
            # This frame may already be the start of the next contact.
            target = target_at(i)
            if target is not None:
                candidate = (target, i)

    if current is not None:
        contacts.append(close(current))
    return contacts


# ---------- the whole climber ----------


def detect_contacts(
    pose: CleanPose,
    route: VideoRoute,
    rules: ContactRules = ContactRules(),
    track: PoseTrack | None = None,
) -> VideoContacts:
    """Contacts of both hands and both feet, sorted by start time.

    With `track` (the raw PoseTrack that `pose` was cleaned from), the weak
    evidence fills the frames where the confident point is unknown.
    """
    torso = body_size(pose.xy)
    if np.isnan(torso):
        raise ValueError("the climber's torso is never seen: no distance unit")
    ground_y = ground_level(pose, route.holds, rules.ground_window)
    contacts = []
    for part in PARTS:
        points = contact_points(pose, part, rules.hand_reach)
        speed = speeds(pose.t, points)
        weak = np.zeros(len(pose), dtype=bool)
        if track is not None:
            fallback = weak_points(track, pose, part, rules)
            weak = np.isnan(points[:, 0]) & ~np.isnan(fallback[:, 0])
            # Speeds from each source separately: switching from one to the
            # other (wrist pushed toward the fingers or not) is not a movement.
            speed[weak] = speeds(pose.t, fallback)[weak]
            points[weak] = fallback[weak]
        contacts += detect_part(
            pose.t, points, speed, part, route.holds, torso, ground_y, rules, weak
        )
    contacts.sort(key=lambda c: (c.t_start, PARTS.index(c.body_part)))
    return VideoContacts(
        video=route.video, route_id=route.route_id, torso_px=torso, ground_y=ground_y, contacts=contacts
    )


# ---------- checking by eye ----------

PART_LABELS = {
    BodyPart.LEFT_HAND: "LH",
    BodyPart.RIGHT_HAND: "RH",
    BodyPart.LEFT_FOOT: "LF",
    BodyPart.RIGHT_FOOT: "RF",
}
KIND_COLORS = {"hold": (60, 180, 60), "wall": (160, 160, 160), "ground": (40, 90, 160)}  # BGR


def contact_label(c: PixelContact) -> str:
    return c.hold_id if c.kind == "hold" else c.kind


def draw_timeline(contacts: VideoContacts, duration: float, width: int = 1600) -> np.ndarray:
    """An image with one row per body part and one bar per contact."""
    left, top, row, bottom = 80, 40, 60, 50
    height = top + row * len(PARTS) + bottom
    image = np.full((height, width, 3), 255, np.uint8)
    scale = (width - left - 20) / max(duration, TIME_TOLERANCE)

    def x_of(t: float) -> int:
        return int(round(left + t * scale))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for s in range(int(duration) + 1):  # one tick per second
        x = x_of(s)
        cv2.line(image, (x, top - 10), (x, height - bottom), (225, 225, 225), 1)
        if s % 2 == 0:
            cv2.putText(image, f"{s}s", (x - 10, height - 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for r, part in enumerate(PARTS):
        y0 = top + r * row
        cv2.putText(image, PART_LABELS[part], (20, y0 + row // 2 + 5), font, 0.7, (0, 0, 0), 2)
        for g in contacts.regrips:
            if g.body_part == part:
                y = y0 + row // 2
                cv2.line(image, (x_of(g.t_off), y), (x_of(g.t_on), y), KIND_COLORS[g.target.kind], 3)
        for c in contacts.of(part):
            x0, x1 = x_of(c.t_start), max(x_of(c.t_end), x_of(c.t_start) + 2)
            cv2.rectangle(image, (x0, y0 + 8), (x1, y0 + row - 8), KIND_COLORS[c.kind], -1)
            if c.hidden_s > 0:  # partly guessed (body part hidden): red outline
                cv2.rectangle(image, (x0, y0 + 8), (x1, y0 + row - 8), (0, 0, 220), 2)
            cv2.putText(
                image, contact_label(c), (x0 + 4, y0 + row // 2 + 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
    return image


def print_table(contacts: VideoContacts) -> None:
    print(f"torso {contacts.torso_px:.0f} px, ground at y={contacts.ground_y}")
    print(f"{'part':4} {'on':>6} {'from':>6} {'to':>6} {'hidden':>6}")
    for c in contacts.contacts:
        print(
            f"{PART_LABELS[c.body_part]:4} {contact_label(c):>6} "
            f"{c.t_start:6.2f} {c.t_end:6.2f} {c.hidden_s:6.2f}"
        )
    if contacts.moves or contacts.regrips:
        print(f"\n{'part':4} {'move':>15} {'from':>6} {'to':>6} {'takes':>6} {'torsos':>6}")
        for m in contacts.moves:
            move = f"{m.from_target.label} -> {m.to_target.label}"
            a, b = m.from_target, m.to_target
            far = np.hypot(b.x - a.x, b.y - a.y) / contacts.torso_px  # contact point to contact point
            print(
                f"{PART_LABELS[m.body_part]:4} {move:>15} {m.t_start:6.2f} {m.t_end:6.2f} "
                f"{m.duration:6.2f} {far:6.2f}"
            )
        for g in contacts.regrips:
            regrip = f"regrip {g.target.label}"
            print(f"{PART_LABELS[g.body_part]:4} {regrip:>15} {g.t_off:6.2f} {g.t_on:6.2f} {g.duration:6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find hand and foot contacts in a climbing video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--pose", type=Path, help="default: <video>_pose.npz")
    parser.add_argument("--route", type=Path, help="default: <video>_route.json")
    parser.add_argument("--out", type=Path, help="default: <video>_contacts.json")
    args = parser.parse_args()

    stem = args.video.with_suffix("")
    track = PoseTrack.load(args.pose or f"{stem}_pose.npz")
    pose = clean(track)
    route = VideoRoute.model_validate_json(Path(args.route or f"{stem}_route.json").read_text())
    contacts = find_moves(detect_contacts(pose, route, track=track))

    out = args.out or Path(f"{stem}_contacts.json")
    out.write_text(contacts.model_dump_json(indent=2))
    timeline = out.with_suffix(".png")
    cv2.imwrite(str(timeline), draw_timeline(contacts, float(pose.t[-1])))
    print_table(contacts)
    print(
        f"wrote {out} ({len(contacts.contacts)} contacts, {len(contacts.moves)} moves, "
        f"{len(contacts.regrips)} regrips) and {timeline}"
    )


if __name__ == "__main__":
    main()
