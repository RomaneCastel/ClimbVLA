"""Climbing metrics from the cleaned pose, the contacts and the moves (Step 2.11).

Everything stays in the image, in pixels, and distances are given in torso
lengths, so the numbers compare within one video. Comparing two videos needs
the meters of step 2.14.

The metrics, each a translation of coaching advice:

    - straight-arm ratio: of the time a hand is on a hold (and its elbow is
      seen), the share with the elbow open at least `straight_arm` degrees.
      "Hang on straight arms": the skeleton holds the weight, not the biceps.
    - lock-off time: seconds a hand holds a hold with the elbow bent under
      `bent_arm` degrees while the other hand is off the holds, so that one
      bent arm holds the body. Bent arms with both hands on are already
      counted by the straight-arm ratio.
    - center-of-mass path: how far the center of mass (COM) travels during a
      move, and that length divided by the straight line from where it started
      to where it ended (1 = straight, more = detours).
    - hesitation: the still time (COM slower than `still_com`) just before a
      body part leaves for a move, since the last contact change.
    - smoothness: SPARC (spectral arc length) of the COM speed during a move.
      Around -1.4 for one smooth push, more negative for jerky, stop-and-go
      motion. The plan said "jerk"; jerk is the third derivative of position,
      and on this pose (30 fps, a few pixels of noise) it changed with the
      filter settings more than with the climbing. SPARC barely changes.

Angles in a flat image: a straight arm always looks straight, whatever its
direction, because a straight line projects to a straight line. A bent arm
can look more open or more closed than it is. And when a segment points at
the camera it looks short, and its angle means little: then the angle is left
unknown (shorter than `min_segment` of its usual length).

Not here: the balance margin (how far the COM is sideways inside the span
of the supports). It was built, then removed after watching the overlay of
the first real video: "sideways" can only be the image's x, and that video
films the start wall from the side (image x is then mostly the distance from
the wall) and the overhang from below (image x mixes along the roof and away
from it). The number jumped by 1 torso length when a foot left the ground,
with the body still. It needs wall coordinates (step 2.14) and a roughly
frontal camera, because the COM is off the wall plane.

The COM path has a milder form of the same limit: it is measured in the
image, so motion toward or away from the camera is partly missing.

A move's "phase" is the time from the last contact change of any body part
before the move (the moment the body was last settled) until the move lands.
It holds the preparation (the push, the hesitation) and the travel.

Try it:
    uv run python -m climbvla.metrics climb.mp4
    -> reads climb_pose.npz and climb_contacts.json, writes climb_metrics.json
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from climbvla.clean import CleanPose, clean
from climbvla.contacts import INDEX, PART_LABELS
from climbvla.schemas import (
    TIME_TOLERANCE,
    BodyPart,
    KeypointName,
    Metrics,
    PixelMove,
    PixelMoveMetrics,
    VideoContacts,
    VideoMetrics,
)
from climbvla.track import PoseTrack

K = KeypointName
HANDS = [BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND]
ARM = {
    BodyPart.LEFT_HAND: (K.LEFT_SHOULDER, K.LEFT_ELBOW, K.LEFT_WRIST),
    BodyPart.RIGHT_HAND: (K.RIGHT_SHOULDER, K.RIGHT_ELBOW, K.RIGHT_WRIST),
}
OTHER_HAND = {BodyPart.LEFT_HAND: BodyPart.RIGHT_HAND, BodyPart.RIGHT_HAND: BodyPart.LEFT_HAND}

# Share of the body's mass in each segment, and where its center lies along it
# (from the first joint toward the second), from Winter's table (Biomechanics
# and Motor Control of Human Movement). The forearm includes the hand and the
# shank includes the foot, because the pose has no fingers or toes.
TRUNK_MASS, TRUNK_AT = 0.497, 0.5  # shoulders' middle -> hips' middle
HEAD_MASS = 0.081  # head and neck, centered at the ears
LIMBS = [  # (joint, joint, joint, mass 1, center 1, mass 2, center 2)
    (K.LEFT_SHOULDER, K.LEFT_ELBOW, K.LEFT_WRIST, 0.028, 0.436, 0.022, 0.682),
    (K.RIGHT_SHOULDER, K.RIGHT_ELBOW, K.RIGHT_WRIST, 0.028, 0.436, 0.022, 0.682),
    (K.LEFT_HIP, K.LEFT_KNEE, K.LEFT_ANKLE, 0.100, 0.433, 0.061, 0.606),
    (K.RIGHT_HIP, K.RIGHT_KNEE, K.RIGHT_ANKLE, 0.100, 0.433, 0.061, 0.606),
]


@dataclass(frozen=True)
class MetricRules:
    """Thresholds, chosen on RobertClimbs.mp4 (see each comment)."""

    # Elbows while a hand was on a hold: 120-174 degrees on that video (10th to
    # 90th percentile per contact); arms that look straight on the overlay
    # read 165-178.
    straight_arm: float = 150.0  # degrees: at least this open counts as straight
    bent_arm: float = 120.0  # degrees: under this, the arm is clearly bent
    # A segment at 0.6 of its length points about 53 degrees toward the camera.
    # 4 to 9% of the on-hold frames were shorter on that video.
    min_segment: float = 0.6  # share of the segment's usual (90th percentile) length
    # The COM moved 0.1-0.4 torso lengths/s between moves (pose jitter and
    # small shifts), 1 to 4 during the big moves.
    still_com: float = 0.5  # torso lengths per second
    min_direct: float = 0.1  # torso lengths: shorter, the path ratio is not computed
    min_phase: float = 0.3  # seconds: shorter phases are too short for SPARC


# ---------- per-frame quantities ----------


def _point(pose: CleanPose, name: KeypointName) -> np.ndarray:
    return pose.xy[:, INDEX[name]]


def elbow_angle(pose: CleanPose, hand: BodyPart, rules: MetricRules = MetricRules()) -> np.ndarray:
    """(N,) degrees, 180 = straight. NaN where a joint is unknown or a segment
    looks too short to trust (pointing at the camera)."""
    shoulder, elbow, wrist = (_point(pose, name) for name in ARM[hand])
    upper, fore = shoulder - elbow, wrist - elbow
    upper_len, fore_len = np.linalg.norm(upper, axis=1), np.linalg.norm(fore, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(upper * fore, axis=1) / (upper_len * fore_len)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    for length in (upper_len, fore_len):
        if np.isnan(length).all():
            return np.full(len(pose), np.nan)
        usual = np.nanpercentile(length, 90)
        with np.errstate(invalid="ignore"):
            angle[length < rules.min_segment * usual] = np.nan
    return angle


def center_of_mass(pose: CleanPose) -> tuple[np.ndarray, np.ndarray]:
    """The center of mass in pixels (N, 2), and the share of the body's mass
    placed by a guess (N,).

    A limb whose end is unknown (often a hidden wrist) gets its mass at its
    last known joint: the forearm and hand are 2.2% of the body, so even a
    wrong guess moves the COM very little. NaN where the trunk is unknown.
    """
    shoulders = (_point(pose, K.LEFT_SHOULDER) + _point(pose, K.RIGHT_SHOULDER)) / 2
    hips = (_point(pose, K.LEFT_HIP) + _point(pose, K.RIGHT_HIP)) / 2
    ears = np.stack([_point(pose, K.LEFT_EAR), _point(pose, K.RIGHT_EAR)])
    known_ears = ~np.isnan(ears[..., 0])
    with np.errstate(invalid="ignore"):
        head = np.nansum(np.where(known_ears[..., None], ears, 0), axis=0) / known_ears.sum(axis=0)[:, None]
    nose = _point(pose, K.NOSE)
    head_guessed = np.isnan(head[:, 0])
    head = np.where(head_guessed[:, None], nose, head)
    head_guessed &= np.isnan(nose[:, 0])
    head = np.where(head_guessed[:, None], shoulders, head)  # no face at all

    com = TRUNK_MASS * (shoulders + TRUNK_AT * (hips - shoulders)) + HEAD_MASS * head
    guessed = HEAD_MASS * head_guessed
    for root, middle, end, m1, c1, m2, c2 in LIMBS:
        a, b, c = _point(pose, root), _point(pose, middle), _point(pose, end)
        b_missing, c_missing = np.isnan(b[:, 0]), np.isnan(c[:, 0])
        b = np.where(b_missing[:, None], a, b)
        c = np.where(c_missing[:, None], b, c)
        com = com + m1 * (a + c1 * (b - a)) + m2 * (b + c2 * (c - b))
        guessed = guessed + np.where(b_missing, m1 + m2, np.where(c_missing, m2, 0.0))
    return com, guessed


def active(contacts: VideoContacts, part: BodyPart, t: np.ndarray, holds_only: bool = False) -> np.ndarray:
    """(N,) bool: the body part is in contact (with a hold, if holds_only) at each time."""
    on = np.zeros(len(t), dtype=bool)
    for c in contacts.of(part):
        if not holds_only or c.kind == "hold":
            on |= (t >= c.t_start - TIME_TOLERANCE) & (t < c.t_end - TIME_TOLERANCE)
    return on


def com_speed(t: np.ndarray, com: np.ndarray, torso_px: float) -> np.ndarray:
    """(N,) torso lengths per second."""
    if len(t) < 2:
        return np.full(len(t), np.nan)
    return np.linalg.norm(np.gradient(com, t, axis=0), axis=1) / torso_px


def sparc(speed: np.ndarray, fs: float, max_freq: float = 10.0, threshold: float = 0.05) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al., 2015).

    The speed's frequency spectrum, normalized to 1 at its peak, is cut where
    it falls under `threshold` (up to `max_freq` Hz), and the length of that
    curve is measured. One smooth bell gives a short, simple spectrum (about
    -1.4); several pushes add bumps to the spectrum, so a longer curve and a
    more negative value. Unknown speeds are filled by straight lines.
    """
    speed = np.asarray(speed, dtype=float)
    known = ~np.isnan(speed)
    if known.sum() < 3:
        return float("nan")
    idx = np.arange(len(speed))
    speed = np.interp(idx, idx[known], speed[known])
    n = int(2 ** (np.ceil(np.log2(len(speed))) + 4))  # zero padding: a finer spectrum
    freqs = np.arange(n) * fs / n
    magnitude = np.abs(np.fft.fft(speed, n))
    if magnitude.max() == 0:
        return float("nan")
    magnitude = magnitude / magnitude.max()
    keep = freqs <= max_freq
    freqs, magnitude = freqs[keep], magnitude[keep]
    above = np.nonzero(magnitude >= threshold)[0]
    freqs, magnitude = freqs[above[0] : above[-1] + 1], magnitude[above[0] : above[-1] + 1]
    if len(freqs) < 2:
        return float("nan")
    df = np.diff(freqs) / (freqs[-1] - freqs[0])
    return float(-np.sum(np.hypot(df, np.diff(magnitude))))


# ---------- metrics ----------


def phase_start(contacts: VideoContacts, move: PixelMove, t0: float) -> float:
    """When the body was last settled before this move: the last contact
    change (any body part landing or leaving) before it starts, or t0."""
    changes = [x for c in contacts.contacts for x in (c.t_start, c.t_end) if x < move.t_start - TIME_TOLERANCE]
    return max(changes, default=t0)


def _seconds(mask: np.ndarray, t: np.ndarray) -> float:
    """Total time of the frames in mask, each frame counting one frame step."""
    if len(t) < 2:
        return 0.0
    step = float(np.median(np.diff(t)))
    return float(mask.sum() * step)


def _path(com: np.ndarray, torso_px: float) -> tuple[float, float]:
    """Path length and straight distance of the known COM points, in torso lengths."""
    known = com[~np.isnan(com[:, 0])]
    if len(known) < 2:
        return float("nan"), float("nan")
    path = float(np.sum(np.linalg.norm(np.diff(known, axis=0), axis=1)))
    direct = float(np.linalg.norm(known[-1] - known[0]))
    return path / torso_px, direct / torso_px


def _put(metrics: Metrics, name: str, value: float) -> None:
    """Store a metric only if it could be measured."""
    if not np.isnan(value):
        metrics[name] = round(float(value), 4)


@dataclass(frozen=True, eq=False)
class Series:
    """The per-frame quantities the metrics are computed from (also drawn on the overlay)."""

    t: np.ndarray
    com: np.ndarray  # (N, 2) pixels
    com_guessed: np.ndarray  # (N,) share of the mass placed by a guess
    speed: np.ndarray  # (N,) COM speed, torso lengths per second
    elbow: dict[BodyPart, np.ndarray]  # (N,) degrees, per hand
    on_hold: dict[BodyPart, np.ndarray]  # (N,) bool, per hand
    straight: dict[BodyPart, np.ndarray]  # (N,) bool: on a hold, elbow seen and straight
    lock_off: dict[BodyPart, np.ndarray]  # (N,) bool: locked off, see the module docstring


def series(pose: CleanPose, contacts: VideoContacts, rules: MetricRules = MetricRules()) -> Series:
    t = pose.t
    com, guessed = center_of_mass(pose)
    elbow = {hand: elbow_angle(pose, hand, rules) for hand in HANDS}
    on_hold = {hand: active(contacts, hand, t, holds_only=True) for hand in HANDS}
    with np.errstate(invalid="ignore"):
        straight = {h: on_hold[h] & (elbow[h] >= rules.straight_arm) for h in HANDS}
        lock_off = {h: on_hold[h] & ~on_hold[OTHER_HAND[h]] & (elbow[h] < rules.bent_arm) for h in HANDS}
    return Series(
        t=t,
        com=com,
        com_guessed=guessed,
        speed=com_speed(t, com, contacts.torso_px),
        elbow=elbow,
        on_hold=on_hold,
        straight=straight,
        lock_off=lock_off,
    )


def _arm_metrics(s: Series, frames: np.ndarray) -> tuple[float, float]:
    """Straight-arm ratio and lock-off seconds over the selected frames."""
    loaded = sum(int((s.on_hold[h] & frames & ~np.isnan(s.elbow[h])).sum()) for h in HANDS)
    straight = sum(int((s.straight[h] & frames).sum()) for h in HANDS)
    ratio = straight / loaded if loaded else float("nan")
    lock_off = sum(_seconds(s.lock_off[h] & frames, s.t) for h in HANDS)
    return ratio, lock_off


def move_metrics(
    s: Series, contacts: VideoContacts, move: PixelMove, rules: MetricRules = MetricRules()
) -> Metrics:
    t = s.t
    start = phase_start(contacts, move, float(t[0]))
    phase = (t >= start - TIME_TOLERANCE) & (t <= move.t_end + TIME_TOLERANCE)
    before = (t >= start - TIME_TOLERANCE) & (t < move.t_start - TIME_TOLERANCE)
    m: Metrics = {}
    _put(m, "duration_s", move.t_end - move.t_start)
    _put(m, "phase_s", move.t_end - start)
    with np.errstate(invalid="ignore"):
        _put(m, "hesitation_s", _seconds(before & (s.speed < rules.still_com), t))
    path, direct = _path(s.com[phase], contacts.torso_px)
    _put(m, "com_path_torso", path)
    if direct >= rules.min_direct:
        _put(m, "com_path_ratio", path / direct)
    if move.t_end - start >= rules.min_phase - TIME_TOLERANCE:
        _put(m, "smoothness_sparc", sparc(s.speed[phase], 1 / float(np.median(np.diff(t)))))
    ratio, lock_off = _arm_metrics(s, phase)
    _put(m, "straight_arm_ratio", ratio)
    _put(m, "lock_off_s", lock_off)
    return m


def climb_metrics(
    s: Series, contacts: VideoContacts, moves: list[Metrics], rules: MetricRules = MetricRules()
) -> Metrics:
    """Whole-climb metrics, from the first move's start to the last move's end
    (the whole video if there are no moves)."""
    t = s.t
    if contacts.moves:
        start, end = min(m.t_start for m in contacts.moves), max(m.t_end for m in contacts.moves)
    else:
        start, end = float(t[0]), float(t[-1])
    span = (t >= start - TIME_TOLERANCE) & (t <= end + TIME_TOLERANCE)
    c: Metrics = {"moves": float(len(contacts.moves)), "regrips": float(len(contacts.regrips))}
    _put(c, "duration_s", end - start)
    ratio, lock_off = _arm_metrics(s, np.ones(len(t), dtype=bool))  # the whole video: holds are holds
    _put(c, "straight_arm_ratio", ratio)
    _put(c, "lock_off_s", lock_off)
    # No path ratio for the whole climb: a climber who climbs back down ends
    # near the start, and the ratio explodes (7.4 on the first real video).
    path, _ = _path(s.com[span], contacts.torso_px)
    _put(c, "com_path_torso", path)
    _put(c, "hesitation_s", sum(m.get("hesitation_s", 0.0) for m in moves))
    sparcs = [m["smoothness_sparc"] for m in moves if "smoothness_sparc" in m]
    if sparcs:
        _put(c, "smoothness_sparc_median", float(np.median(sparcs)))
    _put(c, "com_guessed_mass_mean", float(np.nanmean(s.com_guessed[span])) if span.any() else float("nan"))
    return c


def compute_metrics(
    pose: CleanPose, contacts: VideoContacts, rules: MetricRules = MetricRules()
) -> tuple[VideoMetrics, Series]:
    """The metrics of one video, and the per-frame series they come from."""
    s = series(pose, contacts, rules)
    per_move = [move_metrics(s, contacts, m, rules) for m in contacts.moves]
    result = VideoMetrics(
        video=contacts.video,
        route_id=contacts.route_id,
        torso_px=contacts.torso_px,
        climb=climb_metrics(s, contacts, per_move, rules),
        moves=[
            PixelMoveMetrics(body_part=m.body_part, t_start=m.t_start, t_end=m.t_end, metrics=mm)
            for m, mm in zip(contacts.moves, per_move)
        ],
    )
    return result, s


# ---------- command line ----------


def print_tables(contacts: VideoContacts, result: VideoMetrics) -> None:
    def show(m: Metrics, name: str, fmt: str) -> str:
        return format(m[name], fmt) if name in m else "-"

    print(f"{'part':4s} {'move':14s} {'t':>6s} {'air':>5s} {'phase':>5s} {'hesit':>5s} "
          f"{'path':>5s} {'ratio':>5s} {'sparc':>6s} {'straight':>8s} {'lockoff':>7s}")
    for move, mm in zip(contacts.moves, result.moves):
        m = mm.metrics
        label = f"{move.from_target.label} -> {move.to_target.label}"
        print(
            f"{PART_LABELS[move.body_part]:4s} {label:14s} {move.t_start:6.2f} {show(m, 'duration_s', '5.2f')} "
            f"{show(m, 'phase_s', '5.2f')} {show(m, 'hesitation_s', '5.2f')} {show(m, 'com_path_torso', '5.2f')} "
            f"{show(m, 'com_path_ratio', '5.2f')} "
            f"{show(m, 'smoothness_sparc', '6.2f')} {show(m, 'straight_arm_ratio', '8.0%')} {show(m, 'lock_off_s', '7.2f')}"
        )
    print("\nwhole climb:")
    for name, value in result.climb.items():
        print(f"  {name:30s} {value:8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Climbing metrics from the pose and the contacts.")
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    stem = args.video.with_name(args.video.stem)
    track = PoseTrack.load(f"{stem}_pose.npz")
    contacts = VideoContacts.model_validate_json(Path(f"{stem}_contacts.json").read_text())
    result, _ = compute_metrics(clean(track), contacts)
    out = Path(f"{stem}_metrics.json")
    out.write_text(result.model_dump_json(indent=2))
    print_tables(contacts, result)
    print(f"\nwrote {out}: {len(result.moves)} moves")


if __name__ == "__main__":
    main()
