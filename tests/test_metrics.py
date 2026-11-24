"""Tests for the climbing metrics (step 2.11), on made-up poses with known answers."""

import numpy as np
import pytest

from climbvla.clean import CleanPose
from climbvla.metrics import (
    HEAD_MASS,
    LIMBS,
    TRUNK_MASS,
    MetricRules,
    active,
    center_of_mass,
    compute_metrics,
    elbow_angle,
    phase_start,
    sparc,
)
from climbvla.moves import find_moves
from climbvla.pose import COCO_ORDER
from climbvla.schemas import (
    BodyPart,
    KeypointName,
    PixelContact,
    VideoContacts,
    VideoMetrics,
)

FPS = 30
TORSO = 100.0  # pixels: shoulders' middle to hips' middle in BODY below
K = KeypointName
I = {name: j for j, name in enumerate(COCO_ORDER)}
LH, RH, LF, RF = BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND, BodyPart.LEFT_FOOT, BodyPart.RIGHT_FOOT

# A climber facing the wall, arms straight up, in pixels (y down).
BODY = {
    K.NOSE: (500, 280), K.LEFT_EYE: (490, 275), K.RIGHT_EYE: (510, 275),
    K.LEFT_EAR: (480, 280), K.RIGHT_EAR: (520, 280),
    K.LEFT_SHOULDER: (470, 300), K.RIGHT_SHOULDER: (530, 300),
    K.LEFT_ELBOW: (470, 250), K.RIGHT_ELBOW: (530, 250),
    K.LEFT_WRIST: (470, 200), K.RIGHT_WRIST: (530, 200),
    K.LEFT_HIP: (480, 400), K.RIGHT_HIP: (520, 400),
    K.LEFT_KNEE: (480, 460), K.RIGHT_KNEE: (520, 460),
    K.LEFT_ANKLE: (480, 520), K.RIGHT_ANKLE: (520, 520),
}  # fmt: skip


def make_pose(n: int, changes: dict | None = None) -> CleanPose:
    """n frames of BODY; `changes` maps a keypoint to a point or an (n, 2) path."""
    xy = np.zeros((n, len(COCO_ORDER), 2))
    for name, point in BODY.items():
        xy[:, I[name]] = point
    for name, where in (changes or {}).items():
        xy[:, I[name]] = where
    false = np.zeros((n, len(COCO_ORDER)), dtype=bool)
    t = np.arange(n) / FPS
    return CleanPose(t=t, xy=xy, confidence=np.ones((n, len(COCO_ORDER))), filled=false, swapped=false)


def on(part, t0, t1, hold=None, x=500.0, y=200.0, kind=None) -> PixelContact:
    kind = kind or ("hold" if hold else "wall")
    return PixelContact(body_part=part, kind=kind, hold_id=hold, x=x, y=y, t_start=t0, t_end=t1)


def video(*contacts: PixelContact) -> VideoContacts:
    return find_moves(VideoContacts(video="v.mp4", route_id="r", torso_px=TORSO, contacts=list(contacts)))


# ---------- elbow angles ----------


def test_straight_and_right_angle_elbows():
    pose = make_pose(2, {K.LEFT_WRIST: [(470, 200), (520, 250)]})  # frame 1: forearm sideways
    angle = elbow_angle(pose, LH)
    assert angle[0] == pytest.approx(180)
    assert angle[1] == pytest.approx(90)


def test_elbow_unknown_when_a_joint_is_unknown():
    pose = make_pose(2, {K.RIGHT_WRIST: [(530, 200), (np.nan, np.nan)]})
    angle = elbow_angle(pose, RH)
    assert angle[0] == pytest.approx(180)
    assert np.isnan(angle[1])


def test_elbow_unknown_when_a_segment_points_at_the_camera():
    # 20 frames at the usual forearm length (50 px), then one at 25 px: 0.5 of it
    wrist = np.array([(470, 200)] * 20 + [(470, 225)], dtype=float)
    angle = elbow_angle(make_pose(21, {K.LEFT_WRIST: wrist}), LH)
    assert angle[0] == pytest.approx(180)
    assert np.isnan(angle[20])
    assert elbow_angle(make_pose(21, {K.LEFT_WRIST: wrist}), LH, MetricRules(min_segment=0.4))[20] == pytest.approx(180)


# ---------- center of mass ----------


def test_segment_masses_add_up_to_the_whole_body():
    limbs = sum(m1 + m2 for *_, m1, _, m2, _ in LIMBS)
    assert TRUNK_MASS + HEAD_MASS + limbs == pytest.approx(1.0, abs=1e-3)


def test_center_of_mass_of_a_symmetric_body_is_on_its_middle_line():
    com, guessed = center_of_mass(make_pose(1))
    assert com[0, 0] == pytest.approx(500)
    assert 300 < com[0, 1] < 400  # between shoulders and hips: arms up, legs down
    assert guessed[0] == 0


def test_a_hidden_wrist_moves_the_center_of_mass_very_little():
    full, _ = center_of_mass(make_pose(1))
    com, guessed = center_of_mass(make_pose(1, {K.RIGHT_WRIST: (np.nan, np.nan)}))
    assert guessed[0] == pytest.approx(0.022)  # forearm and hand
    assert np.linalg.norm(com[0] - full[0]) < 0.01 * TORSO


def test_center_of_mass_unknown_without_the_trunk():
    com, _ = center_of_mass(make_pose(1, {K.LEFT_HIP: (np.nan, np.nan)}))
    assert np.isnan(com[0]).all()


# ---------- smoothness ----------


def bell(n: int = 31) -> np.ndarray:
    """Speed of a minimum-jerk movement: the smoothest point-to-point move."""
    tau = np.linspace(0, 1, n)
    return 30 * tau**2 * (1 - tau) ** 2


def test_sparc_of_one_smooth_push_two_pushes_and_noise():
    one = sparc(bell(), FPS)
    assert one == pytest.approx(-1.4, abs=0.1)
    assert sparc(np.r_[bell(), bell()], FPS) < one - 0.5  # stop and go
    noisy = np.abs(bell() + np.random.default_rng(0).normal(0, 0.19, 31))  # 10% of the peak
    assert sparc(noisy, FPS) == pytest.approx(one, abs=0.3)


def test_sparc_of_no_movement_is_unknown():
    assert np.isnan(sparc(np.zeros(20), FPS))
    assert np.isnan(sparc(np.array([1.0, np.nan]), FPS))


# ---------- phases ----------


def test_phase_starts_at_the_last_contact_change_before_the_move():
    contacts = video(
        on(LH, 0, 2, "h1"), on(LH, 2.5, 4, "h2"),
        on(RF, 0, 1, kind="ground", x=500, y=600), on(RF, 1.5, 4, "h3"),
    )  # fmt: skip
    left_move = next(m for m in contacts.moves if m.body_part == LH)
    assert phase_start(contacts, left_move, 0.0) == 1.5  # the right foot landed then
    first = next(m for m in contacts.moves if m.body_part == RF)
    assert phase_start(contacts, first, 0.0) == 0.0  # nothing changed before


def test_active_with_and_without_holds_only():
    contacts = video(on(LF, 0, 1, kind="wall"), on(LF, 1.5, 2, "h1"))
    t = np.array([0.5, 1.2, 1.7])
    assert active(contacts, LF, t).tolist() == [True, False, True]
    assert active(contacts, LF, t, holds_only=True).tolist() == [False, False, True]


# ---------- whole metrics on a made-up climb ----------


def climb(n: int = 90, changes: dict | None = None, contacts=None):
    """3 s: both hands on holds, the left hand moves from h1 to h2 at 1.5-2.0 s."""
    contacts = contacts or video(
        on(LH, 0, 1.5, "h1", x=470), on(LH, 2.0, 3, "h2", x=470, y=150),
        on(RH, 0, 3, "h3", x=530),
        on(LF, 0, 3, "h4", x=480, y=520), on(RF, 0, 3, "h5", x=520, y=520),
    )  # fmt: skip
    return compute_metrics(make_pose(n, changes), contacts)


def test_straight_arms_all_the_time():
    result, _ = climb()
    assert result.climb["straight_arm_ratio"] == pytest.approx(1.0)
    assert result.climb["lock_off_s"] == 0
    assert result.climb["moves"] == 1


def test_lock_off_needs_a_bent_arm_and_the_other_hand_off():
    # the right elbow bends to 90 degrees all along; the left hand is off at 1.5-2.0 s
    result, _ = climb(changes={K.RIGHT_WRIST: (580, 250)})
    assert result.climb["lock_off_s"] == pytest.approx(0.5, abs=1 / FPS)
    assert result.climb["straight_arm_ratio"] < 0.6  # the right arm is never straight
    [move] = result.moves
    assert move.metrics["lock_off_s"] == pytest.approx(0.5, abs=1 / FPS)


def test_hesitation_counts_only_still_time_before_the_move():
    still, _ = climb()
    assert still.moves[0].metrics["hesitation_s"] == pytest.approx(1.5, abs=1 / FPS)
    # the whole body sways sideways (2 torso lengths per second) before the move
    sway = 500 + 200 * np.abs(((np.arange(90) / FPS) % 0.5) - 0.25)
    shifted = {name: np.c_[sway - 500 + x, np.full(90, y)] for name, (x, y) in BODY.items()}
    moving, _ = climb(changes=shifted)
    assert moving.moves[0].metrics["hesitation_s"] < 0.2


def test_com_path_ratio_is_one_for_a_straight_path_and_absent_when_still():
    lift = np.r_[np.zeros(45), np.linspace(0, 50, 15), np.full(30, 50)]  # 0.5 torso up, in a line
    rising = {name: np.c_[np.full(90, x), y - lift] for name, (x, y) in BODY.items()}
    result, _ = climb(changes=rising)
    assert result.moves[0].metrics["com_path_ratio"] == pytest.approx(1.0, abs=0.01)
    assert result.moves[0].metrics["com_path_torso"] == pytest.approx(0.5, abs=0.01)
    still, _ = climb()
    assert "com_path_ratio" not in still.moves[0].metrics  # moved less than min_direct


def test_unmeasurable_metrics_are_left_out_and_the_result_round_trips():
    result, _ = climb(changes={K.LEFT_ELBOW: (np.nan, np.nan), K.RIGHT_ELBOW: (np.nan, np.nan)})
    assert "straight_arm_ratio" not in result.climb
    assert VideoMetrics.model_validate_json(result.model_dump_json()) == result


def test_moves_are_named_like_in_the_contacts():
    result, _ = climb()
    [move] = result.moves
    assert (move.body_part, move.t_start, move.t_end) == (LH, 1.5, 2.0)
