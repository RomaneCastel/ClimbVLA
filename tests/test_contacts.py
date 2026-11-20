"""Tests for contact detection (step 2.9), on made-up paths with known answers."""

import numpy as np
import pytest
from pydantic import ValidationError

from climbvla.clean import CleanPose
from climbvla.contacts import (
    ContactRules,
    contact_points,
    detect_contacts,
    detect_part,
    ground_level,
    rolling_median,
    speeds,
)
from climbvla.track import PoseTrack
from climbvla.pose import COCO_ORDER
from climbvla.schemas import (
    BodyPart,
    KeypointName,
    PixelContact,
    PixelHold,
    VideoContacts,
    VideoRoute,
)

FPS = 30
NAN = np.nan
TORSO = 100.0  # pixels: the distance unit in these tests
RULES = ContactRules()
HAND, FOOT = BodyPart.LEFT_HAND, BodyPart.LEFT_FOOT
K = KeypointName
I = {name: j for j, name in enumerate(COCO_ORDER)}
HOLD = PixelHold(id="h1", x=500, y=500, radius=30)


def times(n: int) -> np.ndarray:
    return np.arange(n) / FPS


def path(*pieces: tuple[float, tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Points moving in straight lines: (seconds, from, to) pieces, end to end."""
    out = []
    for seconds, a, b in pieces:
        n = round(seconds * FPS)
        share = np.arange(n)[:, None] / n
        out.append(np.array(a) + share * (np.array(b) - np.array(a)))
    return np.concatenate(out)


def run(points: np.ndarray, part=HAND, holds=(HOLD,), ground_y=None) -> list[PixelContact]:
    t = times(len(points))
    return detect_part(t, points, speeds(t, points), part, list(holds), TORSO, ground_y, RULES)


AWAY, ON = (100.0, 100.0), (500.0, 500.0)


# ---------- speeds ----------


def test_speeds_constant_motion():
    points = path((0.5, (0, 0), (150, 0)))  # 10 px per frame
    assert np.allclose(speeds(times(len(points)), points), 10 * FPS)


def test_speeds_next_to_unknown_points():
    points = np.array([[0, 0], [10, 0], [NAN, NAN], [40, 0]], dtype=float)
    v = speeds(times(4), points)
    assert v[1] == pytest.approx(10 * FPS)  # only the frame before is known
    assert np.isnan(v[2])
    assert np.isnan(v[3])  # its only neighbor is unknown


# ---------- one hand on one hold ----------


def test_hand_resting_on_a_hold():
    # 0.5 s reaching, 1 s on the hold, 0.5 s leaving
    points = path((0.5, AWAY, ON), (1.0, ON, ON), (0.5, ON, AWAY))
    [c] = run(points)
    assert (c.kind, c.hold_id, c.body_part) == ("hold", "h1", HAND)
    assert c.t_start == pytest.approx(0.5, abs=2 / FPS)
    assert c.t_end == pytest.approx(1.5, abs=2 / FPS)
    assert (c.x, c.y) == pytest.approx(ON, abs=1)
    assert c.hidden_s == 0


def test_hand_passing_over_a_hold_is_not_a_contact():
    points = path((1.0, (300, 500), (700, 500)))  # crosses the hold at 400 px/s = 4 torso/s
    assert run(points) == []


def test_too_short_a_touch_is_not_a_contact():
    points = path((0.5, AWAY, ON), (0.1, ON, ON), (0.5, ON, AWAY))
    assert run(points) == []


def test_jitter_at_the_edge_does_not_split_the_contact():
    """Hysteresis: after landing, the hand wobbles between 0.12 and 0.2 torso
    lengths outside the circle: outside the enter margin (0.10), inside the
    exit margin (0.25). It stays one contact."""
    edge = HOLD.radius
    landed = path((0.5, ON, ON))
    # a slow wobble (about 0.5 torso lengths per second at most): pose jitter, not a move
    phase = 2 * np.pi * times(60) / 0.5
    wobble = np.stack([500 + edge + 16 + 4 * np.sin(phase), np.full(60, 500.0)], axis=1)
    points = np.concatenate([path((0.5, AWAY, ON)), landed, wobble, path((0.5, ON, AWAY))])
    [c] = run(points)
    assert c.t_end == pytest.approx(3.0, abs=3 / FPS)  # lasts until the hand really leaves


def test_nearest_of_two_holds():
    far = PixelHold(id="h2", x=540, y=500, radius=30)  # also overlaps ON, but its center is further
    points = path((0.5, AWAY, ON), (1.0, ON, ON))
    [c] = run(points, holds=(far, HOLD))
    assert c.hold_id == "h1"


def test_contact_still_open_at_the_end():
    points = path((0.5, AWAY, ON), (1.0, ON, ON))
    [c] = run(points)
    assert c.t_end == pytest.approx(times(len(points))[-1])


# ---------- hidden body parts ----------


def test_hidden_on_the_hold_keeps_the_contact():
    points = path((0.5, AWAY, ON), (1.5, ON, ON), (0.5, ON, AWAY))
    points[30:45] = NAN  # 0.5 s hidden while on the hold
    [c] = run(points)
    assert c.hidden_s == pytest.approx(0.5)
    assert c.t_end == pytest.approx(2.0, abs=2 / FPS)


def test_hidden_then_seen_elsewhere_ends_after_last_seen():
    points = path((0.5, AWAY, ON), (1.0, ON, ON), (1.0, AWAY, AWAY))
    points[40:60] = NAN  # last seen on the hold at frame 39, reappears away at frame 60
    [c] = run(points)
    assert c.t_end == pytest.approx(40 / FPS)
    assert c.hidden_s == 0  # hidden frames after the last sighting are not part of it


# ---------- feet: wall and ground ----------


def test_still_hand_off_the_holds_is_not_a_contact():
    points = path((1.0, AWAY, AWAY))
    assert run(points, part=HAND) == []


def test_still_foot_off_the_holds_is_on_the_wall():
    points = path((1.0, AWAY, AWAY))
    [c] = run(points, part=FOOT)
    assert c.kind == "wall" and c.hold_id is None


def test_foot_sliding_on_the_wall_ends_the_smear():
    # still, then slides 0.5 torso lengths slowly (0.5 torso/s), then still again
    points = path((1.0, AWAY, AWAY), (1.0, AWAY, (150, 100)), (1.0, (150, 100), (150, 100)))
    contacts = run(points, part=FOOT)
    assert len(contacts) == 2
    assert all(c.kind == "wall" for c in contacts)


def test_foot_on_the_ground():
    points = path((1.0, (100, 900), (100, 900)))
    [c] = run(points, part=FOOT, ground_y=920.0)  # 20 px = 0.2 torso above the ground level
    assert c.kind == "ground"


def test_foot_wider_margin_than_hand():
    # 0.3 torso lengths outside the circle: too far for a hand, close enough for a foot
    beside = (500 + HOLD.radius + 30, 500)
    points = path((1.0, beside, beside))
    assert run(points, part=HAND) == []
    [c] = run(points, part=FOOT)
    assert c.hold_id == "h1"


# ---------- the whole climber ----------


def make_pose(n: int, moves: dict[KeypointName, tuple[float, float] | np.ndarray]) -> CleanPose:
    """A standing pose (torso of 100 px), with the given keypoints replaced."""
    xy = np.full((n, len(COCO_ORDER), 2), 300.0)
    xy[:, I[K.LEFT_SHOULDER]] = (280, 600)
    xy[:, I[K.RIGHT_SHOULDER]] = (320, 600)
    xy[:, I[K.LEFT_HIP]] = (280, 700)
    xy[:, I[K.RIGHT_HIP]] = (320, 700)
    for name, where in moves.items():
        xy[:, I[name]] = where
    false = np.zeros((n, len(COCO_ORDER)), dtype=bool)
    return CleanPose(t=times(n), xy=xy, confidence=np.ones((n, len(COCO_ORDER))), filled=false, swapped=false)


def test_hand_point_is_past_the_wrist():
    pose = make_pose(3, {K.LEFT_ELBOW: (400, 600), K.LEFT_WRIST: (500, 600)})
    point = contact_points(pose, HAND, hand_reach=0.4)
    assert tuple(point[0]) == (540, 600)  # 0.4 forearm lengths past the wrist


def test_hand_point_without_elbow_is_the_wrist():
    pose = make_pose(3, {K.LEFT_ELBOW: (NAN, NAN), K.LEFT_WRIST: (500, 600)})
    assert tuple(contact_points(pose, HAND, hand_reach=0.4)[0]) == (500, 600)


def test_ground_level_from_a_standing_start():
    pose = make_pose(30, {K.LEFT_ANKLE: (280, 900), K.RIGHT_ANKLE: (320, 880)})
    assert ground_level(pose, [HOLD], window=0.5) == 900  # the lower of the two feet


def test_no_ground_when_already_on_the_wall():
    pose = make_pose(30, {K.LEFT_ANKLE: (280, 520), K.RIGHT_ANKLE: (320, 520)})
    assert ground_level(pose, [HOLD], window=0.5) is None  # feet not below the holds


def test_detect_contacts_whole_climber():
    n = 60
    start = PixelHold(id="h1", x=520, y=600, radius=40)
    top = PixelHold(id="h2", x=900, y=200, radius=40)
    route = VideoRoute(
        route_id="r", video="v.mp4", width=1000, height=1000, frame_t=0,
        holds=[start, top], start_hold_ids=["h1"], top_hold_id="h2",
    )
    pose = make_pose(
        n,
        {
            K.LEFT_ELBOW: (400, 600), K.LEFT_WRIST: (480, 600),  # hand point at 512: on h1
            K.RIGHT_ELBOW: (360, 640), K.RIGHT_WRIST: (360, 680),  # hanging, off the holds
            K.LEFT_ANKLE: (280, 950), K.RIGHT_ANKLE: (320, 950),  # standing
        },
    )
    contacts = detect_contacts(pose, route)
    assert contacts.torso_px == pytest.approx(100)
    assert contacts.ground_y == pytest.approx(950)
    found = {(c.body_part, c.kind, c.hold_id) for c in contacts.contacts}
    assert found == {
        (BodyPart.LEFT_HAND, "hold", "h1"),
        (BodyPart.LEFT_FOOT, "ground", None),
        (BodyPart.RIGHT_FOOT, "ground", None),
    }


# ---------- weak evidence (low-confidence points) ----------


def test_rolling_median_needs_half_the_window():
    points = np.full((15, 2), NAN)
    points[:8] = (10, 20)  # 8 of the 15 frames known
    out = rolling_median(times(15), points, window=0.5)  # 0.5 s = 15 frames at 30 fps
    assert tuple(out[7]) == (10, 20)  # its window holds 8 known frames of 15
    assert np.isnan(out[14]).all()  # its window holds 1 known frame of 8


def weak_route() -> VideoRoute:
    return VideoRoute(
        route_id="r", video="v.mp4", width=1000, height=1000, frame_t=0,
        holds=[HOLD, PixelHold(id="h2", x=900, y=100, radius=30)],
        start_hold_ids=["h1"], top_hold_id="h2",
    )


def make_track(pose: CleanPose, confidence: np.ndarray) -> PoseTrack:
    """The raw track `pose` could have been cleaned from, with these confidences.

    A model always places every keypoint: unknown points become (0, 0) with
    confidence 0.
    """
    unknown = np.isnan(pose.xy[..., 0])
    confidence = np.where(unknown, 0.0, confidence)
    keypoints = np.concatenate([np.nan_to_num(pose.xy), confidence[..., None]], axis=2)
    n = len(pose)
    return PoseTrack(
        video="v.mp4", width=1000, height=1000, models="test", frame_index=np.arange(n),
        t=pose.t, boxes=np.full((n, 5), 1.0), keypoints=keypoints,
    )


def hand_on_hold_seen_weakly(scatter: np.ndarray) -> tuple[CleanPose, PoseTrack]:
    """Left hand reaches the hold, stays 3 s, leaves. In the middle 2 s the model
    gives only low-confidence points (0.3), placed at `scatter` (60, 2); cleaning
    hid them."""
    hand = np.concatenate([path((0.5, AWAY, ON)), path((3.0, ON, ON)), path((0.5, ON, AWAY))])
    raw = make_pose(len(hand), {K.LEFT_WRIST: hand, K.LEFT_ELBOW: (NAN, NAN)})
    confidence = np.ones((len(hand), len(COCO_ORDER)))
    weak_frames = slice(30, 90)
    raw.xy[weak_frames, I[K.LEFT_WRIST]] = scatter
    confidence[weak_frames, I[K.LEFT_WRIST]] = 0.3
    cleaned_xy = raw.xy.copy()
    cleaned_xy[weak_frames, I[K.LEFT_WRIST]] = NAN
    cleaned = CleanPose(t=raw.t, xy=cleaned_xy, confidence=confidence, filled=raw.filled, swapped=raw.swapped)
    return cleaned, make_track(raw, confidence)


def left_hand(contacts: VideoContacts) -> list[PixelContact]:
    return contacts.of(BodyPart.LEFT_HAND)


def test_weak_points_around_the_hold_keep_the_contact():
    rng = np.random.default_rng(0)
    scatter = np.array(ON) + rng.uniform(-40, 40, size=(60, 2))  # guesses all around the hold
    pose, track = hand_on_hold_seen_weakly(scatter)
    [c] = left_hand(detect_contacts(pose, weak_route(), track=track))
    # one contact, and the 2 s seen only weakly are counted as not seen confidently
    assert c.hold_id == "h1"
    assert c.t_end == pytest.approx(3.5, abs=3 / FPS)
    assert c.hidden_s == pytest.approx(2.0, abs=2 / FPS)


def test_weak_points_find_a_contact_the_confident_points_miss():
    """The hand lands while seen only weakly: without the weak evidence there
    is no contact at all (the h5 case on the first real video)."""
    rng = np.random.default_rng(1)
    n = 90
    hand = np.array(ON) + rng.uniform(-40, 40, size=(n, 2))
    raw = make_pose(n, {K.LEFT_WRIST: hand, K.LEFT_ELBOW: (NAN, NAN)})
    confidence = np.ones((n, len(COCO_ORDER)))
    confidence[:, I[K.LEFT_WRIST]] = 0.3
    cleaned_xy = raw.xy.copy()
    cleaned_xy[:, I[K.LEFT_WRIST]] = NAN
    pose = CleanPose(t=raw.t, xy=cleaned_xy, confidence=confidence, filled=raw.filled, swapped=raw.swapped)
    assert left_hand(detect_contacts(pose, weak_route())) == []
    [c] = left_hand(detect_contacts(pose, weak_route(), track=make_track(raw, confidence)))
    assert c.hold_id == "h1"


def test_jumping_guesses_are_not_a_contact():
    """Low-confidence points jumping between the hold and a place 3 torso
    lengths away (the wrong-guess pattern of step 2.7): the median falls
    between the two, on no hold."""
    n = 90
    jumps = np.array([ON if (i // 2) % 2 else (800.0, 500.0) for i in range(n)])
    raw = make_pose(n, {K.LEFT_WRIST: jumps, K.LEFT_ELBOW: (NAN, NAN)})
    confidence = np.ones((n, len(COCO_ORDER)))
    confidence[:, I[K.LEFT_WRIST]] = 0.3
    cleaned_xy = raw.xy.copy()
    cleaned_xy[:, I[K.LEFT_WRIST]] = NAN
    pose = CleanPose(t=raw.t, xy=cleaned_xy, confidence=confidence, filled=raw.filled, swapped=raw.swapped)
    assert left_hand(detect_contacts(pose, weak_route(), track=make_track(raw, confidence))) == []


def test_weak_points_make_no_wall_contact():
    n = 60
    raw = make_pose(n, {K.LEFT_ANKLE: (200, 800), K.RIGHT_ANKLE: (320, 950)})
    confidence = np.ones((n, len(COCO_ORDER)))
    confidence[:, I[K.LEFT_ANKLE]] = 0.3  # a still foot, off the holds, seen weakly
    cleaned_xy = raw.xy.copy()
    cleaned_xy[:, I[K.LEFT_ANKLE]] = NAN
    pose = CleanPose(t=raw.t, xy=cleaned_xy, confidence=confidence, filled=raw.filled, swapped=raw.swapped)
    contacts = detect_contacts(pose, weak_route(), track=make_track(raw, confidence))
    assert contacts.of(BodyPart.LEFT_FOOT) == []


# ---------- the saved format ----------


def contact(**overrides) -> dict:
    data = {"body_part": "left_hand", "kind": "hold", "hold_id": "h1", "x": 1, "y": 2,
            "t_start": 0.5, "t_end": 1.5}
    data.update(overrides)
    return data


def test_contacts_round_trip():
    saved = VideoContacts(
        video="v.mp4", route_id="r", torso_px=100, ground_y=900,
        contacts=[
            PixelContact(**contact()),
            PixelContact(**contact(body_part="left_foot", kind="wall", hold_id=None)),
        ],
    )
    assert VideoContacts.model_validate_json(saved.model_dump_json()) == saved


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"t_end": 0.5}, "must be after"),
        ({"hold_id": None}, "hold_id must be set"),
        ({"kind": "wall"}, "hold_id must be set"),
        ({"kind": "ground", "hold_id": "h1"}, "hold_id must be set"),
    ],
)
def test_pixel_contact_rejects_bad_input(overrides, message):
    with pytest.raises(ValidationError, match=message):
        PixelContact(**contact(**overrides))


def test_video_contacts_rejects_overlap():
    with pytest.raises(ValidationError, match="overlapping"):
        VideoContacts(
            video="v.mp4", route_id="r", torso_px=100,
            contacts=[PixelContact(**contact()), PixelContact(**contact(t_start=1.0, t_end=2.0))],
        )
