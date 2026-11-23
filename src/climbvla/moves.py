"""Moves from contacts, in pixels (Step 2.10).

A move is one body part going from one contact to the next. It starts when
the body part leaves the first contact and ends when it lands on the second.

Each body part's contacts follow one another in time, with free gaps between
them. Every gap between two consecutive contacts is one of two things:

    - a move, when the two contacts are on different targets;
    - a regrip, when the body part comes back to the same target: re-gripping
      a hold, chalking up, shaking out, or re-placing a foot. Not a move: the
      climber ends up where they were.

So regrips are "merged": they never appear as moves. The contacts themselves
are kept exactly as step 2.9 found them, and the regrips are listed next to
the moves, so the time spent chalking or shaking out stays available for the
metrics (2.11).

Same target means: the same hold; the ground; or two spots on the wall less
than `same_wall` torso lengths apart. A wall contact is any still foot off the
route's holds (a smear, or another route's hold), so a foot re-placed a few
centimeters away on the same smear is a regrip, not a move.

Moves of different body parts are independent and can overlap in time: a
dyno is both hands moving at once. What the other body parts touch during a
move is not stored with it: it can be read from the contacts.

The free time before a body part's first contact and after its last one
(dropping off at the end) is neither a move nor a regrip.

Try it: moves are added by the contact detection,
    uv run python -m climbvla.contacts climb.mp4
"""

from dataclasses import dataclass

import numpy as np

from climbvla.schemas import (
    BodyPart,
    PixelContact,
    PixelMove,
    PixelRegrip,
    PixelTarget,
    VideoContacts,
)


@dataclass(frozen=True)
class MoveRules:
    """The one threshold. Default chosen on the first real video (RobertClimbs.mp4).

    There, the right foot left a smear and landed 0.39 torso lengths away
    0.34 s later (t=17.5 s, a re-placement), while its real moves between
    wall spots covered 1.3 and 1.5 torso lengths. 0.5 is also the foot's exit
    margin in contact detection: closer than that, it would have stayed on.
    """

    same_wall: float = 0.5  # torso lengths


def same_target(a: PixelContact, b: PixelContact, torso_px: float, rules: MoveRules = MoveRules()) -> bool:
    """Are two contacts of one body part on the same target?"""
    if a.kind != b.kind:
        return False
    if a.kind == "hold":
        return a.hold_id == b.hold_id
    if a.kind == "ground":
        return True
    return float(np.hypot(a.x - b.x, a.y - b.y)) <= rules.same_wall * torso_px


def find_moves(contacts: VideoContacts, rules: MoveRules = MoveRules()) -> VideoContacts:
    """A copy of `contacts` with its moves and regrips filled in, sorted by start time."""
    moves: list[PixelMove] = []
    regrips: list[PixelRegrip] = []
    for part in BodyPart:
        mine = sorted(contacts.of(part), key=lambda c: c.t_start)
        for prev, cur in zip(mine, mine[1:]):
            if same_target(prev, cur, contacts.torso_px, rules):
                regrips.append(
                    PixelRegrip(body_part=part, target=PixelTarget.of(prev), t_off=prev.t_end, t_on=cur.t_start)
                )
            else:
                moves.append(
                    PixelMove(
                        body_part=part,
                        from_target=PixelTarget.of(prev),
                        to_target=PixelTarget.of(cur),
                        t_start=prev.t_end,
                        t_end=cur.t_start,
                    )
                )
    order = list(BodyPart)
    moves.sort(key=lambda m: (m.t_start, order.index(m.body_part)))
    regrips.sort(key=lambda r: (r.t_off, order.index(r.body_part)))
    return contacts.model_copy(update={"moves": moves, "regrips": regrips})


def move_at(contacts: VideoContacts, part: BodyPart, t: float) -> PixelMove | None:
    """The move this body part is making at time t, if any."""
    for m in contacts.moves:
        if m.body_part == part and m.t_start <= t < m.t_end:
            return m
    return None


def regrip_at(contacts: VideoContacts, part: BodyPart, t: float) -> PixelRegrip | None:
    """The regrip this body part is in the middle of at time t, if any."""
    for r in contacts.regrips:
        if r.body_part == part and r.t_off <= t < r.t_on:
            return r
    return None
