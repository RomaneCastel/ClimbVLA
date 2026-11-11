# ClimbVLA

**Film yourself on a boulder problem. See how your climb compares with a
reference climb of the same route, move by move.**

ClimbVLA takes a phone video of someone climbing an indoor boulder problem.
It reads the route from the wall, follows the climber's body through the
video, and builds a simulated reference climber that climbs the same route.
It then compares the two and reports a **Beta Score**, with a breakdown for
each move.

> Early stage: the shared data format is being built. Nothing runs end to end yet.

## The idea

In climbing, *beta* is the way you climb a route: which hold each hand and foot
goes to, in what order, and how you move your body between them. Two people
can climb the same route very differently, and some ways cost much more
strength than others.

Climbers usually improve their beta by watching others or by asking a coach.
ClimbVLA tries to give similar feedback from a single video:

- **Which holds you used, and in what order**, compared with a reference
  sequence planned for the route.
- **How you moved**, measured from your skeleton: straight or bent arms, how
  long you held a lock-off, how far your center of mass drifted from your
  supports, how smooth or hesitant each move was.
- **Where you lost the most**, as a per-move breakdown and short written
  feedback.

## How it works

1. **Read the wall.** A detector finds the holds in the first frame and groups
   them into routes by color. The user confirms the route, since one wrong hold
   would spoil everything after it. A fine-tuned vision-language model labels
   each hold's type (jug, crimp, sloper, pinch...).
2. **Follow the climber.** A pose model finds the climber's joints in every
   frame. They are placed on the wall in meters, using the climber's height as
   the scale. A hand or foot that stays still near a hold counts as being on it.
3. **Build a reference.** A planner searches for a sensible sequence of moves
   for the route, using reach (computed by a small C++ kinematics module),
   balance and hold types. A simulated climber, scaled to the user's body,
   then performs it.
4. **Compare.** Both climbs are cut into moves at each change of contact and
   compared move by move, so climbing slower or faster is not penalized by
   itself.
5. **Show it.** A web app plays the video with the skeleton overlaid, next to
   the simulated climber, with a timeline of moves and the metrics.

## What it is not

- **Not "the optimal beta".** The reference is the best sequence *under the
  planner's model* of climbing, not the truth. A different, valid beta can
  score lower. The app calls it a reference for that reason.
- **Not 3D.** One camera sees the wall as a flat picture, so anything
  happening in depth is invisible: hips close to the wall, drop-knees, body
  twist.
- **Not for every video.** It expects a phone on a tripod, roughly facing the
  wall, with one climber and the whole route in frame.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)

```bash
uv sync          # create the environment and install dependencies
uv run pytest    # run the tests
```

## Project layout

```
src/climbvla/   the Python package
tests/          tests (pytest)
examples/       small hand-written example files
PLAN.md         design decisions and development plan
READING_LIST.md related papers
```
