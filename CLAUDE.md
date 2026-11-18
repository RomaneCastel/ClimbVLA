# ClimbVLA

The plan is in `PLAN.md`; work goes step by step and sub-step by sub-step
(1, 2.1, 2.2, ...), with an explanation of each new concept along the way.

## Decisions that change the plan

When a decision affects the rest of the plan (later steps, the design
decisions in `PLAN.md` §1, the data format, the order of steps), say so
explicitly when it comes up: what changes in the plan and why. Update
`PLAN.md` only after Romane agrees, then keep its step list (with ✅ for
finished sub-steps) in line with the work.

## After each finished step: development notes

When a step or sub-step of the plan is finished (its code works and its tests
pass), run the `dev-notes` agent before moving on to the next one. Pass it:

- the step id and title (e.g. "2.6, clicking the holds");
- what was built, the decisions and why, the alternatives that were rejected;
- the bugs and surprises, with their cause and fix;
- any change of plan or step back, and why;
- the numbers measured (speed, accuracy, sizes, test count);
- the full paths of images or videos made during the step, and who is visible in them.

The agent writes one entry into the "Development notes" tab of the ClimbVLA Dev
Notes doc and adds a row to its progress log. Those notes are the raw material
for blog posts, so give it the reasoning, not just the result.
