# ClimbVLA — Project Plan

ClimbVLA compares how a person climbs a boulder problem with how a simulated
reference climber would climb the same route, and scores the difference
(the **Beta Score**).

This plan replaces the original specification. Technology versions were chosen
from memory, not a fresh search — check current versions at install time.

## 1. Design decisions

| Decision | What it means | Why |
|---|---|---|
| Wall-plane 2D | Holds, pose and simulation all live in one 2D frame, in meters, in the plane of the wall | A phone on a tripod facing the wall sees roughly that plane; depth from one camera is unreliable |
| Scale from climber height | The user enters their height; the skeleton in the first frames sets pixels-per-meter | No calibration target needed |
| Planner + single-move controller | A search planner picks the limb-to-hold sequence; a learned policy executes one move at a time | One policy learning whole routes from torques is a research problem; single moves are learnable |
| Holds as attachment points | A hand or foot attaches when it is within a small radius of a hold | Avoids simulating finger friction |
| Move-by-move comparison | Both climbs are cut into moves at contact changes, then compared inside each move | The user and the reference never move at the same speed |
| Static, roughly frontal camera | One climber, tripod, whole route in frame | Keeps pose-to-wall mapping a single fixed transform |

Two parts are kept deliberately as goals in themselves:

- **C++ module** (`climb_ik`, pybind11) — reachability and inverse kinematics
  for the planner.
- **VLM fine-tuning** — hold-type classification from crops, with LoRA.

## 2. Pipeline

```
video ──► first frame ──► hold detector ──► color grouping ──► route holds
  │                                              │
  │                                  VLM (fine-tuned): hold types
  │                                              │
  └──► pose model ──► wall-plane skeleton ──► contacts ──► user moves + metrics
                                                 │
                         route holds ──► beta planner (uses climb_ik)
                                                 │
                                     reference sequence of moves
                                                 │
                              IK baseline ──► PPO single-move policy (MuJoCo)
                                                 │
                                        reference trajectory
                                                 │
                    user moves + reference ──► scorer ──► Beta Score
                                                 │
                                   VLM: coaching text ──► web app
```

## 3. Steps

### Step 1 — Shared data format
Pydantic models for `Hold`, `Route`, `PoseFrame`, `ContactEvent`, `Move`,
`Trajectory` and `Audit`. Every stage reads and writes these, as JSON on disk.

*Done when:* a hand-written example route and trajectory validate and round-trip.

### Step 2 — Climber tracking
- Pose estimation per frame, then smoothing (One-Euro or Savitzky–Golay filter).
- Convert pixels to wall-plane meters.
- Contact detection: a wrist or ankle is "on" a hold when it stays within a
  radius of it at near-zero speed for several frames.
- Metrics:
  - center-of-mass path and its length relative to the direct path;
  - **straight-arm ratio** — share of loaded frames with elbows near 180°;
  - **lock-off time** — time spent holding a bent elbow under load;
  - **balance margin** — horizontal distance of the center of mass from the
    line between the current supports;
  - smoothness (jerk) and hesitation time per move.

*Done when:* a real video, with hand-clicked holds, produces a move list and
metrics that look right on the overlay.

### Step 3 — Wall reading
- Fine-tune a segmentation detector on public climbing-hold datasets plus a
  small set of own photos.
- Group detected holds into routes by color (clustering on mask pixels in Lab
  color space).
- The user confirms or corrects the route in the UI — detection will not be
  perfect, and one wrong hold breaks everything downstream.

*Done when:* hold detection reaches a usable mAP on held-out walls and the
route for a test video needs at most a couple of manual fixes.

### Step 4 — VLM fine-tuning: hold types
- **Task:** given a crop of one hold, output JSON
  `{"type": "crimp|sloper|jug|pinch|volume|foothold", "confidence": ...}`.
- **Data:** crops come from the step 3 detector, so 100–200 wall photos yield
  thousands of crops. Label a few hundred to start.
- **Method:** LoRA on a small Qwen3-VL. Locally with MLX on the Mac, or with
  `peft` + 4-bit quantization on a cloud GPU.
- **Baselines, so the fine-tune means something:** the same VLM zero-shot, and
  a DINOv2 linear probe.
- **Report:** accuracy and macro-F1 on held-out walls, tracked in W&B.

*Done when:* the fine-tuned model beats zero-shot clearly, and the comparison
with the linear probe is written down honestly whichever way it goes.

### Step 5 — C++ kinematics module (`climb_ik`)
- Analytic two-link inverse kinematics for arms and legs, with joint limits.
- Reachability: given the attached limbs, can limb *L* reach hold *H*? Returns
  the feasible hip/shoulder region or "unreachable".
- Batched queries, since the planner asks thousands per route.
- Built with pybind11, CMake and scikit-build-core; Eigen for the math.
- Tested against a NumPy reference implementation, with a benchmark.

*Done when:* results match the NumPy reference and the speed-up is measured.

### Step 6 — Beta planner
- State: which hold each limb is on. Action: move one limb to a new hold.
- Search (A*) with feasibility from `climb_ik` and a cost built from reach
  distance, balance margin and hold type (from step 4).
- Output: the reference sequence of moves.
- First feedback product: compare the user's sequence with the planner's.

*Done when:* the planner produces a sensible sequence on several real routes.

### Step 7 — Reference motion
- **7a. Kinematic baseline.** Interpolate between planned stances with inverse
  kinematics. Guaranteed to exist; good enough to build the scorer and the UI.
- **7b. Learned controller.** MuJoCo planar climber (torso plus four two-segment
  limbs: 9 bodies, 8 actuated joints). PPO trained on *single moves* with
  randomized hold positions, wall tilt and limb lengths. Reward: progress
  toward the target hold, minus effort, minus center-of-mass drift, bonus on
  reach, penalty on fall. Chain the policy along the planned sequence.

*Done when:* 7a works on every planned route; 7b completes most single moves in
evaluation and can be chained over at least a short route.

### Step 8 — Scoring
- Match user moves to reference moves by hold and limb.
- Time-warp (DTW) inside each matched move, then compare joint angles and
  center-of-mass path.
- Beta Score = weighted mix of sequence agreement, metric gaps and motion
  deviation, with per-move breakdown.

*Done when:* a deliberately sloppy climb scores visibly worse than a clean one
on the same route.

### Step 9 — App
- Backend: upload endpoint, background job, progress over WebSocket, results as
  JSON plus the overlay data.
- Frontend: video with skeleton overlay next to the simulated climber, one
  shared scrubber, per-move timeline and metric charts, route correction UI.

### Step 10 — Coaching text
The VLM turns the numeric audit into a short written review per move.

## 4. Research track

The steps above build the product. The research track turns parts of it into
careful experiments, released artifacts and a written report, aimed at
research engineer roles. Each item extends a step above; none adds a new
pipeline stage.

Rules for the whole track:

- Fix held-out splits **before** running experiments, and split by gym or
  wall, never by frame or crop.
- At least 3 seeds per configuration; report mean and spread.
- Every number in a report traces back to a W&B run and a config in the repo.
- Report negative results and failed runs, not only the best one.

### R1 — Hold-type study (extends step 4)
- Compare zero-shot VLM, DINOv2 linear probe, and LoRA at several ranks
  (e.g. 4, 16, 64).
- Data-scaling curve: 50, 100, 200, 500 labeled crops.
- Per-class results and a confusion matrix; note which types humans also
  disagree on.

*Done when:* a results table and a scaling plot with error bars exist, measured
on unseen walls.

### R2 — Tracking ablations (extends step 2)
- Pose model: ViTPose++ vs RTMPose on climbing footage, scored (PCK) against a
  small hand-annotated keypoint set.
- Smoothing: none vs One-Euro vs Savitzky–Golay, measured by its effect on
  contact detection.
- Contact detection precision and recall against hand-labeled contacts.

*Done when:* the chosen pose model, filter and contact thresholds are each
justified by a measured comparison.

### R3 — Dataset release
- Hold crops with type labels, and annotated videos with contacts and moves.
- Fixed train/val/test splits, a dataset card (sources, licenses, labeling
  protocol, known biases) and the baseline tables from R1 and R2.
- Only images whose source licenses allow redistribution; no identifiable
  people without their consent.

*Done when:* published on Hugging Face and loadable with `datasets`.

### R4 — RL study (extends step 7b)
- Log every training run, including the ones that fail.
- Ablate reward terms, curriculum and domain-randomization ranges.
- Success rate per move type, and generalization to unseen hold layouts and
  limb lengths.

*Done when:* a write-up explains what made the single-move policy converge,
with training curves over several seeds.

### R5 — Beta Score validation (extends step 8)
- A few experienced climbers or coaches rate or rank a set of climbs on the
  same routes, without seeing the Beta Score.
- Measure rank correlation (Spearman or Kendall) between the Beta Score and
  the human ratings, with a confidence interval.
- Use agreement between the human raters as the ceiling, and ablate the score's
  weights against it.

*Done when:* the correlation is reported next to inter-rater agreement.

### R6 — Paper reproduction
- Reproduce one published result used by the project, e.g. ViTPose accuracy on
  COCO val, or a standard Stable-Baselines3 PPO MuJoCo benchmark.

*Done when:* the reproduced numbers match the published ones within their
reported tolerance, or the gap is explained.

### R7 — Technical report
- 6–10 pages: method, experiments R1–R5, limitations.
- Posted on arXiv; submitted to a relevant workshop (e.g. a computer vision in
  sports workshop at CVPR or ICCV).

*Done when:* submitted.

Alongside the track: contribute fixes or examples upstream to the libraries
the project uses (`transformers`, `peft`, Stable-Baselines3, MuJoCo, `rtmlib`)
when the project runs into their gaps.

## 5. Technology

| Area | Choice | Notes |
|---|---|---|
| Language / packaging | Python 3.12, `uv` | |
| Data contracts | Pydantic v2 | |
| Pose | ViTPose++ via Hugging Face `transformers` | Fallback: RTMPose via `rtmlib` |
| Hold detection | RF-DETR segmentation | Apache 2.0. YOLO11-seg is an alternative but AGPL-licensed |
| VLM | Qwen3-VL, small size, LoRA | MLX locally, or `peft` on a cloud GPU |
| Classifier baseline | DINOv2 linear probe | |
| Physics | MuJoCo | |
| RL | Stable-Baselines3 PPO, Gymnasium | Vectorized CPU environments |
| C++ | C++17, Eigen, pybind11, scikit-build-core | |
| Experiment tracking | Weights & Biases | |
| Backend | FastAPI, in-process job runner, local disk | Add a queue only if needed |
| Frontend | React, TypeScript, Vite, react-three-fiber, Tailwind, Recharts | Browser replays recorded trajectories; no physics in the browser |
| Packaging | Docker Compose | Last, once the pipeline is stable |

## 6. Order and rough timing

| Block | Steps | Research track | Rough time | Result |
|---|---|---|---|---|
| A | 1–3 | R2 | 2–3 weeks | Video in, moves and metrics out, holds detected |
| B | 4–5 | R1, R3 | 2–3 weeks | Hold-type study; dataset released; tested C++ module |
| C | 6–8 | R4, R5 | 4–6 weeks | Planner, reference motion, validated Beta Score |
| D | 9–10 | R7 | 2–3 weeks | Web app, coaching text, technical report |

Steps 4 and 5 are independent of each other and of step 2, so they can be
moved earlier or done in parallel. R6 can be done at any point. The research
track roughly adds a week per block.

## 7. Main risks

- **Pose quality on climbing footage.** Back-facing, stretched, self-occluded
  poses are hard. Check on real videos in step 2 before building on top.
- **Single-move policy not converging.** Step 7a exists so the rest of the
  project never waits on RL.
- **"Optimal" is a strong word.** The reference is the best sequence under the
  planner's cost model, not ground truth. The UI should say "reference".
- **Body differences.** The simulated climber must be scaled to the user's
  limb lengths, or reach-limited moves will be judged unfairly.
- **Research track crowding out the product.** Ablations and seeds multiply
  compute and time. Keep each R item scoped to what its "Done when" asks.
- **Few human raters for R5.** Line up climbers or coaches early; a small but
  consistent panel beats a large noisy one.
