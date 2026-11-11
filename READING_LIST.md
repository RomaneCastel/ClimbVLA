# ClimbVLA — Reading List

Related work on climbing-hold detection, climber pose analysis and beta modeling.
Compiled 2026-11-11 from web search summaries. The full texts have not been
checked yet: confirm venues, authors and numbers before citing.

Priority: **P1** read first · **P2** read before writing the paper · **P3** background.
Tick the box once read and add a one-line note.

## P1 — Closest to ClimbVLA

- [ ] **The Way Up: a Dataset for Hold Usage Detection in Sport Climbing**
  — Maschek et al., 2025.
  [ResearchGate](https://www.researchgate.net/publication/395627441_The_Way_Up_a_Dataset_for_Hold_Usage_Detection_in_Sport_Climbing)
  - 22 annotated climbing videos: hold positions, order of use, time of use.
    Evaluates 2D pose models for detecting hold use.
  - *Relevance:* almost the same task as Step 2 contact detection. Candidate
    benchmark for evaluating ours. Check the dataset license and access.
  - Notes:

- [ ] **Training-Free Hold-Usage Detection in Sport Climbing with Foundation Pose Models**
  — Bakar et al., arXiv 2609.30026, 2026.
  [arXiv](https://arxiv.org/abs/2609.30026v1)
  - Frozen Sapiens pose model, fingertip and toe keypoints, proximity test
    to holds, one limb per hold, short persistence rule.
  - *Relevance:* very close to our contact approach and recent. Contact
    detection alone is not a new contribution; compare against it.
  - Notes:

- [ ] **Using Machine Learning for move sequence visualization and generation in climbing**
  — Rimbot, Jaggi, Barba (EPFL), 2025.
  [arXiv](https://arxiv.org/abs/2503.00458)
  - MediaPipe landmarks to detect stable hand and foot contacts; Transformer
    models to predict move sequences. The authors call their results inconclusive.
  - *Relevance:* covers both Step 2 (contacts) and Step 6 (sequence prediction).
  - Notes:

## P2 — Hold detection (Step 3)

- [ ] **Towards Automatic Object Detection and Activity Recognition in Indoor Climbing**
  — Vrzáková et al., *Sensors* 24(19):6479, 2024. doi:10.3390/s24196479.
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11479384/)
  - YOLOv5 hold detection combined with eye tracking to recognize grasps.
  - *Relevance:* the most established hold-detection paper found.
  - Notes:

- [ ] **YOLO vs Edge Detection for Climbing-hold Detection** — Stockholm University thesis.
  [PDF](https://su.diva-portal.org/smash/get/diva2:1955778/FULLTEXT01.pdf)
  - YOLO beats classic edge detection.
  - *Relevance:* baseline reference; may list public datasets.
  - Notes:

## P2 — Pose and technique analysis (Steps 2 and 8)

- [ ] **Climbing Technique Evaluation by Means of Skeleton Video Stream Analysis** — 2023.
  [ResearchGate](https://www.researchgate.net/publication/374409476_Climbing_Technique_Evaluation_by_Means_of_Skeleton_Video_Stream_Analysis)
  - Technique evaluation from skeleton data.
  - *Relevance:* closest to our technique metrics. Check which metrics
    they use and how they validate them.
  - Notes:

- [ ] **Climbing with Virtual Mentor by Means of Video-Based Motion Analysis** — 2023.
  [PDF](https://www.scitepress.org/Papers/2023/119593/119593.pdf)
  - 2D skeleton from an iPad, flags typical beginner mistakes.
  - *Relevance:* closest to our coaching goal (Step 10).
  - Notes:

- [ ] **Automated Human Movement Segmentation by Means of Human Pose Estimation in RGB-D Videos for Climbing Motion Analysis**
  — Beltrán et al., 2022.
  [ResearchGate](https://www.researchgate.net/publication/358596722_Automated_Human_Movement_Segmentation_by_Means_of_Human_Pose_Estimation_in_RGB-D_Videos_for_Climbing_Motion_Analysis)
  - Splits climbs into movements from pose data.
  - *Relevance:* compare with our split at contact changes.
  - Notes:

- [ ] **Automated Video Assistant Referee in Lead Climbing** — CISS, 2025.
  [CISS](https://ciss-journal.org/article/view/12047)
  - BlazePose and RTMO on competition footage; low detection rates on
    uncropped video.
  - *Relevance:* evidence for our main risk, pose quality. Motivates a tripod
    with the climber filling the frame.
  - Notes:

## P3 — Datasets and background

- [ ] **CIMI4D: A Large Multimodal Climbing Motion Dataset under Human-scene Interactions** — 2023.
  [arXiv](https://arxiv.org/abs/2303.17948)
  - Multimodal 3D climbing-motion dataset.
  - *Relevance:* extra pose data or 3D references if needed.
  - Notes:

- [ ] **Computer Vision Based Indoor Rock Climbing Analysis** — Ekaireb, UCSD.
  [PDF](https://kastner.ucsd.edu/ryan/wp-content/uploads/sites/5/2022/06/admin/rock-climbing-coach.pdf)
  - Earlier holds-plus-coach system.
  - Notes:

## P3 — Beta and route modeling (Step 6)

- [ ] **Recurrent Neural Network for MoonBoard Climbing Route Classification and Generation**
  — Duh & Chang, 2021.
  [arXiv](https://arxiv.org/abs/2102.01788)
  - BetaMove turns routes into move sequences; DeepRouteSet generates routes,
    GradeNet predicts grades.
  - *Relevance:* BetaMove is a hand-made beta model to compare with our A* planner.
  - Notes:

- [ ] **Board-to-Board: Evaluating Moonboard Grade Prediction Generalization** — 2023.
  [arXiv](https://arxiv.org/abs/2311.12419)
  - Notes:

- [ ] **Strange Beta** — 2011, early route generation.
  [arXiv](https://arxiv.org/abs/1110.0532)
  - Notes:

## Gaps this list suggests

- Contact and hold-use detection from pose is already actively published.
  It can't be the main contribution.
- Less covered: **technique metrics validated against coach ratings**, and
  **comparing the user's moves with a planner's reference**.
- To check while reading: does any paper validate technique metrics against
  expert judgment? If not, that is the paper's angle.
