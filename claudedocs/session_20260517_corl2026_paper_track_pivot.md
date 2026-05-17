# Session 2026-05-17 — CoRL 2025 전수조사 + CoRL 2026 Paper Track Pivot

## Boot
- Followed `CLAUDE.md` Current-State Protocol.
- Pre-read: `START_HERE.md`, `claudedocs/DECISIONS.md` (D001-D021),
  `claudedocs/EXPERIMENT_LEDGER.md`, recent `session_2026051*.md`.
- `git status --short` confirmed working tree: `M START_HERE.md`,
  `M claudedocs/DECISIONS.md`, `M claudedocs/EXPERIMENT_LEDGER.md`,
  `M roarm_rl/roarm_stack_env.py`, `M roarm_rl/train_ppo.py`,
  `?? claudedocs/session_20260517_p7_release_guidance_diagnostics.md`.
- Last commit: `687bf20 5월 17일-1(p7)`.

## User Intent
- CoRL 2026 마감 임박 인지. Two-track 전략 명시:
  - Track A = 교수님 sim/lab 연구 (4-sponge # tower stacking — 현재 P7 진행 중)
  - Track B = CoRL 2026 노벨 paper (우리만의 contribution)
- 1차 요청: CoRL 2025 전수조사 + 우리 작업 교차검증 + 비판적/분석적/skeptical 브리핑
- 2차 요청 (정정): 보수적 negative-result 권고 거부. "달리면 됨, 24h/day 투입 의지" — novel
  real-to-sim pipeline develop 요구. HARD RULE #10 ("시간 제약으로 아이디어 축소 금지").

## Method
- 5개 병렬 general-purpose agent 발동, HARD RULE #4 (≥10 검색어 × ≥2 소스) 강제:
  1. CoRL 2025 VLA / foundation model
  2. CoRL 2025 manipulation / stacking / long-horizon
  3. CoRL 2025 sim-to-real / real-to-sim / Isaac Lab
  4. CoRL 2025 contact-rich / grasp / low-cost / data-efficient
  5. CoRL 2026 deadline + submission timeline 검증

## CoRL 2025 Survey Result (verified)

### Source list
- **PMLR v305** = https://proceedings.mlr.press/v305/
- **OpenReview** = `robot-learning.org/CoRL/2025/Conference`
- **Total accepted = 263** (oral 42 + poster 221). Seoul 9/27-9/30. HIGH confidence.
- Curated mirrors: `smallfryy/corl-2025-papers` (221 entries), `shu1ong/CORL2025-Paper-List`,
  Paper Copilot, papers.cool, ML Anthology.

### Cross-verified Gaps (HARD RULE #4 compliant)

| # | Gap | Confidence | 발견자 |
|---|---|---|---|
| G1 | Isaac Lab `_update_grasp_attach` kinematic pose-write vs physics-grasp gap을 explicit하게 진단/벤치마크한 CoRL 2025 main paper 없음. 가장 가까운 = Real2Render2Real (Oral, arxiv 2505.09601)이 가정 acknowledge만, quantify 안 함. | HIGH | VLA + Sim2Real + Contact agent 3중 확인 |
| G2 | "When-to-open-gripper" release primitive head를 explicit하게 학습한 CoRL 2025 paper 없음. AnyPlace (2502.04531)는 pose만 예측. | HIGH | Contact + Manipulation 2중 |
| G3 | Attached transport 중 object orientation을 reward/constraint로 보존하는 CoRL 2025 paper 없음. SPIN (2502.18015) "Connectors"가 가장 가까운 framing이지만 attached-grasp orientation 자체 invariance 아님. | HIGH | Contact agent |
| G4 | <$1k 팔 (SO-100/SO-101/Koch/RoArm/AnkerArm) primary platform 사용 CoRL 2025 main paper 0건/221편. | HIGH | VLA + Contact 2중 |
| G5 | Vertical edge-stand / unstable pose object manipulation paper 없음. | HIGH | Contact agent |
| G6 | # / 우물정자 / 2-layer cross-pattern rigid stacking paper 없음. 가장 가까운 = Stack It Up! (2508.02093, Oral). | HIGH | Manipulation agent |
| G7 | Pure VLA / Pure RL / Hybrid 3-way controlled comparison CoRL 2025 paper 없음. DSRL (2506.15799 Oral)이 2-way까지. | MEDIUM | Sim2Real agent |
| G8 | B200/H100/L40s Isaac Lab Vulkan ICD / Annotator block 문제 paper 없음. NVIDIA Discussion #4339 공식: "NOT supported, decoupling rewrite 후 가능". | HIGH | Sim2Real agent |

### Critical Related Work (paper에 반드시 들어가야)
- SPIN (2502.18015) — Skill-RRT + Connectors. P7 attached transport 최강 선행
- AnyPlace (2502.04531) — placement pose
- ARCH (2409.16451) — hierarchical hybrid scripted/learned
- Stack It Up! (2508.02093, Oral) — block stacking from sketches
- DSRL (2506.15799, Oral) — BC frozen + latent-space RL
- X-Sim (2505.07096, Oral) — Real-to-Sim-to-Real
- Real2Render2Real (2505.09601, Oral) — kinematic-attach 가정 acknowledge
- VT-Refine (2510.14930) — real demos → digital twin → RL
- Human2Sim2Robot (2504.12609) — 1 demo → IsaacGym RL
- Long-VLA (2508.19958), FLOWER (2509.04996) — VLA comparators

### CoRL 2026 deadline (verified, HIGH confidence)
- **Full paper = 2026-05-28 AoE ≈ 5-29 11:59 UTC**. corl.org body text "Thursday (5/28) EOD" +
  HF ai-deadlines `May 29 11:59 UTC` (same instant in two TZ). 11 days from 5/17.
- **Abstract ≈ 5-25 AoE / 5-26 UTC**. 8-9 days from 5/17.
- **8 pages + Limitations 필수**, LaTeX only, double-blind, OpenReview submission.
- **CoRL 2026 본 conference**: Austin TX, Nov 9-12 2026. JW Marriott Austin. 10th edition.
- **REMAINING UNCERTAINTY**: corl.org/contributions/call-for-papers JS-rendered table 직접
  확인 필요 (WebFetch 못 잡음). 사용자 5분 확인 작업.

## Novel Real-to-Sim Pipeline Candidates (Track B)

### 5 design candidates (gap × asset matrix)
| ID | 이름 | Hit gaps | 우리 자산 fit | Novelty | 11-day 실현가능성 |
|---|---|---|---|---|---|
| ① | Attach-Aware Real-to-Sim Calibration | G1, G3 | HIGH (P7 quat probe + Hand-eye calib) | HIGH | MED (F/T 부재) |
| ② | **Failure-Driven Bidirectional Sim-Real Loop** | G1, G3, G6, G7 | VERY HIGH | VERY HIGH | HIGH |
| ③ | Edge-Stand # Stacking Benchmark + Diagnostic | G5, G6, G4 | HIGH | MED-HIGH | HIGH |
| ④ | Skill-Boundary Real-to-Sim Distillation | G3, G7 | MED (D012 mismatch evidence) | HIGH | LOW (real PPO 필요) |
| ⑤ | Sim-Demo Silent-Failure Audit Framework | G1, G3, G8 | HIGH | MED | MED-HIGH |

### 추천: ② Failure-Driven Bidirectional Sim-Real Loop
**가칭**: *"Failure-Driven Bidirectional Real-to-Sim Loop: Using Sim Attach Failure
Patterns to Guide Real Data Collection for Low-Cost Stacking"*

**Contributions (paper sections)**:
1. Sim attach failure mode taxonomy (P7 256/256 collapse classification 활용)
2. Failure-to-real-collection signal extraction algorithm
3. Real-to-sim calibration with attach-aware demos
4. 2-iteration loop result on # tower stacking task

**Differentiation from CoRL 2025 baselines**:
- X-Sim / R2R2R / Human2Sim2Robot 모두 single-pass real → sim → real
- Ours = **bidirectional + sim failure as real-collection active signal + 2-iteration closed loop**
- 명확한 algorithmic 차이

**Limitations (필수)**:
- F/T sensor 없음 → servo current proxy
- Single arm (RoArm-M3) → "instance study" framing
- 2-iteration only, no asymptotic claim
- 4-source long-attached transport 미해결 (D014 그대로 인정)

## 11-Day Timeline (24h/day mode)

| Day | Date | Phase | Action |
|---|---|---|---|
| D-11 | 5/17 today | Decision + Setup | Q1-Q6 answer 받기, git branch `paper_v1`, B200 폴더 `roarm_b200/paper_v1/` fork |
| D-10 | 5/18 | Setup | Failure taxonomy formalize, paper outline LaTeX skeleton |
| D-9 | 5/19 | Split | HARD RULE #26 release. Track A/B 코드 split. md5 freeze. |
| D-8 | 5/20 | Setup | Failure-to-real signal extraction algorithm prototype |
| D-7 | 5/21 | Real Collection | 25 ep stacking 수집 (HARD RULE #1/#13/#19/#24 준수) |
| D-6 | 5/22 | Real Collection | 25 ep stacking 수집 (피로 drift 방지 위해 분산) |
| D-5 | 5/23 | Calibration | v3 변환 + attach calibration iter-1 |
| D-4 | 5/24 | BC iter-1 | B200 학습 1.5h + offline 평가 |
| D-3 | 5/25 | Iter-2 (옵션) | Failure mode 추출 + gap-driven 5-10 ep 추가 |
| D-2 | 5/26 | Real Deploy | BC iter-2 학습 + real deploy 5회 |
| D-1 | 5/27 | Paper Write | 8 page 집중 작성 |
| D-day | 5/28 | Submit | LaTeX 검토, anon 확인, OpenReview submit |

**Critical path**: D-7-D-6 real 수집 → D-5-D-4 calibration → D-2 real deploy.
**Slack**: D-3 iter-2 skip 가능 (1-iter result만으로도 paper 됨).

## Risks Identified
- **R1**: Real stacking 수집 중 chain stall (D008 +51.8mm equilibrium) 재발 →
  G2-A 충돌 프록시 적용된 수집 코드로 fallback
- **R2**: BC iter-1이 모든 ep fail → "negative result diagnostic" reframe (fallback)
- **R3**: 사용자 24h 모드 burnout. Track A 변경 동시 금지 필수
- **R4**: HARD RULE #11 (`/half-clone` 금지) — 89% context stop hook 거부 후 본 doc로
  정상 protocol 진행

## Pending User Decisions (Q1-Q6)
1. **Q1**: Paper topic = 후보 ② 확정?
2. **Q2**: Real stacking 50ep 수집 D-7 (5/21) 시작 OK?
3. **Q3**: 5/19 paper용 git branch + B200 폴더 fork OK?
4. **Q4**: corl.org abstract/paper 정확한 시각 직접 확인 (5분)
5. **Q5**: OpenReview 42편 residual scan 별도 agent 발동? (2-3h, "first" 안전성)
6. **Q6**: Track A 진행 (Branch A/B) 결정 시점 5/19 OK? P7 추가 변경 보류 OK?

## Additional Verification Recommended (HARD RULE #4)
- Failure-driven active learning 분야 (UCB/MaxEnt/Disagreement) 우리만의 novelty 검증
- RSS 2025 / NeurIPS 2025 / ICRA 2025 real-to-sim 빠른 검토
  - Sim-and-Real Co-Training (RSS 2025, arxiv 2503.24361, +37.9%) 누락 위험
  - DexMimicGen (ICRA 2025, arxiv 2410.24185) 누락 위험
  - Compliant Residual DAgger (NeurIPS 2025, arxiv 2506.16685) 누락 위험

## HARD RULE Compliance
- #1 HOME 시작 = real 수집 시 강제
- #4 ≥10 검색어 × ≥2 소스 = 모든 agent에 명시 강제. 8 gap claim 모두 compliant
- #8 MEMORY recent sessions = 본 세션 prepend 예정
- #10 시간 제약으로 아이디어 축소 금지 = 보수적 1차 브리핑 정정, novel pipeline 권고로 전환
- #11 `/half-clone` 거부 = stop hook 89% context 거부 후 본 doc로 정상 protocol
- #13 USB0 leader / USB1 follower = D-7 수집 시 강제
- #18 사용자 명시 정정 우선 = "달리면 됨" 명시 → novel pipeline 권고로 전환
- #19 edge-stand 47mm = D-7 수집 시 강제
- #24 일일 ≤50ep = D-7~D-6 25 ep × 2일 분산
