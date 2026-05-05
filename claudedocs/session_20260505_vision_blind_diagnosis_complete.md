# Phase ST-C v3 진단 완료 — Vision-Blind 이중 결함 확정 (2026-05-05)

## TL;DR

5/05 deploy의 systematic R3 area 직진 + z dive 원인 = **이중 결함**:
1. **Kinematics는 정상** — SDK FK ≠ URDF FK ~125mm offset이 z=-98mm 표시 artifact 만든 것뿐. URDF FK로 보면 모델은 sim 학습 z=+33mm 정확히 출력 중.
2. **그러나 sim demos 설계 결함 + sim-real visual gap이 vision conditioning 학습 차단** → 모델이 image 무관 default trajectory 출력.

## 진단 #1 — Sim ep direct replay (kinematics 검증)

**Script**: [replay_sim_demo_real.py](../replay_sim_demo_real.py) — sim ep 50 (첫 stacking) frame 0~50을 real arm Follower(USB1)에 모델 없이 replay, 3가지 z 동시 측정.

**Output**: [logs/sim_replay_ep50_f50_20260505_160503.csv](../logs/sim_replay_ep50_f50_20260505_160503.csv) + .png

**핵심 수치 (n=51 frames)**:
- Δ(realURDF − simTarget) = **+1.7mm 평균**, abs_max 70.8 → real arm이 sim joint 의도 추종
- Δ(SDK − realURDF) = **−124.7mm 평균**, abs_max 128.8 → 두 FK system origin 차이
- HOME에서도 SDK z=+207 vs URDF z=+328 → 121mm offset 일관

**Verdict**: H1 (sim-real Z gap via kinematics) **기각**. 5/05 deploy의 −98mm는 SDK pose_get 좌표계 차이로 인한 표시 artifact. URDF로 보면 step 50에서 +33mm = sim grasp target 정확.

## Sim demo layout 분석

**Source**: 50 sim_demos_v3/demo_*_layout.json + parquet ep 50~99 frame 15.

**발견**:

| Source | X mean±std (mm) | Y mean±std (mm) | spread |
|---|---|---|---|
| S1 | +194 ± 25 | −178 ± 23 | ±25mm |
| S2 | +198 ± 28 | +138 ± 33 | ±30mm |
| S3 | +382 ± 25 | −150 ± 30 | ±25mm |
| S4 | +385 ± 25 | +119 ± 41 | ±35mm |

**First-grasp distribution across 50 eps**: **S1=50, S2=0, S3=0, S4=0** = procedural script가 **항상 S1부터 grasp** (50/50 fixed).

**의미**: 모델은 "어느 sponge 먼저 잡을지" 결정 학습 X. Sponge 위치 변동도 ±25mm only → vision-based localization 학습 신호 부족.

## 진단 #2 — Vision conditioning probe

**Script**: [vision_conditioning_diagnostic.py](../vision_conditioning_diagnostic.py)

**방법**:
- SmolVLA flow matching = noise 인자 fix 시 deterministic
- σ_det = same image, same noise, 3 forwards (sanity, 0이어야)
- σ_noise = same image, 5 different noises (모델 stochasticity baseline)
- σ_vision = 50 different sim render images, fixed noise (image effect)
- ratio = σ_vision / σ_noise: <1.5=VB, <3=weak, <6=moderate, >=6=STRONG

**Output**:
- v3 5K: [logs/vision_diag_20260505_162218.json](../logs/vision_diag_20260505_162218.json) + .png + .csv
- v6 base: [logs/vision_diag_20260505_162450.json](../logs/vision_diag_20260505_162450.json) + .png + .csv

### v3 5K vs v6 base 결과

| Joint | v3 5K ratio (first / chunk) | v6 base ratio (first / chunk) | Verdict |
|---|---|---|---|
| base | **0.89** / 1.89 | 1.31 / 3.54 | 둘 다 VB on sim render |
| shoulder | 2.60 / 1.50 | 2.09 / 3.09 | weak |
| elbow | 1.65 / 2.14 | 1.85 / 3.51 | weak |
| wrist_p | 1.87 / 1.35 | 2.79 / 4.02 | weak |
| wrist_r | 0.73 / 7.44 | 2.06 / 2.79 | first VB, chunk varied |
| gripper | 1.32 / 0.87 | **0.50** / 3.22 | VB |

**v3 5K first-action distribution across 50 layouts**:
- base mean −1.74°, range [−2.72, −0.61] (모두 거의 동일)
- elbow mean +87.88°, range [+87.30, +88.48] (1.2° spread only)

→ **모델이 image 무관 거의 같은 default action 출력**

**v6 base도 sim render에 vision-blind**:
- v6는 real Kinect 학습 → sim render OOD
- 4/9 deploy 5/5 success는 real Kinect image. **즉 v6 base의 vision conditioning은 real image에서는 작동.**
- 본 sim render 진단은 sim distribution OOD 효과로 약화 측정.

**5/05 real deploy 행동 = real Kinect image에서도 v3 vision-blind**:
- 모델이 real image에서도 sponge 무관 default trajectory 직진 (X+340, Y-32 fixed area)
- → finetune이 v6의 real-image vision capability 손상시킴 (강한 가설)

## VLA Deployment Standard 조사 (참고)

**현재 setup 평가**: per-step closed-loop (n_action_steps=1) = **SmolVLA 공식 비표준**. 공식 권장 = **chunk N=50 + Async + RTC**.

**모델별 표준**:
- π0 (arXiv 2410.24164): chunk N=50 + RTC, 50Hz
- SmolVLA (arXiv 2506.01844): chunk_size=50 default, Async inference 공식 권장
- ACT (2304.13705): N=100 chunk + temporal aggregation
- OpenVLA (2406.09246): per-step (token action, 다른 아키텍처)
- Diffusion Policy (2303.04137): receding horizon T_p=16 T_a=8

**z-dive와 deployment 패턴 관계**: 5K=10K 동일 결과 → **deployment 패턴 차이로 설명 불가**, 진단 #1로 좌표계 artifact 확인됨. 단 stacking sequential task에서는 chunk 실행이 phase 전환에 유리.

**SmolVLA per-step이 flow matching 아키텍처에 비효율**: 10 denoising × 50 actions 계산하고 1개만 사용 (49개 폐기) + 매 호출 noise 독립 → action jitter.

**LeRobot RTC reference**: [lerobot/examples/rtc/eval_with_real_robot.py](../lerobot/examples/rtc/eval_with_real_robot.py) (확인 필요).

## 종합 Root Cause (이중)

🔴 **Cause #1 (확정)**: Sim demo 설계 결함
- 50 ep 모두 first-grasp = S1 fixed
- Source spread ±25mm only
- 4 quadrant fixed → vision-based localization 학습 시그널 부족

🔴 **Cause #2 (강력 의심)**: Sim-real visual gap
- Sim render SigLIP 0.7222 (4/24) = real과 ~70% 유사 only
- v6 base의 real-image vision capability가 finetune 중 sim distribution에 끌려가 손상

## Plan 재평가

| Plan | Cause #1 | Cause #2 | 시간 | 효과 |
|---|---|---|---|---|
| A. Sim diversity↑ (100ep + spread ±60mm + random first-grasp) | ✅ | ❌ | ~3-4h | 부분 |
| B. Real stacking 50ep teleop | ✅ | ✅ | ~6h | 고 |
| C. A+B (real 30 + sim 100 diverse) | ✅✅ | ✅ | ~10h | **최고 권장** |
| D. Domain Randomization | ❌ | ✅ | ~5h | 부족 |
| E. Real 100ep only | ✅ | ✅ | ~10h | sim co-training 이점 포기 |

**강력 권장: Plan C 단계적 실행**

**Step 2.1 (1-2h)**: Sim demos v4 — 100ep + spread ±60mm + random first-grasp source order + EXCLUSION 다양화
- generate_stacking_demos_v3.py 수정 (S1~S4 순서 random shuffle, layout sampling spread 확대)
- 4090 render ~45min + dataset build ~10min

**Step 2.2 (다음 세션, 4-5h)**: Real stacking 30-50ep L-F teleop
- v6 수집 형식 그대로
- 실제 vision-conditioned trajectory

**Step 2.3 (그 다음, ~2h)**: Co-training finetune
- v6 6942fr + sim_v4 100×146fr + real_stacking 30-50ep
- B200 ~85min

**Step 2.4**: ST-C v4 deploy + vision diagnostic 재실행 → 개선 측정

## 작성한 코드 / 산출물

| 파일 | 용도 |
|---|---|
| [replay_sim_demo_real.py](../replay_sim_demo_real.py) | 진단 #1 — sim ep direct replay on real arm, 3-FK 비교 |
| [vision_conditioning_diagnostic.py](../vision_conditioning_diagnostic.py) | 진단 #2 — σ_vision/σ_noise ratio probe |
| logs/sim_replay_ep50_f50_20260505_160503.csv/.png | 진단 #1 결과 |
| logs/vision_diag_20260505_162218.json/.csv/.png | v3 5K vision probe |
| logs/vision_diag_20260505_162450.json/.csv/.png | v6 base vision probe |

## HARD RULE 준수

- #11 `/half-clone` 거부 1회 (Stop hook 101%) — 세션 종료 프로세스로 대체.
- #16 train_config source-of-truth (Follower=USB1) 준수.
- #19/#20 edge-stand / # tower geometry 보존.
- #4 연구 검증 — VLA standard agent 다양 출처 verification.

## 다음 세션 진입 권장

**즉시 가능 (1-2h, 4090 only)**: Step 2.1 — generate_stacking_demos_v4.py 작성
- Sources random ordering (numpy.random.shuffle)
- Layout spread 확대 (±60mm, EXCLUSION zone reformulation)
- 100ep 생성 (vs 50ep)
- Render + dataset 구축

**또는 사용자 confirm 후 (4-5h, 로봇 작업)**: Step 2.2 — Real stacking teleop 직진
