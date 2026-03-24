# HANDOFF — RoArm-M3 Research Project

> Written: 2026-03-23 (session 2) | Previous: 2026-03-23 (session 1), 2026-03-19

---

## Goal

**CoRL 2026 논문 제출 (마감 2026-05-28, D-66)**

Working title: "Data-Efficient VLA Adaptation on Consumer Hardware"

포지셔닝 (수정됨):
- ~~"Consumer hardware VLA 최초"~~ → **"OOD embodiment에서 데이터 효율성 frontier 정량화"**
- 이유: arXiv 2512.11921이 "consumer hardware VLA" 선행
- 차별점: 그들은 LoRA/quantization, 우리는 **scaling laws + data quality**

4가지 기여:
1. OOD Scaling Laws: episodes(10-150) × quality × steps → 성공률 곡선
2. Data Quality Methodology: 7단계 검증, FK depth, gripper phase, static frame
3. Multi-Object Transfer: 4물체(sponge/cup/box/tool) × 50ep
4. Self-Improving Loop (Seed2Scale-lite): 배포→VLM 판별→성공 rollout 재활용

보험: IROS 2026 LBR (마감 7/31)

---

## Current Progress

### 이전 세션 (2026-03-23 session 1)에서 완료

- **collect_data_manual.py**: `--object` 파라미터 추가 (sponge/cup/box/tool)
- **convert_to_lerobot_v3.py**: `--multi-object` 플래그 추가 (에피소드 metadata에서 물체명 자동 읽기)
- **Agent-Team 12개 완성**: `.claude/agents/` 에 12개 파일 (3 workers + 9 research)
- **March 2026 VLA 논문 60+편 조사**: `claudedocs/MARCH_2026_LANDSCAPE_UPDATE.md`
- **Projector-VLA, WiFi CSI 재검토 → 기각 강화**
- **경쟁 분석 완료**: arXiv 2512.11921 확인, ICLR 2025 Data Scaling Laws 확인

### 이번 세션 (2026-03-23 session 2)에서 완료

- **experiment_matrix.py 작성**: Scaling law 실험 자동화
  - `prepare`: 원본 데이터셋 복사 → `delete_episodes()`로 서브샘플링
  - `train`: 서브셋별 200K run (체크포인트 10K 간격)
  - `eval`: 모든 체크포인트 오프라인 L2/diversity 평가
  - `summary`: 에피소드 수 × 스텝 테이블 출력
- **eval_deployment.py 작성**: 배포 성공률 체계적 측정
  - `deploy_smolvla.py` subprocess 호출 → CSV 로그 + 사람 판정
  - `run-matrix`: experiment_matrix 전체 체크포인트 자동 순회
  - failure mode 분류 (gripper_fail/drift/miss/ood/collision)
- **메모리 업데이트**: 포지셔닝 수정, ProbeFlow, 연속학습 저항성 기록

### 이전 성과 (참고)

- SmolVLA + RoArm-M3 스펀지 pick **100% 성공** (74ep, open-loop 4-chunk, 50K steps)
- Isaac Lab 설치 완료 (conda env isaaclab, URDF→USD 변환 성공)
- 데이터 품질 도구: data_episode_quality.py, data_distribution_simple.py

---

## Key Files

### 핵심 파이프라인
| 파일 | 역할 | 상태 |
|------|------|------|
| `collect_data_manual.py` | 데이터 수집 (Azure Kinect + 토크 OFF) | ✅ --object 추가 |
| `convert_to_lerobot_v3.py` | LeRobot v3 포맷 변환 | ✅ --multi-object 추가 |
| `run_official_train.py` | lerobot-train CLI 래퍼 | ✅ v4 설정 (200K steps) |
| `deploy_smolvla.py` | 실시간 로봇 배포 | ✅ open-loop 4-chunk |

### 실험 인프라 (NEW)
| 파일 | 역할 | 상태 |
|------|------|------|
| `experiment_matrix.py` | Scaling law 실험 자동화 | ✅ 작성 완료, 미실행 |
| `eval_deployment.py` | 배포 성공률 측정 | ✅ 작성 완료, 미실행 |

### 문서
| 파일 | 내용 |
|------|------|
| `CLAUDE.md` | 프로젝트 규칙 + agent team |
| `claudedocs/MARCH_2026_LANDSCAPE_UPDATE.md` | 60+ VLA 논문 경쟁 분석 |
| `claudedocs/AGENT_PERSONAS.md` | 9개 에이전트 페르소나 |
| `claudedocs/PROJECTOR_VLA_ANALYSIS.md` | Projector-VLA 기각 분석 |
| `claudedocs/DATA_COLLECTION_STRATEGY.md` | 데이터 수집 전략 |

---

## What Worked

1. **experiment_matrix.py 설계**: LeRobot 내장 `delete_episodes()` 활용 → 데이터 무결성 보장
2. **eval_deployment.py 설계**: deploy_smolvla.py 재활용 → 코드 중복 없음
3. **포지셔닝 수정**: "최초" → "frontier 정량화"로 안전한 전환

## What Didn't Work / 주의사항

1. **"Scaling law" 용어 주의**: ICLR 2025 oral (40,000 demos)과 충돌. "scaling curve" 또는 "data efficiency frontier" 사용
2. **CoRL 4-contribution은 과도**: 수락 확률 5% 미만 (이전 세션 분석). 범위 축소 검토
3. **experiment_matrix.py 아직 미검증**: delete_episodes() API 호환성 확인 필요
4. **백그라운드 에이전트 4개 결과 미확인** (이전 세션의 /tmp 경로 → 만료 가능)

---

## Next Steps

### 즉시 (Step 2 잔여)
1. **물체 확보**: cup, box, tool (실제 물체 준비)
2. **5-zone 배치 확정**: LEFT_FAR/LEFT/CENTER/RIGHT/RIGHT_FAR 좌표
3. **sponge 추가 수집**: 74 → 100+ 에피소드
4. **experiment_matrix.py 검증**: `prepare --source lerobot_dataset_v3 --episodes 10,25` 테스트

### Step 3: Scaling 실험 (D-56~D-46)
5. **서브셋 생성**: `python experiment_matrix.py prepare --source lerobot_dataset_v3 --episodes 10,25,50,74`
6. **학습 실행**: `python experiment_matrix.py train-all` (4 runs × 200K steps ≈ 44시간)
7. **오프라인 평가**: `python experiment_matrix.py eval-all`

### Step 4: 배포 평가 (D-46~D-38)
8. **배포 테스트**: `python eval_deployment.py run-matrix --steps 50000,100000,200000 --trials 5`
9. **결과 분석**: experiments/results.csv + eval_results/summary.csv

### Step 5+: Self-improving loop, Multi-task transfer, 논문 작성

---

## 70일 타임라인

```
D-70~D-68: Agent personas + 연구 계획 ✅
D-68~D-66: 코드 준비 (--object, --multi-object, experiment_matrix, eval_deployment) ✅
D-66~D-56: 데이터 수집 (물체 확보 + sponge 추가 + multi-object) ← NEXT
D-56~D-46: Scaling 실험 매트릭스 (학습 + 오프라인 평가)
D-46~D-38: 배포 평가 (real robot trials)
D-38~D-30: Self-improving loop
D-30~D-24: Multi-task transfer
D-24~D-10: 논문 작성
D-10~D-0:  제출 (5/28)
```

---

## Hardware Summary

```
보유:
├── RoArm-M3-Pro × 3 (6-DOF, ~$200/ea)
├── Azure Kinect DK × 3 (RGB+Depth+IMU+7-mic array)
├── RTX 4090 Laptop (15.6GB VRAM, CUDA 12.6)
└── Isaac Lab 셋업 완료 (conda env isaaclab)

추가 검토 중:
├── ZED Mini (stereo, eye-in-hand 장착 가능)
├── LiDAR sensor
└── Cloud GPU ($50-100/월, OpenVLA 비교용)
```

## Key Findings (이번 세션)

- **ProbeFlow (2603.17850)**: SmolVLA flow matching 10→2-3 steps 가속 가능 (training-free)
- **VLA 연속학습 저항성** (2603.03818): 사전학습 VLA는 catastrophic forgetting에 강함 → multi-object 순차 학습 근거
- **"Scaling law" 대신 "data efficiency frontier"** 용어 사용 (ICLR 2025 oral 충돌 회피)
