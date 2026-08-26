# Reference — Research Verification Rules · External/Evidence Links

> 출처: 분리 전 `AGENTS.md` **657-724행**. 원본 전체는 `docs/archive/AGENTS_full_20260825_pre_split.md`.
> 아래 본문은 원본에서 **바이트 동일**하게 이동했다 (2026-08-25).
> 이 절은 `AGENTS.md` HARD RULE **#4**("없다/최초" 주장 = 10개+ 검색어 × 2개 소스 검증)의 상세 절차다.
> **연구 갭·"최초"·"없다"를 주장하기 직전에 이 파일을 읽는다.**

---

## Research Verification Rules (연구 검증 — 2026-03-10 실수에서 배운 것)

> **배경**: 2026-03-10에 "연구 갭" 5가지를 제시했으나 4/5가 거짓이었음.
> 원인: 충분한 검색 없이 "없다"고 단정. 논문 제목의 단어를 잘못 해석.

### 절대 규칙

| Rule | Why | 위반 사례 |
|------|-----|----------|
| **"없다/최초"는 반드시 10개+ 검색어로 검증** | 한두 번 검색으로 "없다"고 단정하면 거짓 positive | "RGBD-VLA 없음" → 실제 8개+ 존재 |
| **논문 제목의 단어를 문맥 없이 해석 금지** | "Depth"가 depth 카메라인지 network depth인지 확인 필수 | RD-VLA의 "Depth" = 네트워크 깊이 |
| **"갭 발견" 시 반증 검색 먼저** | 갭을 주장하기 전에 그 갭을 채운 논문을 적극 검색 | "adaptive chunking 없음" → MoH 존재 |
| **arXiv ID 있으면 반드시 확인** | 논문 실존 여부 + 내용 일치 검증 | pi0.6 → 실제 π\*₀.₆ (5B, RECAP) |
| **"X가 유일/최초" 주장 전에 경쟁자 최소 5개 검색** | 주장의 강도에 비례하는 검증 필요 | "SmolVLA가 유일한 로컬 학습 VLA" 등 |
| **분야별 최신 서베이/메타분석 먼저 확인** | 개별 검색보다 서베이가 전체 그림 제공 | ICLR 2026 VLA 메타분석 활용 |

### 검증 프로세스 (연구 갭 주장 시)

```
1. "X가 없다" 주장하려면:
   ├── 최소 3가지 다른 검색어로 검색
   ├── 최소 2개 소스 (arXiv, Google Scholar, Semantic Scholar)
   ├── 2024-2026 논문 중심으로 확인
   └── 반증 논문 1개라도 발견 시 → 주장 철회

2. "세계 최초" 주장하려면:
   ├── 위 1번 + 관련 학회 proceedings 확인
   ├── 유사 논문의 Related Work 섹션 확인
   └── 확신도를 명시: HIGH/MEDIUM/LOW

3. 검증 실패 시:
   ├── 즉시 정정 (정정 경위 + 올바른 정보)
   ├── ResearchPlan.md에 ⚠️ 정정 마크 추가
   └── 이전 주장을 삭제하지 말고 정정 기록 유지
```

### 근본 원인 분석 (2026-03-10 실수)

| 실수 유형 | 원인 | 방지책 |
|-----------|------|--------|
| 확증 편향 | "갭을 찾고 싶다" → 갭이 아닌 증거 무시 | 반증 검색을 먼저 수행 |
| 검색 부족 | 1-2개 키워드만 검색 | 최소 3개 검색어 × 2개 소스 |
| 용어 오해 | "Depth" = depth camera라고 가정 | 논문 abstract/method 반드시 확인 |
| 시간 지연 | 2025 중반 기준 지식으로 2026 주장 | 최신 arXiv (최근 6개월) 필수 확인 |
| 과대 주장 | "zero papers" 같은 절대적 표현 | "우리 검색 범위 내에서" 등 한정어 사용 |

## Reference

### External
- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA: https://huggingface.co/docs/lerobot/en/smolvla
- RoArm M3 PR: https://github.com/huggingface/lerobot/pull/820

### Sim env (4/24, 4/28)
- [sim_scripts/stacking_scene.py](sim_scripts/stacking_scene.py) — Stacking 씬 spawn (Layout A/B/Temp)
- [sim_scripts/replay_v6_sim.py](sim_scripts/replay_v6_sim.py) — V6 trajectory sim replay (50ep ✓)
- [sim_scripts/sim_to_lerobot.py](sim_scripts/sim_to_lerobot.py) — Sim → LeRobot v3 변환기
- [sim_scripts/kinect_calib.yaml](sim_scripts/kinect_calib.yaml) — Kinect intrinsic + extrinsic
- [sim_scripts/table_plane.json](sim_scripts/table_plane.json) — Table plane fit (-12.12mm)
- [sim_scripts/sponge_poses.json](sim_scripts/sponge_poses.json) — 50 ep 별 sponge 위치
- [sim_v1/](sim_v1/) — Sim replay LeRobot v3 dataset (87MB)
- [sim_renders_v2/](sim_renders_v2/) — 50ep frame PNGs + tracking RMSE

### Calibration (4/15, 4/24)
- [claudedocs/marker_real_photo.png](claudedocs/marker_real_photo.png) — 실제 빨간 마커 사진
- [claudedocs/marker_urdf_truth.png](claudedocs/marker_urdf_truth.png) — URDF 정답 비교
- [claudedocs/stepDE_siglip50_sim_v1_20260424.md](claudedocs/stepDE_siglip50_sim_v1_20260424.md) — SigLIP 0.7222 GO 분석
- [claudedocs/session_20260424_stacking_design_pivot.md](claudedocs/session_20260424_stacking_design_pivot.md) — N=2 stacking design
