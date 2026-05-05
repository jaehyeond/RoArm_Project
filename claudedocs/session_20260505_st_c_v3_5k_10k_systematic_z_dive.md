# Phase ST-C v3 — 5K & 10K deploy: systematic R3 knock + Z dive (2026-05-05)

## Verified facts (n=3 deploys, all 300 steps)

| Run | Ckpt | First visit | TCP z min | Y- steps | CENTER steps | Knock | MP4 |
|---|---|---|---|---|---|---|---|
| Retry-1 | 5K | R3 (step ~50) | −98.6mm | 300 | 0 | R3 ✓ | logs/frames_st_c_5k_20260505_140735.mp4 |
| Retry-A | 5K | R3 (step ~50) | −98.6mm | 50 | 250 | R3 ✓ | logs/frames_st_c_5k_retryA_20260505_152657.mp4 |
| Retry-D | **10K** | R3 (step ~30) | **−99.8mm** | 39 | 261 | R3 의심 | logs/frames_st_c_10k_retryD_20260505_153525.mp4 |

## Conclusions

1. **R3 first visit = 3/3 systematic** (far-Y- area, X≈+380, Y≈−160).
2. **TCP z dive = −98 to −99.8mm = systematic** (책상 표면 −12mm보다 86mm 아래).
3. **10K가 5K 대비 z dive 개선 못함** → ckpt-step 가설 기각.
4. **Variance 가설 기각** — 같은 ckpt가 동일 z 산출 (BC 재현성 매우 높음).
5. **CENTER phantom place behavior** — Retry-A/D는 grasp 실패 후 (+280, 0) 근처로 이동 → place 동작 시도 (sponge 없는 상태).

## Root cause hypothesis (가장 강한 후보)

**H1 (가장 강) — Sim-real Z calibration gap**:
- Sim demo의 `Z_TCP_GRASP_L1 = +33mm world` 위치에서 grasp 학습.
- Real에서 같은 joint angles → TCP z = −98mm.
- **131mm 차이** (sim에서 의도한 grasp z vs real 실측).
- 가능 원인: (a) sim FK ≠ real FK, (b) sim의 z=0 원점이 real과 다름, (c) sim 학습 데이터가 z 정합 잘못.

**H2 (중)**: lerobot_dataset_v6_stacking_v3 stats 분포 문제.
**H3 (약)**: sim render와 state 매칭 오류.

## Setup info

- 4 sponges placed: R1(+200,−175)Y, R2(+200,+135)X, R3(+380,−160)X, R4(+380,+125)Y — edge-stand 47mm tall.
- Kinect calibration unchanged (4/15, RMSE 10.13mm).
- Init pose [0,0,90,0,0,5], task="Stack four pink sponges into a # pattern".
- Follower=USB1 (per HARD RULE).

## Next session — diagnostic plan

**우선순위 1: Sim demo direct replay test** (모델 없이 sim ep0 joint sequence를 real arm에 그대로 명령 → TCP z 측정).
- 결과 z = −98mm → **H1 확정** (FK or 좌표계 mismatch). 다음: real ↔ sim FK 비교.
- 결과 z = +33mm → **H1 기각**, normalization/모델 출력 의심. 다음: model action chunk 분석.

**우선순위 2 (병행)**: 15K, 20K ckpt deploy로 ckpt-step 전 범위 systematic 확정.

**우선순위 3**: lerobot_dataset_v6_stacking_v3 sim ep parquet 분석 (action joint dist, FK z target 분포).

## Files & commands

```bash
# Sim ep replay (no model) test:
conda run -n roarm python -c "
import pandas as pd
import numpy as np
from roarm_sdk.roarm import roarm
df = pd.read_parquet('lerobot_dataset_v6_stacking_v3/data/chunk-000/episode_000.parquet')
# Take only stacking eps (task_index=1, episode_index >= 50)
# Actual replay code TBD
"

# 4090 outputs structure:
# outputs/smolvla_v6_stacking_v3_b200/checkpoints/{005000,010000,015000,020000}/pretrained_model/
```
