# Phase ST-A Sub-A4 + Sub-A5 PASS (visual) → 🔴 사용자 정정으로 v2 lying flat 설계 자체가 잘못됨 (2026-04-30 evening)

> **결론**: 본 세션 작성한 코드 모두 lying flat 가정 → 무효. 다음 세션에 UPRIGHT 재설계 필수.

---

## 0. 본 세션 산출물

| 파일 | 상태 |
|---|---|
| `sim_scripts/stacking_scene_v2.py` | 🔴 lying flat 가정 → 무효 |
| `sim_scripts/render_stacking_demos_v2.py` | 🔴 lying flat 가정 → 무효 |
| `sim_renders_v4/stacking_initial.png` | 🔴 lying flat 시각화 → 무효 |
| `sim_renders_v4_dryrun/episode_000/*.png` (128 PNG, 37MB) | 🔴 무효 (사용자 4 sponge 모두 lying flat 보고 정정) |

**선행 4/30 afternoon 세션 산출물도 잘못된 가정 기반**:
- `sim_scripts/generate_stacking_demos_v2.py` — TCP grasp z=+32.5mm (lying flat top-down)
- `sim_demos_v2/` 50 demos × 128 frames = 6400 frames — 무효

---

## 1. 사용자 정정 (4/30 evening)

> "아니 왜 스펀지를 또 눕혀놨어. 지금 이렇게 2번째 이미지처럼 세웠잖아. 그리고 우물정도 저렇게 세운상태에서 1층 2층한거잖아."

- 사용자 첨부 이미지 2: v6 실제 robot setup, 분홍 sponge **upright** (125mm 수직)
- 정정: **V2 sponge = UPRIGHT (v6와 동일)** ≠ lying flat
- # 우물정자 = upright 상태에서 1층 + 2층 (구체 기하는 사용자 추가 답변 대기)

---

## 2. 4/30 afternoon 결정의 root cause

session_20260430_phase_st_a3_n4_well_demos_v2.md line 36-40:
> 1.3 V6 → V2 매핑 결정
> - V2 = lying flat # (사용자 의도, 사진 매치, 물리 안정성)

session_20260429_phase_st_a_redesign_n4_well_pattern.md (4/29 evening):
> 사용자 사진 10장 분석 → N=4 sponge lying flat # 패턴 확정 (1층 평행 2 + 2층 직교 2 = 위에서 # 모양)

**오류**: 사용자 사진 10장 lying flat로 잘못 해석. 실제로는 upright sponge로 # 우물정자 구성.

---

## 3. Step-by-step 검증 (사용자 요청)

### Step 1: GPU 위치 — HARD RULE #17 ✓ NOT 위반
- 호스트: `cgxr-Legion-Pro-7-16IRX9H` (이 노트북)
- GPU: `RTX 4090 Laptop GPU` (UUID `05b1a3f8-b7cf-dc57-06aa-741fe2daa4b4`) — **로컬**
- B200 (NHN/Sogang, UUID `c553ca20-...`) **사용 X**
- 명령: `conda run -n isaaclab python ...` — SSH/원격 0
- → HARD RULE #17 준수

### Step 2: Sponge orientation 코드 검증
```
generate_stacking_demos_v2.py:1: """N=4 well-pattern (#) lying flat stacking
generate_stacking_demos_v2.py:76: SPONGE_THICK = 0.020   # lying flat thickness
stacking_scene_v2.py:46: # V2 = lying flat (thickness = z = 20mm)
stacking_scene_v2.py:53: SIZE_LENGTH_X = (0.125, 0.047, 0.020)  # 125,47,20
render_stacking_demos_v2.py:4: 4 lying flat sponges tracked per frame
```
모든 v2 코드가 lying flat 전제. 무효.

### Step 3: TCP grasp z OOD 정량
- V6 grasp z mean **+114mm** range [+74, +194] (upright lateral grasp)
- V2 (lying flat 가정) grasp z = **+32.5mm** (top-down)
- **OOD -82mm** — 4/30 afternoon에서 "acceptable: v6 release region 활용"이라 판단했지만
  → upright로 재설계 시 V2 grasp z도 v6과 동일 +114mm 범위로 재조정 필요

### Step 4: Sub-A4/A5 dry render 시각 검증 결과 (참고용 — 무효지만 코드 구조는 재사용 가능)
- 7 phase boundary frames (HOME, S1.close/open, S2.close/open, S3.close/open, S4.close/open, HOME_end) 시각 확인 PASS
- HOLD_INTERVALS_V2: S1[12,28) S2[41,57) S3[73,89) S4[102,118) — 32-anchor 구조 자체는 재사용 가능
- Held offset TCP+(0,0,-34.6mm) — upright 시 재계산 필요 (sponge center to TCP grasp z 관계 변경)
- 50-demo render ETA ~19min 예상 — 무효라 launch 안 함

---

## 4. Upright # 우물정자 재설계 — 사용자 답변 대기

**Option A** — 1층 4 sponges 바닥 평면에 # 모양 (2층 없음, 인접 4 pillar)
```
║ ║       (평행 2 upright)
═══       (직교 2 upright 사이)
║ ║
```

**Option B** — 1층 2 + 2층 2 (upright sponge 위에 upright sponge 쌓기)
- 1층: 2 sponges upright (top z=+112mm)
- 2층: 2 sponges upright on top (top z=+237mm)
- 총 높이 ~25cm

**Option C** — 다른 배치

사용자 답변 + 가능 시 # 패턴 사진 1장 첨부 후 재설계.

---

## 5. 다음 세션 진입 지점

### 5.1 사용자 답변 대기 (Q1-Q3)
- Q1. # 우물정자 정확한 기하 (Option A/B/C 또는 새 옵션, 사진 첨부)
- Q2. v6와 동일하게 sponge upright + lateral grasp 재사용 OK?
- Q3. 1층/2층 모두 upright = ABCD 4 sponge 동일 stance? 아니면 layer별 stance 변경?

### 5.2 사용자 답변 후 작업 순서
1. **Cleanup** (사용자 답변 받으면): `rm -rf sim_demos_v2/ sim_renders_v4/ sim_renders_v4_dryrun/`
2. **재설계 generate_stacking_demos_v2.py** (upright 기반, v6 grasp z +114mm 매치)
3. **재설계 stacking_scene_v2.py + render_stacking_demos_v2.py** (sponge size 22×47×125mm upright)
4. **TCP-to-sponge offset 재계산** (lateral grasp on upright = TCP_to_sponge에서 z 작은 차이만)
5. Sub-A6 50-demo render → Sub-A7 sim_to_lerobot → Sub-A8 merge_v6_stacking_v2

### 5.3 코드 재사용 가능 부분 (lying flat에 비종속)
- `roarm_kinematics.py` 100% (FK/IK/V6WarmStart)
- `sim_to_lerobot_stacking.py` 100% (LeRobotDataset.create + add_frame)
- `merge_v6_stacking.py` 100% (Approach C aggregate)
- `generate_stacking_demos_v2.py` 구조 ~50% (32-anchor structure + IK loop + interp)
- `render_stacking_demos_v2.py` 구조 ~70% (Isaac Sim boot + 4-sponge prim + held offset state machine)
- `stacking_scene_v2.py` 구조 ~50% (Isaac Sim boot + URDF + camera + sponge spawn)

---

## 6. HARD RULES 적용 기록

- **#11 NO /half-clone** — 본 세션 3회+ 거부 (Stop hook 85%, 86%, 87%, 91%)
- **#12 데이터 먼저 확인** — 사용자 정정 받자마자 v6 frame PNG 분석으로 즉시 검증 (이미 4/30 afternoon에서 v6=upright 확인됨)
- **#17 sim render = 4090** — Sub-A4/A5 dry render 모두 로컬 4090 (UUID 05b1a3f8) 사용. B200 0회.

---

**기록자**: Claude Opus 4.7 (1M context).
본 세션 = 잘못된 가정 기반 작업 시각 PASS → 사용자 정정으로 무효화. 다음 세션 진입점은 사용자 # 패턴 답변.
