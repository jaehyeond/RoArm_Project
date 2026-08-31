# hm1_s2 — heightmap 공통 모듈 + 출력 계약 (s2)

> **후속**: s1 의 실제 DEME 정착 pile 로 재검한 결과는
> [`../hm2_s1pile/README.md`](../hm2_s1pile/README.md).
> 실 pile 의 펠릿 지름은 여기서 가정한 5.0 mm 가 아니라 **4.16 mm** 였고(계약 셀
> 5 mm 는 유지 = 1.20 × 지름), 실 사면은 79.5° 가 아니라 p95 **30°** 라 실물 오차
> 예산은 4~10 mm 다.  또한 이산 입자 표면에서는 **median 집계가 포락면 추정자가
> 아니며**(아래 §4 의 연속 표면 결론이 전이되지 않는다), 부피는 펠릿 crown
> r-오프셋 때문에 계통적으로 +63 % 뜬다 — hm2 §3 참조.
> s1 pile 5개 앙상블 비교 + **전체 오차 예산 순위**는
> [`../hm3_pile_ensemble/README.md`](../hm3_pile_ensemble/README.md).
> 경로 (A) 다분산(입자별 반지름) 확장 + 검증은
> [`../hm4_polydisperse/README.md`](../hm4_polydisperse/README.md).

작성 = s2 워커, 2026-08-31.  **이 폴더는 신규다** — `yard_track/` 및 `sim_scripts/p26·p27·p29·p31`
은 읽기 전용으로만 참조했고 수정·재실행 0건.

| 항목 | 경로 |
|---|---|
| 모듈 (numpy 전용) | `roarm_rl/heightmap.py` (sha256:16 `032df9da432a76a5`) |
| 검증 probe | `sim_scripts/p33_hm1_heightmap_contract_probe.py` |
| 결과 | `hm1_results.json` |
| 진단 그림 (D324) | `hm1_parity.png`, `hm1_slope_error.png` |
| Rerun (D341) | `hm1_timeline.rrd`, `hm1_timeline.rbl`, `hm1_inspection.png` |
| 샘플 출력 | `hm1_pathA_particles.{npz,json}`, `hm1_pathB_depth_{nadir,kinectpose}.{npz,json}` |

재현: `python3 sim_scripts/p33_hm1_heightmap_contract_probe.py` (Rerun 포함은
`/home/cgxr/miniconda3/envs/isaaclab/bin/python ... --rerun`).  실행 전 기존
`hm1_inspection.png`/`hm1_timeline.*` 를 지워야 한다 (스크린샷 덮어쓰기 거부).

---

## 1. 출력 계약 — `spec_version = "roarm-heightmap-v1"`

s3 예측 모델의 **입력 계약**이다.  정본은 `roarm_rl/heightmap.py` 모듈 docstring이고,
모든 산출물은 `.npz` 안에 `header_json` 으로, 그리고 `.json` 사이드카로 헤더를 동봉한다.

| 항목 | 값 |
|---|---|
| 좌표계 | `roarm_base` — RoArm 베이스 프레임, z 위쪽.  `kinect_calib.yaml` 외부파라미터가 `p_base = R @ p_cam + t` 로 사상하는 그 프레임 |
| 셀 크기 | **5.0 mm** (`cell_m = 0.005`), 정사각 |
| 격자 원점 | `origin_xy_m` = 셀 `[0,0]` 의 **좌하단 모서리** (셀 중심 아님) |
| 축 / 인덱싱 | `height[row, col]`, `row → +y`, `col → +x` |
| 셀 중심 | `x = origin_x + (col+0.5)·cell_m`,  `y = origin_y + (row+0.5)·cell_m` |
| 측정량 | **셀 정사각 풋프린트 안의 최고 표면점** (셀 중심 샘플 아님) |
| 단위 | 미터.  `height` = 베이스 프레임 z − `z_datum_m` (`z_datum_m` = 지지면 z) |
| dtype | `height` float32 / `valid` bool / `counts` int32 |
| 빈 셀 | `valid=False` ⇒ `height = empty_cell.height_fill_m` (기본 0.0 = 지지면).  **`valid` 가 권위**, fill 은 패딩이지 측정값이 아니다 |
| `counts` | 셀에 들어온 표본 수(신뢰도).  단독 유효성 판정에 쓰지 말 것 — 입자 경로에서 `counts==0` 은 "위에 입자 없음 = 바닥 판독"이라는 **유효** 측정이다 |
| `agg` | 경로 (B) 전용.  `max`(기본, (A)와 동일 의미) / `p95` / `p90` / `median`, nearest-rank(round-half-up) |

`indexing`·`cell_center_formula` 는 동결된 yard_track `region_cells()`
(`[row=y, col=x]`) 와 **같은 규약**이라 과거 증거가 그대로 읽힌다.

### 왜 "셀 중심"이 아니라 "셀 풋프린트"인가
성긴 펠릿 층에서 셀 **중심** 샘플은 입자 사이 틈으로 빠져 바닥(0)을 읽을 수 있는데,
같은 셀의 풋프린트에는 펠릿 crown 이 들어 있다.  차이가 더미 전체 높이(≈60 mm)까지
벌어진다.  초기 구현(중심 샘플)에서 (A)−(B) 최대차 **63.5 mm** 가 실측됐고, 풋프린트
연산자로 바꾼 뒤 **6.67 mm** 로 떨어졌다.  두 경로를 교환 가능하게 만드는 핵심 결정.

### 두 생산자
```python
from roarm_rl.heightmap import GridSpec, heightmap_from_particles, heightmap_from_depth
spec = GridSpec.centered(center_xy_m=(0.20, 0.00), extent_m=0.150, cell_m=0.005)  # 30x30

# (A) 시뮬: DEME 펠릿 중심 (N,3) [m] + 반지름 (스칼라 또는 (N,) 배열 — hm4 확장)
hm = heightmap_from_particles(centers_m, radius_m=0.0025, spec=spec)
hm = heightmap_from_particles(d["positions_m"], d["radii_m"], spec)   # DEME npz 직결

# (B) 실물: Kinect depth [m] (0/NaN = no return) + 캘리브
from roarm_rl.heightmap import load_kinect_calib
c = load_kinect_calib("sim_scripts/kinect_calib.yaml")
hm = heightmap_from_depth(depth_m, c["intrinsics"], c["R"], c["t"], spec, agg="max")

hm.save("out/foo.npz")               # npz(+embedded header) + foo.json
hm.height, hm.valid, hm.counts       # float32 / bool / int32, all (30,30)
```
`roarm_rl/__init__.py` 는 gymnasium/Isaac 을 끌어온다.  그 의존성이 없는 환경에서는
`importlib.util.spec_from_file_location` 로 `roarm_rl/heightmap.py` 를 직접 로드하면 된다
(probe 상단이 그 예시).  모듈 자체는 numpy 만 쓴다(`load_kinect_calib` 만 pyyaml).

---

## 2. 셀 크기 5 mm 의 근거

* **하한** — 펠릿 지름(3~5 mm) 미만이면 셀당 펠릿이 1개 미만이라 crown/골이 교대로
  잡힌다.  셀별 변동이 ±펠릿 반지름(2.5 mm)의 순수 앨리어싱으로 채워지고 더미
  스케일 정보는 늘지 않는다.
* **상한** — yard_track 의 10 mm 는 "최소 물체 폭 22 mm 의 절반 이하"(D453 설계
  `p26_y1_testbed_design_author.py:28`)라는 **이산 물체 논거**다.  연속 입자에는
  전이되지 않고, 펠릿 지름의 2배라 스쿱 트렌치 벽이 뭉개진다.  "퍼낸 뒤 남을 형상"
  예측이 연구 목표이므로 트렌치 모서리는 이산화에서 살아남아야 한다.
* **실물 센서** — 캘리브된 Kinect 표준거리 0.9 m 에서 색정렬 depth 픽셀 풋프린트는
  `z/fx = 0.9/608.33 = 1.48 mm`.  5 mm 셀은 약 11 표본, 3 mm 셀은 약 4 표본이라
  NFOV 무작위오차 평균화가 부족하다.
* **모델 입력** — 150×150 mm 영역 기준 5 mm → 30×30, 3 mm → 50×50(대부분 센서
  잡음), 10 mm → 15×15(트렌치 소실).
* **D454 는 근거가 아니다** — 착지 분산 p95 43~63 mm 는 **이산 물체 place 결정층**
  수치다.  배출 위치가 고정된 현 방향에서는 관측 셀 크기의 근거가 되지 않는다.

`cell_m` 은 인자다.  바꾸면 헤더에 기록되고 s3 입력 계약이 바뀐다.

---

## 3. 가파른 면 오차 실측 (D453 후속)

D453: 레이캐스트 높이맵은 평탄 셀에서 raw-메쉬 GT와 bit-일치하지만 **cooked convex
수평 오차 ~1.2 mm 가 tan(θ) 로 증폭**(79.5° 셀 6.4 mm)된다.  이 probe 는 그 법칙을
재현하고, 같은 법칙이 **캘리브 오차와 셀 풋프린트에도 그대로 적용**됨을 보인다.
증폭 법칙: 임의의 수평 오차 `ε_h` → 수직 오차 `ε_h · tan(θ)`.

셀 5 mm, 해석 평면, 측정값 [mm]:

| θ [°] | (a) 풋프린트 편향 max집계 | (a') median집계 | (b) cooked ε=1.2 mm | (c) Kinect calib RMSE 10.13 mm | (d) 펠릿 경로 편차 rms | 권장 slope-aware tol |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 0.000 | 0.000 | 0.000 | 0.00 | 2.50 | 0.50 |
| 10   | 0.432 | 0.009 | 0.212 | 1.79 | 2.35 | 2.73 |
| 20   | 0.892 | 0.018 | 0.437 | 3.69 | 2.41 | 5.10 |
| 30   | 1.415 | 0.029 | 0.693 | 5.85 | 2.79 | 7.79 |
| 40   | 2.056 | 0.042 | 1.007 | 8.50 | 2.99 | 11.10 |
| 50   | 2.920 | 0.060 | 1.430 | 12.07 | 3.04 | 15.55 |
| 60   | 4.244 | 0.087 | 2.078 | 17.55 | 2.97 | 22.38 |
| 70   | 6.731 | 0.137 | 3.297 | 27.83 | 2.83 | 35.20 |
| **79.5** | **13.219** | 0.270 | **6.475** | **54.66** | 2.64 | 68.65 |
| 80   | 13.895 | 0.284 | 6.806 | 57.45 | 2.63 | 72.13 |

* **(a)** 계약 측정량(풋프린트 최댓값)이 셀 중심 높이보다 높은 결정적 편향.
  실측 = 해석 예측 `((c − p)/2)·tanθ` 와 전 슬로프에서 ≤1e-4 mm 일치 (p = 서브샘플 피치).
  연속 극한 설계값은 `(c/2)·tanθ`.  `median` 집계는 이 편향을 없애지만 그러면
  계약 측정량이 아니라 셀 중심 높이를 재는 것이다.
* **(b) D453 교차검증** — 1.2 mm 수평 시프트 주입 시 79.5°에서 **6.475 mm** 측정.
  D453 보고치 **6.4 mm** 와 일치 ⇒ 증폭 법칙 확인.
* **(c) 실물에서 지배적인 항** — 캘리브 RMSE 10.13 mm 가 수평 성분으로 들어가면
  30°에서 5.85 mm, 60°에서 17.5 mm, 79.5°에서 54.7 mm.  **실제 더미 안식각
  (~30~40°)에서도 6~9 mm** 로, (a)·(b)보다 한 자릿수 크다.  실물 heightmap 게이트는
  이 항이 결정한다 (RMSE < 5 mm 재캘리브가 최우선 개선점).
* **(d) 입자 경로는 tan(θ) 증폭이 없다** — 펠릿 경로의 편차는 0°에서 80°까지
  2.35~3.04 mm rms 로 사실상 슬로프 무관하고 펠릿 반지름(2.5 mm) 스케일에 묶인다.
  구는 cooked convex 근사가 없어 (b) 항이 아예 존재하지 않기 때문.
  **sim 관측이 실물보다 기하적으로 깨끗하다**는 뜻이고, 이 비대칭 자체가 sim2real 갭이다.
* **게이트 설계** — 단일 전역 max 임계 금지.  `slope_aware_tol_m(cell, ε_h, θ)` =
  `평탄 tol + (c/2)tanθ + ε_h·tanθ` 를 쓰거나 셀을 슬로프로 분류할 것.
  D453 의 G-hmap FAIL 은 정확히 이 두 항을 무시한 게이트 설계 인공물이었다.

---

## 4. (A)/(B) 규격 일치

* **계약 헤더 완전 동일** (`contract_diff = {}`): spec_version, frame, cell, origin,
  shape, indexing, 수식, 단위, dtype 3종, 빈 셀 규약까지 키·값 일치.
* **값 비교** — 동일 펠릿 원뿔(514개, H60/R60 mm)을 (A) 입자 폐형해 / (B) nadir 0.90 m
  합성 깊이렌더로 통과: 전체 rms **1.23 mm**, max **6.67 mm**, p95 3.30 mm.
  슬로프 밴드별 rms: 0–15° **0.000**, 15–35° 0.67, 35–60° 1.56 mm.
  잔차는 원근 광선 + depth 픽셀 이산화이며 평탄부에서는 정확히 0.
* **실제 Kinect 포즈의 그림자 비용** — `kinect_calib.yaml` 외부파라미터(카메라
  x=0.72 m, z=0.62 m, 수평 기준 앙각 약 43°)로 렌더하면 원뿔 **반대편(−x)** 셀
  **15/900 = 1.67%** 가 무효가 된다 (`hm1_parity.png` 3번 패널의 검은 셀).
  더 크고 가파른 실제 더미에서는 이 비율이 커진다 — 카메라를 더 nadir 쪽으로
  올리거나 2-뷰 융합이 필요한지는 s1 정착 결과로 재측정할 것.
* **깊이 잡음 하 집계 선택** (nadir 무잡음 max 대비 mm):

  | σ | max | p95 | p90 | median |
  |---|---|---|---|---|
  | 3 mm (근거리 통상, 참고치) | mean +3.50 / rms 4.19 | +2.75 / 3.59 | **+1.75 / 2.74** | −1.96 / 3.99 |
  | 17 mm (NFOV unbinned 스펙 상한) | **mean +24.99 / rms 26.81** | +21.12 / 23.10 | +16.15 / 18.11 | **−1.08 / 7.10** |

  셀당 표본이 ~11개뿐이라 `max` 는 잡음의 극값을 그대로 집는다 — σ=17 mm 에서
  **+25 mm 계통 과대**.  실 depth 에는 `median`(또는 `p90`) 을 쓰고 (a) 풋프린트
  편향을 해석적으로 되더하거나, depth 영상을 먼저 공간 필터링할 것.  **어느 쪽이든
  `agg` 는 헤더에 기록되며 sim/real 이 같은 값을 써야 한다.**

---

## 5. 검증 게이트 · 재현성

`hm1_results.json` verdict = **`HM1_ALL_GATES_PASS`**

| 게이트 | 내용 | 결과 |
|---|---|---|
| G1 analytic | 평면(z=40 mm) / 경사면 0–80° x축·대각 / 원뿔.  해석 예측 대비 잔차 | PASS, 최대 잔차 **1.49e-5 mm** (평탄 8.94e-7 mm) — float32 저장 양자화 수준 |
| G2 steep | 증폭 법칙 실측 vs 예측, 풋프린트 편향 실측 vs 예측 | PASS, 전 슬로프 ≤1e-4 mm |
| G3 parity | (A)/(B) 계약 헤더·dtype·shape·빈셀 규약 동일 | PASS, `contract_diff = {}` |

* **교차 환경 bit-재현** — base(`python 3.13.11 / numpy 2.4.2`) 와
  `isaaclab`(`python 3.11.14 / numpy 1.26.0`) 두 환경에서 **수치 필드 487개 전부 동일**.
* **save/load 왕복** — height/valid/counts/헤더/GridSpec 전부 bit-동일.
* **D341 Rerun** — `hm1_timeline.rrd` (sha256 `094b806b172c4fd0…`, 74,994 B):
  버전 0.34.1 일치, footer-enabled `rrd verify` PASS, exact entity(7)·timeline
  (`blueprint`/`log_time`/`slope_deg`)·component 계약 전부 PASS, 고정 blueprint 임베드 +
  `.rbl` 검증 PASS, headless 스크린샷 생성 PASS → `validation.pass = true`.

### 육안 확인 기록 (D341 "실제 시각 검사")
* `hm1_parity.png` — (A)/(B nadir) 모두 apex ≈63.6/63.5 mm 의 축대칭 원뿔, r>60 mm
  에서 정확히 0.  육안 구분 불가.  (B kinect 포즈) 는 같은 원뿔에 **−x 쪽(카메라
  반대편)에만** 검은 무효 셀이 뭉쳐 있어 사각 시점 그림자임이 확인된다.
  차분 패널은 전부 ≥0(붉은 계열)이고 평탄 외곽은 정확히 0, 최대 6.7 mm 가 원뿔
  사면에 링 형태로 분포 — 보고 수치와 일치.
* `hm1_inspection.png` — 좌측 3D 뷰에 주황(A)/파랑(B nadir)/자홍(B kinect) 세 점군이
  겹쳐 원뿔+평탄 스커트를 이루고, 주황·파랑은 사면에서 일치, 자홍만 정점 부근에
  결손이 보인다.  우측 시계열은 4개 스칼라가 슬로프 증가에 따라 단조 증가하고
  calib_rmse 가 가장 가파르다 — matplotlib 그림과 동일.
* ⚠ 미미한 표시 이슈: `slope_deg` 를 duration 타임라인으로 로그해 뷰어 축이
  "초" 로 표시된다(값은 도 단위로 정확).  판정에 영향 없음.

---

## 6. 미주장 (non-claims)

* **실제 Kinect 프레임 미사용.**  경로 (B) 는 합성 깊이 렌더로만 검증했다.
  실 depth 잡음 모델·다중경로·표면 반사율·NFOV FOV 클리핑 충실도 미주장.
  카메라가 물리적으로 없어 저장된 프레임도 확보 못 함 — 경로만 세워 둔 상태다.
* **DEME 정착 결과 미수신.**  `claudedocs/runtime_logs/sim_deme/` 에는 s1 의
  타이밍 smoke(`n_particles`, `radius_m` 등)만 있고 입자 위치 배열이 없다.
  경로 (A) 는 합성 원뿔/경사면 펠릿으로 검증했고 실제 정착 분포 통계는 미주장.
  s1 산출이 오면 `heightmap_from_particles(centers, radius, spec)` 에 그대로 넣으면 된다
  (반지름 2.5 mm 는 s1 smoke 값과 일치).
* 안식각 현실성, 스쿱 물리, 실물 트레이/배출 용기 기하 미주장.
* `kinect_calib.yaml` 자체의 재검증 없음 — RMSE 10.13 mm 는 2026-04-24 값을 인용했고
  카메라 재설치 시 무효다.
