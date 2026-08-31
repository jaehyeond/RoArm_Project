# hm4_polydisperse — 경로 (A) 다분산 확장 + 검증

작성 = s2 워커, 2026-08-31.  계보: [`../hm1_s2`](../hm1_s2/README.md) (계약) →
[`../hm2_s1pile`](../hm2_s1pile/README.md) (실 pile) →
[`../hm3_pile_ensemble`](../hm3_pile_ensemble/README.md) (앙상블) → **hm4** (다분산).

hm2 에서 확인한 모듈 한계 — `heightmap_from_particles()` 가 단일 반지름만 받아
다분산 pile 을 `POLYDISPERSE_UNSUPPORTED` 로 거부했다 — 를 해소한다.

| 항목 | 경로 |
|---|---|
| 모듈 | `roarm_rl/heightmap.py` (확장 대상) |
| probe | `sim_scripts/p36_hm4_polydisperse_contract.py` |
| 결과 | `hm4_results.json` — verdict **`HM4_ALL_GATES_PASS`** (P1~P6) |
| 그림 (D324) | `hm4_polydisperse.png` |
| Rerun (D341) | `hm4_timeline.rrd` (sha256 `773de5047b47e084…`), `.rbl`, `hm4_inspection.png` |
| heightmap | `hm4_pathA_polydisperse`, `hm4_pathA_monodisperse_reference`, `hm4_pathB_depth_nadir` |

재현: `python3 sim_scripts/p36_hm4_polydisperse_contract.py [--rerun]`

---

## 1. API 변경 (하위 호환)

```python
heightmap_from_particles(centers_m, radius_m, spec, *, fill_m=..., floor_z_m=0.0,
                         extra_meta=None)
#   radius_m : 스칼라  또는  (N,) 배열
```

DEME 는 `radii_m` 을 (N,) 배열로 준다 — **그대로 넘기면 된다**:

```python
d = np.load("pile_....npz")
hm = heightmap_from_particles(d["positions_m"], d["radii_m"], spec)
```

측정량·출력 규격은 hm1 계약 그대로다.  달라진 것은 반지름이 입자별이라는 것뿐:

> `height[cell] = max over spheres i of  p_z,i + sqrt(r_i² − dist(p_xy,i, cell_rect)²)`

**헤더 변화** (계약 키가 아니라 source-meta 키라 (A)/(B) 규격 비교에는 영향 없음):

| 키 | 값 |
|---|---|
| `polydisperse` | bool — 신규 |
| `particle_radius_min_m` / `_max_m` / `_mean_m` | 신규, 항상 기록 |
| `particle_radius_m` | **균일할 때만** 유지 (기존 독자 보호) |

`p33`의 합성 깊이 렌더러(`render_sphere_depth`)와 `p34`의 `load_pile` 도 같은 규약으로
확장해, 실제 다분산 pile 이 오면 hm2/hm3 파이프라인이 그대로 돈다.
`p34.load_pile` 은 이제 `r` 을 배열로 반환하고 `r_uniform / r_min / r_max / r_mean /
r_med` 를 함께 준다.  파생 스칼라는 **어느 통계인지 이름에 박아** 뒀다
(`pellet_diameter_median_mm`, `cell_per_pellet_diameter_max`, …).

### 구현: 입자별 창(window) 버킷팅
구 하나는 **자기 반지름** 안의 셀 사각형에만 닿는다.  그래서 창 반폭
`k_i = ceil(r_i / cell) + 1` 을 입자마다 잡고, 같은 `k` 끼리 묶어 그룹별로 스캐터한다.
`r_max` 일괄 창을 쓰면 좁은 입자가 가장 큰 구의 창을 통째로 훑게 된다 (§P5).

### 셀 크기 지침 (다분산)
단일 "그" 지름이 없으므로 두 경계를 따로 읽는다 — **앨리어싱 하한은 최대 반지름**이
정하고(셀 crown 이 평균 표면 위로 뜰 수 있는 양), 형상 해상도 요구는 그대로다.
대표(중앙) 지름으로 셀을 잡되 헤더의 `2 × particle_radius_max_m` 로 확인할 것.

---

## 2. 검증 — 6 게이트 전부 PASS

허용치 `TOL = 1e-7 m`.  계약이 height 를 float32 로 저장하므로 h ~ 0.05 m 에서
양자화 스텝이 약 6e-9 m 다.  1e-7 m 를 넘는 차이만 실제 구현 오차다.

### P1 하위 호환 — 균일 배열 == 스칼라
`height` / `valid` / `counts` **bit-동일**, 헤더도 완전 동일, `particle_radius_m` 키
양쪽 모두 존재, `polydisperse=False`.
→ **기존 monodisperse 호출자는 이 확장의 영향을 전혀 받지 않는다.**

### P2 독립 브루트포스 대조
셀마다 **전** 구를 순회하는 별도 구현(창 버킷팅·벡터 스캐터 미사용)과 비교:

| 분포 | 반지름 [mm] | max\|Δ\| |
|---|---|---:|
| narrow | 1.00 – 1.50 | 1.82e-9 m |
| wide | 1.00 – 7.99 | 1.86e-9 m |
| bimodal | 1.00 / 7.00 | 1.85e-9 m |

전부 float32 양자화 수준 → 최적화가 결과를 바꾸지 않았다.

### P3 손으로 푼 2-구 배치
셀 5 mm, 셀[0,0] 중심 (2.5, 2.5) mm / 셀[0,1] 중심 (7.5, 2.5) mm.
- big: 중심 (2.5, 2.5, 0) mm, r = 6 mm
- small: 중심 (7.5, 2.5, 4) mm, r = 1 mm → 자기 셀에서 꼭대기 5.000 mm

| 셀 | 기대 | 실측 |
|---|---:|---:|
| [0,0] | 6.0000 mm | 6.0000 mm |
| [0,1] | 5.4544 mm (= √(6²−2.5²)) | 5.4544 mm |

**셀[0,1] 안에 작은 구가 있는데도 이웃 셀의 큰 구가 이긴다** (5.454 > 5.000).
반지름이 균일하면 생길 수 없는 상황이고, 창을 입자별로 잡아도 이 교차 기여를
놓치지 않음을 못 박는다 (`counts = [[1, 2]]` — 셀[0,1] 은 두 구에서 기여받음).

### P4 반지름 그룹 중첩 불변식
다분산 heightmap == 반지름 레벨(1.2 / 3.0 / 6.5 mm)별 heightmap 들의 **원소별 max**.
max\|Δ\| = **정확히 0.0**, counts 합도 일치.
→ 구현이 그룹을 어떻게 버킷팅하든 결과가 같아야 한다는 구조적 강제.

### P5 창 버킷팅 비용 (결정적 후보쌍 카운트)

| 분포 | cell | k 범위 | 버킷 | 버킷팅 | r_max 일괄 | 절감 |
|---|---:|---|---:|---:|---:|---:|
| s1 유사 균일 r=2.08 mm | 5 mm | [2,2] | 1 | 50,000 | 50,000 | **1.00×** |
| moderate 1–4 mm | 5 mm | [2,2] | 1 | 50,000 | 50,000 | **1.00×** |
| wide 0.5–10 mm | 1 mm | [2,11] | 10 | 481,664 | 1,058,000 | **2.20×** |
| bimodal 0.5 / 10 mm | 1 mm | [2,11] | 2 | 156,344 | 1,058,000 | **6.77×** |

좁은 분포에서는 **비 1.00×** — 확장이 기존 경로에 비용을 얹지 않는다.
폭이 넓을수록 이득이 커지고, bimodal(대부분 작고 소수만 큰) 이 가장 크다.
⚠ 후보쌍 수는 결정적 상한 지표이지 벽시계 시간이 아니다.  버킷 수만큼 파이썬 루프가
늘어나므로 아주 좁은 분포에서는 이득이 상수 오버헤드에 묻힐 수 있다.

### P6 실 pile 좌표 + 크기 분포
s1 정착 좌표 512개에 log-normal 크기 분포를 입혔다.  **반지름을 줄이기만** 하므로
(i) 겹치지 않던 배치에서 구를 줄이면 여전히 안 겹치고 (ii) 다분산 heightmap 은 균일
참조보다 결코 높을 수 없다 — 둘 다 검사한다.

| 항목 | 값 |
|---|---|
| 반지름 [mm] | min 0.832 / median 1.502 / mean 1.535 / max 2.080 (원래 균일 2.080) |
| (A)/(B) 계약 헤더 | **완전 동일** (`contract_diff = {}`) |
| peak vs 독립 계산 `max(p_z + r_i)` | 기대 11.987286 mm, 실측 11.987287 mm, **오차 0.0** |
| 축소 전용 단조성 (poly ≤ uniform) | **성립** |
| poly − uniform | mean −0.562 mm, rms 0.699, max 2.754 mm |
| (A)−(B) pile 위 | rms **1.185 mm**, max 5.497 mm |

(A)/(B) rms 1.19 mm 는 hm2 의 균일 pile 값(0.90 mm)보다 약간 크다 — 작은 구가
많아져 표면이 더 거칠어졌으니 예상되는 방향이다.

---

## 3. 육안 확인 (D324 / D341)

`hm4_polydisperse.png` — 왼쪽부터 균일 참조(peak 12.44 mm) / 다분산(11.99 mm) /
(B) 깊이(11.94 mm) / 차분.  다분산 지도는 전체적으로 균일 참조보다 **한 톤 낮고**
발자국 가장자리가 더 성기다 (작은 구는 셀 사각형까지 덜 닿으므로 가장자리 셀 몇 개가
빠진다) — 축소 전용 단조성과 일관.  차분 패널은 전부 ≥ 0 이고 구조 없이 흩어져
최대 5.5 mm 인 단일 셀이 하나 보인다.

`hm4_inspection.png` (Rerun) — 주황(다분산)·회색(균일 참조)·파랑(깊이) 점군이 겹쳐
있고 주황이 회색 아래로 일관되게 깔린다.  우측 시계열은 P5 의 후보쌍 카운트 4 케이스.
D341 계약: `validation.pass = true`, errors 없음 — 버전 0.34.1 일치, footer-enabled
`rrd verify` PASS, exact entity(6)·timeline(`blueprint`/`log_time`/`cost_case`)·
component 계약 PASS, 고정 blueprint 임베드 + `.rbl` 검증 PASS, headless 스크린샷 PASS.

## 4. 회귀 확인

확장 후 기존 probe 3개를 모두 재실행했고 판정·수치 변화 없음:
`p33` → `HM1_ALL_GATES_PASS`, `p34` → `HM2_ALL_GATES_PASS`, `p35` → `HM3_OK`
(peak 범위 0.996 mm, 부피 범위 1.91 %, rms 1.608 / 1.613 mm — 확장 전과 동일).

한 가지만 값이 바뀌었다: `p34.load_pile` 의 `surface_max_z_m` 을
`positions_max_m[2] + R[0]` 에서 **s1 의 `surface_max_m[2]` 직접 인용**으로 바꿨다.
다분산에서는 최고 z 입자가 최대 반지름 입자가 아닐 수 있어 전자가 틀리기 때문이다.
그 결과 hm2 H1 의 peak 오차가 `0.0` → `5.66e-9 mm` 로 표기가 바뀌었다 (둘 다 float32
양자화 수준, 허용치 1e-4 mm 의 1/17000).  내 산술을 빼서 **교차검증이 더 독립적**이 됐다.

## 5. 미주장 (non-claims)

* **진짜 다분산 DEME pile 은 아직 없다.**  s1 이 낸 5개는 전부 균일(r span < 1e-12).
  P6 은 정착 좌표에 크기 분포를 **사후에** 입힌 기하 스트레스 테스트다 —
  다분산 입자의 정착 물리·안식각·패킹률은 전부 미주장.  진짜 다분산 pile 은
  s1 이 DEME 로 생성해야 하고, 오면 `p34 --npz`, `p35 --glob` 로 바로 돈다.
* hm2 §3 의 부피 편향 기전(`excess ≈ 점유면적 × r`)은 **균일** pile 에서 세운
  경험식이다.  다분산에서는 대표 반지름을 어떤 통계로 잡아야 하는지 미검증 —
  probe 는 평균 반지름을 쓰지만 그 선택의 정당성은 진짜 다분산 pile 로 재검할 것.
* 실제 Kinect 프레임 미사용 (합성 깊이 렌더).
* P5 는 후보쌍 카운트이지 벽시계 벤치마크가 아니다.
