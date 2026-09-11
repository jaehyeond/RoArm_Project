# GATES_w8 — 사전 등록 (2026-09-10, 실행 전에 작성)

W8 = 렌즈형 실측 펠릿 더미(W7, 20,000알 = 구 140,000행)에 S1 그랩 실물 절차(D481)로 퍼내기.
판정은 전부 셀 결과 JSON(`cell_*/scoop_s1_seed460.json`)과 회귀 JSON 을 `gates_w8_writer.py` 가 읽어 `gates_w8.json` 에 쓴다.
이 문서는 실행 전에 고정하며, 실행 후 수정하지 않는다.

## 사전 고정 설정 (변수 사다리: 신규 변수 = 더미(구 능선 → 렌즈 클럼프 언덕) 1개)

| 항목 | 값 | 출처 |
|---|---|---|
| 더미 | `pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz` (sha256 `659d6b0b…b818812`) | W7 |
| 알 | 템플릿 질량 20.257 mg, 7구(중심 r 1.2502 · 링 r 1.1564 mm), 외접 지름 4.5007 mm | npz `clump_template_json` |
| 물성 (임시 = MEASURE) | mu 0.45 · Crr 0.06 · CoR 0.3 · nu 0.3 · E 5e6(W3b 수치 안정값; 더미는 1e7 로 정착 → t=0 미세 재안착, W3b 와 같은 규약) · 툴 E 3 GPa | W7 metadata + W3b params |
| 절차 | 립→문 서보 30°(관절 27.5°)→잠김 25 mm(하강 25 mm/s, 팔 힘 상한 6 N)→닫기(서보 정지 1.96×0.9=1.764 N·m, 45°/s)→+80 mm(150 mm/s)→재닫기 | D481 · W3b `params_fixnorm_plunge25.json` |
| 적분 | dt 1e-5 · sync 4 ms · CD 20 · error-out 60 m/s · 천장 0.5 m | W3b |
| 셀 | seed 460 고정. 중앙 (0, 0) mm · +x 50 mm · −x 50 mm (문은 세계 −y→+y 로 닫히므로 옆 셀은 쓸어 담는 방향과 직교하는 x 축으로 둔다; y 벽 220 mm 에서 멀다) | 이 문서 |
| 셀당 상한 | 벽시계 2400 s (`timeout -k 30 2400`). 첫 셀 하강 구간에서 벽시계/sim-초를 재어 초과 예상이면 ask | 지시서 |
| heightmap | `roarm_rl.heightmap.heightmap_from_particles`, 5 mm 셀, 전개된 구(중심 + R(q)·offset, 구별 반경) | 지시서 |

## 게이트

| # | 게이트 | PASS 조건 | 비고 |
|---|---|---|---|
| G1 | 발산 0 | 3셀 모두 `diverged == false`, DEME abort/예외 0 | `pops.steps_over_pop_speed`·`v_particle_max` 는 보고값 |
| G2 | 회귀 PASS | 구 더미(`pile_practical_fast_d4p16_n18796_seed460.npz`, sha16 `e26b214f336a4e97`) + W3b `params_fixnorm_plunge25.json` 그대로 seed 460 재실행 → `capture.n_in_cavity == 315` (W3b `scoop_fixnorm/scoop_s1_seed460.json`) | 엄격 동일. DEME GPU 는 최종 상태 bit-동일을 보장하지 않으므로(W7 §7-7) 다르면 FAIL 로 적고 Δ개수·Δ질량·문 정지 위상 일치 여부를 함께 보고한다(완화 금지) |
| G3 | 3셀 완주 | 셀 3개 모두 rc 0 · 상한 2400 s 안 · 결과 JSON 존재 · `steps_actual` 에 settle/descend/close/lift 전부 > 0 | reclose 0 은 허용(문이 끝까지 닫힌 경우) |
| G4 | 포획 질량 > 0 | 3셀 모두 `capture.mass_g > 0` | 충전율(공동 45.7 cm³ × 벌크 0.55 대비)은 보고값 |
| G5 | 절단면 각 보고 | 3셀 × 4방위 각각 값 또는 None(사유) 이 JSON 에 있음 | **판정 아님**. 정의 = 코드 `crater_angles()` 도크스트링 (아래 요약) |
| R | D341 Rerun 1셀 | SDK/CLI 0.34.1 핀 · footer `rrd verify` PASS · 엔티티/타임라인/컴포넌트 정확 계약 PASS · 고정 blueprint `.rbl` · 헤드리스 스크린샷 · 실제 육안 검수 기록(JSON) | 완결 항목. 실패해도 과학 판정은 안 바꾼다 |
| S | D470 source sha | 읽은 파일(더미 npz·STL·design.json·스크립트·params) sha 가 결과 JSON `inputs_sha16` + `gates_w8.json` 에 있음 | |

`all_pass` = G1 ∧ G2 ∧ G3 ∧ G4 (G5·R·S 는 완결/보고 항목).

## 절단면 각 정의 (사전 고정 — 코드 `crater_angles()` 가 정본)

- dh = h_pre − h_post (m, > 0 = 깎여 나간 깊이). h_pre = 재안착(settle 25스텝) 직후 하강 전, h_post = 최종 프레임의 남은 입자(딸려 올라간 것 제외).
- 구덩이 중심 = 퍼내기 위치 반경 80 mm 안에서 dh ≥ 2 mm 인 셀의 dh 가중 무게중심(없으면 퍼내기 위치).
- 4 방위 = 세계 +x, +y, −x, −y. 방위마다 ±22.5° 쐐기 안 · r ≤ 80 mm 셀을 5 mm 고리로 묶어 고리 평균 dh(r), h_post(r).
- r_peak = dh(r) 최대 고리. 그 바깥쪽으로 dh 가 0.2·d_max 아래로 처음 떨어지기 전까지의 고리 중 0.2·d_max ≤ dh ≤ 0.8·d_max 인 고리에 최소제곱 직선. **옆면 각 = atan(|기울기|)** (수평 기준). 같은 고리에 h_post(r) 직선도 맞춰 절대 표면 경사각을 보조로 낸다.
- 고리 2개 미만이면 None + 사유. 벽(±155/±110 mm) 인접 방위는 그대로 보고하되 R1 §5 "벽 마찰로 더 가파름" 을 주석한다.
