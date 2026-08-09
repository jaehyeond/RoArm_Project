# 2026-08-06 (29th) — G0b T3R: Gate-0 시각 메시 감사 스크립트 저작 + 실행 전 4-렌즈 적대검증 발사 + p9 분석 발사 (context 95% 비상 중단 — 실행 0)

이번 case의 신규 변수: [없음 — 저작·검증 발사 계층만. **Gate-0 스크립트는 저작만 되고
실행 0회**(실행 전 적대검증 게이트, 22nd p9 선례). Isaac 0, 자산 변경 0, 로봇 HW 0,
lerobot-train 0, git 0(사용자 직접 push 확인만). P·F 설계 확정 상태 유지, 실행 미개시.]

## §1 부트 + git 정합

1. Current-State Protocol 6단계 이행(28th판 기준). D426 발효 상태(확인 대기 0) 확인.
2. **git**: HEAD = `79df2b3` "8월 6일자 변경" — **사용자 직접 커밋·푸시**(세션 시작 전
   고지 수령). 23rd~28th분 전체 포함, working tree clean. START_HERE 28th판 "미커밋"
   목록은 이 커밋으로 전건 해소 — 본 세션 산출물만 신규 미커밋.

## §2 정찰 — 소스 신원 전건 검증 (Gate-0 착수 전제)

| 대상 | sha256 (선두) | 기대 핀 | 판정 |
|---|---|---|---|
| `local_assets/roarm_m3/urdf/meshes/gripper_link.stl` | `7946a374e24a2f46…` | 24th §4-4 핀 `7946a374` | ✅ |
| `local_assets/roarm_m3/urdf/meshes/link5.stl` | `1d63f374a78c1419…` | 24th §4-4 핀 `1d63f374` | ✅ |
| `gripper_left_link.stl` (죽은 자산) | `1dfb77228316baf6…` | URDF 전문 grep 비참조 확인 | ✅ 사용 금지 유지 |
| `sim_scripts/p9_…grasp_probe.py` | `99c99c65da75d5b7…` | D424 계보 최종 | ✅ |
| 23rd 감사 스크립트 | `bca4f898023f63f2…` | D425 핀 | ✅ |

- `mesh_audit/` 사본 2종 = local_assets 원본과 sha 동일(동일 파일).
- **URDF 사실 확정**(`local_assets/roarm_m3/urdf/roarm_m3.urdf`): ① link5/gripper_link
  visual origin = 항등, scale 0.001(STL 단위 = mm) ② gripper_link **collision은 별도
  파일**(`gripper_link_collision_g2a.stl`) — 시각≠충돌 소스 분리 실증 ③ joint
  `link5_to_gripper_link` origin xyz=(0, 0.018821, 0.052035)/rpy=(−90°,−90°,0), axis Z,
  limits [0, 90°](URDF:225-231) ④ TCP = link5 +z 0.115428 m(URDF:234-239, 23rd
  TCP_LOCAL 일치) ⑤ 양 STL 모두 binary(삼각형 13,698/14,092개) → 순정 numpy 파서로
  파싱 가능 = **신규 패키지 설치 0(D326 무저촉)**.
- **audit3 대조 앵커 재확인**(`t3_jaw_audit3_results.json`): collision
  `assembly_max_depth_mm = 4.4576` **전 14각 상수**(플러그 지배) / `min_r_mm_in_rim_band
  = None` **전 14각**(rim band 5–15mm 완전 공백) — Gate-0의 비교 기준선.

## §3 Gate-0 스크립트 저작 (미실행 리비전)

- 파일: `sim_scripts/g0b_t3r_gate0_visual_mesh_distal_depth_readonly_audit.py`
  (706줄, **sha256 `81259edc666b6d7c4586ad10d6cfb3ecfb09025d7771d592616ca047e992c9ba`**
  — 패널 수리 전 리비전, 수리 시 sha 갱신·계보 기록 의무)
- **사전 고정 게이트(실행·프로파일링 전 확정 — no peeking)**: L_MIN=5.5mm(24th §4-1
  관계식 x=L−m, m=5.5 선례, T1 정합 x∈(0,12] ⇒ L∈(5.5,17.5]) / indeterminate 반폭
  ±0.5mm(24th 허용오차 규칙) / 손가락 탐색창 r≤30mm(tool axis 기준) / wall annulus
  12.5–20mm(보고 전용) / 샘플링 0.5mm 피치 / 스윕 88.31→0° 0.25° + 피크 ±1° 0.02° 정밀.
- verdict 4클래스: `GATE0_SOURCE_PRESENT`(양 body PASS) / `GATE0_INDETERMINATE` /
  `GATE0_SOURCE_ABSENT`(어느 한 body FAIL — 분기는 수제 저작 승인 or 정지·재상의
  **둘뿐**, C 소멸[D426 ①], 사용자 재질의 의무) / `GATE0_ABORT`(신원·프레임·계약).
- 설계 결정: ① STL은 **비-hull 원 삼각형 표면 샘플링**(시각 메시는 비볼록 — 23rd의
  ConvexHull 경로 비적용) ② 이동 조 배치 프레임 = **USD joint(audit3 앵커)** + URDF
  origin과의 witness 게이트(θ∈{0,45,88.31}, tol 1e-5) ③ 단일 timeline `sweep_index`
  (23rd audit3 TextLog 교훈 반영) ④ 뷰 데시메이션 1/16 기본(28th 가속 ⑤) ⑤ results
  JSON 네이티브 타입 전수(D425 ④ — default 없는 json.dumps로 위반 시 즉시 크래시)
  ⑥ 산출물 선존재 abort + tag `t3r_gate0*` 강제(동결 t3_*/t3r_* 보호) ⑦ 죽은 자산은
  sha 기록만·데이터 흐름 비진입 ⑧ collision 대조(audit3 수치 필드만 소비 — 문자열
  불리언 함정 회피).
- 산출 예정 tag: `t3r_gate0_vismesh` → results/bands.csv/rrd/rbl/png/validation 6종.

## §4 병렬 발사 2건 (28th 가속 ② 이행) — 양건 모두 **미회수** 상태로 세션 종료

1. **Gate-0 스크립트 실행 전 적대검증** `wf_f6c65ef4-b50` (4 lenses 병렬, schema
   강제 + 자기판정 의무: ① geometry-math[STL 파싱 바이트 배치·샘플링 커버리지·joint
   수학·witness 3각 충분성·스윕 격자] ② contract-compliance[read-only 전수·동결
   보호·D415 ③·D425 ④ 직렬화 크래시 경로·단일 실행 규율] ③ prereg-methodology
   [L_MIN 도출 타당성·false-PASS(비손가락 구조물 r≤30 오인)·false-FAIL(r>30 손가락
   누락)·verdict 매핑·실패 가능성·층위 규율] ④ rerun-d341[엔티티/타임라인/컴포넌트
   계약·anchor 인덱스 오프바이원·RRD 크기·blueprint·validate 시그니처]).
   - **봉인 직후 완주·세션 내 회수 완료**(26th "봉인 직후 완주" 패턴 4회째): 4/4 agents,
     에러 0, 525,420 tok, 835s. 전문 영속화 =
     `g0b_d420/t3r_gate0_script_review_wf_f6c65ef4_findings_raw.json`(53,386 B,
     sha256 `c26fa4322d547bff725ce80a622317bae21ed4374f99baab82641606b468c6ab`).
   - **판정: FATAL 0 / MAJOR 3 / MINOR 11 / INFO 9** — Lens1(geometry) SCRIPT_OK ·
     Lens2(compliance) SCRIPT_OK · Lens3(prereg-methodology) NEEDS_CHANGES(MAJOR 2) ·
     Lens4(rerun-d341) NEEDS_CHANGES(MAJOR 1). Lens1 실증 하이라이트: STL 파서
     바이트 정확성 독립 재구현 대조 max|diff|=0(양 메시 전건), 정규 row 낙하 = normal
     확인, audit3 소비 필드 native float 확인, validate 시그니처 정합 확인.
   - **MAJOR 3건 수리 지시(전부 실행 전 합법 사전등록 수정 — 게이트 불변)**:
     ① [L4:616] 이동 조 3D 포즈가 anchor 6각에만 기록 → 판정 주체(피크 각)가 RRD에
     없을 수 있음(D341 "decision subject" 위반 소지). 수리 = `anchor_ids =
     set(anchor_idx.values()) | {k}`(coarse argmax 인덱스 포함) 후 그 집합으로 로깅.
     ② [L3:481] moving_peak에 판별 지표 부재(r_at_l_vis·annulus/footprint 분해·피크
     각 bands 행 없음 — 고정 조와 비대칭). 수리 = 정밀 피크 q5에서 body_metrics 재계산
     → results `moving_gripper_link.peak_metrics` + bands CSV `peak_<q5>` 행 + 피크점
     방위각(atan2)·body-local 좌표 기록.
     ③ [L3:245] r≤30 창이 비손가락 축상 구조물(플러그 유사 boss)·개방각 전용 피크로
     허위 PRESENT 가능. 수리 = 게이트 불변 + 사전등록 2차 보고 지표: `l_vis_wall_mm`
     (12.5≤r≤20 한정), 피크 방위각, `l_vis_grasp_range_mm`(이동 조 q5∈[0,45] 한정) +
     results JSON에 해석 규칙 명문화(PRESENT인데 wall-annulus 공백 시 육안검수에서
     소스 위치 명시 확인 의무).
   - MINOR 주요(수리 재량, 다음 세션 판단): 단일 피크 refine의 이웃 극대 ~0.2mm 과소
     가능(경계 인접 시 해석 병기) / 빈 창 -inf → FAIL 오분류·JSON -Infinity·None
     포맷 크래시 경로(abort 가드 권고) / GV4가 URDF `<axis>` 원소 미검증(+z 가정
     명시 게이트 권고, URDF sha 기록 권고) / bands CSV 선기록 순서(부분 산출물 시
     수동 정리 필요 — t3r_gate0_vismesh_* 한정 삭제 후 재실행).
2. **p9 하드코딩 전수 분석** (read-only 백그라운드 에이전트) — **세션 내 완주·회수·
   영속화 완료**: `g0b_d420/t3r_p9_parameterization_analysis_readonly_report.md`
   (276줄, sha256 `a95086508ebf6d047444e35c63a43e6e35b56daa782b9cea90e98f2943d40008`).
   핵심: ① p9 sha 재검증 PASS(`99c99c65…`) ② 자산-신원 결합 지점 **26 행동 + 10 문서**
   — 알려진 3구간(L138-151/612-631/815-832) 외 banner L863·유효경로 게이트 L901-904·
   스테이지 감사 호출 L952-957·summary_md L1529-1530·results JSON L1632-1636·물리
   레이어 상대 서브경로(L144 내포) ③ `EXPECTED_PART_COUNT=64`는 양 body 공통 단일
   상수, 리터럴 "64"가 JSON **키 이름** `part_count_64`에 소성(L622→953→1636)
   ④ `d338_convex_parts`는 p9에 부재 — 파트 판정은 leaf-only `"part_" in leaf`(L617)
   → **mode-B(증분 자산)는 leaf 판정 전 네임스페이스 분리 필수** ⑤ 제안 CLI 11종
   (core 6 + mode-B 4 + per-body 옵션 1), 전 게이트 `==` 유지·mode-B는 게이트 추가만
   (orig==64 ∧ new==선언값 ∧ no-stray ∧ 네임스페이스 disjoint) ⑥ 리스크: sha 핀 skip
   경로 금지(fail-closed) / **순서 불변식 sha게이트→env-var→roarm_rl import**
   (`roarm_stack_env.py:96-99`가 import 시 `ROARM_M3_USD_PATH` 스냅샷) / summary_md가
   측정 sha 아닌 핀 상수 출력 중 / provenance `asset_params` 블록 vs 바이트 동일성
   긴장 → artifact V1→V2 승격은 prereg 결정 사항 / L686/695 "Attempt3" 주석은 run
   번호(자산 아님 — 개명 금지).

## §5 규칙 이행

- **실패 가능 실험 justification**(session progress rule): 본 세션의 실패 가능 실험 =
  Gate-0 단일 실행으로 설계·저작까지 완료했으나, ① 실행 전 적대검증 게이트(22nd p9
  선례 — 저작물 전건 D423 강도 검증 후 실행, 23rd leg 소모 재발 방지) + ② 실행 직전
  stop-hook context 103% → **AGENTS.md 95% 비상 프로토콜 발동(신규 구현·실행 중단
  조항)**으로 실행이 다음 세션 서두로 이월됨. 발사된 패널 자체도 실패 가능 검증
  (NEEDS_CHANGES/FATAL 적발 시 수리 강제). NO_PPO_PROMOTION 해당 없음(물리 verdict
  없는 저작 세션).
- **/half-clone 거부 19회째(#11 — 18회=stop-hook 103%, 19회=세션 봉인 후 stop-hook
  152% 재지시. 양회 모두 end-of-session update + continuation prompt로 대체).**
  HANDOFF 미생성(#7). DECISIONS append 없음
  — 사유: 실행/verdict 0, durable lesson 신규 없음(D426 발효는 28th 기록).

## §6 산출물

- `sim_scripts/g0b_t3r_gate0_visual_mesh_distal_depth_readonly_audit.py`(신규, 미실행)
- `g0b_d420/t3r_p9_parameterization_analysis_readonly_report.md`(신규, sha `a9508650…4008`)
- `g0b_d420/t3r_gate0_script_review_wf_f6c65ef4_findings_raw.json`(신규, sha `c26fa432…c6ab`)
- 본 doc / START_HERE 29th판 / LEDGER 29th row / MEMORY 29th entry(23rd verbatim 회전 +
  24th/26th/27th/28th archive verbatim 보존 후 압축 인덱스화 — hook 용량 요구, #8 준수)
- **미회수 0** — 패널·p9 분석 양건 세션 내 회수 완료.

## §7 다음 부트 순서 (고정 — 패널 회수 완료로 1단계 소멸)

1. ~~패널 회수~~ **완료(§4-1)**. MAJOR 3 수리 적용(①anchor_ids∪{k} ②peak_metrics+
   bands peak 행 ③2차 보고 지표 3종+해석 규칙) + MINOR 재량 판단(빈 창 abort 가드
   권고 채택 여부) → Gate-0 스크립트 sha 갱신(계보: `81259edc…` → 수리판) →
   수리 diff 자체검증(22nd 라운드-3 "기계 수리" 선례 — 전면 재패널 필요 여부 판단
   기록 의무: 3건 모두 게이트·판정 로직 무접촉 additive면 생략 정당화 가능)
2. (병합됨 — 1에 포함)
3. **Gate-0 단일 실행**(isaaclab env python, 산출물 6종) → 결과 JSON 판독
4. D341 육안검수(PNG 실제 열람 + 관찰 기록) — validation PASS ≠ 검수(D425 ①)
5. verdict 분기: PRESENT → p9 파라미터화+게이트 v2 저작 착수 / ABSENT →
   **정지·사용자 재질의**(수제 저작 승인 or 정지·재상의 둘뿐) / INDETERMINATE →
   측정 한계 보고 후 사용자 재질의
6. 게이트 v2(파트-제외 프로파일·밴드 확장·수평 간극·자기-관통 q5∈[23°,90°]) +
   p9 파라미터화 저작(§4-2 영속 보고서 = 설계 입력) → D423 동일 강도 적대검증 → sha 핀
7. 별건: 25th scratchpad 118MB 처분 지시 대기 불변.
