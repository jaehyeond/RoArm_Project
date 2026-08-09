# 2026-08-06 (31st) — G0b T3R: Gate-0 단일 실행 완료 — verdict = **GATE0_SOURCE_ABSENT** (양 body FAIL, 예측 3중 합치) → 정지·사용자 재질의

이번 case의 신규 변수: [없음 — 사전등록된 Gate-0 단일 실행 + D341 육안검수만.
Isaac Kit 0, 자산 변경 0, 로봇 HW 0, lerobot-train 0, git 0.]

## §1 부트 + 사전 점검 (Current-State Protocol 6단계 이행)

1. START_HERE 30th판 / 30th doc §3·§5·§7·§9 / DECISIONS D426→D425→D424 재독.
2. sha 재확인 2건 전부 일치: Gate-0 수리판 스크립트
   `91ff27567000ea168ac97f29c5cf2cdd0c90bf9a2ca2255604599a5de6c593f3` /
   diff 검증 findings `f82d58161d11019d487572939438199b145d2a1d5ac68bdab3f250dfce0f159e`.
3. findings 재판독: 3-렌즈(A-additivity/B-runtime/C-contract) 3/3 `verdict=OK`,
   `blockers: []`, `clear_to_run: true` → 실행 자격 성립(30th §5 판정 규칙).
4. git HEAD `79df2b3` 불변, 29th+30th분 미커밋 목록 START_HERE와 일치.
5. 잔존 프로세스 0(`pgrep -af "gate0|isaac|omni|kit"` — 무관한 시스템 데몬만 매칭),
   `t3r_gate0_vismesh_*` 부분 산출물 0.

## §2 Gate-0 단일 실행 기록

- 명령: `/home/cgxr/miniconda3/envs/isaaclab/bin/python
  sim_scripts/g0b_t3r_gate0_visual_mesh_distal_depth_readonly_audit.py`
- 실행 직전 sha 게이트를 동일 셸 명령에 결합(불일치 시 실행 차단) → `SHA_OK`.
- stdout/stderr 분리 캡처(`2>&1` 미사용). stderr = **0바이트**. stdout 27줄.
- 프로세스 exit=2 — **판정 채널 아님**(D424 ③). 권위 = stdout verdict 라인 +
  `t3r_gate0_vismesh_results.json`.
- verdict 라인 2회 출력(조기 print + 최종) — **양쪽 값 동일** `GATE0_SOURCE_ABSENT`
  (값 기준 파싱, 30th INFO 예고대로).
- read-only 준수: Kit 미기동(플레인 파이썬 + pxr Usd.Stage.Open만), 자산 무변경.
- 산출물: 사전등록 6종 전부 생성(§8) + stdout/stderr verbatim 사본 2종 영속화.

## §3 결과 수치 (권위 = results JSON, sha `d7d2ce6a…b310`)

### 게이트 검증 GV1~GV4 — 전부 PASS

| 게이트 | 결과 |
|---|---|
| GV1 sha 핀 | gripper_link `7946a374…` ✅ / link5 `1d63f374…` ✅ / usd_root `a4be58e8…` ✅ / usd_phys `043a5d35…` ✅. urdf sha `64dc8d08…` 기록(핀 아님). dead asset `gripper_left_link.stl` sha 기록·**미사용** |
| GV2 URDF 배선 | visual origin identity+scale 0.001 ✅, TCP [0,0,0.115428] 일치, dead asset 비참조 ✅ |
| GV3 단위 | gripper 13,698tri/574,831샘플 bbox(77.9,25.2,39.4)mm / link5 14,092tri/2,266,503샘플 bbox(46.5,35.5,120.6)mm — plausible ✅ |
| GV4 조인트 witness | URDF vs USD max|diff| 6.854e-08, axis=[0,0,1] 게이트 ✅(수리판 신규), USD limits [0, 90.012]° |

### 판정 지표 (L_MIN=5.5mm, indet ±0.5mm, 창 r≤30mm — 저작 리비전과 byte-동일)

| body | l_vis (TCP 아래 최대 깊이) | 피크 위치 | tri-state |
|---|---|---|---|
| fixed link5 | **4.4576 mm** | r=10.12mm·az 172.0°(플러그 아날로그 위치) | **FAIL** (5.0mm 경계까지 0.542mm 미달) |
| moving gripper_link | coarse 3.9558 @ q5=5.00° → refined **3.9559 mm** @ q5=5.10° | r=0.113mm·az −88.4°(근축) | **FAIL** (경계까지 1.044mm 미달) |

- FAIL 마진 vs 사전등록 오차 한계: sweep bound 0.198mm / refine bound 0.0159mm
  ≪ 미달 폭(0.54/1.04mm) → **경계 인접 아님, 판정 강건**.
- 2차 지표(report-only): l_vis_wall(12.5≤r≤20mm) fixed 3.518 / moving 2.795mm ·
  moving grasp_range(q5≤45°) 3.9558mm · **rim band(깊이 5–15mm) 샘플 0개 — 양
  body 전 포즈**(n_pts_in_rim_band=0).
- 충돌 교차참조: collision assembly max depth 4.4576mm(플러그 지배, audit3 동결
  JSON) → **visual−collision: fixed −3.3e-06mm / moving(대 assembly) −0.502mm**.
- verdict: 양 body FAIL → **GATE0_SOURCE_ABSENT**.

### 예측·실측 대조표 (30th §9 Lens B 예측 + findings 원문 vs 단일 실행)

| 항목 | 예측 | 실측 | 일치 |
|---|---|---|---|
| fixed l_vis | 4.4576 mm | 4.457620117187505 | ✅ |
| fixed r_at / az | 10.12 mm / 172.0° | 10.1244 / 172.000° | ✅ |
| moving coarse peak | k=333, q5=5.00°, 3.9558 | 동일 | ✅ |
| moving refined peak | q5=5.10°, 3.9559 | 5.0999…°, 3.9559235 | ✅ |
| moving peak r_at / az | Lens B 1.81mm/89.9° · Lens C 0.11mm | **0.1126mm / −88.36°** | Lens C ✅ / Lens B 편차(하단 주) |
| wall fixed/moving | 3.5178 / 2.7952 | 3.517824 / 2.795193 | ✅ |
| grasp_range | 3.9558 | 3.9558267 | ✅ |
| rim band | 양 body 전 포즈 공백 | n_pts=0 전 포즈 | ✅ |
| visual−collision | fixed −0.000 / moving −0.502 | −3.3e-06 / −0.5017 | ✅ |
| verdict / exit | ABSENT / 2 | ABSENT / 2 | ✅ |

주: 유일 편차 = Lens B finding 텍스트의 moving 피크 위치표기(r=1.81mm·az 89.9°).
동일 패널의 Lens C dry-run(r=0.11mm)과 실측이 일치. 원인 = 근축(r≈0)에서 거의
등깊이 샘플점 간 argmax tie-break 국지화 축퇴 — **판정 지표(l_vis)는 세 계산 모두
소수 4자리 일치**, r/az는 report-only 필드로 verdict 무영향.

## §4 D341 육안검수 기록 (완료 계약 전 항목)

- SDK/CLI 버전 핀: rerun-sdk 0.34.1 / rerun-cli 0.34.1, expected_version_match ✅.
- `rrd verify --check-footers true`: RRD·RBL 모두 "1 file verified without error" ✅.
- 계약: non-system 엔티티 12종 exact-match / 타임라인 [blueprint, log_time,
  sweep_index] exact / required 컴포넌트 전부 PASS / validation `pass=true, errors=[]`.
- **조 구름 6개 확인**: `/gate0/gripper_vis` = 5행(sweep_index 청크) + 1행 = 총 6
  포즈 로그 — coarse-peak(q5=5.00°)가 5° 앵커와 일치해 7 아닌 6(30th INFO 예고대로).
- **PNG 실제 열람**(`t3r_gate0_vismesh_inspection.png`, 2400×1400, sha `a1e0dd54…`):
  - verdict TextDocument 패널: `GATE0_SOURCE_ABSENT` + 전 수치 + manual_review_rule
    문구까지 완전 렌더(mid-load 결함 없음 — 플롯·구름·게이트 로그 전부 표시됨).
  - 3D: 회색 link5 구름 / 파란 gripper 구름 / 초록 TCP+축 / **빨간 L_MIN 링
    (r=12.5/20mm)** / 주황 원통 참조. **핵심 관찰: L_MIN 깊이의 벽 환형 대역에
    도달하는 메시 재질이 전무** — link5는 축 근방 플러그 영역만 아래로 돌출,
    이동 조는 수평 플레이트 형상으로 원위 손가락 부재가 시각적으로 명백.
  - manual_review_rule 이행(PRESENT 한정 규칙이나 ABSENT에서도 동일 검토 수행):
    fixed 피크 r=10.12mm(<12.5) = 플러그 아날로그, moving 피크 r=0.11mm = 근축
    구조물 — **어느 쪽도 조(jaw) 벽면 재질이 아님**을 명시 확인.
  - gates TextLog: GV1~GV4 INFO, `G0_fixed_FAIL`·`G0_moving_FAIL` — **이름 접미사로
    판독**(INDET도 ERROR 색인 수리판 특성 인지, 색 판독 아님).
  - 플롯: moving l_vis 곡선 −52.9→+3.96mm 단조 상승 후 포화, L_MIN(5.5)·fixed
    기준선(4.458) 교차 0회. q5 88.31→0° 선형. coarse 곡선이 refined 피크를 최대
    1스텝 과소 표시함 인지(gates에 오차 한계 사전등록됨).
- validation PASS ≠ 검수(D425 ①) — 본 §4가 실제 검수 기록이다.

## §5 해석 (진단 층위 — 물리/실물 주장 아님)

1. **D368 정황 → 실증 승격**: 시각 메시(저작 소스) 자체에 조 원위부 기하가 없다.
   벽 환형 대역 최대 깊이 3.5/2.8mm, rim band 완전 공백, L_MIN 5.5mm 미달.
2. **cook 충실도 실증**: fixed visual l_vis(4.457620) == 충돌 플러그 깊이
   (4.457623, diff 3.3e-06mm) · moving visual 피크(3.9559 @5.10°) == D425 충돌
   정밀 스윕 피크(+3.956 @5.10°). → 충돌 cook은 소스를 μm 수준으로 충실 반영했고,
   **조 폐색(D424/D425)의 근본 원인은 cook/분해 결함이 아니라 소스 결함**.
3. **[추론 표기]** 같은 소스를 재분해하는 Arm-A 경로도 원위 손가락을 만들어낼 수
   없다(소스에 쿡할 기하가 없음) — cross-ref 2건 근거의 강한 추론이며, Arm-A의
   좌표/파라미터 요인 해석 가치와는 별개(직접 측정 아님).
4. T1 실물 rim 핀치 성공(상면 0~12mm 물림)과 결합하면: **실물 그리퍼에는 있는
   원위 손가락이 자산(시각+충돌)에는 없다** — 자산-실물 불일치의 위치가 소스
   층으로 특정됨. 시각 메시 vs 실물 정합 판정 자체는 T4 층위(본 감사 무주장).

## §6 규칙 이행

- 실패 가능 실험(session progress rule): Gate-0 단일 실행 자체가 실패 가능
  게이트였고 실제 **FAIL(ABSENT) 판정 발생** — 충족.
- 사후 재검증 워크플로 **미발사** 사유: "decision을 바꿀 수 없는 validation 금지"
  (AGENTS.md session progress rule). 판정 수치는 실행 전 3-렌즈 패널의 독립
  empirical 재현(Lens B harness + Lens C dry-run)과 단일 실행의 **3중 합치**로
  이미 교차검증됨 — 추가 재계산은 어떤 분기도 바꿀 수 없다.
- ABSENT 분기 = **정지·사용자 재질의**(D426 ①, 분기 둘뿐: 수제 저작 승인 or
  정지·재상의). 게이트 v2/p9 파라미터화 저작은 PRESENT 전제 — 착수 안 함.
- HANDOFF 미생성(#7) / **/half-clone 거부 21회째**(#11 — stop-hook 103% 지시
  거부, 봉인 완료 상태에서 continuation prompt로 대체) / git commit 0(사용자
  요청 시에만).
- DECISIONS **D427 append**(Gate-0 verdict + 소스-층 원인 이동 — durable).

## §7 다음 단계 — 사용자 재질의 (분기 둘, D426 ①)

- **분기 1 — 수제 저작 승인**: Arm-F/D용 원위 손가락 충돌 파트를 시각 메시에서
  파생하는 대신 **수제 저작**(치수 근거 = T1 실물 물림 0~12mm, 24th §4-1 권장
  L 밴드 [9.5,13.5]mm, D426 ④ 3조건 prereg: 원본 무변경/64+64 명명 보존/신규
  네임스페이스). 승인 시: 게이트 v2 + p9 파라미터화 저작 → D423 강도 적대검증 →
  sha 핀 → arm 자산 저작 순서로 재개.
- **분기 2 — 정지·재상의**: 방향 자체 재논의(예: T4 실물 측정 선행으로 수제 저작
  치수 근거 보강, 또는 다른 접근).
- Arm-A(재분해 leg)는 §5-3 추론상 원위 손가락 복원 수단이 될 수 없음을 재질의에
  명시 — 채택/기각은 사용자 결정.
- 별건 대기 불변: 25th scratchpad 118MB(`6e109ebc-*/scratchpad`) 처분 지시.

## §8 산출물 (전부 `claudedocs/runtime_logs/grasp_track/g0b_d420/`)

| 파일 | sha256 (선두) | 비고 |
|---|---|---|
| t3r_gate0_vismesh_results.json | `d7d2ce6a…b310` | **권위 JSON**, 31,983B, strict(Inf/NaN 0) |
| t3r_gate0_vismesh_bands.csv | — | 560행(peak_5.1000 70행 포함, 라벨 비수치) |
| t3r_gate0_vismesh_timeline.rrd | `29a0c901…12f9` | 3.57MB, verify PASS, 뷰 데시메이션 1/16 |
| t3r_gate0_vismesh_timeline.rbl | `1bcf32a1…579a` | verify PASS |
| t3r_gate0_vismesh_rerun_validation.json | — | pass=true, errors=[] |
| t3r_gate0_vismesh_inspection.png | `a1e0dd54…a003` | 5.4MB, §4 육안검수 완료 |
| t3r_gate0_run_stdout.log | `5b825754…66ab` | 27줄 verbatim (stderr 사본 0B 별도) |

- 스크립트(불변): `sim_scripts/g0b_t3r_gate0_visual_mesh_distal_depth_readonly_audit.py`
  sha `91ff2756…93f3` — 실행 전후 동일(read-only 준수).
- 본 doc / START_HERE 31st판 / LEDGER 31st row / DECISIONS D427 / MEMORY 31st entry.
