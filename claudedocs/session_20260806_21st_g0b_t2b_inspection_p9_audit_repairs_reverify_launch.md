# 2026-08-06 (21st) — G0b: T2b 육안검수 완료 + p9 감사 confirmed 10건 전건 수리 + 사전비행 PASS + 재검증 워크플로우 발사 (context 비상 종료)

이번 case의 신규 변수: [기존 ①②(19th) 범위 내 — 신규 변수 축 없음. 본 세션은 ②(T3)의
수리·검증 단계만 수행]

Case: `g0b_d420` 계속. 로봇 HW 0 · lerobot-train 0 · git commit/push 0 · **Isaac 기동 0**
(p9는 수리·컴파일·numpy 사전비행만, 실행 전). 승인 신규 0 (기존 "T2/T3 진행 승인" 범위 내).
**세션 진행 규칙 이행 주석**: 본 세션의 실패 가능 검사 = numpy 사전비행(자기검증 게이트,
FAIL 가능형). 주 실험(T3 Isaac)은 "감사 미반영/미검증 p9 실행 금지"(D422 Impl ②)와
재검증 워크플로우 미회수 + context 비상이 겹쳐 **의도적으로 연기** — 규칙상 정당화 명기.

## §1 부트 + 감사 전문 정독

- 부트 프로토콜 완주(START_HERE 20th판 → D422~D419 → 20th doc → git status).
- `g0b_d420/p9_audit_wf_78b1adfd_findings_raw.json`(82.5KB) 전문 정독: **confirmed 10 /
  refuted 0 / minor 8** — confirmed 10 = 고유 결함 7종(LIFT follow FATAL이 3렌즈 중복,
  LATCH stall MAJOR가 2렌즈 중복·1건은 verify에서 MINOR 하향). :150 이후는 워크플로우
  진행 메타데이터(findings 아님).

## §2 T2b 육안검수 (D341 이월 의무 이행 → **T2b 완전 종결**)

- 대상: `g0b_d420/t2b_ik_reachability_inspection.png` (8.0MB, 2400×1400 헤드리스 캡처).
- **관찰 기록**: ① 헤더 패널 verdict **T2B_PASS**, z_offset +0.012117(descend 0.050500 /
  approach 0.090500), self-check 23.019°∈(21.7,24.4) + tcp_z 0.013521∈(0.013486,0.013628),
  grid URDF 264/513·v6clip 250 — **stdout log(t2b_ik_stdout.log:2-23)와 수치 전건 일치**.
  ② 3D 뷰: 베이스 근측 green(≤5°) 밴드 → 외곽 yellow→orange→red 단조 전이 = **annulus
  위상 시각 확인**(T2와 동형). PASS 4후보(S1 0.3658/S2 0.2081/R1c 0.5249/R2c 0.4204)
  전부 green 내부, FAIL 4후보(S4 33.5578/S3 20.1709/R3c 20.1165/R4c 16.8594) 전부 외곽
  red/orange. best 해(seed0_S2) 위치 orange ring + 로봇 체인 스켈레톤 마커 존재.
  **수치-시각 불일치 0.** ③ 사소 1건: 뷰 탭 라벨이 blueprint 템플릿의 "T2 verdict +
  gates"로 잔존(내용 텍스트는 T2B 정확) — 표기 전용, 계약 저촉 없음.
- 교차검증: `t2b_ik_results.json` 직접 파싱 — verdict/게이트/self_check/named 8종
  pass·tilt 전부 stdout·PNG·START_HERE 인용값과 일치. 통과 셀 목록상 외곽 경계는
  y≈0에서 x≈0.32 (seed0_S1 r≈0.290 내부 여유).
- **판정**: T2b는 이제 "검증기 PASS + 육안검수 완료" — D341 계약 전 항목 이행, 이월 0.

## §3 p9 수리 (감사 confirmed 10 + minor 2 반영, 1,082→1,605줄)

| 감사 항목 | 수리 구현 |
|---|---|
| **FATAL ① lift follow 구간별 측정** | `run_resampled_path`가 phase 시작 물체 z(`phase_start_obj_z`)를 앵커, 매 waypoint의 `run_to_q` 반환 직후 `object_follow_delta_m`을 **phase-누적값으로 덮어씀**(early-kill/중단 반환 경로 포함). verdict/aggregate/JSON이 누적값 소비 |
| **FATAL ② transit 수직 게이트** | `_solve_q_vertical(require_tilt)` + `run_resampled_path(vertical_scope)` — approach는 `"arrival"`(transit=위치만, **도착점만 pos+tilt**), descend/lift는 `"all"`. plan 3타깃 REACH 게이트는 require_tilt=True 유지. waypoint 로그에 `vertical_gate=` 필드 추가 |
| MAJOR ⓐ stall vs 0.75° | stall 검출: \|Δq5\|<`--gripper_stall_rate_deg_per_step`(0.02) **`--gripper_stall_min_steps`(5)연속** ∧ err>gate → reached 인정 + `gripper_stalled` 기록(StepResult/close_records/JSON/이벤트 로그 전파) |
| MAJOR ⓑ marker 절단 | p7 `--continue_close_after_grasped_until_angles_done` 복원(BooleanOptionalAction, **기본 ON** — p7 기본 False에서 의도적 상향) → 전 밴드 완주로 close_records 밴드 증거 생성 + HOLD 41.4→24° 점프 해소(마지막 close=24=q_hold). latch phase 종료 시 `grasped_seen`을 close_records **phase-any로 집계**(p7 등가 의미; 물리 증거는 여전히 LIFT follow) |
| MAJOR ⓒ episode 예산 | 기본 60s(=6000 step) — 최악 3615(=30+44×60+5×60+11×45+30+2×60) 대비 여유. verdict 직후 `episode_truncated=YES` 경고 라인 추가 |
| MAJOR ⓓ 산출 경로 충돌 | `--tag`(기본 `t3_grasp`, `[A-Za-z0-9_]+` 검증) → 6개 산출 경로 tag-유도, **어느 하나라도 기존 존재 시 즉시 abort(return 3, AppLauncher 전)**. 기본 tag = 사전등록된 t3_grasp_* 이름과 동일(attempt1 정본), 감도 leg는 별도 tag |
| MAJOR ⓔ 조기 return 행 | 모듈 `_CLEANUP` + `_close_all()`(idempotent) — version mismatch/유효 USD 가드/USD_AUDIT_FAIL/정상 종료 전부 경유 + `__main__` try/finally 안전망(예외 시에도 Kit close) |
| MINOR vacuous 가드 | 동일 문자열 검사 삭제 → `cfg = RoArmStackEnvCfg()` 직후 **`cfg.robot.spawn.usd_path` 유효값 검증**(attempt3 일치 + /NHNHOME 부재, 불일치 시 USD_GUARD_FAIL) — env:97-100/149 소비 경로 실측 근거 |
| MINOR JSON Infinity | `_finite_or_none`로 aggregate 6종 sanitize(빈 results→null), RFC 8259 정합 |
| MINOR docstring | REACH/APPROACH 매핑·수직 게이트 범위·수리 내역 docstring 정합화. descend→APPROACH_FAIL 표기는 D-8 승계 유지(③ prereg 등재 예정) |

- py_compile OK. **잠정 sha `1e7f1907b98794028a11016d491442ab2804fd71e10655f60cf70b31d792fa31`**
  — **"재검증 미회수" 층위. ③ prereg 핀 금지**(재검증 생존 이슈 반영 시 변경 가능).

## §4 numpy 사전비행 (실패 가능 검사 — PASS)

- 스크립트: `g0b_d420/t3_preflight_ik_chain.py`(영속 사본, sha `471a6d0a…6a07`) —
  수정판 p9 모듈을 직접 import, `run_resampled_path`의 IK 체인(chain-seed, per-waypoint
  게이팅)을 정확 재현. isaaclab env python 실행.
- 결과: plan 3타깃 **양 높이(settled 0.025 / planned 0.012883) 전부 수직 IK OK**
  (settled tilt 1.300/0.718/0.199°). **approach 44wp**: transit 최악 tilt 82.24°(위치
  수용 — 원 감사 preflight wp001 82.24°와 소수점 일치 = 솔버 동작 재현 검증), **도착점
  pe 0.016mm/tilt 0.200° 수직 PASS**. descend 5wp·lift 2wp 전 waypoint 수직 PASS.
  최악 예산 3615 < 6000. → FATAL ②의 "매 실행 APPROACH_FAIL" 구조 해소 수치 확인.
- 실행 환경 점검: 4090 여유(879MiB/16.4GB), 디스크 56G, `g0b_d420/` t3_grasp_* 0건,
  isaaclab pin 무결(rerun 0.34.1 / numpy 1.26.0 / psutil 5.9.8 — D326).

## §5 적대 재검증 워크플로우 (발사 → **미수령**)

- **`wf_3cea04db-7c2`** — 9 agents: 수리 7건(고유 결함별) 각각 "수리가 정말 해소했는지
  반박 + 수리의 영향 반경에서 신규 결함 탐색" + 회귀 사냥 2렌즈(런타임 버그 / 계약 정합).
  스키마 강제(repair_effective / residual_or_new_issues).
- journal: `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/c22d8714-8587-4180-baed-c36e317a5e99/subagents/workflows/wf_3cea04db-7c2/journal.jsonl`
  (agent 전사 8+개 동폴더). 스크립트: 동 세션 `workflows/scripts/p9-repair-adversarial-verify-wf_3cea04db-7c2.js`.
- 세션 종료 시점 상태: agent 파일 12:05~12:07 갱신 중 = **실행 중**. 17th 교훈 적용 —
  **다음 부트에서 journal mtime 확인·회수가 재발사보다 우선**. 필요 시
  `Workflow({scriptPath, resumeFromRunId: "wf_3cea04db-7c2"})` resume 가능.

## §6 산출물

- p9 수정판(1,605줄, 잠정 sha 위), `g0b_d420/t3_preflight_ik_chain.py`(신규),
  본 doc, START_HERE 전면 갱신, LEDGER 1행, DECISIONS D423, MEMORY 21st 엔트리
  (+16th/17th 회전). ③ prereg는 **미발행**(sha 미확정 상태 발행 금지).

## §7 다음 세션 (순서 고정)

1. **`wf_3cea04db-7c2` 회수** (journal/agent 전사; 재발사 금지) → 생존 이슈 FATAL→MAJOR→
   MINOR 순 반영 → py_compile → 사전비행 재실행(`t3_preflight_ik_chain.py`) →
   **p9 최종 sha 확정**.
2. **③ prereg/hash/attestation 발행** — 최종 sha + 전체 CLI(기본 인자 = 스폰 seed0_S1,
   근거 D421/D422 + tag t3_grasp) + 게이트 전수 + D341 계약 + **감사 반영 내역 매핑** +
   supersession 2건 등재(D-4 스폰 (0.300,0)→seed0_S1 [T2 FAIL annulus 경계 r=0.30] /
   D-6 local_assets→attempt3 [D420-R1]) + p7-parity 의도적 델타 등재(descend_path_ok,
   approach 3-IK 전건, continue 기본 ON, stall-aware reached, grasped_seen phase-any,
   descend→APPROACH_FAIL 승계) + 접촉 화살표 생략 정당화(감사 MINOR — 사용자 확인 항목
   플래그) + 마찰 주 leg 0.40/0.30(감도 leg 별도 tag 별도 실행).
3. **T3 Isaac 실행** (승인 기수령; 실행 직전 tuple 요약 브리핑; stdout→
   `g0b_d420/t3_grasp_stdout.log`, stderr 별도 파일 — 2>&1 금지 규칙) →
   `*_FAIL` = 다음 수리 지점. 육안검수(D341)까지가 완료 조건.

## §8 [세션 말미 회수] 재검증 `wf_3cea04db-7c2` 완주 (10/10 agents, 에러 0, 1,301,532 tok, 1831.8s)

- 봉인 직후 완주 통지 도착(20th와 동일 패턴) — 세션 내 회수. 전문 repo 영속 사본 =
  `g0b_d420/p9_reverify_wf_3cea04db_findings_raw.json` (65.6KB, 구조 =
  {result:{repairVerdicts, notFixed, issues}}).
- **수리 유효성: 7/7 전건 repair_effective=true, notFixed 0** — confirmed 10건의 원 결함은
  전부 해소 확인 (FATAL ① 데이터 흐름 추적·FATAL ② 8구성 44~47wp 완주 수치 재현 포함).
- **신규/잔존 이슈 13건 = MAJOR 2종(고유) + MINOR 10 + regression 계약 표기 2**:
  - **MAJOR ① (실행 전 수리 필수)**: wp002 관절 재구성 46° 점프 — min-tilt 랭킹이 transit
    초입에서 원거리 반복해를 선택, 속도클램프 슬루 적분 시 스텝당 TCP 이동 최대
    seed0_S1=9.90mm(S2 9.60/R1c 9.71/R2c 9.37) vs `max_tcp_step_m` 10mm early_kill —
    마진 1~6%, PhysX 적분이 수 % 어긋나면 APPROACH_FAIL. **권고 스폰 seed0_S1이 최악.**
  - **MAJOR ② (동근원)**: transit 랭킹이 밴드(3mm) 내 최소-tilt 우선이라 wp003에서
    pe=2.52mm 명령 선택 → 실측 reached 3mm 게이트에 0.48mm 마진(중력 처짐 가산 시 칼끝).
  - 수리 방향(다음 세션): `require_tilt=False`일 때 랭킹을 **pos 우선**으로 전환(또는
    내부 밴드를 게이트보다 좁힘) → wp002 점프·wp003 잔차 동시 해소 기대 — 수정 후
    **사전비행 재실행으로 per-waypoint 관절 delta + pe + 도착 tilt 재검증 필수**
    (pos-first가 사전 회전을 줄여 도착점 재배향 부담이 커지는지 확인).
  - MINOR 10 요지: lift path_ok 미소비(기본값 무해) / target_error_gate 좁힘 시 내부 밴드
    3.0mm 불일치 / stall 채터 잔존 창(보수적) / latch 누적 drift 세그먼트 기준(사각 확대) /
    recording_id 고정(다중 leg 병합 혼동) / `_close_all` env.close 예외 시 sim_app 미도달 /
    NaN이 phases·close_records에 잔존 가능 / D-8 "episode 20s" 문구 vs 60s 3중 불일치
    (supersession 등재 필요) / docstring이 미존재 t3_prereg.md를 현재형 인용.
- **귀결**: D423 ①(회수 전 sha 핀 금지)이 정확히 적중 — 잠정 sha `1e7f1907…fa31`은 폐기
  예정. 다음 세션 = MAJOR 2종 수리(랭킹 전환) → 저렴한 MINOR 동시 수리(§8 목록 중
  _close_all 순서/lift path_ok/NaN/D-8 문구) → py_compile → 사전비행 재실행(관절 delta
  검사 추가) → **최종 sha 확정** → ③ prereg(나머지 MINOR 등재) → T3 Isaac.
