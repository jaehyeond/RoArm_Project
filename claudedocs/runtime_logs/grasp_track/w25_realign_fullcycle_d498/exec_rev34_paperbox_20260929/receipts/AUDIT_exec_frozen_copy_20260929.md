# 독립 감사(independent-auditor, NumPy+표준 라이브러리, 읽기 전용) — exec_rev34_paperbox_20260929 동결본, 2026-09-29 06:0x

사전등록 A1~A9 **FAIL 0**. 워커(메인) 판정을 뒤집은 항목 0. 물리 산출 0 시점의 순수 파일/해시/스키마 감사(D341 RRD 면제).

| 항목 | 판정 | 근거(감사자 독자 재계산) |
|---|---|---|
| A1 핀 대조 | PASS | REVISION_PIN 54항목 일치 54/불일치 0/누락 0. 미등재 7개(n67737 증거·PB 영수증) 원본 대비 sha 동일. manifest 606 vs 디스크 606/606, vs tar 606/606, EXEC_PIN 77/77(sha+size) |
| A2 criteria diff | PASS(주의) | 값 변경 = `runner.physics_wall_cap_s` 32400→115200, `runner.sim_soft_wall_cap_s` 31200→114000 **2건**. 그 외 문구 4건(purpose/origin/scientific_limitation)·메타 3건(artifact·revision·derived_from 추가). 나머지 43 threshold value 불변, `no_threshold_change_after_outcomes` 원문 유지. derived_from.sha256 `71a7d239…` = rev34/criteria.json = rev32_frozen_copy/criteria.json(바이트 동일 검증) |
| A3 argv vs 템플릿 | PASS | 15토큰 중 6토큰 상이(경로 접두어·NPZ 채움·--out·numeric_evidence basename `_n67737`·--max-wall-s 31200→114000). podA↔podB 상이 토큰 1개(`--out`) |
| A4 cap 정합 | PASS | cap_s 115200 = physics_wall_cap_s · --max-wall-s 114000 = sim_soft_wall_cap_s = cap−grace · grace 1200 = graceful_grace_s · auto_retry false · timeout≠success |
| A5 NPZ sha | PASS | `31dd2897…`(34,475,600 B) = COMMANDS frozen_inputs. source_result_sha256 `5745cca5…` 일치. domain JSON 에 npz sha 필드 없음(N/A), `.pile` 경로 문자열 바이트 동일. NPZ 독립 판독 clump (67737,3), 구 474,159, z span 40.432 mm, xy span 307.5×217.5 mm |
| A6 러너 diff | PASS | run_w19(sha `0198decd…` 검증) → run_w25: −10/+14줄(--commands·artifact 이름·pod_tag·gpu_count). **실행 루프 53줄 바이트 동일** |
| A7 부트스트랩 diff | PASS(주의) | deme 휠 sha·정적 lib sha·pip 핀·conda 채널·검사 순서 동일. 경로 외 변경 1 = `nvidia-smi` 질의에 `index` 추가 |
| A8 꾸러미 | PASS | tar sha `c0137a2a…`·46,047,485 B 일치, 멤버 606 = manifest 606(집합 동등), 멤버 내용 sha 606/606, 외부 13/13 = W19 manifest, groups 74/13/1/489/30/2 = EXEC_PIN |
| A9 params 인용 | — | travel_cm 45.0 · 규약 A · open_at_surface true · chatter true · tray 310×220·230 · pellet 23.9 · dt 1e-6 · E 5e6 · 토크 1.96 |

## 문구·기록 결함(판정 불변, 정오표 — 핀 파일은 고치지 않는다)
1. `COPY_RECEIPT.json.rule` 산문 "56개…56/56" → 실제 54(기계 필드 `n_pinned=54` 가 맞음).
2. `EXEC_PIN.criteria_file` "…만 변경" → 값 2건 외 문구 4건·메타 3건 동반.
3. `bootstrap_pod_w25.sh` 도크스트링 "경로만" → `nvidia-smi` `index` 필드 추가 포함.
4. `EXEC_PIN.not_in_bundle` 에 `receipts/*`·`local_tools/*` 누락(핀 뒤 생성; 핀 77 파일은 sha+size 불변 확인).
5. `numeric_inputs_w25_paperbox_n67737.json.source_result_json` 이 orca worktree 절대경로(출처 메타 전용, 런타임 미참조 — 코드 0건, 실행 무관).

## 경계
물리 성패 판정 없음. 본 실행 뒤 별도 감사 항목: 회수 해시 전수·32 h 상한 도달/`abort_class`·정착 cadence(≥6프레임·≤0.05 s).
