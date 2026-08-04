# Session 2026-08-04 (3rd = 8th) — D409 warning 처분·정적 fixture PASS·attempt1 runtime 완주

이번 case의 신규 변수: [없음 — 3rd 세션 선언분 1개(실물 원통 기하) 불변]

## 0. 승인 / 규칙

- 사용자 승인 2단계 모두 획득: (i) 기존 "설계 착수" (tuple까지),
  (ii) **attempt1 runtime 명시 승인 — tuple sha `de79bc78…efcc9a60` 인용**
  ("승인할게", 본 세션). retry 0 준수 — 실행 정확히 1회.
- /half-clone 미사용 (stop-hook 98% 경고에도 HARD RULE #11 준수 —
  auto-memory 연속성 스냅샷으로 대체 후 계속).

## 1. Warning 6건 독립 재검증 → 처분 (전건 수용-기록, 소수리 0)

보존 verbatim (`g0a_d409/design_inputs/microrepair_diff_reverify_wf_311d5910-658/`,
result `77927381…d7ed31` / journal `f912919…dff5f612` — 본 세션 sha 재계산
bit-일치 후 전문 판독). 고유 이슈 4건:

| ID | 독립 재검증 (방법 → 결과) | 처분 |
|---|---|---|
| A-W1≡B-W2 | predicate 형상-기반 확인 (controller:1956-2093) + **scratchpad 재현**: deadline 4블록 전삭 + 죽은 decoy 1함수 → 표면 PASS (P3). 대조군 P3b(decoy 없음) → **FAIL** = M2 판별력 동시 실증 | 수용-기록 (완화 = `_validate_approval_tuple` :3549-3569 현재 bytes 재해시 — 코드 실재 확인) |
| A-W2 | writer:1427 Compare 형상(Name vs Attribute)이 두 predicate 불충족 분석 + **재현**: :1427-1428 단독삭제 → 9표면 전부 PASS (P2) | 수용-기록 (동일 sha-결박 완화) |
| A-W3≡B-W1 | `_write_json_x` dumps(:479)→open(:481) 순서 판독 + **재현**: NaN/Inf → ValueError·파일 미생성, finite 정상 (P1) | 수용-기록 (fail-closed 유지; 실발동 불능) |
| B-W3 | 순서 판독: env gate :2108 → exists-guard :2112 → prereg build :2118 → prereg write :2120 — 선기록 전 실패 = 재시도 가능 | 수용-기록 (설계 의미론 재확인) |

probe: scratchpad `reverify_probe/probe_warning_reverify.py` (repo 무변경).
M1 사실성(worker:755-756 GJK 1e-9 + rule 문장) 별도 재확인.

## 2. 정적 fixture 1회 — PASS_STOP

- 선확인: sha v2 3파일 bit-일치, ATTEMPT_ROOT 부재, git 형상 md5
  `4d01d3ea…` 기록. 세션 초입 실행 (신규 repo 파일 저작 전 — §9 운영 주의 준수).
- `isaaclab python -B controller --mode static-prep` →
  **`D409_G0A_ZERO_STEP_STATIC_PREP_PASS_STOP` exit 0** (lens B 예측 적중).
- 산출물 4파일 (경로: `g0a_d409/attempt1_zero_step_dual_jaw_contact_region_enumeration/`):
  prereg `46e31049…863a0d` / static results `1780ede4…5d363a` /
  attestation `17e16f91…7592fc` / **tuple `de79bc7867b9a6c89752e3477a5af411118db2b0b51a54cd5f030528efcc9a60`**.
  tuple 내부 해시 = 3 스크립트 sha v2 bit-일치 검증.

## 3. attempt1 runtime 완주 (사용자 승인 후 1회)

- 배관: stdin FIFO(scratchpad) + 백그라운드 실행 + Monitor. repo 무변경
  유지 (모든 로그 scratchpad; attempt 산출물은 기존 untracked `g0a_d409/` 내).
- 타임라인 (phase markers 12행): tuple gate → admission → prerun inventory
  → run1 → run2 → canonical promotion (run1/run2 **bit-exact PASS**) →
  RRD/RBL/screenshot/decision sheet 저작·검증 → manual prompt → manual
  published → **`RUNTIME_COMPLETE_STOP` exit 0**. 완주 audit 9항 전부 true.
- **과학 결과 (evidence JSON float64 권위)**: POSES 1239 / IK 1239 /
  ADMIT 1239 / **A 665 / B 0 / A∧B 0** / PINCH 1146 / **FULL 0** /
  REGIONS 1 (domain-censored 1, rep (2500,9000), rho_R 2.750mm) /
  질의 4,479,296 ≤ 7.0M. anchor gate 4채널 ≤0.000137mm (reject 0.0005mm).
  **6th doc §5-3 사전 브리핑된 top-rim 전 격자 (B)-fail이 실제 확정** —
  witness 결함 아님, 정당한 과학 결과. tolerance 수리 금지 (D354).
- 해시: verdict `D409_G0A_ZERO_STEP_DUAL_JAW_CONTACT_REGION_ENUMERATION_COMPLETE_STOP`,
  canonical evidence `ccc8197bc6c74cd87bb2475ad276464de2d6bfacfd5020792845b55b23f16750`,
  region CSV `d5a51cfab5897d7b0057f1d3b337261e901e1eb2a80fc9db37e978cbecf6cff7`,
  completion `6ce9218c7e8e2eb6036b45571cac1eb91f04dc66d93dc6b6a62d68add8d638a8`.

## 4. Manual 시각 검사 — 정직 FAIL (manual_pass=false)

두 PNG 실제 판독 (prompt 전 선판독 — 파일은 exclusive-create 최종본):
- **decision sheet**: 전부 판독 가능 — 59×21 전 셀 GREEN, REP/컨트롤 마크
  픽셀 좌표 역산 (2500,9000)/(7000,11000) 일치, 카운트·anchor gate·NULL
  문구 명료 → 관련 필드 true.
- **region map screenshot**: 격자·원통·마커는 확인되나 ① 알림 토스트
  4개(gRPC/rrd/rbl/headless)가 우측 텍스트 패널 덮음 ② representative
  뷰 탭에 빨간 오류 배지 ③ 흰 메시 포화로 양 jaw·witness 마커 식별 불가
  → 4필드 정직 false (`anchor_pose_both_jaws_visible`,
  `q5_star_or_witness_markers_visible`, `no_notification_or_text_overlap`,
  `no_error_banner_visible`).
- 11필드 = 7 true / 4 false → **manual_pass=false**. D341 의미론: 관측성
  계약 FAIL은 과학 verdict를 무효화하지 않음 (권위 = evidence JSON).
  runtime completion은 정상 기록·audit 통과. 검사 기록 원문 = manual
  document `d409_manual_visual_inspection.json` notes + scratchpad
  `attempt1_runtime_reply_sent.json`.

## 5. 과학 상태 / 다음 단계

- **A∧B 0셀 확정** → 기하 라벨 학습 승격 없음 (원래도 금지), z 설계
  재고는 **별도 case·별도 사용자 승인** 사안 (Open Risks 예고대로).
- g0a_pass=false 불변 (D407 FAIL-STOP 유지 — attempt1은 열거 완주이지
  grasp 성공 아님). null claim 목록 불변.
- attempt1 소모 완료 — **재실행 절대 금지** (retry 0, DECISIONS D409).
- 관측성 수리 후보 (차기 attempt 있을 시에만): headless screenshot 토스트
  억제 / 뷰 오류 배지 원인 / witness 마커 가시성. 본 attempt에는 소급 불가.
- (사용자 병행) 기울임/전도힘 손측정 — 보고 시 기록만.

## 6. Session progress rule 충족

실패 가능 실험 실행: attempt1 runtime 자체 (admission/bit-exact/audit/
manual 전부 FAIL 가능이었고 manual은 실제 FAIL 기록), warning 재현 probe
4종 (P3b 대조군은 실제 FAIL 재현). 로봇 HW·Isaac runtime·lerobot-train 0.
동결 침범 0. git commit/push 0.
