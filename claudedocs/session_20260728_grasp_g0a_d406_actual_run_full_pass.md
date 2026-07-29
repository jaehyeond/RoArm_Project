# D406 actual run — D400 게이트 사상 최초 전체 PASS (기술 + 관측성 + 라이브 육안검수)

Date: 2026-07-28 밤 KST. **D406 attempt1 소진·동결 (worker 1회, retry 0, exit 0).**
승인 근거: 유저 명시 지시("D405 attempt1로 소진 — D406은 정적 준비부터 새 명시
승인. step-by-step으로 순차적으로 사고하면서 진행해") — 이 attempt로 소진.
인용 tuple SHA: `bc54e7c51c0ca5ef367595c53dcb5e06b7a9afbb2907a607642a65ebf9adf435`
(발사 브리핑에 명시 후 실행).

## 1. 실행 절차 (감사 가능 step-by-step)

1. 정적 준비 완주 (상세: static prep doc): prereg status 리터럴을 동결 worker
   소스에서 도출 저작(sha `c4980157...`), wrapper 2종 순수 rebind, 정적 runner
   체크 47/47+양성 26/26+음성 56/56 PASS (신규 stage K = 동결 admission replay,
   실물 D405 prereg로 라이브 실패 bit-exact 재현 포함), 4-lens 적대 리뷰
   (wf_fc718cbc-23d) blocker 1 → 원자적 검수 작성기 dry-run으로 해소, 실물
   attestation/tuple로 승인 게이트 오프라인 복제 10/10 PASS.
2. 실행 직전 점검 13/13 PASS: **dirty == allowlist 정확 64/64** (outside 0,
   unused 0), HEAD==origin/master==a69a96d, roarm_rl 2파일 pin 일치, tuple
   바인딩 4종 최종 sha 일치, VRAM 14672≥8192MiB, 충돌 프로세스/9876 리스너 0,
   forward-only 선존재 0, ppp=2.0 재확인 렌더(960x540→정확 1920×1080).
3. controller **단독** 백그라운드 발사 → argv-청정 감시(/tmp 캡처 stdout의
   프롬프트/실패 시그니처만; OUT_DIR·wrapper 파일명 문자열 0) → t+~25s 육안검수
   프롬프트 출현 → 라이브 스크린샷 실물 육안 검수 → 원자적(임시명→같은 fs
   rename) 검수 JSON 기록(잔여 ~240s) → controller가 0.25s 폴러로 수락 →
   completion 수집, exit 0.

## 2. 결과 (source: `claudedocs/runtime_logs/grasp_track/g0a_d406/attempt1_d405_prereg_status_literal_repair/`)

### 전체 PASS — canonical verdict

`D400_GRIPPER_LINK_SDF_RES256_CONFIGURATION_LOAD_ADMISSION_OWNER_ENUMERATION_PREFLIGHT_PASS_NO_PHYSICS`
(`d400_completion_summary.json`: `pass=true`, `technical_pass=true`,
`observability_pass=true`, `observability_error=null`,
`runtime_preflight_pass=true`)

- **D405 실패 지점 통과**: worker admission이 D406 prereg(sha `c4980157...`)를
  수락 — raw summary `exception: null`, `preregistration_sha256` 일치.
- **기술 체인**: derivative 저작(A64 64 비활성·SDF API 1·SDF attr 7), SDF cook
  **136 task 전부 완료** (scheduled 136 = finished 136, cache hit 136/miss 0 —
  D404 cook 결과 캐시 재사용; baseline/final running 0, checks 6/6), PhysX
  property query **link5 65·gripper_link 66 collider 전부 pass**, mass
  invariance PASS, counter gate PASS, worker protocol PASS.
- **관측성 (사상 첫 라이브 완주)**: RRD 1,214,078B(sha `07b164cf...`) footer
  verify PASS, RBL 83,160B, validation 계약 PASS(sha `98c7f086...`), 헤드리스
  스크린샷 **정확 1920×1080**(sha `8b7308ee...` — 수리 R2 라이브 실증), board
  1920×1080(sha `58a2c2c5...`), headless_viewer_invocations=1.
- **라이브 육안검수 (사상 첫 통과)**: 검수 8체크 전부 true (`manual_inspection.
  pass=true`, JSON sha `10e706a4...`). 실물 관측: 3D 뷰에 source(파랑)/live SDF
  input(주황)/link5 A64(초록) 전부 가시; status 4패널+phase 3패널 실제 JSON
  렌더 — **"Can only show one text document" 배너 0 (수리 R3 라이브 실증)**;
  토스트 3개는 3D 뷰 빈 하늘에만 (사전 고정 §5c 정의 기준 겹침/잘림 0).
- **phase 감사**: 35 phase 전원 required exactly-once·등록 순서·owner 정확,
  `technical_pass_branch=true`, rerun marker 완결, 감사 22체크 전부 true.
- 인프라: freeze manifest 14체크 전부 PASS, kit log 71행 오류 0, supervisor
  rc 0·잔존 signal 0, 잔존 Isaac/rerun 프로세스 0.
- scope counters (계약 준수): worker 1/retry 0, derivative 1, A64 disable 64,
  SDF attr 7, property query 2, physics/contact/q5/cylinder/public_forward/
  timeline play/커밋 전부 0.

### 권한 한계 (완주 verdict의 정확한 의미 — 과대 주장 금지)

completion `authority_limit` 명문: **configuration / stage-load admission /
global cook-queue drain / rigid-owner·property enumeration까지만**. per-prim
SDF 내부 identity, collision participation(충돌 참여), contact, tipping, grasp
주장 없음. 따라서 `scientific_or_physics_verdict=null`, `g0a_pass=false` 유지 —
과학 질문(SDF 표현이 D362 전도를 바꾸는가)은 여전히 미측정이며, 그것이 다음
rung이다.

## 3. 판정과 다음 단계

- **D400 게이트 체인(D400 prereg → D401 freeze → D402 harness → D403 gate
  contract → D404 observability import → D405 prereg literal → D406) 완결.**
  라이브 미판정으로 이월됐던 D405 수리 R2(크기)/R3(blueprint)가 이번 run에서
  실증 — `observability_first_live_render_repair_v1` 변수 라이브 판정 완료.
- durable lesson = DECISIONS **D406** (순수 admission replay 의무의 실효 확인 +
  라이브 육안검수 운영 계약 확립).
- 다음 최소 rung = **D407 [sdf_physics_ab_d362_remeasure]** (후보, **미승인** —
  물리는 별도 승인 필요): 동결 34×90mm/0.72kg D362 계약 그대로, gripper_link
  A64↔SDF 단일 변수 A/B 전도 재측정. D406 산출 derivative가 그 입력.
