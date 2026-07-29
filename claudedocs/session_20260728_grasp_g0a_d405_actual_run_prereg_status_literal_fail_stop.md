# D405 actual run — prereg status 리터럴 저작 결함 FAIL_STOP (관측성 수리 3건 미도달)

Date: 2026-07-28 밤 KST. **D405 attempt1 소진·동결 (worker 1회, retry 0, 85.7s).**
승인 근거: 유저 순차 지시("다음 최소 승인할테니...") — 이 attempt로 소진.
인용 tuple SHA: `8f43c9679548e10ca65b2607e6aff15f677e157c9f685ba9e747536ff4f68a78`
(발사 브리핑에 명시 후 실행).

## 1. 실행 절차 (감사 가능 step-by-step)

1. 실행 직전 점검(리뷰 반영 확장판) 전부 PASS: 호스트 경계(pid 2099513,
   /dev/nvidiactl), HEAD==origin/master==a69a96d, **allowlist 47 = dirty 47
   완전 일치**, roarm_rl 2파일 sha == prereg pin, 잔존 Isaac/rerun/9876 0,
   발사 직전 scratchpad 재렌더로 ppp=2.0 재확인(960×540→1920×1080).
2. controller **단독** 백그라운드 발사(리뷰 blocker 3 준수) 후 ≥3s 뒤 task
   출력 파일만 참조하는 감시 시작(OUT_DIR 문자열 미포함 — 환경 게이트 안전).
3. t+85.7s 종료(exit 1) — 육안검수 프롬프트 미출현. 사후 진단 read-only.

## 2. 결과 (source: `claudedocs/runtime_logs/grasp_track/g0a_d405/attempt1_d404_observability_import_path_repair/`)

### 인프라·선행 게이트 전부 PASS

- **D405 pre-delegation probe 라이브 통과** — 수리 R1(sys.path repo-root) +
  roarm_rl.rerun_contract 해석 + rerun SDK 0.34.1 + numpy 1.26.0 + CLI 존재가
  라이브에서 실증됨 (probe 실패 시 위임 전 정지 설계였고, 통과 후 위임).
- 승인 게이트(동결 실게이트) PASS — tuple 4필드/attestation 스키마/negative
  59행 superset 전부 수락 (오프라인 복제 10/10과 일치).
- freeze manifest 게이트 PASS: pre-write snapshot의 dirty 47 == allowlist 47,
  `unexpected=[]`; frozen inputs/installed sources/d334 sidecar 전부 exact.
- 환경 게이트 PASS(비조상 프로세스 충돌 0), offline negative controls PASS,
  worker 1회 spawn, **Isaac 기동 정상**(simulation_app_launch_end), kit log
  오류 0, supervisor 잔존 signal 0.

### FAIL 지점 — 동결 worker의 prereg admission (worker.py:2517-2518)

```
RuntimeError: D400 preregistration status is not frozen
  (prereg.get("status") != "PREREGISTERED_NOT_EXECUTED")
```

- D403/D404 prereg는 status가 동결 리터럴 `"PREREGISTERED_NOT_EXECUTED"`;
  **D405 prereg만** 저작 시 `"STATIC_PREP_PREREGISTERED_RUNTIME_PENDING"`으로
  "개선" — 동결 소비자 리터럴을 도출하지 않은 저작 결함.
- 실패 시점이 derivative 복사 **전**이라 D405 수리 R2(크기)/R3(blueprint),
  SDF 저작/cook, 관측성 분기 전부 미도달 — 라이브 판정 이월.
- worker rc -9는 close hang의 SIGTERM/SIGKILL 정리(D402-R1 기지 패턴);
  raw summary/preclose sentinel은 정상 기록됨.
- phase 감사 pass=true (15 phase, technical fail 브랜치 정상 기록),
  scope counters: worker 1/retry 0/derivative·SDF·physics·contact 전부 0.

### 정적 준비의 명암 (정직 평가)

- **잡은 것**: 실물 replay가 라이브 차단급 결함 2건(headless ppp 2.0 물리
  크기, TextDocumentView 1문서 제약)을 attempt 소모 전에 적발·수리; 리뷰
  blocker 3건(negative superset/300s 창 능동 폴링/환경 게이트 프로세스 스캔)
  전부 사전 해소 — 이번 run에서 해당 실패 0.
- **놓친 것**: 동결 worker의 **2행짜리 순수 admission**(prereg sha+status)을
  replay 범위에 넣지 않았다. replay가 닿은 결함은 전부 잡혔고, 닿지 않은
  유일한 순수 체크에서 실패 — D403 lesson("라이브 전 미검증")의 세 번째 반복,
  이번엔 무거운 분기가 아니라 가장 가벼운 체크가 사각.

## 3. 판정과 다음 단계

- canonical `D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP`; descriptive
  **`D405_PREREGISTRATION_STATUS_LITERAL_FAIL_STOP`** — Isaac/PhysX/체인/수리
  R1-R3 실패 아님 (그렇게 부르지 말 것). durable lesson = DECISIONS **D405**.
- `scientific_or_physics_verdict=null`, `g0a_pass=false` 유지.
- 다음 최소 rung = **D406 [d405_prereg_status_literal_repair]**: prereg
  status만 동결 리터럴로 수정(신규 변수 0 — 계약 준수 수정; 관측성 수리
  3건은 D405 변수 그대로 이월) + wrapper 2종 sha 재내장 + 새 attestation/
  tuple + 정적 runner에 **동결 prereg-admission replay fixture** 추가.
- **D406 runtime은 새 명시 승인 필요** — 순차 지시는 D405 attempt1로 소진.
