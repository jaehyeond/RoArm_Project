# session 2026-09-19 — W20: 사용자 결정 5항목 교차 검증 + 실행 (D492)

> 이번 case 의 신규 변수: **없음(물리 실행 0)**. 이 세션은 규약·도구·역할 정의와
> case ② 의 CPU 사전 관문만 다룬다. 물성·형상·알 수·경로·문 속도·보호선·`timestep_s` 1e-06·
> `cd_update_freq`·seed 460·원자료 전부 불변.
> 새 물리 0 · GPU 0 · DEME 0 · RunPod pod 0 · 실물 0 · 설치 0 · commit/push 0 · LFS 0.

메인 세션 = Claude Opus 5 (1M context). 작업자 = Claude `claude-opus-5` high, 새 worktree 2개.
Codex 는 주간 한도로 이번 세션 미사용(사용자 지시).

---

## 1. 부팅 (관찰 가능한 절차)

`AGENTS.md`(자동 로드) → `START_HERE.md` → `DECISIONS_ACTIVE.md`(D485~D491) →
`LEDGER_RECENT.md`(`:596~603`) → `relay/from_codex.md`(9/16)·`from_claude.md`(9/18) →
`CONTINUE_20260918_W19_DECISIONS.md` 전체 → `session_20260917_w19_runpod_afternoon.md` §9~§15 →
설계 브리핑 2건 전문(외장 SSD 심링크 경유) → `git status --short` · `git worktree list`.

relay 검증 명령 4건 실측 결과:

| 검증 | 기대 | 실측 | 판정 |
|---|---|---|---|
| `DECISIONS.md` 앞 30,235줄 md5 | `ffdfc7a5bb0fd5b83592d0bcb2db7dfc` | 동일 | 불변 |
| `DECISIONS.md` 총 줄 | 30,243 | 30,243 | 일치 |
| Orca workspaces 심링크 | 12 | 12 | 일치 |
| 1단계 감사(Codex) | 9/9 | `overall_verdict PASS`, 9/9, fail 0 | 일치 |
| 2단계 재생 감사(opus-5) | 9/9 | `verdict PASS`, 9/9 | 일치 |
| worktree | 7 + main | 7 + main | 일치 |
| 외장 SSD 아카이브 | 마운트 | 12폴더 + INDEX 접근 가능 | OK |
| RunPod pod | 없음 | 조회·생성 0 | OK |

---

## 2. 결정 5항목 교차 검증 — 무엇이 바뀌었나

사용자가 "권고대로 진행"을 승인하기 전, 5항목을 **근거 파일로 다시 검증**했다.
그 결과 **1번 권고는 철회**, **4-2 는 불필요**로 판명됐다. 나머지는 유지.

### 2-1. 🔴 결정 1 권고 철회 — 규약 개정은 `hard_fail` 정책 위반

동결 `rev32_frozen_copy/criteria.json`(sha `71a7d239…646196`)을 읽고 확인한 것:

| 임계 id | severity | 내용 |
|---|---|---|
| `policy.no_threshold_change_after_outcomes` | **hard_fail** | "결과를 본 뒤 임계를 조정하지 않는다." 합법 경로 = "새 revision + 새 criteria 파일 + 코디네이터 검토" |
| `delivery.settlement_window_UNCALIBRATED` | uncalibrated_report_only | 프레임 부족 시 "하한/상한 + 누락 증명"을 보고하라고 **이미 지시**. "pass 를 만들기 위한 추가 대기 금지" |
| `delivery.three_layers_separate` | policy | "단일값은 `definite == possible` 일 때만 허용" |

두 가지가 동시에 드러났다.
① W19 A 기록을 **본 뒤** 0.05 → 0.1 로 푸는 것은 정면 위반이다(D485 ⑤ "소급 PASS 금지"와 같은 취지).
② 애초에 **규약을 바꿀 필요가 없었다.** 정착 창 임계는 `hard_fail` 이 아니고, 프레임이 모자랄 때
무엇을 보고할지가 규약 안에 이미 적혀 있다.

→ 대체 조치 2개로 분리:
- **지금**: 규약이 지시하는 형식대로 W19 A 배출량을 **구간**으로 보고
  (`w20_decisions_d492/DELIVERY_REPORT_W19A.md`). 규약 변경 0, 원자료 변경 0, 소급 판정 0.
- **앞으로만**: 새 revision rev33b + 새 criteria 에서 기록 간격을 0.046 s 로 **정정**
  (`retroactive: false`). 0.046 은 0.05 보다 **촘촘**하다 — 완화가 아니라 달성 가능하게 만드는 정정이다
  (4 ms 동기 격자에서 0.05 를 요청하면 실제 0.052 가 된다, rev33 테스트로 고정).

**W19 A 배출량 공식 표기**: **272~327알(5.51~6.62 g), 정착 미증명.**
`exact_single_value_allowed = false` 가 원자료에 기록돼 있어 "272알" 단일 문장은 규약상 불가.
상한 식은 동결 소스 `sim_w13_full_cycle.py:937`~`:942`(애매/공중 알 중 용기 평면 범위 + 높이 띠 안).
정착 창 5프레임 동안 그 272알의 최대 속도 2.391e-05 m/s, 최대 이동 0.003311 mm — 즉 **사실상 정지**,
모자란 것은 움직임이 아니라 증거 프레임 수다.

### 2-2. 실행 시간 상한 — 올릴 이유 없음(부가 결정 해소)

영수증 `cap_s = 43200.0`, 동결 규약 `runner.physics_wall_cap_s = 32400`(hard_fail, 사용자 승인 9시간),
실제 소요 `wall_s = 20977.836`. 실제 위반은 없었다. 계획된 모든 셀(설계 ② 13,204 s, 설계 ① 16,934 s)이
32,400 s 안쪽이다. → **32,400 유지.** 올리는 것은 별도 명시 승인 사항이며 이번에 하지 않았다.

### 2-3. 🔴 결정 4-2 불필요 — 안전 hook 은 이미 4종을 차단 중

역할 agent README 는 "`safety-check.sh` 는 commit/push 계열만 차단"이라 적었다. **실측은 다르다.**
스크립트는 이미 ① git 계열 15종 ② 로봇 시리얼(`serial.Serial`·`/dev/ttyUSB`·`torque_set`·
`joints_angle_ctrl`·`move_init`·`T:106`) ③ `rm -rf` ④ `lerobot-train` 을 차단한다.
→ **확대 불필요.** 대신 실제로 비어 있던 구멍을 메웠다(§3-2).

### 2-4. 유지된 판단

| 항목 | 검증 내용 | 결론 |
|---|---|---|
| 결정 2 rev33 | AST 범위 검사 PASS·CPU 30/30·옵션 기본 OFF 확인. **단 rev33 의 `criteria.json` 이 rev32 와 바이트 동일**(같은 sha)임을 발견 — 새 criteria 가 아직 없다 | 유지 + rev33b 로 확장 |
| 결정 3 case ② | n=5,000 NPZ 실재·sha `ac29d533…9d294` 일치, n=20,000 sha `659d6b0b…8812` 일치, `recover_deme_lattice.py` 실재 | 유지, 수단 (ii) |
| 결정 3 case ① A1 | 실물은 **같은 토크 900/1000 에서 관절 1.2° 도달**, 시뮬은 fraction 0.9 에서 3.05~3.63° 정지 → 차이는 토크 천장이 아니라 **절차**(재시도·chatter) 쪽을 먼저 의심하는 것이 맞다 | 유지, 1순위 A1 |
| 결정 5 `_obj` | `retrieve_A.sh` 에 `--exclude "_obj"` + 해시 대조 2곳 `-not -path` 확인, `run_01/_obj` 부재 확인 | 유지 |

---

## 3. 실행 결과 — 메인이 직접 한 것

### 3-1. 결정 5 — 회수 스크립트 (forward-only 신규 파일)

W19 원본 `w19_runpod_d487/pod/retrieve_A.sh`(sha `67ef4f0e…255af`)는 **그대로 두었다.**
그 파일은 "그 세션에서 실제로 무엇이 돌았나"의 증거이고, 고쳐 덮으면 기록이 깨진다.
대신 `w20_decisions_d492/tools/retrieve_run.sh` 를 새로 만들었다.

| 바뀐 점 | 효과 |
|---|---|
| `--exclude "_obj"` 제거 | 생산 중 생성되는 OBJ(트레이·용기·고정부·문) 회수 |
| 해시 대조 2곳의 `-not -path "*/_obj/*"` 제거 | `_obj` 도 pod↔로컬 sha256 대조 대상 |
| 영수증에 `obj_included`·`n_obj_files`·`obj_files` 추가 | 다음 세션이 회수 완전성을 눈으로 확인 |
| 불일치·누락 시 `rc 1` + "pod 를 종료하지 말 것" 출력 | **W19 실패의 재발 차단** |
| pod/포트/호스트/경로를 환경변수로 | W19 전용 하드코딩 제거 |

검증: `bash -n` 통과. 합성 해시 2건으로 동작 확인 —
정상(`_obj` 포함 2/2) → `RETRIEVAL_COMPLETE` rc 0 / `_obj` 누락 → `RETRIEVAL_INCOMPLETE` rc 1.

### 3-2. 결정 4 — 역할 agent 4개 채택

| 결정 | 처리 |
|---|---|
| 4-1 소유권 hook 등록 | `.claude/agents/` 에 `deme-runner`·`raw-accountant`·`replay-renderer`·`independent-auditor` 복사(원본과 sha 동일), 4개 전부 frontmatter 에 `Write\|Edit → file-ownership-check.sh <이름>` 연결. hook 에 산출 경계 `claudedocs/runtime_logs/**` case 추가 |
| 4-2 안전 hook 확대 | **불필요**(§2-3). 코드 변경 0 |
| 4-3 12-agent 표 편입 | `CLAUDE.md` 에 "Execution Roles (4개)" 절 추가, 제목 `12 agents` → `16 agents`, Safety hooks 설명 실측대로 정정 |

**추가로 메운 구멍**: 어떤 agent 이름이든 상태 원장
(`START_HERE.md`, `claudedocs/{DECISIONS,DECISIONS_ACTIVE,EXPERIMENT_LEDGER,LEDGER_RECENT}.md`,
`claudedocs/relay/`)에 쓰면 **exit 2** 로 차단한다. `AGENTS.md` 의 "원장 배타 소유" 규칙이 그동안
문서로만 있고 기계적 강제가 없었다.

hook 동작을 8가지 경우로 실측 검증:

| agent | 대상 | 기대 | 실측 |
|---|---|---|---|
| deme-runner | `…/claudedocs/runtime_logs/grasp_track/w20/out.json` | 허용 | exit 0 |
| deme-runner | `START_HERE.md` | 차단 | exit 2 |
| raw-accountant | `claudedocs/DECISIONS.md` | 차단 | exit 2 |
| replay-renderer | `claudedocs/relay/from_claude.md` | 차단 | exit 2 |
| independent-auditor | `sim_scripts/hack.py` | 차단 | exit 2 |
| data-agent | `data_collect.py` | 허용(회귀) | exit 0 |
| data-agent | `claudedocs/LEDGER_RECENT.md` | 차단(신규) | exit 2 |
| unknown-agent | `…/runtime_logs/a.json` | 차단(fail-closed) | exit 2 |

---

## 4. 작업자 배정 (Orca)

Orca run `run_88fcbaaeaaf1`, 코디네이터 터미널 `term_f09509f7-…f6`.
비-Orca 터미널이라 `terminal create` 로 발신 터미널을 먼저 만들어야 했다(기존 함정 재확인).
새 worktree 2개 모두 master `fc557db`, `--no-parent`, `--setup skip`.

| worktree | task / dispatch | 모델 | 과제 |
|---|---|---|---|
| `w20-particle-count` | `task_718411a9651c` / `ctx_ba9f1da70896` | claude-opus-5 high | case ② n=5,000 CPU 사전 관문(도메인 유도 + DEME 격자 재구성 fail-closed 스크리닝) |
| `w20-rev33b` | `task_e25eae1ff493` / `ctx_4e9a187b6954` | claude-opus-5 high | rev33b = rev33 바이트 사본 + 새 criteria(`frame_dt_s` 한 필드) + 옵션 연결 + 검사 |

과제서: `w20_decisions_d492/coordinator/TASK_SPEC_w20_{particle_count_screen,rev33b_criteria}_claude.md`.
둘 다 **CPU 전용**, 상태 원장 쓰기 금지, 산출은 각자 worktree 안 `w20_decisions_d492/` 하위.

### 4-1. 작업자 결과

#### (A) `w20-particle-count` — case ② n=5,000 CPU 사전 관문 → **REUSABLE (관문 통과)**

`worker_done` outcome `succeeded`, 산출 18파일, dispatch `ctx_ba9f1da70896` release 완료.
보고서 `w20-particle-count/.../particle_count_screen/REPORT_particle_count_screen.md`.

**워커가 한 일(관찰 가능한 절차)**
1. 읽기 전용 입력 50개 sha256 기록(BEFORE).
2. 동결 `sim_w13_full_cycle.py` 를 **수정 없이** `ast` 로 파싱해 `run()` 본문 중 `dom_z` 대입(`:225`)
   까지의 문장만 잘라내고, 호출만 기록하는 **가짜 DEME 모듈**을 꽂아 원본 파일명·줄번호로 실행.
   재구현이 아니라 생산 소스 바이트 그대로의 실행이다. 경계 검사(잘린 마지막 문장이 `dom_z` 대입인지,
   다음 문장이 `InstructBoxDomainDimension` 인지)를 fail-closed 로 걸었다.
3. **방법 검증 먼저**: n=20,000 으로 돌려 동결 `numeric_inputs.json` 의 `rev11_recomputed_domain_m` 과
   대조 → 세 축 전부 binary64 **완전 일치** → `METHOD_VALIDATED`.
4. n=5,000 도메인 확정.
5. `recover_deme_lattice.py` 를 문서 `:68`~`:71` 명령 형식 그대로 CPU 실행.
6. 후보 `numeric_inputs` 가 생산 fail-closed 대조(`:236`~`:252`)를 실제로 통과하는지 + **음성 대조**.

**수치**

| 항목 | n=20,000 (동결) | n=5,000 (신규) |
|---|---|---|
| 도메인 x | [-0.4449132573934248, 0.22] | **동일** |
| 도메인 y | [-0.175, 0.4612066423643583] | [-0.175, **0.46121241555032444**] |
| 도메인 z | [0.0, 0.5881087722936942] | [0.0, **0.5750117499666003**] |
| 더미 표면 높이 `z_surf_pre` | 41.18 mm | **28.08 mm** |
| 격자 `l` | — | 5.5548655952808446e-12 m |
| `voxel_size` | — | 3.6404367165232543e-07 m |
| 축 voxel 수 2의 거듭제곱 | — | [22, 21, 21] |
| 좌표 절대 상한 | — | 1.0155052542686462 m |
| 인접 추가 비트 | — | **[0, 0]** = 인증된 분기 안 |

y/z 상한만 다른 이유는 더미가 얇아져(41.18 → 28.08 mm) 웨이포인트와 공구 스윕이 내려앉기 때문이다.
x 가 같은 것은 x 방향 범위를 용기 위치가 지배하기 때문이다.

**판정**: `REUSABLE` — rc 0, stderr 0바이트, 인증 분기 안.
**음성 대조**: 같은 자리에 동결 n=20,000 증거를 넣으면 `SystemExit`("도메인이 수치 증거의 고정 입력과
다르다 — lattice 재사용 금지")로 **거부**된다. 즉 이 관문은 살아 있고 통과가 공짜가 아니다.

**메인 교차 검산(전부 메인이 직접 실행)**

| 검산 | 결과 |
|---|---|
| 읽기 전용 입력 50개를 메인이 **재해시** | 불일치 0, 누락 0 |
| BEFORE vs AFTER 영수증 대조 | 불일치 0 |
| 동결 `numeric_inputs.json` 의 도메인 3축 vs 워커 유도값 | **세 축 exact True** |
| `recover_deme_lattice.py` 를 **메인이 같은 입력으로 재실행** | stdout **바이트 동일**(sha `3b563f94…c42e`), rc 0, stderr 0바이트 |
| 동결 src sha vs `REVISION_PIN.json` 핀 | `fc8d4a87…059f3` 일치 |
| 워커가 지적한 `--stop-after-phase` 도움말 불일치 | **사실 확인** — 도움말은 `:1281` "스모크 전용", 그러나 W16 **생산** 실행은 `--stop-after-phase reclose --max-wall-s 13800` 으로 돌았다(`COMMANDS_w16.json` `steps.profile.argv`, `--max-particles` 없음) |

**GPU GO 꾸러미(초안, 실행 0)**: P2 단독(n=5,000, `settle`~`reclose`), 소프트 상한 13,204 s =
P0 실측 11,003.3 s × 1.2, 러너 하드 상한 13,804 s(유예 600 s), 재시도 0, `setsid` 분리.
**둘 다 동결 규약 32,400 s 안쪽이다**(§2-2 판단과 일치). P1(n=10,000)은 NPZ 부재로 전제조건 문서만.
P0 반복 셀은 "미결 옵션 — 사용자 결정, 기본값 추가 안 함"으로 분리.

#### (B) `w20-rev33b` — 새 revision + 새 criteria

`worker_done` 2회(1차 `ctx_4e9a187b6954`, 후속 `ctx_906ffcc335bf`) 전부 `succeeded`, 둘 다 release 완료.
보고서 `w20-rev33b/.../rev33b/REPORT_rev33b.md`, 산출 50파일.

**1차 — rev33b 생성**

| 항목 | 결과 |
|---|---|
| rev33 → rev33b 바이트 사본 | 39/39 동일 |
| criteria 재귀 diff | **정확히 5자리** |
| thresholds 개수 | 45 → 45 |
| severity·operator·boundary·id 전수 | 동일 |
| `runner.physics_wall_cap_s` | **32400 불변** |
| AST 범위 검사 | PASS(동결 함수 전수 동일, 모듈 상수 동일) |
| CPU 단위 테스트 | 56/56 PASS |

바뀐 5자리: `thresholds[26].value.frame_dt_s`, 같은 항목 `origin`(기존 문장 보존 + 근거 덧붙임),
`revision`(rev11 → rev33b), `supersedes_criteria_sha256`(신규), `retroactive`(신규 `false`).

**워커가 제 지시보다 낫게 한 설계 1건(코디네이터 채택)**: `--settle-window-frame-dt-s` 를
**값 인자 → 켜고/끄기 플래그**로 바꾸고 값은 `--criteria` 가 가리키는 동결 파일에서만 읽게 했다.
이유가 정확하다 — 값이 명령줄에 있으면 운영자가 **결과를 본 뒤 숫자를 바꿔 넣을 수 있고**,
그것이 바로 `policy.no_threshold_change_after_outcomes` 가 막으려는 행위다. 값을 동결 파일이 쥐어야
정책이 실제로 작동한다. 원자료 메타데이터에 `settle_window_frame_dt_source` 와
`settle_window_criteria_sha256` 2건도 추가됐다.

**🔴 워커가 코디네이터 과제서의 수치 오류를 escalate**

과제서 §2-4 ④ 가 "옵션 지정 시 정착 창 최대 간격 ≤ 0.046" 을 요구했는데 **4 ms 동기 격자에서 도달 불가**다.

| 요청 간격 | 실제 간격 집합 | 실제 최대 | 창 안 프레임 | 감사 게이트(≤0.05 · ≥6) |
|---|---|---:|---:|---|
| 0.05 (옛 값) | {0.048, 0.052} | 0.052 | 8 | ❌ |
| 0.046 (1차) | {0.012, 0.044, 0.048} | 0.048 | 10 | ✅ |
| **0.044 (확정)** | {0.040, 0.044} | **0.044** | 10 | ✅ |

워커는 추정으로 맞추지 않고 단언을 **규약이 실제 요구하는 조건**(최대 간격 ≤0.05 · 프레임 ≥6)으로
바로잡은 뒤 값 선택을 올렸다. **이것이 정상 동작이다** — 과제서 단언이 물리적으로 도달 불가일 때
워커는 숫자를 맞추지 말고 escalate 한다.

**후속 — 코디네이터가 0.044 로 결정**

근거 4건: ① 0.044 = 11 × dt_sync 의 정확한 배수라 **요청값 == 실제값** ② 최대 간격이 0.048 → 0.044 로
내려가 게이트 0.05 에 여유 증가 ③ 창 안 프레임 10개로 동일 = 기록 비용 증가 0
④ 워커가 §3-2 에서 지적한 오독 위험(누가 `frame_dt_s` 를 "최대 허용 간격"으로 읽는 경우)이 0.044 에서는
실제로 참이 되어 사라진다. **메인이 격자 산술을 독립 재계산해 위 표의 간격 집합과 최대값이 일치함을 확인.**

값을 criteria 가 쥐는 설계 덕분에 **후속 왕복에서 생산자 소스는 한 줄도 바뀌지 않았다**(src sha 1차와 동일).
최종: 재귀 diff 여전히 5자리, wall cap 32400, AST PASS, 테스트 **57/57 PASS**(상속 30 + 신규 27).

**메인 교차 검산(전부 메인이 직접 실행)**

| 검산 | 결과 |
|---|---|
| 읽기 전용 입력 **78개를 메인이 재해시** | 불일치 0, 누락 0 |
| BEFORE vs AFTER 영수증 | 불일치 0 |
| rev32/rev33 criteria 원본 sha | 둘 다 `71a7d239…646196` 불변 |
| W19 원자료 NPZ sha | `414633fb…9074` 불변 |
| **메인 독립 재귀 diff**(rev32 → rev33b) | **5자리**, thresholds 45→45, severity/operator/boundary/id 전수 동일, wall cap 32400, `frame_dt_s` 0.044 |
| **메인 독립 격자 산술** | 0.05→max 0.052(게이트 FAIL) · 0.046→max 0.048 · 0.044→max 0.044 — 워커 표와 일치 |

**GPU 스모크 꾸러미(초안, 실행 0)**: 알 300개, sim 소프트 상한 3,000 s, 러너 하드 3,600 s.
**정본 승격은 스모크 통과 + 사용자 승인 뒤.** 승인 전 물리 실행 정본은 계속 rev32.


---

## 5. 한계 · 다음 승인 경계

### 5-1. 이 세션이 **주장하지 않는** 것

- case ② 관문 통과(`REUSABLE`)는 **실행 가능성**만 말한다. P2 셀이 완주한다·수렴한다·비용이 N 에 비례해
  내려간다는 증거가 **아니다**. 그것이 P2 실행의 검증 대상이다.
- **n=5,000 통과는 n=10,000 의 증거가 아니다.** 도메인이 또 달라지므로 같은 CPU 관문을 다시 통과해야 한다.
- 격자 값은 설치된 바이너리로부터 되살린 **정적 재구성**이며 새 solver 초기화에서 실제로 읽은 값이 아니다.
- rev33b 는 **초안**이다. GPU 스모크 0회이므로 이 판으로 물리가 돈 적이 없다. 정본은 계속 rev32.
- "272~327알" 구간을 다른 실행이나 실물 계량에 일반화하지 않는다(실행 간 변동 517/517/489, 292 vs 144 — D490).
- "없다/최초" 류 주장을 하지 않는다(HARD RULE #4).

### 5-2. 이 세션이 남긴 공정 교훈 (D492 Implication 으로 승격)

1. **규약 변경을 제안하기 전에 해당 임계의 `severity` 와 `boundary` 를 먼저 읽는다.** 이번 권고 1건이
   `hard_fail` 정책에 걸려 철회됐고, 애초에 개정이 필요 없었다는 것도 같은 파일에 적혀 있었다.
2. **판정 문턱 값을 CLI 인자로 두지 않는다.** 명령줄에 있으면 결과를 본 뒤 바꿔 넣을 수 있어
   사후 조정 금지 정책이 무력해진다.
3. **저장 간격은 동기 격자의 정확한 배수로 고른다.** 비배수는 요청값과 실제 간격을 어긋나게 만들어
   나중에 그 숫자를 게이트로 오독할 여지를 남긴다.
4. **워커가 과제서의 도달 불가 단언을 escalate 하는 것이 정상 동작이다.** 추정으로 맞추지 않는다.
5. **배출량은 `definite == possible` 일 때만 단일값으로 쓴다.** 아니면 구간 + 미증명 사유.

### 5-3. 다음 승인 경계 (사용자 결정)

1. **P2(n=5,000) GPU 실행 GO** — 소프트 상한 13,204 s, 러너 하드 13,804 s, 재시도 0, `setsid` 분리.
   출력 폴더 신설·경로 확정 + `START_HERE.md` `Active Case` 등재 필요.
2. **rev33b GPU 스모크 GO**(알 300개, 소프트 3,000 s / 하드 3,600 s) → 통과 시 정본 승격.
3. **설계 ① 문 이음새** — 같은 N 에서, 1순위 수단 A1(재닫기 chatter 1회), 반복 셀 n≥3.
4. 미결 옵션 3건(기본값 = 하지 않음): P0(20,000알) 반복 셀 · P1(n=10,000) 더미 생성 ·
   포획량 ±15 % 를 판정 문턱으로 승격.
5. 실행 상한 32,400 s 유지에 동의하는지(올리려면 명시 승인, 계획상 올릴 이유 없음).
6. 실물·학습·PBD·설치·commit/push·LFS·RunPod 신규 pod 는 새 명시 승인 전 금지.

### 5-4. Orca 정산

run `run_88fcbaaeaaf1`, dispatch 3건(`ctx_ba9f1da70896`·`ctx_4e9a187b6954`·`ctx_906ffcc335bf`) 전부 settled.
`ctx_4e9a187b6954` 는 같은 터미널을 후속 dispatch 로 재사용(완료 회계 1번 경로), 나머지 2건 release.
최종 `worker-list --terminal-state reclaimable` **0**, 코디네이터 inbox **0**, 코디네이터 터미널 close 완료.
worktree 2개(`w20-particle-count`·`w20-rev33b`)는 산출 보존을 위해 유지하며 **원장 merge 하지 않았다**.

**운영 함정(재확인·신규)**
- 비-Orca 터미널에서는 `terminal create` 로 발신 터미널을 먼저 만들어야 orchestration 명령이 선다.
  플래그가 명령마다 다르다: `run-create`·`worker-start`·`reply` 는 `--from`, `check` 는 `--terminal`,
  `worker-release` 는 둘 다 받지 않는다.
- 같은 Run 에 actionable waiter 는 **하나뿐**이다. 두 번째 `check --wait` 는 `waiter_exists` 로 거절된다.
- `check --wait --json` 출력은 keepalive 줄이 섞인 **여러 JSON 문서**다. 마지막 문서만 파싱해야 한다.
