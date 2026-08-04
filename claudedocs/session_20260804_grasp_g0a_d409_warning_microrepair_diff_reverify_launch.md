# Session 2026-08-04 (2nd = 7th) — D409 warning 9건 처분·소수리 + diff-한정 재검증 발사

이번 case의 신규 변수: [없음 — D409 정적 준비 단계 지속. 신규 변수는 3rd
세션 선언분 1개(실물 원통 기하) 불변]

과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`, 과학 verdict 없음.
attempt1 산출물 생성 0 / ATTEMPT_ROOT 미생성 (본 세션 종료 시점 재확인).
로봇 HW·Isaac runtime·lerobot-train 실행 0. 정적 fixture **미실행**
(117% stop-hook — /half-clone 거부 HARD RULE #11, end-of-session 전환).

## 0. 승인 범위 / 규칙 준수

- 사용자 "설계 착수" 승인 지속 (설계+정적 준비+tuple까지; attempt1은
  tuple SHA 인용 별도 승인, retry 0). D399 금지.
- 처분 원칙 준수: 6th doc §2.3 warning 9건 전건을 본 세션에서 **독립
  재검증 후** 처분 (재검증 verbatim
  `g0a_d409/design_inputs/repair_reverify_wf_bc577f9b-dfa_result_verbatim.json`
  sha `6d458f76…dbce5c` 본 세션 재계산 bit-일치 확인 후 전문 판독).

## 1. Warning 9건 처분 (독립 재검증 → 반증 0 → 전건 확정)

| ID | 본 세션 독립 재검증 (방법 → 결과) | 처분 |
|---|---|---|
| SCI-R1-W1 | worker:1346 `max(hi,lo) > bound_mm` strict 비교 + worker:755-757 `DistanceRequest(True,1e-9,1e-9)`·`gjk_tolerance=1e-9` 직접 판독 — 주장 재현 | 수리 **M1** (rule 텍스트 정밀도 공개 1문장) |
| SCI-R1-W2 | 재검증 자체가 게이트 누출 0 실증 (scenario-D: gated 전항 false) | **수용-기록** (CSV `first_crossing_order_certified` 인접 컬럼 join으로 필터 가능 — 소비자 규칙으로 기록) |
| OPS2-W1 | controller:1946-1974 `needle in writer_source` 구조 판독 + writer:190-191(Assign)/:1186(contract Load)/:1160·1164(계약 문자열)로 needle 충족 경로 확인 — 주장 재현 | 수리 **M2** (실제 AST 노드 검사로 교체) |
| OPS2-W2 | fixture 내부 raise 4곳 실측 (controller:1584 N4 / :1663 audit / :1691 positive / :1795 equivalence) + fixture dict 일괄 생성 구조(무 try) 판독 — 주장 재현 | 수리 **M3** (try-wrap + STATIC_FAIL 기록 후 전파) |
| OPS2-W3 | controller:3523-3527 radius/height 2채널만 + `_build_prereg` geometry 5 pin(:1199-1203) 판독 — 주장 재현 | 수리 **M4** (x/table_z/z_center 3채널 추가) |
| L3-W1 | controller:478 `json.dumps` allow_nan 부재 (대조: `_canonical_bytes`:448은 있음) — 주장 재현 | 수리 **M5** (`allow_nan=False` 추가) |
| L3-W2 | 절차 규칙 — 소수리로 byte-결박 실효 | **절차 이행**: 소수리 후 diff-한정 적대 재검증 `wf_311d5910-658` 발사 (§5) |
| L3-W3 | note-only (prereg는 fixture 전 기록 — 무영향 기실증) | **수용-기록** |
| L3-W4 | ATTEMPT_ROOT 현재 부재 재확인 — moot | **수용-기록** |

## 2. 소수리 M1~M5 diff (파일별)

**worker** (`sim_scripts/cyld29h50_..._worker.py`, 2,390→2,395줄):
- M1: `order_certification["rule"]` 문자열 말미에 정밀도 공개 1문장 추가 —
  "raw hppfcl GJK values (DistanceRequest/GJK tolerance 1e-9 m = 1e-6 mm),
  no additional numerical-error allowance; margin below that tolerance
  would be theoretically unsound (observed margins are orders of magnitude
  larger)". 텍스트 전용 — 수치 경로 무변경 (§3 스모크 bit-exact로 실증).

**controller** (`..._controller.py`, 3,847→3,998줄):
- M1 동기: prereg `gates.order_certification.rule`에 동일 취지 절 추가.
- M2 (OPS2-W1): `_fixture_w_ops3_reject_surfaces`의 'ast' 표면 5개를
  substring needle → **실제 AST 검사**로 교체. 검사 형태 =
  raising-If 내 Compare 형상 매칭: ① identity: `os.getppid()` NotEq +
  `_proc_start_ticks(...)` NotEq ② tuple_sha_binding: ProtocolError raise
  인자 Constant "approved tuple-file SHA mismatch" + JoinedStr
  "tuple {key} mismatch" ③ deadline: `time.monotonic_ns()` 비교 guard +
  `manual_deadline - WRITER_DEADLINE_LEAD_NS` Sub-BinOp 비교 guard
  ④ exclusive_create: `RENAME_NOREPLACE` Call 위치 인자(Load) +
  `os.fsync` Call ≥2 + `os.O_EXCL` Attribute ⑤ no_writer_repair:
  Constant 'controller_started' 비교 guard + `_sha_path(WRITER_PATH)`
  비교 guard. 결과 키 `enforcing_source_fragments` →
  `enforcing_ast_nodes` (method='ast' 이제 정직). 계약 문자열/상수
  Assign/죽은 코드는 이 노드 형상을 만들 수 없음 — 강제 블록 삭제 시
  해당 표면 FAIL (§3 D 차등검증 실증).
- M3 (OPS2-W2): `run_static_prep` fixture 5건을 `fixture_specs` thunk
  리스트 + try-wrap으로 재구성. 내부 raise 시: `fixture_raise`
  {group/error_type/error} 기록 + `fixture_failures`에
  `<group>.raised:<type>` 추가 + **STATIC_FAIL_STATUS results 파일 기록
  후** `D409Error ... from 원본예외` 전파. catch는 `Exception` 한정
  (KeyboardInterrupt/SystemExit 의미 보존). attestation/tuple은 모든
  실패 경로에서 미도달 유지. static_results에 `fixture_raise` 키 신설
  (happy path에서는 null).
- M4 (OPS2-W3): `run_runtime` R12 재대조를 5채널로 확장 —
  `CYL_X_M`/`TABLE_Z_M`/`Z_CENTER_M` repr vs prereg
  `x_m_repr`/`table_z_m_repr`/`z_center_m_repr` 추가 (오류 메시지
  "R12/OPS2-W3").
- M5 (L3-W1): `_write_json_x`의 json.dumps에 `allow_nan=False` 추가.

**manual_writer**: 무변경 (sha 유지 `7f044583…ea7f01`).

## 3. 검증 실측 (본 세션)

| 검증 | 결과 |
|---|---|
| py_compile 3/3 (isaaclab python -B) | PASS |
| 8-pose 스모크 (4 코너 + (7250,9000)+(7000,11000)+(250,6750)+(14250,11250)) | **인증 8/8**, unresolved 0, rejected 0, 질의 3,456~3,840 ≤ 5,400 |
| bit-exact 회귀 (6th doc §2.2 값 대비) | (7000,11000) q5*=0.6596315690985648/part_042, (0,6500) q5*=0.6097505786983675 — **bit-일치** |
| 결정성 (PYTHONHASHSEED 1 vs 31337, 2 독립 프로세스) | 3 pose canonical bytes sha 3/3 bit-일치: (0,6500) `d7cc1bdc…`, (7250,9000) `563adc09…`, (14500,11500) `a3c42e72…` |
| 정적 dry-run **A** (실상수, 무기록) | env gate / interface(cap 48) / writer contract 9표면 / prereg build+audit / strict re-parse(40,193B, allow_nan=False) / rule 공개문 확인 — 전항 PASS |
| 정적 dry-run **B** (scratch 패치 `run_static_prep` 실제 실행) | rc 0 + `D409_G0A_ZERO_STEP_STATIC_PREP_PASS_STOP` + 4파일 + w_ops3 9표면+coverage 전부 PASS + ast 표면 5개 `enforcing_ast_nodes` 전부 true + tuple 해시 = 디스크 3파일 sha bit-일치 |
| 정적 dry-run **C** (M3 raise 경로: equivalence fixture 합성 raise) | STATIC_FAIL results **기록됨** (fixture_raise 기록 + failures에 raised 항목), D409Error cause 체인, attestation/tuple 미생성 — M3 의도 실증 |
| 정적 dry-run **D** (M2 차등검증: writer 사본에서 deadline 강제 3블록 삭제 — 재검증 probe 시나리오 재현) | `common_monotonic_deadline` 표면 **FAIL** (lead-arithmetic guard false — 구 needle은 통과했던 시나리오), 나머지 8표면 PASS — M2 판별력 실증 |
| repo 무변경 | git status 형상 세션 시작과 동일, `g0a_d409/`는 design_inputs/만, ATTEMPT_ROOT 부재 |

dry-run 산출물/probe = 세션 scratchpad(`smoke_8pose.py`,
`smoke_run{1,2}.txt`, `dryrun_static.py`, `dryrun_static_tree/`) — repo 외부.
dry-run B의 산출 sha들(prereg `5eb25cac…` 등)은 scratch 산출물이며
created_utc 포함이라 **실제 fixture 산출물과 bit-동일 기대 금지** (참고 전용).

## 4. 최종 as-repaired sha256 (v2 — 6th doc §2.4를 대체; 정적 fixture는 이 bytes로 1회)

| 파일 (sim_scripts/) | 줄수 | as-repaired-v2 sha256 |
|---|---|---|
| cyld29h50_..._worker.py | 2,395 | `d82cee18b6282b857f0a762896bc43ee4a23a2a954a680b22735669dc6f827d9` |
| cyld29h50_..._controller.py | 3,998 | `2af457ecfbd5df3d6a223de991fee4108e8ec7d243bcf74e37287d69f622eb1a` |
| cyld29h50_..._manual_writer.py | 1,540 | `7f04458f3e44042be297fefcdd71638f9e62f2ef35cf0114df59ea5c76ea7f01` (무변경) |

## 5. diff-한정 적대 재검증 발사 — **미회수** (다음 세션 1순위)

- run `wf_311d5910-658`, 2-lens 병렬 (A: diff-correctness — M2 AST fixture
  우회 시도·M1 무영향·M4 필드명·M5 직렬화·interface drift / B:
  static-exec safety — scratch 시뮬레이션 재실행·M3 양 실패경로·쓰기
  경로 감사·PASS_STOP 예측). default-refute, read-only, 실경로
  static-prep 실행 금지 명시. sha v2 3파일 결박 선확인 지시.
- **회수 절차** (전례 = 2nd/3rd doc 동일):
  1. journal 판독: `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/87f79385-b471-4f3e-8ea0-f8b3bd431f27/subagents/workflows/wf_311d5910-658/journal.jsonl`
     의 `{"type":"result"}` 2행 (lens A/B).
  2. 부재/불완전 시: 같은 폴더 `agent-*.jsonl` 전사에서 StructuredOutput
     `tool_use` input 직접 회수 (프롬프트 원문 오탐 주의).
  3. 그래도 부재 시 resume 재발사: `Workflow({scriptPath: "~/.claude/projects/.../87f79385-*/workflows/scripts/d409-warning-microrepair-diff-reverify-wf_311d5910-658.js", resumeFromRunId: "wf_311d5910-658"})`
     (세션 경계 캐시 미이월 전례 — 전량 재실행 예상).
  4. 회수 후 verbatim 보존 (`g0a_d409/design_inputs/microrepair_diff_reverify_wf_311d5910-658/`
     result/journal/manifest + sha) → 전문 판독 → **독립 재검증 후 처분**
     (무비판 수용 금지). blocker 0 확인 전 정적 fixture 착수 금지.
- **알려진 위양성 후보 (처분 시 참작)**: lens B는 "post-probe git status
  == pre-probe" 검사를 수행하는데, 본 세션 종료 state 갱신(본 doc 신규
  `??` 항목 + START_HERE/EXPERIMENT_LEDGER 수정)이 워크플로 진행 **중**
  발생 — lens B가 자기 probe와 무관한 repo 변화를 보고할 수 있음. 이
  항목은 본 세션 원인이므로 blocker 아님 (단, 그 외의 쓰기 흔적 주장은
  실제 검증 필요).

## 6. 정적 fixture (미실행) / 다음 단계

1. **wf_311d5910-658 회수·처분 (§5)** → blocker 0 확인. blocker 존재 시:
   처분 → 수리 → sha 재기록 → diff-한정 재검증 재실행 (L3-W2 절차).
2. 정적 fixture **1회** (sha v2 bytes 그대로 — §4 선확인 후):
   ```
   /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
     sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_controller.py \
     --mode static-prep
   ```
   prereg 선기록 = 실패 시 attempt1 네임스페이스 영구 소모 fail-closed.
   M3 수리로 내부 raise도 STATIC_FAIL 기록 후 정지. 성공 시
   attestation/tuple 자동 저작 → **tuple SHA 보고 후 정지**.
3. attempt1 = tuple SHA 인용 사용자 별도 승인 (retry 0). **승인 전
   브리핑 의무**: top-rim 기하로 전 격자 (B)-fail(A∧B 0셀)이 정당한
   과학 결과일 수 있음 (6th doc §5-3; tolerance 수리 금지 D354).
4. (사용자 병행) 기울임/전도힘 손측정 — 2nd doc §2. 보고 시 기록만.

## 7. DECISIONS.md 미갱신 사유

본 세션은 기존 처분(6th doc §2.3)의 집행 + 절차 준수(L3-W2)로, 신규
durable lesson/do-not-repeat 규칙 발생 없음 — append 없음 (규칙 준수).

## 8. Session progress rule 충족

실패 가능 검증 실행: (i) dry-run D 차등검증은 변조 writer에서 신규 AST
fixture가 실제 FAIL을 내는지 검증 (구 needle 체계는 이 시나리오를
통과했음 — 판별력 개선 실증), (ii) dry-run C는 M3 실패 경로가 실제
STATIC_FAIL을 기록하는지 검증, (iii) 8-pose bit-exact 회귀와 2-프로세스
결정성은 각각 FAIL 가능 게이트였음.

과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`. 동결 침범 0.
git commit/push 0.

## 9. §5 Addendum — 세션 말미 완주·verbatim 보존 (처분 미실행)

end-of-session 갱신 직후 `wf_311d5910-658` 완주 통지 수신 (2/2 lens, agent
오류 0, 320,492 tokens). **verbatim 보존 완료**:
`g0a_d409/design_inputs/microrepair_diff_reverify_wf_311d5910-658/`
{result_verbatim `77927381…d7ed31` / journal `f912919…dff5f612` /
manifest}. §5의 회수 절차 1~3단계는 불요 — 다음 세션은 보존 verbatim
전문 판독부터.

**Headline (처분 아님 — 전문 판독+독립 재검증 후 처분)**:
- **blocker 0 / 2 lens.** 양 lens 모두 3파일 sha v2 bit-일치를 probe
  전후 재검증. 실제 ATTEMPT_ROOT 미생성·probe는 scratchpad 한정 확인.
- warning 6건 (전부 완화 실증 동봉, 채택-차단 아님 주장):
  - A-W1≡B-W2: M2 AST 검사는 형상-기반·도달성 아님 — 전 강제블록 삭제
    + 죽은 decoy 함수로 9표면 전부 우회 가능 실증. 완화 = writer sha가
    tuple에 결박 + runtime 재해시 (fixture는 저작-시점 증거). 리뷰 자평:
    구 needle 대비 엄격히 강함(블록 삭제는 검출).
  - A-W2: deadline 표면 내 :1427-1428 단독 삭제는 미검출 (지시된 3블록
    조합 삭제는 검출됨). 동일 sha-결박 완화.
  - A-W3≡B-W1: fixture 결과에 NaN/Inf 존재 시 STATIC_FAIL 증거 파일
    저작 자체가 ValueError로 소실 (fail-closed는 유지, attestation/tuple
    미도달). 완화 실증: 실제 static_results 재귀 스캔 비유한 float 0 —
    실제 실행에서는 발동 불능.
  - B-W3: fixture 실패(flag/raise 공히)=네임스페이스 소모는 설계·문서화
    된 fail-closed 의미론 재확인 (신규 결함 아님). env-gate/prereg-build
    실패는 prereg 선기록 **전** abort라 재시도 가능 확인.
- 확인(발췌): M1 텍스트-전용(두 rule 문자열 비교/해시 경로 전무) + 수치
  bit-exact 2-pose 재현; M4 5채널 키 일치+실측 repr 5종 bit-일치
  (runtime 도달 불능 재확인); M5 유한 payload byte-동일+NaN 시 파일
  미생성; M3 양 실패 경로 실증(cause 체인 보존); 역방향 차등검증 6종
  전부 해당 표면 FAIL; 쓰기 경로 감사 = mkdir+4 exclusive-create뿐,
  diff 신규 쓰기 0; CLI 거부 3종; env gate 금일 PASS.
- **Lens B(e) 예측: 정확히 이 bytes로 실제 static-prep 실행 시
  PASS_STOP·exit 0** (probe 산출물 sha는 실제 실행 값과 다름 — 인용
  금지 명시).
- git-status 위양성 = §5 예고대로 정확히 1행(본 7th doc 신규 ??) —
  probe 쓰기 아님을 쓰기-사이트 감사로 귀속 확인.
- **운영 주의 (pre-existing, diff 밖)**: static-prep는 실행 시점
  git-dirty 스냅샷을 prereg에 소성(bake). static-prep 이후 approved
  runtime 전에 생긴 repo churn(신규 세션 doc·커밋)은 runtime admission
  fail-closed를 발동시킴 — **static-prep → 승인 → attempt1 runtime
  순서를 repo churn 없이 계획할 것** (다음 세션: 세션 초입에 fixture
  실행 권장, 상태 doc 저작은 그 후).

**다음 세션 1순위 재정의**: 보존 verbatim 전문 판독 → warning 6건 독립
재검증 → 처분 (소수리 재발생 시 sha 재기록+diff-재검증 재실행; 수용-기록
이면 sha v2 그대로) → 정적 fixture 1회 → tuple SHA 보고 후 정지.
