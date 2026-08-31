# 71st — 부트 7단계 재검증 + 후속 과제 3건 진단·수리 (원장 소급 등재·앵커 재계산·상태문서 정정·부트 이중지시 차단) — 문서 무결성 전용, 물리 0

날짜: 2026-08-26 (70th 이후 첫 RoArm 세션. 08-25~08-26 구조 수리 작업은
`/home/cgxr`에서 수행돼 이 repo에 세션문서가 없다 — 근거는 `/home/cgxr/NEXT_SESSION_PROMPT.md` 5장).
성격: **문서/원장 무결성 수리 전용.** 연구 실험 0, 물리 0, Isaac 0, 로봇 0, git 커밋 0.

**이번 case의 신규 변수: 없음** (D322~). yard_track 변수 사다리와 무관한 문서 무결성 작업이며,
`START_HERE.md` `Active Case`(`y3_d455` 완료)는 이 세션이 바꾸지 않았다.

**Session progress rule 정당화**: 사용자가 "승인 없이 파일 수정 금지 — 진단과 선택지 제시까지만"을
명시했고, 이후 승인 범위도 문서 수리로 한정됐다. 물리 실험 대상 case(y4)는 사용자 결정 대기다.
다만 **실패 가능한 검증 게이트는 두었다** — 26항목 검증 스위트가 사전 보장 없이 돌았고
실제로 결함 1건을 잡아냈다(§4).

**D341 Rerun 계약**: 순수 파일/해시/스키마 감사로 기하·자세·좌표계·궤적·시간 판단이 없다 →
계약의 명시적 면제 조항 해당. 본 절이 그 정당화 기재다.

## 1. 부트 7단계 (AGENTS.md `### Session boot procedure`)

사용자 지시로 `docs/reference/session_protocol.md`의 낡은 boot prompt(통독 지시)를 따르지 않고
`AGENTS.md` 7단계를 실행했다. **`DECISIONS.md`·`EXPERIMENT_LEDGER.md` 통독 0회.**

| 단계 | 결과 |
|---|---|
| 1 `START_HERE.md` | 70th판(113줄), Last updated 2026-08-17 |
| 2 `DECISIONS_ACTIVE.md` | 앵커 D450~D455 **6/6 실측 일치**(`grep -n '^## D45[0-5]'`) |
| 3 `LEDGER_RECENT.md` | 원장 531줄 = 전제값 일치 → `:512`~`:531` 앵커 유효 확인 |
| 4 `relay/from_codex.md` | **§2 비어 있음** — Codex 인계 없음(파일 신설 2026-08-26 이후 미사용) |
| 5 세션문서 | 69th·68th·63rd·70th + `ARCHIVE_INDEX.md` |
| 6 `git status --short` | 출력 없음 = 워킹 트리 깨끗, HEAD `0606201` |
| 7 수치 재확인 | 아래 §2 |

## 2. 7단계 — 수치 교차 검증 (원본 JSON 직접 파싱, 불일치 0)

권위 = `claudedocs/runtime_logs/yard_track/y3_d455/a{1..8}_results.json`.

| 인용 수치 (출처) | 원본 실측 | |
|---|---|---|
| blind 242 vs 게이트 32 = 7.56× (69th:64-67) | a5 `total_actions`=242·`n_noop`=210, 나머지 7 arm 전부 32 (242/32=7.5625) | ✅ |
| a5 물리 부분수열 ≡ a2 (69th:66) | 분산·reshape·위반·`total_steps`(2404) **전 필드 동일** | ✅ |
| release 33.9→21.7 / 78.1→91.6 (69th:68-73) | a8 p95 33.92·hmax 78.05 → a1 p95 21.66·hmax 91.56 | ✅ |
| reshape Σ 92/190/238/346 (69th:75-76) | a1 92·a2 190·a4 238·a3 346 (3.76×) | ✅ |
| place 단독 위반 5→0 (69th:84-86) | a1 `hmax_violations_final`=5 → a6=0 | ✅ |
| 예측2 반증 a7=4 (69th:80) | a7 final=4 (>예측 ≤2) | ✅ |
| rep2 bit-재현 2302 Δ=0.0 (69th:48-49) | `max_dh_bin_mm`=0.0, rep1/rep2 `total_steps` 둘 다 2302, `branch`=`i_repeatable` | ✅ |
| 8/8 전 게이트 PASS | 8 arm × 게이트 5종 전부 `True` | ✅ |
| outputs 79.1GB/467파일 (70th:29) | 79,058,293,814 B / 467 files | ✅ |

**신규 인용 주의 1건**: a3는 `g_final_hmap: True`인데 `max_abs_diff_mm`=**31.54mm**다. 모순이 아니라
D453 ②에 따라 **max 절대 게이트를 의도적으로 제외**하고 p95/비율(0.99)만 게이트로 쓰기 때문
(68th:19-20). "G-final-hmap PASS"를 "높이맵 오차가 작다"로 읽으면 오독이다.

## 3. 진단 3건 → 승인 → 적용 (근거: `/home/cgxr/NEXT_SESSION_PROMPT.md:657-665`)

### 3-1. 원장 결함 — 소급 등재 2행 + `## Schema errata` (승인안 ⓓ+ⓨ)

**진단 중 초판 프레이밍을 정정했다.** `LEDGER_RECENT.md` §2 ①이 "등재 누락 3건"이라 썼는데
세션문서를 열어 보니 셋이 같은 성질이 아니었다:

| 세션 | 자기 기록 | 판정 |
|---|---|---|
| 56th | doc `:72` "LEDGER append 0 (실험 없음), DECISIONS append 0" | 명시적·정당화된 미등재. 산출물 0 → **등재 안 함** |
| 57th | doc `:67-68` "LEDGER append 0 (물리 실행 없음 — 실행 세션에서 fg1 row 기록 예정)" | 명시 결정이나 **D444 개시가 원장에서 소실** → 소급 등재 |
| 70th | 원장/LEDGER 언급 **0회** | **유일한 무기록 누락** → 소급 등재 |

**신규 발견 — 등재 관행 자체가 비일관이었다.** 물리 0인데 등재된 행이 이미 있다: `:525` 63rd(조사
전용)·`:528` 66th(저작 전용). 즉 "실험 0이면 미등재"는 지켜진 적 없는 암묵 규칙이다.
새 기준을 원장 `### 등재 관행 메모`(`:553~`)에 기재했다 — **`Dxxx`를 낳았거나 되짚어야 할
산출물·상태 변경을 만든 세션은 물리 실행 여부와 무관하게 등재**, 미등재 시 세션문서에 사유 기재.

적용 (append만, Python으로 파일 끝 `|\n` 확인 후 이어붙임):
```
1,062,466 B → 1,071,997 B (+9,531)   531줄 → 564줄
앞 1,062,466 B 바이트 불변: md5 0a6d7071539051e99de33d00bd0a8608 == 백업   ✅

:532  57th  G0B_D444_CASE_OPENED__PREREG_FROZEN__NO_PHYSICS  (D444)      6열(필드8)
:533  70th  T1_T2_COLD_ARCHIVE_MIGRATED__SHA256_ALL_MATCH__
            SINGLE_COPY_NOT_BACKUP__T3_PENDING                            6열(필드8)
:535  ## Schema errata      — :529~:531의 누락 Run/Path·Goal + 소급 판정 토큰 3종 (표 밖)
:553  ### 등재 관행 메모
```
⚠️ **판정 토큰은 전부 2026-08-26 소급 부여**이며 원 세션문서·`DECISIONS.md`에 없다. 각 행/절이
그 사실을 스스로 명시한다. 판정의 정본은 계속 `DECISIONS.md` 원문이다.
`:529`~`:531` 원행은 필드수 6(=4열) 그대로 **무수정 확인**. `:92-103` 죽은 상태 절도 삭제 안 함.

### 3-2. `LEDGER_RECENT.md` 앵커 재계산

```
앵커 :512~:531 → :514~:533  (20건 유지, 연속·누락 0·중복 0, 원장 실제 줄과 20/20 기계 대조)
144줄 → 191줄 (상한 200)
```
- §2 ① 판정을 위 3-1대로 정정, §2 ②에 errata 보정 기재 추가.
- 상한 밖으로 밀린 `:513` 51st·`:512` 50th-b는 **삭제하지 않고** 앵커+교훈 한 줄로 회전 처리
  (원문은 `DECISIONS_ACTIVE.md` §8 D437-R1 `:26932`·D438-R1 `:27181`가 보유).
- ⚠️ **append 순 ≠ 시간 순**이 됐다 — `:532`(57th, 08-13)가 `:531`(69th, 08-16)보다 뒤. §1·§3에 경고 기재.
- §1 재확인 명령을 `wc -l` → `grep -n '^## Schema errata'`로 교체(표 밖 절이 생겨 `wc -l`이 표 끝이 아니다).

### 3-3. `START_HERE.md` 미커밋 절 정정 (승인안 ⓐ)

`:73-77` 5줄이 **전부 거짓**이었음을 실측 확인:
```
git status --short                                → 출력 없음
git log -1 -- sim_assets/ | yard_track/ | ARCHIVE_INDEX.md | 70th doc  → 4a38896
git log -1 -- .gitignore                          → 0606201
git ls-files: sim_assets 54 · yard_track 115 · p26~p32 7  전부 추적됨
.gitignore에 두 경로를 막는 규칙 없음 → "whitelist 추가 필요"도 무의미
```
→ 8줄로 교체 + **`:92`의 다음행동 후보 (f) "git 커밋"도 완료 처리**했다.
승인 문구의 줄 범위를 1줄 넘긴 확장이며, 그대로 두면 다음 세션이 **이미 끝난 커밋 작업을
시작**하기 때문이다. 사용자에게 명시 보고했다.

### 3-4. `session_protocol.md` 부트 이중 지시 차단 (승인안 ⓑ)

**진단 중 "바이트 동일 보증"의 실제 범위를 확정했다** — 파일 전체가 아니다:
```
sed -n '9,86p'    session_protocol.md              md5 983c186da129a74e03c5e62904babbdf
sed -n '155,232p' AGENTS_full_20260825_pre_split.md  md5 983c186da129a74e03c5e62904babbdf
→ 본문 :9-86(78줄/3,461 B)만 보증 대상. 머리말 :1-7은 분리 세션이 새로 쓴 안내문 = 보증 밖.
```
따라서 **머리말에만** 경고를 넣어 보증을 전혀 깨지 않고 해결했다(본문 md5 수정 전후 동일 검증).
경고 내용 3건: ⓐ 보증 범위가 본문 78줄뿐 ⓑ 2·3단계 통독 지시가 낡았고 현행 단일 소스는
`AGENTS.md` 7단계 ⓒ **신규 발견 — `Rules` 마지막 줄 "RoArm Isaac Lab hierarchical chain skills
work"가 죽은 상태**(현재 pivot은 포스코 야드). 이 ⓒ는 후속 과제 3건 목록에 없던 항목이고
`AGENTS_full:179` 원본에도 있으므로 이동이 만든 결함이 아니다.

부수 확인: 종료 프롬프트 2번 항목은 **이미 6열**(`Date | Run | Goal | Key Result | Verdict |
Source`)을 지시하고 있었다 — `:529`~`:531`의 4열 드리프트는 이 프롬프트를 안 지킨 결과였다.

## 4. 검증 (26항목 → 실질 FAIL 0)

초기 24 PASS / 2 FAIL → 재검증. **검사기 오탐이 또 나왔다((라)·(마)·(바)·(사)에 이어 5·6번째):**

| 항목 | 판정 | 처리 |
|---|---|---|
| `LEDGER_RECENT.md:187`(§5)에 낡은 앵커 `:512~:531` 잔존 | **진짜 결함** | 수정 + "표 끝은 `wc -l`이 아니다" 주의 추가 |
| `HANDOFF.md` 존재 | **검사기 오탐 5** | 2026-03-23 커밋 `3eda0f4`의 기존 추적 파일. `git status` 출력 0·mtime 불변 = 이번 세션 미변경. 검사 항목이 "미생성"이 아니라 "미변경"이어야 했다 |
| `:514~:533` 문구 2회 출현 | **검사기 오탐 6** | §1·§5 양쪽에 있어야 정상. 기대값 1이 틀렸다 |

무수정 확인: `DECISIONS.md` md5 `bfb324fd…` 불변 / `AGENTS_full_20260825_pre_split.md` md5
`670b5c6d…` 불변 / 원장 앞 1,062,466 B 불변 / `session_protocol.md` 본문 md5 불변.

## 5. 순응 확인

- 물리 0, Isaac 0, 로봇 0, lerobot-train 0, git 커밋 0(사용자 전담), HANDOFF 생성·편집 0.
- 동결 case 편집 0, `DECISIONS.md` 무수정, 원장은 **append만**(기존 행·죽은 상태 절 삭제 0).
- 수정 4파일 전부 사전 백업(`*.bak_20260826_pre_retro`/`_pre_gitfix`/`_pre_note`) — 커밋 대상 아님.
- 사용자 승인 범위: 진단 → D절 3건 → 세션 마감 3건(본 doc + relay + MEMORY). 각 단계 별도 승인.

## 5-1. Stop-hook /half-clone 요구 → 거부 (64~67회째 [가정])

- 08-26 네 차례(86% / 117% / 122% / **150%**) 전부 **HARD RULE #11 + AGENTS.md Context 95%
  protocol #4로 거부**. harness 토큰 카운터는 매번 ≈14.8~14.99M/15M 잔여(0.1~1.3% 사용)로 모순 —
  51~63회째와 동일한 `check-context.sh` 오탐 패턴.
- **단조 증가 반증**: 요구 %가 86→117→122→150으로 커지는 동안 harness 실사용은 오히려 줄었다
  (1.3%→0.3%). 즉 hook이 읽는 값은 harness 컨텍스트 점유율과 **상관관계 자체가 없다.**
- 수리는 사용자 승인 사안이라 손대지 않았다. (67회째는 본 doc 작성 후 사후 추가 — 57th doc §9 전례)

## 6. 다음 (사용자 결정 대기)

1. **git 커밋** — 수정 4파일 + 본 doc(백업 4개 제외). 미커밋 상태다.
2. **논문 완독 (사용자 지시, 다음 세션 1순위)** — 로컬 PDF 3파일. 산출물 = 세션문서 + auto-memory
   토픽 파일(사용자 승인 완료):
   - `/home/cgxr/Downloads/p048.pdf` (11p) = **CoDeGa**, Zhu·Thangeda·Ornik·Hauser (UIUC), RSS 2023
   - `/home/cgxr/Downloads/입상물질(후속논문).pdf` (9p) = **arXiv 2311.17405v2**, 같은 그룹 + NASA JPL
     OWLAT 실배포 경험 보고 (위 논문의 후속)
   - `/home/cgxr/Downloads/2501.06583v3.pdf` (18p) = **arXiv 2501.06583v3** "Optimizing wheel loader
     performance", Aoshima(Umeå+**Komatsu**)·Wadbro·Servin, 2025-07-10
   🔴 **세 번째 논문은 갭 판정에 직결된다.** 초록이 이미 *"each loading's performance depends on the
   pile state, which depends on previous loadings"* + *"look-ahead tree search is **6% more efficient
   than a greedy strategy** over a horizon of **15 sequential loadings**"* 라고 말한다. 우리 **RQ2**
   ("재형성 더미에서 greedy가 완주까지 최소 동작수인가")와 겹칠 수 있으므로 **HARD RULE #4 +
   D450 3-결합 갭 문구 재판정**이 필요하다. 확인할 것: 학습 대상이 world model+tree search인가
   정책 RL인가 / place(load receiver) 선택을 학습했나 고정했나 / "완주" 개념이 있나 15-loading
   유한 지평선뿐인가 / 이산 물체 집기인가 연속 벌크 스쿱인가. **갭이 무너지면 무너졌다고 기록한다.**
3. **사용자가 "달라진 것들"을 설명 예정** — 논문 완독 후. 그 전에 y4 등 신규 case 착수 금지.
4. 이월: T3 45G 2사본화 / Kinect depth 비교 / 실물 제작 / 파일럿 E:\ 이관 / 프로포절 v2 검토.
