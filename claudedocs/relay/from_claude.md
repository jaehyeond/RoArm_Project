# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약

- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo를 여는 **다른 도구**(Codex CLI / Cursor).
  Claude가 연속으로 두 번 열려도 이 파일이 아니라 `START_HERE.md`로 재개한다.
- **덮어쓰기.** append-only 아님. 최신 인계 1건만 §2에 둔다. 과거 인계는 보존 대상이 아니다
  (실제 기록은 `claudedocs/session_*.md`가 소유).
- **상태 정본이 아니다.** 현재 상태·active case·다음 행동·수치는 전부 `START_HERE.md`가 소유하며
  **여기에 한 줄도 베끼지 않는다.** 어긋나면 `START_HERE.md`가 이긴다.
- **`HANDOFF.md`가 아니다.** HARD RULE #7은 `HANDOFF.md`라는 **파일명**을 금지한다
  (PreToolUse 후크로 기계 강제 중). 이 파일은 그 이름을 쓰지 않으며, 상태 대시보드를 대체하지도 않는다.
- **중복 금지 지도**: 현재 상태 → `START_HERE.md` / 활성 결정 → `claudedocs/DECISIONS_ACTIVE.md` →
  정본 `claudedocs/DECISIONS.md` / 최근 실험 → `claudedocs/LEDGER_RECENT.md` → 정본
  `claudedocs/EXPERIMENT_LEDGER.md` / 규칙 → `AGENTS.md`.
  **여기 적을 것은 그 어디에도 안 들어가는 것뿐** — 직전 도구가 무엇을 만졌나, 무엇을 만지지 말아야 하나,
  다음 도구가 밟을 함정, 사용자 승인 대기 항목.

## §1 템플릿 (§2를 덮어쓸 때 이 골격을 쓴다)

```
## §2 <YYYY-MM-DD> Claude → Codex

**한 일 (repo에 남은 변경)**: 파일 단위로. 없으면 "repo 무수정".
**만지지 말 것**: 진행 중이거나 사용자 승인 대기라 다음 도구가 건드리면 안 되는 경로.
**함정**: 문서가 사실과 어긋나는 지점 + 실측 근거(명령/커밋 해시/줄번호).
**승인 대기**: 사용자 결정 없이는 못 하는 항목.
**검증 방법**: 다음 도구가 위 주장을 스스로 재확인할 명령.
```

---

## §2 2026-08-27 Claude → Codex (73rd 세션)

세션 목적은 연구가 아니라 **도구 아키텍처 정비**(Orca 도입 · 교차도구 규칙 · 메모리 구조 · 후크 버그).
연구 산출물 0, 물리 0, Isaac 0, 로봇 0.

⚠️ **먼저 읽을 것**: 이 §2는 **72nd 세션이 relay를 쓰지 않고 닫힌 뒤 73rd가 사후 작성**한 것이다
(함정 ①). 72nd의 인계 내용은 여기 없으며
`claudedocs/session_20260826_72nd_paper_readthrough_codega_owlat_aoshima_d450_reverdict.md`가 유일 권위다.

**한 일 (repo에 남은 변경)** — RoArm repo 안은 **`AGENTS.md` 한 파일뿐**(`+6 −2`).

| 변경 | 내용 |
|---|---|
| `AGENTS.md` `:14~` | **규칙 범위 축소.** "같은 프로젝트에서 Claude와 Codex 편집 세션 **동시 실행 금지**" → "**상태 원장 소유권은 배타적이다**". worktree가 분리된 코드 편집·분석·읽기 전용은 도구가 달라도 동시 실행 허용. relay 인계 문장은 불변 |
| repo 밖 · `~/.claude/CLAUDE.md` | 전역에도 같은 취지로 교체 (`Never run Claude and Codex edit sessions concurrently` → state-ledger 배타 소유) |
| repo 밖 · `~/.claude/scripts/check-context.sh` | **버그 수리 2줄** — 창 크기 폴백 `200000`→`1000000`, 발동 조건에 `&& $pct -le 100` 가드. 백업 `check-context.sh.bak_20260827_pre_window_fix` |
| repo 밖 · auto-memory `MEMORY.md` | **26,411 B → 12,786 B 압축 + 현행 방향으로 재조준.** 원문 verbatim 전문 = `MEMORY_archive_20260827.md`(HARD RULE #8 보존 조항 준수) |
| repo 밖 · `~/.agents/skills/orchestration` | Orca 오케스트레이션 스킬 신규 설치(심링크 = `~/.claude/skills/orchestration`). Codex는 `.agents/skills` 네이티브 탐색으로 자동 인식 |

**만지지 말 것**

- **git commit 금지.** 미커밋 9건 전부 사용자 명시 지시 시에만 커밋한다.
- `claudedocs/DECISIONS.md` · `DECISIONS_ACTIVE.md` — **72nd(D456) 산출물이며 73rd는 무수정.**
  73rd는 원장 소유권이 비어 있는 동안에도 이 두 파일을 열지 않았다.
- `*.bak_20260826_pre_d456` / `_pre_71st` / `_pre_gitfix`, `MEMORY_archive_20260827.md`,
  `check-context.sh.bak_20260827_pre_window_fix` — 전부 되돌리기 경로. 커밋·삭제 대상 아님.
- `docs/reference/session_protocol.md` 본문 `:9-86` — 분리 전 `AGENTS_full:155-232`와 바이트 동일
  (md5 `983c186d…`). 낡았지만 **의도적으로 안 고친 상태 유지**(71st 결정).

**함정** (전부 이번 세션 실측)

1. **72nd가 relay 없이 닫혔다.** `from_claude.md`가 `08-26 20:25`에 멈춰 있었고 D456 작업분이 반영되지
   않았다. 이 §2가 그 공백을 사후에 메운 것이라 **72nd 내부 사정은 담고 있지 않다.**
2. 🔴 **`START_HERE.md`가 낡았다 — 가장 위험한 항목.** 여전히 `y3_d455`를 Active Case로 가리키는데,
   2026-08-27 확정된 프로포절 최종본(`/home/cgxr/Downloads/박재현 프로포절_직전 랩미팅.pptx`, 18슬라이드)이
   방향을 바꿨다: **이산 물체 → 연속 입자**, **놓기 학습 → 배출 위치 고정**.
   그 결과 `yard_track`의 **place 결정층(3×3 존 40mm·벽-하한 release)은 폐기**됐다.
   `START_HERE.md` 갱신은 **사용자 승인 대기**(아래) — 그전까지 이 파일을 현행으로 읽지 말 것.
3. 🔴 **D455 "관측 제거 = 완주 동작수 32→242 (7.56×)"는 이산 물체 결과다.**
   연속 입자 실험의 근거로 그대로 인용하면 안 된다. yard_track에서 살아남는 것은
   **관측→순차결정 골격과 heightmap 파이프라인**뿐.
4. **실물 경로 전체가 하드웨어에서 막혀 있다.** 프로포절은 입자 **퍼내기(scoop/그랩)**를 요구하는데
   인벤토리에는 2-jaw 클램프뿐이고 **스쿱·그랩 없음, 입자 재료 없음, 배출 용기 없음**
   (`project_hardware_inventory.md`). D452 슬리브 그리퍼도 이 방향에는 못 쓴다.
   시뮬·모델·결정층은 하드웨어 없이 병렬 진행 가능.
5. **stop-hook "Context usage is at N%" 오탐의 원인이 밝혀지고 수리됐다.**
   `check-context.sh`가 컨텍스트 창을 못 받으면 `200000`으로 가정해 실제의 약 5배를 보고했다
   (그래서 150%·135% 같은 불가능한 값이 나왔다). 69회 거부 기록의 원인이 이것이다.
   ⚠️ 다만 이전 세션들이 근거로 쓴 *"harness 카운터 14.70M/15M 잔여 = 2.0%"* 대조는 **부적절한 비교**다 —
   그건 세션 **총 토큰 예산**이고 후크가 재는 것은 **컨텍스트 창 점유율**이라 서로 무관한 양이다.
   결론(오탐)은 맞지만 근거는 *"100%를 넘는 백분율은 정의상 불가능"* 쪽을 쓸 것.
6. **동시 실행 규칙이 완화됐다** (위 `AGENTS.md` 변경). Codex/Cursor 세션은 이제 Claude 세션과
   **동시에 열려도 된다** — 단 `START_HERE.md`·`DECISIONS*.md`·`EXPERIMENT_LEDGER.md`·`relay/`를
   **쓰는 쪽은 언제나 하나**여야 한다. 원장은 worktree로 격리되지 않는다
   (`DECISIONS.md`는 Dxxx 순번 append, `START_HERE.md`는 통째 overwrite).
7. **RoArm 후크 3건은 여전히 Claude에만 걸린다** (71st 결론 불변).
   #7(`HANDOFF.md`)·#5(`JOINT_LIMITS`)·#28(`outputs/` 삭제)는 `~/.claude/settings.json` PreToolUse 경로다.
   Codex 세션에서는 산문 규칙일 뿐이니 `AGENTS.md` HARD RULES를 스스로 지켜야 한다.
8. **`outputs/`·`logs/`·`collected_data*` 등 14경로는 심링크다** (70th 콜드 아카이브 이관).
   외장 `ROBOT_DEV` 미마운트 시 전부 끊긴다 → `ARCHIVE_INDEX.md`의 새경로 열 참조.
   **이관분은 사본 1개이며 백업이 아니다.**

**승인 대기** (사용자 결정 없이는 착수 금지)

- 미커밋 9건의 **git 커밋**.
- 🔴 **`START_HERE.md`를 프로포절 최종본 기준으로 갱신** — 함정 ②. 폐기 자산 판정을 포함해야 한다.
- 🔴 **스쿱/그랩 엔드이펙터 설계 방향** — 슬라이드 16의 크레인 확장을 고려하면 단순 삽이 아니라
  **조개형 그랩(clamshell)** 축소판이어야 전이된다(슬라이드 3: 그랩 = "퍼 올리는 **집게** 장치").
  3D프린터 보유. 함께 정해야 할 것: **입자 재료**(모래/쌀/펠릿 — 입자 크기·마찰이 물성 조건 축이라
  실험 설계에 직결) · **배출 용기** 규격.
- **프로포절 v2 수정 7건 + LEDGER 등재** — 72nd 미실행분.
- 이월: T3 45G 2사본화 / Kinect depth 비교 / 파일럿 `E:\` 이관.

**검증 방법** (이 파일을 믿지 말고 직접 확인)

```
git status --short --untracked-files=all              # 미커밋 9건 실제 목록
git diff --stat AGENTS.md                             # 73rd repo 변경 = 이 한 파일 (+6 −2)
sed -n '12,20p' AGENTS.md                             # 함정 ⑥ 원장 배타 소유 규칙 원문
grep -n 'Active:' START_HERE.md                       # 함정 ② → 아직 y3_d455를 가리키면 낡은 것
md5sum claudedocs/DECISIONS.md claudedocs/DECISIONS_ACTIVE.md   # 73rd 무수정 확인용 기준값 확보
grep -n 'max_context=\|pct -ge 85' ~/.claude/scripts/check-context.sh   # 함정 ⑤ → 1000000 / -le 100
wc -c ~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY*.md
ls -l outputs logs collected_data                     # 함정 ⑧ → 심링크 여부
ls ~/.agents/skills/                                  # orchestration 설치 확인
```
