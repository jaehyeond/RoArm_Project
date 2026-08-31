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

## §2 2026-08-26 Claude → Codex

**한 일 (repo에 남은 변경)** — 세션 목적은 연구가 아니라 **문서/메모리 구조 수리**. 연구 산출물 0건.

| 변경 | 내용 |
|---|---|
| `AGENTS.md` | 724 → 234줄. 레퍼런스 12개 절을 `docs/reference/` 5파일 + `docs/archive/` 1파일로 **이동**(삭제 0건). 그 뒤 부트 절차·relay 배선으로 → 현재 줄수는 파일 확인 |
| `docs/reference/`·`docs/archive/` | 신규. `AGENTS.md`의 참조 표가 가리킴 |
| `claudedocs/DECISIONS_ACTIVE.md` | 신규 161줄. 활성 결정 37건 + `DECISIONS.md` **줄번호** 앵커 |
| `claudedocs/LEDGER_RECENT.md` | 신규 144줄. 원장 최근 20건(`:512`~`:531`) + 줄번호 앵커 |
| `claudedocs/relay/` | 신규 = 이 파일 + `from_codex.md` |
| `.gitignore` | `docs/` 통째 무시 → `docs/*` + `!docs/reference/` `!docs/archive/`. **이걸 안 고치면 새 클론에서 `AGENTS.md` 참조 표가 전부 깨진다** |
| `CLAUDE.md` | 이동한 세션 프롬프트 포인터 3곳 갱신 (Claude 전용 파일) |
| repo 밖 | Claude auto-memory `MEMORY.md` 다이어트 + PreToolUse 후크 3건 — **Codex에는 영향 없음**(아래 함정 ④) |

**만지지 말 것**
- **git commit 금지.** 위 변경 전부 미커밋이며 사용자가 명시 지시할 때만 커밋한다
  (`AGENTS.md` Safety constraints + `START_HERE.md`).
- `AGENTS.md.bak_20260826_pre_relay`·`AGENTS.md.bak_20260825_pre_split` — 되돌리기 경로. 커밋 대상 아님
  (동일 사본이 `docs/archive/AGENTS_full_20260825_pre_split.md`로 추적됨).
- `claudedocs/DECISIONS.md`·`EXPERIMENT_LEDGER.md` — **append-only.** 요약본이 생겼다고 원본을 손보지 않는다.
  이번 세션 무수정 확인(줄수·바이트 불변).

**함정** (전부 이번 세션 실측)

1. **`AGENTS.md` 줄번호 인용이 전부 어긋났다.** 세션문서들이 `AGENTS.md:73`·`:266` 식으로 인용 중인데
   724→234줄 재편성으로 무효. **2026-08-25 이전 인용은
   `docs/archive/AGENTS_full_20260825_pre_split.md`(724줄, 바이트 동일)에서 해석**할 것.
2. **`START_HERE.md`의 미커밋 목록(현재 "Open risks" 절)이 낡았다.** 거기 적힌
   `sim_assets/`(54파일)와 `claudedocs/runtime_logs/yard_track/`(115파일)은 **이미 커밋됐다** —
   커밋 **`4a38896` "ai setting toml" (2026-08-25 16:04)**. `git ls-files`·`git log --diff-filter=A`로 확인.
   `START_HERE.md`는 2026-08-17에 마지막 갱신됐고 그 뒤 커밋이 2건 있었다. **이 절은 아직 수정 안 했다**
   (상태 문서는 종료 세션이 소유 — 구조 수리 세션이 건드릴 범위 밖).
3. **`docs/reference/session_protocol.md`의 부트 프롬프트는 옛 절차를 지시한다.** 그 파일은 분리 전
   `AGENTS.md`의 **바이트 동일 사본**이라 2·3단계가 여전히 `DECISIONS.md`·`EXPERIMENT_LEDGER.md`
   **통독**을 시킨다(각각 28,097줄 / 줄당 평균 2 KB). **최신 절차는 `AGENTS.md`
   "Session boot procedure"** — 요약본 + 줄번호 앵커를 읽는다. 충돌 시 `AGENTS.md`가 이긴다.
4. **RoArm 후크 3건은 Claude에만 걸린다.** HARD RULE #7(`HANDOFF.md`)·#5(`JOINT_LIMITS`)·#28(`outputs/` 삭제)는
   `~/.claude/settings.json`의 PreToolUse로 기계 강제되는데, **Codex는 이 설정을 읽지 않는다.**
   Codex 세션에서 이 3건은 **산문 규칙일 뿐**이므로 `AGENTS.md` HARD RULES를 스스로 지켜야 한다.
   ⚠️ **2026-08-26 정정**: 처음엔 "`~/.codex/config.toml`에 hook 항목 없음"을 근거로 들었는데,
   같은 날 Orca 첫 기동이 Codex에 **자체 후크 8종**(`[hooks.state]` + `~/.codex/hooks.json`,
   SessionStart 등 → `~/.orca/agent-hooks/codex-hook.sh`)을 **자동으로 심었다.** 이제 Codex에도
   후크 배선은 존재하지만 **전부 Orca 오케스트레이션용이고 RoArm 규칙과 무관**하다 —
   결론(#7·#5·#28 미강제)은 그대로다. Codex 후크를 봤다고 RoArm 규칙이 걸린 걸로 오해하지 말 것.

**승인 대기** (사용자 결정 없이는 착수 금지)
- 위 전부의 **git 커밋**.
- **`EXPERIMENT_LEDGER.md` 데이터 결함 4건** — 등재 누락 3건(`56th`·`57th`·`70th`, 원장 언급 0회.
  57th는 `D444` 개시 세션) / 최근 3행(`:529`~`:531` = 현재 야드 피벗 전체)의 4열 스키마 드리프트 /
  `:105`~`:531` 427행이 표 헤더 없이 이어져 렌더 안 됨 / `:92-103`의 죽은 상태 12줄
  ("Active pivot (2026-05-21)" — 현재 피벗과 모순). **append-only 문서라 수정에 사용자 승인 필요.**
- `START_HERE.md` 미커밋 절 정정(함정 ②).

**검증 방법** (이 파일을 믿지 말고 직접 확인)
```
git log --oneline -1 --diff-filter=A -- sim_assets        # 함정 ② → 4a38896
git status --short --untracked-files=all                  # 미커밋 실제 목록
wc -l AGENTS.md docs/archive/AGENTS_full_20260825_pre_split.md   # 함정 ① → 724줄 원본
grep -n "Read claudedocs/DECISIONS.md" docs/reference/session_protocol.md   # 함정 ③
grep -n "PreToolUse" ~/.claude/settings.json              # 함정 ④ (Claude 전용 경로)
```
