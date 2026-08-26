# Reference — Session Protocol (부팅/종료 프롬프트 원문 · 상태 파일 규칙)

> 출처: 분리 전 `AGENTS.md` **155-232행**. 원본 전체는 `docs/archive/AGENTS_full_20260825_pre_split.md`.
> 아래 본문은 원본에서 **바이트 동일**하게 이동했다 (2026-08-25). 요약 규칙은 `AGENTS.md`에 남아 있다.
> 새 세션을 열 때/닫을 때 이 파일을 열어 프롬프트를 그대로 복사해 쓴다.

---

### Session boot prompt (paste this verbatim at new-session start)

```
Read AGENTS.md first, then follow the Current-State Protocol exactly.

Step-by-step:
1. Read START_HERE.md.
2. Read claudedocs/DECISIONS.md.
3. Read claudedocs/EXPERIMENT_LEDGER.md.
4. Read only the claudedocs/session_*.md files referenced by START_HERE.md
   unless missing evidence requires more.
5. Run `git status --short`.
6. Brief me on:
   - Current verified state (with file:line citations)
   - Active pivot vs reserve pivots
   - Open risks / do-not-repeat rules from DECISIONS.md
   - Next concrete action

Rules:
- Be critical, analytical, and skeptical. Cross-verify before claiming.
- Do not rely on memory-only claims. Cite the referenced file/line.
- Verify metrics from logs/data files; flag any mismatch.
- Do not treat HANDOFF.md or TASKS.md as current state.
- If context approaches 95%, stop new work and run the end-of-session update.
- We are continuing the RoArm Isaac Lab hierarchical chain skills work
  (or whatever START_HERE.md says is the active pivot — do not assume).
```

### End-of-session update prompt (paste before closing session)

```
Before ending this session, update the project state system.

Step-by-step:
1. Update START_HERE.md (overwrite) with:
   - Current truth (latest session_*.md link)
   - Current status (key metrics, latest run results)
   - Current direction (active pivot + next concrete step)
   - Must-read first list
   - Do-not-trust-as-current list
2. Append to claudedocs/EXPERIMENT_LEDGER.md any major run/result row
   (Date | Run | Goal | Key Result | Verdict | Source).
3. Append to claudedocs/DECISIONS.md ONLY if a durable lesson, failure rule,
   or do-not-repeat rule changed. Use Dxxx numbered sections with Evidence /
   Implication / Source.
4. Write a new claudedocs/session_YYYYMMDD_short_title.md (append-only) with
   detailed metrics, code paths, evidence, decisions, next steps.
5. Do not overwrite previous session logs.
6. Keep START_HERE.md short (~120 lines). Put history in the ledger and
   detail in the session doc.
7. Cross-verify: re-read all 4 files (START_HERE, DECISIONS, EXPERIMENT_LEDGER,
   new session doc). Check numbers match across files.
```

### Context 95% emergency protocol

If active chat context approaches 95%:

1. Stop new implementation work immediately.
2. Run the end-of-session update prompt above (state files only — no new code).
3. Output a concise continuation prompt for the next session (≤80 lines, lists
   active pivot, next concrete step, files to read, current md5s).
4. Do NOT use `/half-clone` or `/handoff` skills (project rule, see auto-memory
   HARD RULE #11).
5. User starts a new session and pastes the boot prompt above.

### Project rules for state files

- `START_HERE.md` is the current dashboard and is overwritten as work progresses.
- `claudedocs/DECISIONS.md` is append-only durable lessons / do-not-repeat rules.
- `claudedocs/EXPERIMENT_LEDGER.md` is append-only major experiment history.
- `claudedocs/session_*.md` files are detailed append-only session logs.
- `HANDOFF.md` and `TASKS.md` are historical and stale unless `START_HERE.md`
  explicitly points to them.
- Auto-memory `MEMORY.md` is per-user across conversations; it is complementary
  to (not a replacement for) the project-state docs above. Update both when a
  session closes: project-state for repo continuity, MEMORY.md for user-level
  habits/preferences and HARD RULES.
