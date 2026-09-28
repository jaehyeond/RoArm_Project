# from_codex.md — Codex/Cursor → Claude 인계 (relay)

## §0 이 파일의 규약

- **쓰는 쪽 = Codex CLI 또는 Cursor 세션 하나.** 읽는 쪽 = 다음에 이 repo를 여는 **Claude** 세션.
  Codex가 연속으로 두 번 열려도 이 파일이 아니라 `START_HERE.md`로 재개한다.
- **덮어쓰기.** append-only 아님. 최신 인계 1건만 §2에 둔다. 과거 인계는 보존 대상이 아니다
  (실제 기록은 `claudedocs/session_*.md`가 소유).
- **상태 정본이 아니다.** 현재 상태·active case·다음 행동·수치는 전부 `START_HERE.md`가 소유하며
  **여기에 한 줄도 베끼지 않는다.** 어긋나면 `START_HERE.md`가 이긴다.
- **`HANDOFF.md`가 아니다.** HARD RULE #7은 `HANDOFF.md`라는 **파일명**을 금지한다. 이 파일은 그 이름을
  쓰지 않으며, 상태 대시보드를 대체하지도 않는다.
- **중복 금지 지도**: 현재 상태 → `START_HERE.md` / 활성 결정 → `claudedocs/DECISIONS_ACTIVE.md` →
  정본 `claudedocs/DECISIONS.md` / 최근 실험 → `claudedocs/LEDGER_RECENT.md` → 정본
  `claudedocs/EXPERIMENT_LEDGER.md` / 규칙 → `AGENTS.md`.
  **여기 적을 것은 그 어디에도 안 들어가는 것뿐** — 직전 도구가 무엇을 만졌나, 무엇을 만지지 말아야 하나,
  다음 도구가 밟을 함정, 사용자 승인 대기 항목.

### Codex가 이 파일을 쓸 때 특히 남길 것

Claude는 auto-memory(`~/.claude/projects/.../memory/`, 토픽 86개)를 갖지만 **Codex에는 영속 기억이 없다**
(`~/.codex/memories_1.sqlite` 0행, `~/.codex/memories/` 0개 — 2026-08-25 확인).
반대로 **Claude는 Codex 세션에서 무슨 일이 있었는지 알 방법이 이 파일밖에 없다.**
Codex가 repo 파일에 남기지 않은 판단·시도·실패는 그대로 소실된다.

## §1 템플릿 (§2를 덮어쓸 때 이 골격을 쓴다)

```
## §2 <YYYY-MM-DD> Codex → Claude

**한 일 (repo에 남은 변경)**: 파일 단위로. 없으면 "repo 무수정".
**만지지 말 것**: 진행 중이거나 사용자 승인 대기라 다음 도구가 건드리면 안 되는 경로.
**함정**: 문서가 사실과 어긋나는 지점 + 실측 근거(명령/커밋 해시/줄번호).
**승인 대기**: 사용자 결정 없이는 못 하는 항목.
**검증 방법**: 다음 도구가 위 주장을 스스로 재확인할 명령.
```

---

## §2 2026-09-16 Codex → 다음 세션 — 교수 질의 감사와 로컬 보관 설정

**한 일 (repo에 남은 변경)**: 메인의 START_HERE, BACKLOG append, 새 session_20260916_professor_physics_lfs_defer.md, CONTINUE_20260916_PHYSICS_AUDIT_DT.md, LFS_DEFERRED_20260916.md, research/professor_review_20260916/와 이 relay. 메인·pellet-model·w12-isaac-replay·w13-cycle-audit·w13-full-cycle의 .gitignore와 index만 보관 요청 범위에서 변경했다. commit/push/과거 이력 변경 없음.

**만지지 말 것**: 기존 W13 run_01/rev28/post03·모든 미디어/NPZ·기존 실패 보고·pellet의 기존 세 .bak. staged D를 보고 디스크 파일까지 삭제하거나 worktree를 제거하지 않는다. 기존 worker 원장은 main에 merge하지 않는다. 기존 .gitattributes/커밋을 임의 되돌리지 않는다.

**함정**: ignore해도 기존 추적 파일에는 효력이 없어서 이번에는 index만 제외했다. 과거 커밋에는 포인터가 남아 있어 바로 push하면 LFS 전송이 필요할 수 있다. 기존 git_publication.mjs의 stage(force-add)/push를 재사용하지 않는다. 명단은 백업 완료 증명이 아니다. 공식 문서의 기능과 이번 실행의 실제 포함 항목을 혼동하지 말고, 카메라 focal_mm 필드명도 물리 mm 측정값으로 복사하지 않는다. 불변 원시 파일과 새 파생 분류를 혼합하지 않는다.

**승인 대기**: GPU 명령/예산, 물성·형상·제어 변경, 실물 조회·구동·PID/토크·카메라, 설치·학습·PBD 하이브리드·A/B/C, commit/push/이력 정리는 새 명시 승인 없이는 하지 않는다. 다음 행동의 정본은 START와 최신 continuation이며 과거 GO/COMMANDS는 실행 권한이 아니다.

**검증 방법**: research/professor_review_20260916/verify_review.mjs를 읽은 뒤 physics/git/docs 모드를 사용한다. CPU 원자료 감사와 파일 보존/문서 링크 검사이며 새 물리나 원격 전송은 없다. defer_lfs.mjs prepare/apply는 이미 완료한 일회 작업이므로 재실행하지 않는다. HEAD/원본/ignore/index의 정확한 상태는 LFS_BEFORE/AFTER로 확인한다. 다음 세션이 상태 원장 소유권을 넘겨받는다.
