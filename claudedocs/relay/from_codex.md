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

## §2 2026-09-11 Codex → Claude

**만진 파일**: `boot_check_20260911/{plan_outlet_tilt.py,render_outlet_plan.py,hw_outlet_tilt.py,analysis_outlet_run.py,verify_outlet_tilt.py,hw_recorded_arm.py,outlet_tilt_01/}` 및 세션/START/LEDGER/색인. 기존 실행 사본은 보존. `RecordedArm`는 선택적 인스턴스 검증기만 추가했고 기본 guard는 유지된다.

**함정**: 실기 동결 `execution_01/hw_outlet_tilt.py`는 result.completed=false여도 shell0을 반환한다. 정본은 실패2로 수정했고 가짜 로봇 마지막5.01도 편차 회귀가 PASS. 성공 여부는 항상 result/원시 settle을 읽을 것. 실제 마지막 자세는 START 정본이며 HOME/P1로 추정하지 않는다. 이전 place/roll 드라이버에 현재 자세를 넣으면 안 된다.

**자료 함정**: S1은 g18 크랭크가 제거된 직결형. 그 사실과 새 경로 CAD검사를 생략하고 예전g18±14도 또는 새 임의롤범위로 일반화 금지. nominal IK/CAD만 컵/실제접촉 보증으로 확대하지 않는다. 계획·실제 RRD는 별도 폴더이고 사전 plan이 실제20도 달성 증거는 아니다. after 질량이 없으면 기존22.28g를 새 after값으로 대입하지 않는다.

**만지지 말 것**: W1~W10/원시 PID/과거900·790·roll원본·실행별동결소스. 실기 완료 뒤 새 PID/보호값/형상 변수는 추가하지 않았다. 루트 oldManual도 그대로다. Rerun0.34.1 분석은 isaaclab, 하드웨어는roarm을 유지한다.

**승인/입력 경계**: 사용자 다음 동작과 앞서 commit/push를 명시 요청했다. 반복 실행승인 질문은 필요 없다. 현재 필요한 것은 사용자 후 질량/잔류 사실 입력이다. 자동 복귀/5회수집 완료로 해석하지 않는다. 대시보드의 시작 자세와 다음 단계를 먼저 읽는다.

**검증 방법**: `verify_recorded_arm.py`, `verify_outlet_tilt.py`는 fake serial만 사용한다. execution command_audit/result/raw, Rerun validation/inspection, manifest를 읽는다. git origin/master가 올바른 게시 대상이며 임의 다른worktree로 옮기지 않는다.
