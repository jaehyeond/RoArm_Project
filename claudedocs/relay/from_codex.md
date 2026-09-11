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

**만진 파일**: `boot_check_20260911/{plan_scoop_tilt_cycle.py,hw_scoop_tilt_cycle.py,analysis_scoop_tilt_cycle.py,audit_scoop_tilt_cycle.py,verify_scoop_tilt_cycle.py,plan_finish_home.py,verify_finish_home.py,scoop_tilt_cycle_01/}`·직전outlet관찰JSON·START/세션/LEDGER/색인. 기존기본RecordedArm/JOINT_LIMITS/PID코드는이번턴미변경.

**함정**: 실행별completed와전체목표달성을분리한다. combined_01/REPORT_initial_stop은중간보고이며최신은START→REPORT→combined_02. 이전FAIL을고쳐서성공으로만들지말것. 마지막기울임은이전고정립점IK와다른고정지지관절손목경로라원래경로무중단재현으로해석금지. 마지막실제자세는START정본만참조.

**자료함정**: execution05계획62단계중조회2를포함한감사필드명을command_count_erratum.json에보완. RRD각도decision은배출후·현재최종각은result의HOME별도다. 스크린샷의초기커서시간창과전체readback을구분. 포트닫힘공백보간은측정아님. 이전잔류0알/22.28g를새회차측정에대입금지.

**만지지 말 것**: 실행원시/동결소스/계획/기존W9·W10·PID·900·790원본. PLAN04소스복원은원plan의SHA정확일치검증후별도보존했으므로임의수정금지. 새경로의시작복귀예외는일반운용한도확장으로쓰지않는다.

**승인/입력 경계**: 사용자push소유권선언후이번턴commit/push0. Git은사용자가한다. 무게/잔류/영점은사실입력대기이며반복실행승인질문으로바꾸지말것. 다음전체cycle의종료조건은사용자정정에따라HOME이며,열린상태유지로끝내지않는다.

**검증 방법**: 최종case REPORT/command_audit/validation/inspection/manifest를읽는다. fake검증스크립트는하드웨어를열지않는다. 실제실행스크립트는사용자승인범위·최신시작자세를확인후에만사용.
