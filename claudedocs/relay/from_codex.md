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

## §2 2026-09-14 Codex → 다음 세션 — 연구 설명·재개·Git 게시

**한 일 (repo에 남은 변경)**: START_HERE.md, BACKLOG의 후속 후보 append, 새 session_20260914_research_closeout_git.md, CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md, research/closeout_20260914/와 이 relay. Downloads에 새 출력용 Markdown을 전달했다. 사용자 최신 요청으로 이번 commit/push를 승인받아 각 branch로 보존한다. 실제 원격 확인은 GIT_PUBLICATION.md와 최종 START를 따른다.

**만지지 말 것**: 기존 PPT/출력용 Markdown/미디어·W13 run_01/rev28/post03·기존 실패 보고·과거 동결 revision. 모든 실험 원본 바이트는 유지한다. pellet-model의 세 .bak는 제외한 로컬 보존본이며 삭제하지 않는다. worker branch의 오래된 상태/원장을 main에 merge하지 않는다.

**함정**: master에 worker 코드가 자동 통합되지 않았다. branch 매핑/LFS 원자료는 research/closeout_20260914/GIT_PUBLICATION.md와 PUBLICATION_MANIFEST.json을 확인한다. LFS 포인터만 있는 clone은 원자료 확보가 끝난 상태가 아니다. 당시 pin의 이전 HEAD는 이번 게시 때문에 바뀌었지만 원시 해시를 고치거나 오래된 GO를 되살리지 않는다. 새 MD는 최근 질의의 종합 설명이며 옛 verify_delivery.mjs의 두 문서 검사 대상이 아니다. 새로운 verify_closeout.mjs가 delivery/continuation/numbers/publication을 분리해 검사한다.

**승인 대기**: 새 continuation §5를 새 세션에 사용자 요청으로 전달하면 §4 첫 case의 원자료 판정 수정과 CPU 테스트부터 진행한다. 지금 새 연구 실행은0. 이후 재생3결함·운반 원인·성능 계측은 순차 분리한다. GPU 물리·재렌더·학습·A/B/C·하드웨어 조회/구동/PID/토크/카메라·설치는 자동 승인되지 않는다. 이번 Git 게시 승인은 후속 세션에 이월하지 않는다.

**이번 게시의 미완료 인계**: 로컬main+6worker커밋은보존됐지만 원격게시미완료다. 첫workerpush는SSH종료/rc141, 메인push는자동보안검토거부로미실행. 사용자에게공개origin/master+6branch와원자료/영상LFS약3.32GB의구체적게시승인을질문했고답전재시도금지다. 로컬커밋을원격백업으로말하지않는다. 승인후정상push+원격SHA대조만이어가며실험재실행불필요. 세부정본은GIT_PUBLICATION.md.

**검증 방법**: 새 verification 스크립트의 내용을 읽고 필요한 mode만 실행한다. publication은 로컬 전체 증거 해시와 Git 원격 ref를 읽으며 새 물리나 renderer를 실행하지 않는다. 새 세션의 상태 정본은 START_HERE이고 이 relay가 아니다. 이번 종료 뒤 상태 원장 소유권을 넘겨받는다.
