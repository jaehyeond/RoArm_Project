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

**최신 실행 파일/함정**: `hw_release_roll_test.py`, `analysis_release_run.py`, `torque790_01/`, `release_roll_01/`, `session_20260911_790_tilt_execution.md`. 롤 시험의 result.completed=false는 사용자 사진/방향 질문 후 의도적 Ctrl-C 종료다. **P1 복귀를 전제한 scoop 시작 금지**: 현재 마지막 자세/다음 행동은 START_HERE 정본. 포트는 닫혔으며 백그라운드 로봇 프로세스 없음. 측정 대기 프롬프트는 더 이상 실행 중이 아니다.

**사진/계량 해석 함정**: 이번 컵 포함22.28g과 약0.05g tare는 이번 사용자 입력이다. 예전9.65g 사진 tare로 덮어쓰지 말 것. 회전 전 “아까랑 똑같아”는22.28g 유지로 해석해 원문/해석을 남겼다. 회전 후 저울값은 받지 못했고 사진 표시도 가려져 null이다. 사진 잔류를 추가 배출0g으로 변환하지 말 것. 단일 롤은 명목 출구5° 기울임이 아니다.

**시각화 인계**: `release_roll_01/visual_01/`은 전체 실제 타임라인/RRD readback. 첫 overview screenshot의 빈/작은 패널 한계를 inspection에 기록했다. 근접 방향 검수는 `direction_visual_01/`이며 before/after는 실제, ideal_outward5만 미실행 개념이다. 원본/첫 화면/새 화면은 모두 보존한다.

**한 일 (repo 변경)**: W10 실행/관측 완료 후 기존 영상 워커/W9/W10 검토. 이어 `session_20260911_real_boot_measurement.md`, `s1_v1_real/boot_check_20260911/`의 원시 수집·PID 분석/Rerun·실물 scoop 기록 및 문 목표 정책 수정. D484/LEDGER584 이후 사진 계량 후속 LEDGER585·색인/START_HERE 갱신. `torque900_03/operator_photos/`·`operator_measurement_01.json/CSV`·`MEASUREMENT_01.md` 추가. 이어 `release_tilt_review_01/`·`session_20260911_release_tilt_review.md`에 배출 회전축 검토와 별도 RRD 시각 증거를 추가했다. 각 폴더의 plan/source 사본·manifest를 참조.

**추가 변경**: `session_20260911_790_tilt_mass_git.md`, `pre790_feedback_01/`, `git_publish_01/`, `.gitattributes`, `.gitignore`. 기존 LFS pre-push hook을 사용하며 새 hook 설치 없음. 원시 CSV의 CRLF와 동결 소스/OBJ/diff의 기존 공백은 원본 해시 보존을 위해 그대로 둔다.

**Git 인계 주의**: origin은 `git@github.com:jaehyeond/RoArm_Project.git`, 현재 branch는 master다. 사용자 요청으로 산출물을 게시했고 해시 증거는 `git_publish_01/push_result_01.json`에 둔다. 확인 문서는 그 뒤 일반 커밋으로 게시한다. LFS checkout 후 원본 SHA256으로 기존 manifest를 검증하며 pointer 바이트와 비교하지 말 것. 새790/기울임/5회 실기를 완료했다고 해석하면 안 된다.

**만지지 말 것**: 기존 W1~W10 결과/자산/params·사용자 Manual 원본·이전 원시 기록. 첫 실패/수정 전 실험도 원인 증거라 보존. 설치 SDK/펌웨어는 변경하지 않았다. 사용자 “진행해. 그리고 나서 지금 git 제대로 된 위치에 push해봐”로 후속 실기 및 commit/push 요청 있음. 다른 Orca worktree 브랜치·무관한 g16/출력 모니터/발표자료 변경은 이번 게시에서 제외하고 로컬 보존.

**함정**: `hw_s1_manual.goto_q`가 T122에 측정 문각을 넣어 닫힘0° 목표를 완화한다. 새 기록 어댑터의 명시 T121 목표 유지 수정은 모의검사와 수정900_03 실제 송신 검증까지 끝났다. 기존 Manual REPL은 수정 안 했으므로 그대로 돌아가면 교란이 반복된다. 새 어댑터의 기록 메서드는 record; SDK log는 Logger 객체. 설치 roarm env Rerun0.26.2 대신 isaaclab의0.34.1을 분석에 사용하고 CLI PATH를 지정한다. PID 정본 시각화는 visual_02, scoop은 visual_01; 첫 커서/정적 decision 패널의 한계는 inspection 참조.

**승인/입력 경계**: 연결 준비 완료/기존 배치/계속 초기자세·제대로 진행/PID 데이터 수집 지시가 있다. 실물 순차 절차의 재승인은 불필요. 수정900_03 컵 질량/잔류 사진 입력은 기록했다. 기존 measurement_context.json은 답변 전 null 역사 사본이며 새 operator_measurement_01.json을 읽는다. 다음 운동 전 필요한 입력은 사진 촬영 후 실물 배치 복구와 잔류 제거 여부다; 직전 종료 자세를 현재 자세로 추정하지 말 것. 구체적 단계/자세/남은 범위는 START_HERE 정본. 외부 sim변수·학습·펌웨어 변경은 이 승인에 포함되지 않는다.

**배출 검토 함정**: 새 검토의 ideal_outward5는 그랩 전체의 이상적인 자세이며 실행할 손목 명령이 아니다. 실제 입자 배출을 시험하지 않았다. 첫 RRD의 출구면 가림을 보완한 고정 jaw 안쪽 검수 정본은 `release_tilt_review_01/visual_02/`; 첫 결과도 보존한다.

**검증 방법**: `boot_check_20260911/verify_pid.py`는 장치 없는 파일 감사. `verify_recorded_arm.py`는 가짜 접촉 시리얼 회귀(실기0). RRD validation/inspection/manifest 및 원장 append_integrity를 확인한다. 첫 실행 소스 사본과 수정본을 섞지 말 것. 백그라운드 실물 프로세스는 종료했고 포트는 닫았다.
