# Session 2026-07-12 — AGENTS.md 단일 소스 전환 + Codex 지침 결손 해소 (config 세션)

**유형**: 인프라/설정 세션 (연구 실험 없음 — Session progress rule 예외 사유: 사용자가 명시
지시한 cross-tool 지침 재구조화 작업이며 grasp track 코드/실험 미접촉).

## 배경

- 머신 구조 분석(7/11)에서 확인: Codex CLI/Cursor는 전역·프로젝트 어디에도 instruction
  파일이 없어 HARD RULES를 모른 채 병행 작업 중이었음 (Codex `default.rules`에 grasp
  G0a D332/D333 스크립트 실행 allow가 실존 — 병행 사용 증거).
- 다른 머신 실측 검증된 사실(사용자 제공): Claude Code는 CLAUDE.md 안의 `@AGENTS.md`
  임포트를 기계적으로 인라인. Codex는 `~/.codex/AGENTS.md`(전역) + cwd `AGENTS.md` 자동
  로드. Cursor도 AGENTS.md 네이티브 로드.

## 실행 내역 (A → B → C)

### A. 전역 (순수 추가, 완료)
- `~/.codex/AGENTS.md` 신규 (1.9KB): 머신 사실 / 규약 3개(AGENTS.md 단일 소스·상태≠규칙·
  Claude-Codex 동시 편집 금지) / RoArm 부트 요약.
- `~/.claude/CLAUDE.md`: 말미에 "Cross-tool conventions" 섹션만 추가 (기존 무변경).

### B. 프로젝트 분리 (워킹 트리 적용, **미커밋 — 사용자 승인 대기**)
- `AGENTS.md` 신규 마스터 (33.8KB): 구 CLAUDE.md의 도구 중립 내용 전부 verbatim +
  HARD RULES 섹션(auto-memory에서 #1-#7/#9-#10/#12/#18-#20/#27-#28 verbatim 이전,
  **번호 보존**) + Safety constraints + File ownership 요약 신설.
- `CLAUDE.md` 교체 (5.6KB): 첫 줄 `@AGENTS.md` + Claude 전용(Session Workflow,
  12-agent 팀, auto-memory Topic Files)만 잔류.
- 의도적 변경 4건만 non-verbatim: boot prompt "Read CLAUDE.md"→"Read AGENTS.md",
  CLAUDE.md 내 참조 2곳, Current Status에 historical snapshot 주석 1줄, 섹션명 개명 2건.
- auto-memory `MEMORY.md`: 이전 규칙 14개 엔트리를 한 줄 포인터로 대체 (#8/#11 Claude
  전용 잔류, #13-17/#21-26 archive pointer 잔류).
  백업: `memory/MEMORY_BACKUP_20260712_pre_agents_split.md`.
- 검증: 구 CLAUDE.md 전체 섹션 헤더가 두 새 파일 중 하나에 존재함을 스크립트로 확인.
- **승인 후 할 일**: 새 codex 세션에서 "프로젝트 규칙 중 B200 관련 규칙 말해봐"로
  AGENTS.md 자동 로드 실측 검증 → 이후 커밋.
- **원복 방법(거부 시)**: `git checkout CLAUDE.md && rm AGENTS.md` + MEMORY.md 백업 복원.

### C. 정리 (①③④ 완료, ②⑤ 제안만)
- ① `~/.claude/settings.json`·`~/.claude.json` chmod 600; 토큰 포함 확인 후
  `settings.json.bak-cf`·`.claude.json.bak-cf` 삭제.
- ③ `~/.codex/rules/default.rules` 458→362줄 (stale B200 `JHPark` allow 96줄 제거,
  잔여 0). 백업: `~/.codex/default.rules.bak_b200_cleanup_20260712` (rules/ 밖).
- ④ agent-memory 이중 디렉토리 병합: 에이전트 실소환으로 현재 하네스가 하이픈
  디렉토리(`Manipulation---Control-Specialist` 등)를 쓰는 것을 실측 확인 → `&` 이름
  구 디렉토리 3개 내용을 하이픈 디렉토리로 이관 (비충돌 10파일 move, 충돌 4파일
  merge-marker append), 15→12 dirs. git 변경에 포함 (미커밋).
- ② (제안만) 전역 Stop hook `check-context.sh`에 프로젝트 opt-out 게이트 diff 제시
  (`.claude/no-half-clone` 마커). **본 세션 종료 시 실제로 hook이 /half-clone을 지시하는
  충돌이 재현됨** — HARD RULE #11에 따라 거부하고 파일 기반 종료로 대체.
- ⑤ (제안만) `/ac`·`/ac-status` 스킬은 AC247 backend 부재로 동작 불가 →
  `~/.claude/skills-disabled/` 이동 제안.

## 대기 중 결정 3건
1. B diff 승인 (git status: `M CLAUDE.md`, `?? AGENTS.md`, agent-memory 병합 변경) →
   승인 시 codex 실측 검증 후 커밋.
2. ② hook opt-out diff 적용 여부.
3. ⑤ ac 스킬 비활성화 여부.

## 비고
- grasp track (G0a D333/D334) 상태는 본 세션에서 미접촉 — START_HERE.md 그대로 유효.
- 나머지 로봇 프로젝트 6개(IsaacLab, openvla-oft 등)는 의도적으로 미적용 (페인 포인트
  발생 시 B 패턴 적용).
