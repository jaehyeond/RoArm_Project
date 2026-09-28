# TASK_SPEC — 역할 agent 정의 초안 4개(문서만, 본 repo 채택은 사용자 승인 후) — Claude claude-opus-5 high
작성 2026-09-18, 코디네이터 = 메인. 계약 정본 = 이 파일. 목적: worktree 마다 임시 agent 를 만드는 대신, 본 repo `.claude/agents/` 에 반복 역할 4개를 한 번 정의해 모든 worktree 가 git 으로 물려받게 한다. 규칙의 단일 소스는 AGENTS.md — **규칙 복제 금지, 참조만**.
## Target(이 worktree 안에만)
`.claude/agents/{deme-runner,raw-accountant,replay-renderer,independent-auditor}.md` + `claudedocs/research/role_agents_20260918/README.md`(4 역할 ↔ AGENTS.md/CLAUDE.md 규칙·과거 TASK_SPEC 대응표, 배정 시 사용법).
## Change
각 agent 파일 = Claude Code 서브에이전트 정의(frontmatter: `name`, `description`(언제 쓰나), `tools`, `model: claude-opus-5`) + 본문 ≤120줄 한국어: 역할·입력·산출·**계약**(영수증·해시 대조·타임아웃≠성공·재시도 0·setsid·D324 시각·D341 Rerun 완료·독립성 = 생산 모듈 import 금지·사후 완화 금지·"없다/최초" 금지·"실제 경과 시간(wall-clock)" 용어)·금지 사항(상태 원장·다른 worktree·실물·설치·commit)·완료 보고 형식(worker_done 3문장·report-path)·검증 명령 예. 근거로 기존 과제서 `w19_runpod_d487/coordinator/TASK_SPEC_*.md`, `w16_profile_d486/coordinator/TASK_SPEC_*.md`, `w17_replay_fix_d484/coordinator/TASK_SPEC_*.md`, W14/W15 REPORT 의 절차를 읽고 **공통분모만** 추출한다.
## Constraints
AGENTS.md·CLAUDE.md·상태 원장·다른 worktree 무수정. 코드 0. 인용은 파일:줄 확인. 총 1 h 상한.
## Observable acceptance
5파일 존재·frontmatter 유효·각 ≤120줄·README 대응표·worker_done(--report-path README, 3문장).
