# AGENTS.md — RoArm_Project (single source of project rules)

이 repo에서 작업하는 **모든 AI 도구 공통**(도구 중립) **규칙**의 단일 소스. 자동 로드는 규칙만이다.
하드웨어·파이프라인·명령어·세션 프롬프트 **레퍼런스는 2026-08-25에 `docs/reference/`로 분리**했다
(아래 표, on-demand). 원본 삭제 0건 — 분리 전 724줄 원본은 `docs/archive/AGENTS_full_20260825_pre_split.md`.
2026-08-25 이전 문서의 `AGENTS.md:<줄번호>` 인용은 그 원본에서 확인한다.

- Claude Code: `CLAUDE.md` 첫 줄의 `@AGENTS.md` 임포트로 자동 인라인 로드.
- Codex CLI / Cursor: cwd의 `AGENTS.md`로 네이티브 자동 로드.
- Claude 전용 내용(12-agent 팀, 스킬, auto-memory 워크플로우)은 `CLAUDE.md`에 있다.

**상태 ≠ 규칙**: 이 파일에는 규칙만 둔다. 진행 상태·실험 결과의 인계는
`START_HERE.md` → `claudedocs/DECISIONS.md` → `claudedocs/session_*.md`로만 한다.
**상태 원장 소유권은 배타적이다** — `START_HERE.md`, `claudedocs/{DECISIONS.md,
DECISIONS_ACTIVE.md, EXPERIMENT_LEDGER.md, session_*.md, relay/}`를 쓰는 세션은 한 번에 하나뿐이다.
소유 세션이 state doc 갱신 + relay 인계로 끝난 뒤 다음 세션이 원장을 넘겨받는다.
worktree가 분리된 코드 편집·분석·읽기 전용 작업은 도구가 달라도 동시 실행해도 된다
(원장은 worktree로 격리되지 않는다 — `DECISIONS.md`는 Dxxx 순번 append, `START_HERE.md`는
통째 overwrite라 사본이 갈라지면 머지가 아니라 유실이 된다).
세션을 닫는 도구는 `claudedocs/relay/from_<자기 도구>.md`를 **덮어써** 다음 도구에 인계한다
(Claude→`from_claude.md`, Codex/Cursor→`from_codex.md`). 상태 정본은 `START_HERE.md`이며
relay에는 베끼지 않는다 — 만진 파일·건드리지 말 것·함정·승인 대기만 (규약은 파일 §0).

## 참조 문서 (자동 로드 아님 — 해당 작업 착수 전에 읽을 것)

| 파일 | 언제 읽나 |
|---|---|
| `docs/reference/session_protocol.md` | 세션 부팅/종료 프롬프트 원문, 상태 파일 운영 규칙 |
| `docs/reference/pipeline.md` | 개요·환경 스펙·Key Commands·5단계 파이프라인·핵심 파일 표 |
| `docs/reference/hardware.md` | 관절 범위, SDK API/버그, 모터 복구(T:106), Azure Kinect 셋업 |
| `docs/reference/training_deploy.md` | SmolVLA 학습/배포 Critical Rules, L-F 수집, 과거 실패 원인 |
| `docs/reference/research_verification.md` | 연구 갭 주장 검증 절차(HARD RULE #4 상세) + 외부·증거 링크 |
| `docs/archive/` | 분리 전 원본, 2026-04-28 Current Status(**죽은 상태** — 현재값으로 쓰지 말 것) |

## Current-State Protocol

기억이 아니라 rolling state doc으로 재개한다. 파일 역할을 혼동하지 말 것:
`START_HERE.md`=현재 상태 대시보드(주요 갱신마다 덮어쓰기) / `claudedocs/DECISIONS.md`=증거 붙은
지속 교훈·반복 금지 규칙(`D001`~ append-only, superseded 표시만·삭제 금지) /
`claudedocs/EXPERIMENT_LEDGER.md`=주요 실험+verdict(append-only) /
`claudedocs/session_YYYYMMDD_*.md`=세션 상세 로그(세션당 새 파일, append-only) /
`HANDOFF.md`·`TASKS.md`=낡음, `START_HERE.md`가 명시하지 않는 한 신뢰 금지 /
auto-memory `~/.claude/projects/.../memory/MEMORY.md`=사용자 단위 기억(Claude 전용, Recent Sessions prepend,
HARD RULES 삭제 금지). 상태 문서의 **대체가 아니라 보완** — 세션 종료 시 둘 다 갱신한다.

### Session boot procedure

Before answering current project-state questions or making edits:

1. Read `START_HERE.md`.
2. Read `claudedocs/DECISIONS_ACTIVE.md` (활성 결정 요약 + 줄번호 앵커).
   전체 `claudedocs/DECISIONS.md`(28,097줄)는 통째로 읽지 말고, 앵커가 가리키는 줄만 on-demand read.
   판정·수치의 정본은 언제나 `DECISIONS.md` 원문이다 — 요약과 어긋나면 원문이 이긴다.
3. Read `claudedocs/LEDGER_RECENT.md` (최근 실험 20건 + 줄번호 앵커).
   전체 `claudedocs/EXPERIMENT_LEDGER.md`(531줄이지만 줄당 평균 2 KB)는 통째로 읽지 말고,
   앵커가 가리키는 줄만 on-demand read. 원장 자신의 경고대로, 수치 인용 전에는
   원장이 아니라 링크된 session/data 파일까지 내려가 확인한다.
4. Read `claudedocs/relay/from_<직전 도구>.md` — Claude 세션이면 `from_codex.md`,
   Codex/Cursor 세션이면 `from_claude.md`. 직전 도구가 남긴 미커밋 변경·함정·승인 대기 항목.
   상태 정본이 아니다 — `START_HERE.md`와 어긋나면 `START_HERE.md`가 이긴다.
5. Read only the `claudedocs/session_*.md` files referenced by `START_HERE.md`
   unless more evidence is needed.
6. Run `git status --short`.
7. Verify any metric from the referenced log/data file before citing it.

### Context 95% emergency protocol

If active chat context approaches 95%:

1. 새 구현 작업 즉시 중단.
2. `docs/reference/session_protocol.md`의 end-of-session update prompt 실행 (상태 파일만, 새 코드 금지).
3. 다음 세션용 continuation prompt 출력 (≤80줄: active pivot, 다음 행동, 읽을 파일, 현재 md5).
4. `/half-clone`·`/handoff` 스킬 사용·제안 금지 (HARD RULE #7·#11).
5. 사용자가 새 세션을 열고 `docs/reference/session_protocol.md`의 boot prompt를 붙여넣는다.

## NVIDIA Stack Official-Source Verification Rule

For questions or changes involving Omniverse, Isaac Sim, Isaac Lab, Kit, PhysX,
Fabric, Hydra, Warp, CUDA, or RTX, use this evidence order before making a
technical claim:

1. Identify the installed product, extension, SDK, driver, and relevant schema
   versions first.
2. Consult version-matched NVIDIA official documentation, API references,
   schemas, and published source before using forums, blogs, or search summaries.
3. Explicitly distinguish UI authoring ranges, schema defaults, SDK/engine hard
   limits, CPU/GPU compatibility limits, and project-authored settings. Never
   present one category as another.
4. Cross-check the public documentation against the installed package's
   schema/source and this repo's original JSON/log/runtime evidence.
5. In the user briefing, list the official document title, URL, applicable
   version, and the local `file:line` evidence used. Mark version mismatches and
   any inference that is not stated directly by NVIDIA.
6. Third-party material is supplementary only and must not replace an available
   NVIDIA primary source for NVIDIA-stack semantics or limits.

## Session progress rule

- Every research session must run at least one experiment that can fail
  (RL training with real updates, or perturbation evaluation), or explicitly
  justify why not in the session doc.
- Control-contract hardening is REACTIVE only: it is permitted solely in
  response to a failure observed during training or perturbation evaluation.
- A verdict ending in `NO_PPO_PROMOTION` without a training attempt or
  perturbation evaluation in the same session requires explicit justification
  against this rule.
- Validation that cannot change a decision must not be run.

## Research briefing language and teaching rule

- User-facing research briefings must be written primarily in Korean. Keep exact
  code, schema, field, file, and verdict identifiers when needed, but explain each
  unfamiliar English term in plain Korean at first use.
- Never use only a `Dxxx`, attempt number, acronym, or internal check name as the
  explanation. State what it is, why it was checked, what PASS/FAIL means, and
  how it changes the next decision.
- A step-by-step request means reporting auditable actions, observations, and
  evidence in execution order. Do not replace that report with opaque labels.
- The final experiment briefing follows this order: (1) what/why, (2) procedure
  in observable steps, (3) quantified result with source paths, and (4) an
  everyday-language verdict plus the next authorization boundary. Put this
  briefing at the end of the turn so later tool chatter does not bury it.

## Variable Ladder Protocol (D322~)

- Each active case may introduce only one or two new variables. The session doc
  must state near the top: `이번 case의 신규 변수: [...]`.
- Future-looking ideas must not be implemented immediately. Append them to
  `claudedocs/BACKLOG.md`, then return to the current critical path.
- The `START_HERE.md` `Active Case` section is the single source of truth for
  what is in scope. Everything outside it is a non-goal unless the user
  explicitly approves a case change.
- Folders are forward-only. Do not move or rename existing files/folders, so
  old evidence paths remain valid. New grasp outputs must be created only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`, and the path must be
  listed in `START_HERE.md` `Active Case`.

## Visualization Definition of Done (D324~)

- Any probe/evaluation that reasons about geometry, pose, contact, jaw faces, or
  tool frames must emit visual diagnostics through `roarm_rl.viz_debug` when
  practical.
- Required artifacts are: target-vs-actual frame markers, at least one
  decision-time diagnostic snapshot in the run output folder, and explicit
  snapshot paths in the session document.
- This rule is for single-frame debugging only. It does not relax the existing
  ban on large renders, trajectory videos, new data generation, or variable
  ladder advancement without explicit user approval.

## Rerun Observability Completion Contract (D341~)

- A replayable RRD is mandatory when a verdict depends on geometry, pose,
  coordinate frames, collision/contact, a trajectory, or synchronized sensor
  time. Rerun may be omitted only for a pure file/hash/schema audit with no
  spatial or temporal judgment; the session doc must state that justification.
- Deterministic Isaac/batch work uses save-only recording by default. A live
  Viewer is optional for exploration, but the file sink must be attached before
  the first user log in either mode. The recording must be finalized by a
  `RecordingStream` context exit or disconnect before any artifact gate runs.
- The RRD must contain the actual decision subject, not only generic robot/frame
  markers. Cook/representation cases log source, instance, prototype, and
  candidate geometry as separate entities. Physics/settle cases log the full
  executed step timeline plus decision scalars and contact points/force arrows;
  a final or trial-1-only row is insufficient for a trajectory verdict.
- Rerun is an observability/replay layer, not the bit-exact authority. Original
  callback arrays and canonical JSON/hashes decide equality. Float64 metrics may
  be plotted in Rerun, while its Float32 spatial copies are inspection evidence
  only and must never be hashed back into a scientific gate.
- Rerun completion requires all of the following: the exact SDK/CLI version pin;
  footer-enabled `rrd verify` PASS after finalization; exact non-system entity,
  timeline, and required-component contracts PASS; a fixed embedded blueprint
  plus its verified `.rbl` export; a headless decision
  screenshot; and an actual visual inspection whose path and observations are
  recorded in the session doc. Non-empty generation, loadability, or screenshot
  creation alone must never be reported as "inspected".
- RRD, RBL, validation report, and inspection screenshot belong in the active
  run output folder. If any required item fails, the visualization contract
  fails without overriding the scientific verdict or relaxing a gate.

## IsaacLab Environment Package Rule (D326~)

- Any package install into the `isaaclab` conda environment must record the
  dependency impact and verify the known Isaac-compatible pins afterward:
  `numpy==1.26.0` and `psutil==5.9.8`.
- If an install upgrades either package, immediately restore those pins and
  verify imports before running Isaac. This rule comes from D325, where
  installing `rerun-sdk` pulled incompatible `numpy 2.4.6` and `psutil 7.2.2`.

## HARD RULES — 절대 위반 금지 (도구 중립)

> 원 출처: Claude auto-memory
> `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY.md`.
> 도구 중립 규칙을 2026-07-12에 **verbatim 이전** — Claude는 `CLAUDE.md`의 `@AGENTS.md`
> 임포트로, Codex/Cursor는 네이티브로 모두 자동 로드한다. **번호는 원본 그대로 보존** —
> 다른 문서의 "HARD RULE #n" 참조가 계속 유효하다. 삭제 금지, 새 실패 발생 시에만 append.
> 잔류분: #8(MEMORY.md 운영)·#11(/half-clone 금지)은 Claude 전용이라 auto-memory에 잔류.
> #13-#17/#21-#26은 B200-era 비활성 pointer로 auto-memory에 잔류
> (원문: auto-memory의 `hard_rules_b200_era_archive_20260711.md`).

1. **데이터 수집 = HOME [0,0,90,0,0,0] 시작** — v5 136ep 전량 폐기 사유. 유저 직관을 데이터 없이 기각 금지. (상세: feedback_v5_data_collection_failure.md)
2. **학습 = `lerobot-train` CLI + `lerobot/smolvla_base` pretrained 필수** — 커스텀 학습 3회 실패. Action Expert 랜덤 초기화 금지. (상세: tech_critical_lessons.md)
3. **VGST 자동 verdict FAIL ≠ 실제 실패** — 5° 임계값 too conservative. v3(M2=1.73° FAIL→5/5 성공), v6(M2=4.62° FAIL→reach 성공) 2회 전례. open-loop 직접 테스트 필수. (상세: experiment_log.md 2026-04-07)
4. **"없다/최초" 주장 = 10개+ 검색어 × 2개 소스 검증** — 3/10, 3/23에 거짓 갭 주장 3회 반복. 반증 검색 먼저. (상세: research_verification_failures.md)
5. **JOINT_LIMITS 제거 금지** — 하드웨어 보호.
6. **Camera 위치 변경 = 전체 데이터 무효** — 단, "카메라 절대 고정"은 과적합 원인이기도 함 (대형 VLA는 다양 각도 OK). 수집 중에는 고정, 데이터셋 설계 시에는 다양성 확보.
7. **HANDOFF.md 절대 생성/건드림 금지** — MEMORY + continuation prompt 방식만 사용.
8. **[Claude 전용 — auto-memory 잔류]** MEMORY.md 운영 규칙 (오버라이드 금지, Recent Sessions prepend).
9. **VLA 모델은 SmolVLA에 한정 안 됨** — 클라우드 GPU 사용 가능, OpenVLA/Octo/pi0 비교 가능.
10. **문제-중심 연구만 수용** — "X% 향상" 메트릭 논문 구조 거부. 시간 제약으로 아이디어 축소 금지 (유저 20h/day 투입 의지).
11. **[Claude 전용 — auto-memory 잔류]** `/half-clone` 절대 사용/제안 금지.
12. **배포 디버깅 = 데이터셋 이미지/상태 먼저 확인** — 모터/속도/대역폭 같은 HW 추측 전에 학습 parquet의 state 분포, ep 끝 프레임 gripper/pose, Azure Kinect 이미지를 직접 본다. L-F 수집의 action(leader)과 state(follower) gap은 설계 기능이지 버그 아님. (4/9 gripper 실패 오진 → 규칙화)
13. -17. **[B200-era 비활성 — auto-memory 잔류]** 원문: auto-memory `hard_rules_b200_era_archive_20260711.md`.
18. **사용자 명시 정정 > Claude 추론 (절대 우선)** — 사용자가 "X = Y"로 명시 정정한 사항은 후속 분석/PNG/parquet 재해석으로 절대 무효화 금지. 재해석 필요 시 반드시 사용자에게 confirm 후 진행. **근거**: 4/30 evening 사용자 "sponge=세움 (이미지2처럼)" 정정 → 4/30 late-evening Claude가 "v6 PNG 재분석"으로 "vertical pillar + 우물정자=lying-flat" 결론 도출, 사용자 정정 무효화 → 5/01 sim_demos_v2 lying-flat 50ep × 146fr + 5/03 B200 10K finetune (~42min GPU) + ST-C 1차 deploy 모두 폐기. 약 4 세션 손실. **적용**: sponge orientation, # pattern geometry, grasp 방향 등 사용자 한 번이라도 명시한 design 결정은 단독 변경 금지. (5/03 evening 규칙화)
19. -20. **[Track A sponge/stacking 확정 design — 변경 절대 금지 유지, 본문 그대로 보존]** Sponge = edge-stand 47mm tall(#19, lying-flat/vertical-pillar 금지, TCP grasp z +33mm world) / # tower = 2-layer cross, L1 Y c2c=87mm·L2 X c2c=67mm(#20). **sponge/stacking 작업 재개 시 원문 read 필수**: auto-memory `hard_rules_b200_era_archive_20260711.md` + `project_well_pattern_design_v3.md`.
21. -26. **[B200-era 비활성 — auto-memory 잔류]** 원문: auto-memory `hard_rules_b200_era_archive_20260711.md`.
27. **B200 lease retired after 2026-05-22 23:59 KST — no future work may require B200 SSH or `.ssh` secrets.** `JHPark/roarm_b200` 재진입, B200 Isaac 재실행, 추가 파일 pull, `.ssh` 키 복사/요청 전부 금지. 백업 안 된 B200-only 파일이 필요한 경로 = blocked, 로컬 증거로 재설계. Track A/B 백업 검증 해시 + 위치(Track B 완전 checkpoint는 `openvla_oft_b200_pulls`) 원문: auto-memory `hard_rules_b200_era_archive_20260711.md` + `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`, DECISIONS D087-D088.
28. **D232 storage rule — SmolVLA `outputs/` 기본 보존; `collected_data*`/`b200_backup_*`/`openvla_oft_b200_pulls` 삭제 금지 (archive/move-only, 명시 승인 필요).** 디스크 압박 시 무작위 재스캔 금지. 1차 승인 경로 = `outputs/*/checkpoints/*/training_state` cleanup (~25.6GB, manifest+명시 승인 후, pretrained_model 보존). 2차 = run별 keep-one pruning (~90.15GB 총, **무승인 실행 절대 금지**). Run별 keep-one 목록 원문: auto-memory `hard_rules_b200_era_archive_20260711.md` + D232 docs/logs.

## Safety constraints (모든 도구 공통)

- 로봇 하드웨어 직접 제어(`serial` `/dev/ttyUSB*`, `torque_set`, `joints_angle_ctrl`,
  `move_init`, `T:106`)는 사용자 명시 승인 없이 실행 금지.
- `lerobot-train` 실행은 사용자 승인 후에만 (config 설계/검토는 자유).
- `rm -rf` 금지. `JOINT_LIMITS` 코드 제거 금지 (HARD RULE #5).
- git commit/push는 사용자가 요청할 때만.

## i4h 스킬 규칙 (2026-08-25~)

- 이 repo 세션에는 NVIDIA `i4h-*` 스킬 13개가 심링크로 노출되어 있다
  (정본: `/media/cgxr/ROBOT_DEV/i4h-workflows/.agents/skills/`,
  셋업/이식 문서: `/media/cgxr/ROBOT_DEV/AGENT_SKILLS_SETUP.md`).
- **사용자가 스킬 이름으로 명시 요청할 때만 호출한다.** teleop/mimic/convert/finetune 등
  RoArm 작업 어휘와 트리거 키워드가 겹치므로 자동 발동 금지
  (Claude는 `.claude/settings.local.json`의 `skillOverrides`로도 차단됨).
- i4h 파이프라인 본 실행(teleop 녹화, finetune 등)은 이 repo가 아니라
  `/media/cgxr/ROBOT_DEV/i4h-workflows`에서 세션을 열어 수행한다.

## File ownership (요약 — 상세 표는 CLAUDE.md Agent Team)

파일 prefix별 소유 규칙: `data_*`/`collect_data_manual.py`(data), `train_*`/`run_official_train.py`(pipeline),
`deploy_*`(deploy), `trajectory_*`(manipulation), `sim_*`(sim2real), `hw_*`/`calibrate_*`(hardware),
`model_*`(vla-model), `augment_*`/`self_improve_*`(data-efficiency), `monitor_*`/`safety_*`(deployment-safety),
`experiment_*`/`eval_*`(experiment), `analysis_*`/`figure_*`(analysis), `paper/`(writing).
어떤 도구든 이 경계를 존중한다.
