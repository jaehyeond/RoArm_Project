# ARCHIVE_INDEX.md — 콜드 아카이브 이관 대장

> 2026-08-17 사용자 승인(T1+T2 완전 이전)에 따라 **git 비추적 대형 데이터**를
> 외장 `ROBOT_DEV`로 이관한 기록. 각 원경로에는 새 위치를 가리키는 심링크가
> 남아 있어 기존 문서의 증거 경로가 계속 유효하다 (AGENTS.md forward-only
> 규칙의 경로-보존 수단). 외장하드 미마운트 시 심링크가 끊기므로, 그 경우
> 이 표의 새경로 열을 참조해 하드를 연결할 것.
>
> 이관 절차(폴더당): src sha256 전량 매니페스트 → rsync 복사 → dst 매니페스트
> → diff 대조 PASS → 그때만 파일 단위 원본 제거(rm -rf 불사용) → 심링크.
> 매니페스트 원본 = `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/_manifests/`.
> 이관분은 사본 1개(백업 아님). T3(b200_backup 2종·openvla_oft_b200_pulls,
> 유일 사본 45G)는 이관하지 않고 내장 유지 — 별도 2사본화 결정 대기.

| 날짜 | 폴더 | 새 경로 | 규모 | 검증 | 사유 |
|---|---|---|---|---|---|
| 2026-08-17 | `logs` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/logs` | 1034 files / 766024668 bytes | sha256 전량 일치 (`_manifests/logs_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `sim_renders_v4_dryrun` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/sim_renders_v4_dryrun` | 147 files / 44690024 bytes | sha256 전량 일치 (`_manifests/sim_renders_v4_dryrun_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `sim_renders_v5_dryrun` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/sim_renders_v5_dryrun` | 147 files / 45588226 bytes | sha256 전량 일치 (`_manifests/sim_renders_v5_dryrun_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `lerobot_dataset_v4` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/lerobot_dataset_v4` | 7 files / 259402472 bytes | sha256 전량 일치 (`_manifests/lerobot_dataset_v4_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `lerobot_dataset_v3` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/lerobot_dataset_v3` | 7 files / 404871566 bytes | sha256 전량 일치 (`_manifests/lerobot_dataset_v3_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `sim_renders_v3` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/sim_renders_v3` | 4751 files / 1493411333 bytes | sha256 전량 일치 (`_manifests/sim_renders_v3_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `sim_renders_v4` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/sim_renders_v4` | 7302 files / 2257761866 bytes | sha256 전량 일치 (`_manifests/sim_renders_v4_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `sim_renders_v5` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/sim_renders_v5` | 7302 files / 2277948475 bytes | sha256 전량 일치 (`_manifests/sim_renders_v5_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `collected_data_v6_phase0_singlearm_DISCARD` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/collected_data_v6_phase0_singlearm_DISCARD` | 1491 files / 1503707438 bytes | sha256 전량 일치 (`_manifests/collected_data_v6_phase0_singlearm_DISCARD_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `collected_data_v6` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/collected_data_v6` | 13934 files / 13854784917 bytes | sha256 전량 일치 (`_manifests/collected_data_v6_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `collected_data_v2_backup` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/collected_data_v2_backup` | 19540 files / 19780752034 bytes | sha256 전량 일치 (`_manifests/collected_data_v2_backup_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `collected_data_v5` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/collected_data_v5` | 27524 files / 27417715012 bytes | sha256 전량 일치 (`_manifests/collected_data_v5_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `collected_data` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/collected_data` | 26364 files / 27022806256 bytes | sha256 전량 일치 (`_manifests/collected_data_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |
| 2026-08-17 | `outputs` | `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/outputs` | 467 files / 79058293814 bytes | sha256 전량 일치 (`_manifests/outputs_{src,dst}.sha256`) | T1/T2 사용자 승인 이관 |

## 2026-09-28 Orca worktree 보관 6건 (W23 K 계획, 사용자 승인·실행)

절차: 복사+전량 SHA256 목록 대조 → HEAD `archive/<이름>` 태그 + git bundle → `orca-ide worktree rm` → 원경로 심링크 → 심링크 경유 목록 재대조. 계획·명령 `w23-worktree-plan/claudedocs/research/worktree_plan_20260928/{REPORT,COMMANDS}.md`. 메타 `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/_plan_20260928_<이름>/`.

| 날짜 | worktree | 보관 위치 | 파일 / 크기 | 검증 | 태그(HEAD) |
|---|---|---|---|---|---|
| 2026-09-28 | `w19-audit` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w19-audit` (원경로 심링크) | 9565 files / 3675491762 bytes | 원본·사본·심링크 경유 목록 전량 일치, bundle verify ok | `archive/w19-audit` = `fc557db0f1c2` |
| 2026-09-28 | `w19-rev33` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w19-rev33` (원경로 심링크) | 9596 files / 3675971773 bytes | 원본·사본·심링크 경유 목록 전량 일치, bundle verify ok | `archive/w19-rev33` = `fc557db0f1c2` |
| 2026-09-28 | `w19-role-agents` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w19-role-agents` (원경로 심링크) | 9562 files / 3675296496 bytes | 원본·사본·심링크 경유 목록 전량 일치, bundle verify ok | `archive/w19-role-agents` = `fc557db0f1c2` |
| 2026-09-28 | `w20-particle-count` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w20-particle-count` (원경로 심링크) | 9575 files / 3675382746 bytes | 원본·사본·심링크 경유 목록 전량 일치, bundle verify ok | `archive/w20-particle-count` = `fc557db0f1c2` |
| 2026-09-28 | `w21-p2-runs` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w21-p2-runs` (원경로 심링크) | 9595 files / 3675494633 bytes | 원본·사본·심링크 경유 목록 전량 일치, bundle verify ok | `archive/w21-p2-runs` = `fc557db0f1c2` |
| 2026-09-28 | `w21-smoke-evidence` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w21-smoke-evidence` (원경로 심링크) | 9583 files / 3675453525 bytes | 원본·사본·심링크 경유 목록 전량 일치, bundle verify ok | `archive/w21-smoke-evidence` = `fc557db0f1c2` |

## 2026-09-28 밤 Orca worktree 보관 10건 (W23 7 + W24 3, 사용자 권한 부여 후 메인 실행)

절차: 09-28 저녁 6건과 동일(복사+전량 SHA256 대조 → 태그 `archive/<이름>` + git bundle → `orca-ide worktree rm --force` → 원경로 심링크 → 심링크 경유 재대조). 메타 `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/_plan_20260928_<이름>/`. 스크립트 = 세션 scratchpad `archive/stage{1_prepare,2_finalize}.sh`(K 계획 COMMANDS §2·§3 대상만 교체). 세션 문서 `claudedocs/session_20260928_w25_realign_fullcycle_prep.md` §5.

| 날짜 | worktree | 보관 위치 | 파일 / 크기 | 검증 | 태그(HEAD) |
|---|---|---|---|---|---|
| 2026-09-28 | `w23-equipment` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-equipment` (원경로 심링크) | 9381 files / 3448745612 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-equipment` = `3267dcb38369` |
| 2026-09-28 | `w23-placement-final` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-placement-final` (원경로 심링크) | 9388 files / 3450454941 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-placement-final` = `3267dcb38369` |
| 2026-09-28 | `w23-real-reference` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-real-reference` (원경로 심링크) | 9391 files / 3450722505 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-real-reference` = `3267dcb38369` |
| 2026-09-28 | `w23-sim-alignment` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-sim-alignment` (원경로 심링크) | 9401 files / 3449679930 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-sim-alignment` = `3267dcb38369` |
| 2026-09-28 | `w23-sim-reference` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-sim-reference` (원경로 심링크) | 9377 files / 3448705288 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-sim-reference` = `3267dcb38369` |
| 2026-09-28 | `w23-strategy` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-strategy` (원경로 심링크) | 9379 files / 3448718326 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-strategy` = `3267dcb38369` |
| 2026-09-28 | `w23-worktree-plan` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w23-worktree-plan` (원경로 심링크) | 9468 files / 3451647904 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w23-worktree-plan` = `3267dcb38369` |
| 2026-09-28 | `w24-camera-placement` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w24-camera-placement` (원경로 심링크) | 9402 files / 3451072805 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w24-camera-placement` = `3267dcb38369` |
| 2026-09-28 | `w24-container-spec` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w24-container-spec` (원경로 심링크) | 9385 files / 3449112826 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w24-container-spec` = `3267dcb38369` |
| 2026-09-28 | `w24-dt10us-cause` | `/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w24-dt10us-cause` (원경로 심링크) | 9411 files / 3464499020 bytes | 원본·사본·심링크 경유 목록 전량 일치(OK), The bundle records a complete history., orca rm ok=True | `archive/w24-dt10us-cause` = `3267dcb38369` |
