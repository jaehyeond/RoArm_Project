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
