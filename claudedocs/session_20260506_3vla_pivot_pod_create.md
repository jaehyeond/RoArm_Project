# Session 2026-05-06 — 3-VLA Real-to-Sim Pivot + RunPod Pod 생성 + Memory 정정

## 요약 (1줄)
사용자 명시 결정으로 3-way → 3-VLA real-to-sim 비교 추가 (HARD RULE #21 정정), RunPod Pod 생성 PASS, B200 작업폴더 사라짐 발견, forge 기본 변환 실패.

## HARD RULE 변경/검증

- **#21 정정 (5/06)**: 같은 backbone(SmolVLA) variant만 금지, 다른 backbone (π0/OpenVLA-OFT) 3-VLA 비교로 명시 허용. inline 정정 완료 ([MEMORY.md:33](../../../.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY.md#L33))
- **#17 정정 (5/05 검증)**: B200 Vulkan ICD 부재 = 학습 only, sim/RL 막힘 (state-only RL이라도 `app.vulkan = true` 강제). Issue #357로 검증.
- **#11 거부**: 본 세션 1회 (Stop hook 125%, /half-clone X — 세션 종료 프로세스 적용)

## 완료 작업

| # | 작업 | 결과 |
|---|---|---|
| 1 | Memory: HARD RULE #21 정정 inline | [MEMORY.md:33](../../../.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY.md#L33) 끝부분 추가 |
| 2 | Memory: 5/06 entry prepend (Recent Sessions) | 9개 항목 — 사용자 결정 + Pod + B200 + OFT + 매트릭스 |
| 3 | Memory: 5/04 archive 이동 | [MEMORY_archive_20260505.md](../../../.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY_archive_20260505.md) append |
| 4 | Topic file 작성 | [project_3vla_real_to_sim_20260506.md](../../../.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/project_3vla_real_to_sim_20260506.md) |
| 5 | OFT constants.py에 ROARM_M3 추가 | [prismatic/vla/constants.py](../../openvla-oft/prismatic/vla/constants.py) — `ROARM_M3_CONSTANTS` (chunk=8, dim=6/6, BOUNDS_Q99) + detect_robot_platform "roarm" 분기 |
| 6 | **RunPod Pod 생성** | id `az53n8t8alp8pz`, RTX A6000 Secure US-TX-1 $0.49/h, image isaac-lab:2.3.2, **38.147.83.11:34856** |
| 7 | B200 SSH 검증 | sogang_jhki@JHPark-container OK / GPU0 c553ca20 / **~/roarm_b200/ 사라짐 발견** |
| 8 | forge tool 설치 | openvla env (Python 3.10) + forge-robotics 0.2.0 + lerobot/rlds/hub extras |
| 9 | v6 forge inspect | 50ep × 6942fr, top 1280x720, action(6), state(6), language 100% — 정확 |

## 발견 이슈 (다음 세션에서 해결)

### Issue 1: forge 기본 변환 = 사용 불가
- output observation FeaturesDict **완전 비어있음** (image, state 모두 손실)
- action (6,) → (7,) 잘못
- 이미지 1280x720 → 640x480 다운스케일
- **결정**: forge 폐기, 커스텀 RLDS converter 작성 (~2h)
  - 패턴: [Physical-Intelligence/openpi `convert_libero_data_to_lerobot.py`](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py) (역방향이지만 schema 참고)
  - 또는: [Ke-Wang1017/lerobot_rlds](https://github.com/Ke-Wang1017/lerobot_rlds) 참고
  - 또는: openvla `modified_libero_rlds` HF dataset schema 따라가기

### Issue 2: B200 ~/roarm_b200/ 사라짐
- last activity Apr 29
- env.sh, lerobot_dataset_v6_stacking_v3 (115MB), outputs/smolvla_v6_stacking_v3_b200/ 모두 사라짐
- **Lenovo 백업 보존됨**:
  - `outputs/smolvla_v6_stacking_v3_b200/` (4ckpt 4.8GB) — 5/04 night byte-exact rsync 검증
  - `lerobot_dataset_v6_stacking_v3/` (115MB)
  - `lerobot_dataset_v6/` (75MB, 50ep)
- **사용자 결정 옵션**:
  - (a) 운영팀 정리 사유 문의 + 복구 요청
  - (b) **재셋업 (추천)** — Lenovo 백업 rsync 1h + env install 1h = 2h
  - (c) RunPod에 OpenVLA-OFT까지 학습 (~$48 추가, 24h)

### Issue 3: Pod boot 시간
- image isaac-lab:2.3.2 = 30GB+, container disk 100GB
- 폴링 시작 03:25:34 KST → ~03:27까지 still booting
- 예상 5-10분 boot
- Monitor `bngvgphnb` 폴링 중 — ready 시 자동 알림

## 다음 세션 진입 (Continuation Prompt)

```
RoArm M3 SmolVLA 3-VLA Real-to-Sim Pivot 진행 (5/06 lock-in).

전 세션 완료:
- HARD RULE #21 정정 (다른 backbone 허용), #17 정정 (B200 Vulkan 학습 only)
- RunPod Pod az53n8t8alp8pz 생성 (RTX A6000, 38.147.83.11:34856, isaac-lab:2.3.2)
- OFT constants.py ROARM_M3 추가 (action/proprio=6, chunk=8)
- forge tool 설치 + v6 inspect PASS, 그러나 forge 기본 변환 = observation 손실 (사용 불가)
- B200 ~/roarm_b200/ 사라짐 발견 (Lenovo 백업 살아있음)

진입 시 사용자 결정 대기:
1. B200 재셋업 옵션 (a)/(b)/(c) — 추천 (b)
2. forge 폐기 + 커스텀 RLDS converter 작성 OK?

진입 시 즉시 작업:
- A. Pod boot 완료 확인 (mcp__runpod__get-pod az53n8t8alp8pz, ssh -p 34856 root@38.147.83.11)
- B. Pod ready 시: vulkaninfo + git clone (RoArm_Project + isaac_roarm_m3 + IsaacLab) + stacking_scene_v3.py --headless 실행 → 첫 PNG output
- C. 커스텀 RLDS converter 작성 (parquet → tf.data.Dataset, OFT schema 매칭)
- D. B200 재셋업 (사용자 OK 후): Lenovo rsync + nightly cu128 + lerobot install + flash-attn 2.5.5 smoke

핵심 read 파일:
- memory/project_3vla_real_to_sim_20260506.md (5/06 lock-in plan)
- memory/MEMORY.md HARD RULE #21 정정 (line 33)
- memory/tech_b200_server_setup.md (B200 환경 절차)
- claudedocs/session_20260506_3vla_pivot_pod_create.md (본 세션 결과)

코드 진입:
- /home/cgxr/Documents/Robotics/openvla-oft/ (cloned, ROARM_M3 constants 추가됨)
- /home/cgxr/Documents/Robotics/forge/ (cloned, 기본 변환 unusable)
- /home/cgxr/Documents/Robotics/RoArm_Project/lerobot_dataset_v6_rlds/ (forge output 100MB, 검증 실패 — 폐기 후 재변환 필요)

3-VLA 학습 매트릭스 (5/06 lock-in):
- SmolVLA 450M: ✅ done (v6_b200/last)
- π0 3.3B PaliGemma: B200 GPU0, ~10h, lerobot policy.type=pi0
- OpenVLA-OFT 7B: B200 GPU0, ~24h, bs=8 grad_accum=4, OFT recipe + smoke test 강제

HARD RULE 준수: #11 /half-clone X / #13 dual-PC / #14 fail-fast / #15 nightly cu128 / #16 train_config / #17 B200 학습 only / #18 사용자 명시 정정 / #19 edge-stand / #20 # tower / #21 정정.

step-by-step 교차 검증, 검토, 확인하면서 순차 진행.
```

## Reference

- Pod ID: az53n8t8alp8pz / IP 38.147.83.11 / Port 34856 / SSH key local ~/.ssh/id_ed25519
- B200: ssh JHPark (sogang_jhki@59.150.32.1:47110) — IdentityFile ~/.ssh/JHPark-container_key
- OpenVLA-OFT: arXiv [2502.19645](https://arxiv.org/abs/2502.19645)
- π0: arXiv [2410.24164](https://arxiv.org/abs/2410.24164)
- forge: [arpitg1304/forge](https://github.com/arpitg1304/forge) (default convert unusable for our case)
- 3-VLA topic: [project_3vla_real_to_sim_20260506.md](../../../.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/project_3vla_real_to_sim_20260506.md)
