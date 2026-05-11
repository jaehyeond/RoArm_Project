# Phase 0 — B200 Physics-Only Isaac Sim RL Environment Setup

**Date**: 2026-05-07 night
**HARD RULE**: #26 (B200 physics-only Isaac Sim RL priority, 5/19 deadline)
**Status**: ✅ **COMPLETE — V1 hypothesis validated**

---

## TL;DR

B200 + Isaac Sim 5.1 + Isaac Lab v2.3.2 + headless+enable_cameras=False = **PhysX simulation works** without SIGSEGV. Ready for Phase 1 (RoArm M3 RL env wrapper).

---

## V1 검증 결과 (HARD RULE #26 hypothesis)

| Test | Result | Evidence |
|---|---|---|
| **V1.0** SimulationApp launch (headless+no-cameras) | ✅ PASS | exit 0, "Simulation App Startup Complete" + "Shutting Down" 정상 |
| **V1.1** PhysX 100step state-only | ✅ PASS | Cube 자유낙하 (1.0m → 0.05m), sim_time=1.7s, no crash |
| **V1.2** Isaac Lab full stack import | ✅ PASS | DirectRLEnv, ManagerBasedRLEnv, scene, assets, sensors, actuators, controllers 모두 OK |
| **HARD RULE #17 visual RL fail mode 회피** | ✅ Confirmed | Annotator/TiledCamera 미사용 → Vulkan→CUDA tensor mapping SIGSEGV 없음 |
| torch sm_100 (B200 Blackwell) | ✅ Working | torch 2.7.0+cu128, arch_list includes sm_100, matmul fp32+bf16 PASS |

**결론**: 교수님 + B200 회사 직원 발언 ("Isaac Sim rendering OFF가 코드로 가능") = **B200에서 검증 완료**.

---

## 환경 매트릭스 (확정 lock-in)

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 22.04.5 (Docker container) | NHN B200 server |
| GPU | NVIDIA B200 (sm_100, Blackwell) | UUID `c553ca20-377c-49dd-c30b-f5c530b3ff69` |
| CUDA driver | 580.95.05 | NVML mismatch 경고 있지만 non-blocking |
| Python | 3.11.15 | micromamba env `isaacsim_5_1` |
| **PyTorch** | **2.7.0+cu128** | sm_100 + sm_120 + compute_120 ✓ |
| **Isaac Sim** | **5.1.0.0** (pip install) | EULA accepted via `OMNI_KIT_ACCEPT_EULA=YES` |
| **Isaac Lab** | **v2.3.2** (git clone, editable) | `code/IsaacLab/` |
| Warp | 1.13.0 | NVIDIA Warp DSL for sim kernels |
| Gymnasium | 1.2.1 | Isaac Lab pinned |
| Vulkan ICD | 1.4.312 (sysadmin 5/07 install) | headless에서도 init되지만 RTX renderer 미사용 |

---

## Install 절차 (재현 가능)

### Step 1: torch cu126 → cu128 (sm_100 회복)

```bash
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1

pip install --upgrade --force-reinstall \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: isaacsim deps 8개 pin 복원 (force-reinstall이 latest 강제 → isaacsim 호환 깨짐)

```bash
pip install --no-deps --force-reinstall \
    "numpy==1.26.0" "Pillow==11.3.0" "typing_extensions==4.12.2" \
    "filelock==3.13.1" "fsspec==2024.6.1" "markupsafe==2.1.3" \
    "networkx==3.3" "sympy==1.13.3"
```

### Step 3: Isaac Lab v2.3.2 clone + setuptools downgrade + isaaclab.sh --install

```bash
cd $ROARM_B200_ROOT/code
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2

# setuptools 70+은 pkg_resources 제거 → flatdict 4.0.1 source build fail
pip install "setuptools==69.5.1"

# flatdict 4.0.1 (isaaclab pin) source build with --no-build-isolation
pip install --no-build-isolation "flatdict==4.0.1"

# isaaclab.sh --install (none = skip RL frameworks for now)
./isaaclab.sh --install none

# flatdict 4.0.1 트랩 회피용 추가:
pip install --no-build-isolation -e source/isaaclab
```

### Step 4: V1.0/V1.1/V1.2 검증

`scratch/test_v1_minimal_launch.py`, `scratch/test_v1_state_only_physx.py`, `scratch/test_v1_isaaclab_apilauncher.py` (B200에 저장됨).

---

## 알려진 Warning / Non-Blocking Issues

1. **NVML mismatch**: `Could not initialize NVML: NVML_ERROR_LIB_RM_VERSION_MISMATCH` — container의 NVML library가 host driver와 미일치. CUDA/PhysX 작동에 영향 없음. 5/07 evening MEMORY entry로 알려진 이슈.
2. **NGX context fail**: DLSS 비활성화 (`--enable_cameras=False` 모드에서 RTX renderer 안 씀). 무관.
3. **carb.audio**: 컨테이너에 audio device 없음. 무관.
4. **GLFW init fail**: 컨테이너에 X11 없음. headless이므로 무관.
5. **psutil 7.2.2 vs isaacsim-kernel pin 5.9.8**: psutil API 안정 → 미충돌 예상.

---

## 다음 작업 (Phase 1, Day 2-4)

### 1. RL framework 설치
```bash
# isaaclab.sh --install rsl_rl  # rsl_rl만 설치 (light + Isaac Lab 공식 PPO)
# 또는 --install all  # rsl_rl, rl_games, skrl, sb3 모두
```
권장: `rsl_rl` 먼저 (가장 light + Isaac Lab 공식 우선 지원). 학습 안정화 후 다른 framework 비교.

### 2. RoArm M3 URDF → USD 변환
- 기존 v6 sim 작업에서 사용한 URDF (`sim_scripts/` 참조)
- Isaac Sim Asset Converter (Python API or `convert.py` CLI)
- 검증: USD 로드 후 joint count 6 + 각 link inertia 정상

### 3. Custom DirectRLEnv 작성 (RoArm M3 stacking)
- `RoArmStackingEnv` (DirectRLEnv subclass)
- Observation: joint pos[6] + joint vel[6] + N×sponge SE(3) + tower SE(3) (이미지 X)
- Action: joint position target [6] (delta or absolute)
- Reward shaping (curriculum: 1-cube → 2-cube → # tower)
- Termination: tower built / fall / timeout

### 4. PPO baseline (Day 5-9)
- rsl_rl PPO config
- 1-cube intermediate task 먼저 → success rate >50% → 2-cube → # tower
- 3 seeds, B200 활용 (massive parallel envs, 4096+ envs)

### 5. 평가 (Day 10-12)
- Sim 평가 metric: success rate, episode length, reward curve
- 5/19 deadline까지 결과물 + 보고서

---

## Abort Criteria (Phase 1 진행 시 대기)

- **Day 3 EOD**: env wrapper + 1 episode rollout 미완 → scope 축소 (1-cube only?) 결정
- **Day 7 EOD**: PPO 1K step 미통과 → 5/19 deadline risk 보고
- **Day 12 EOD (5/19)**: 결과물 못 내면 plan 재논의

---

## 환각 방지 — 다음 세션 진입 시 반드시 확인

| 확인 | How |
|---|---|
| roarm_b200 env (lerobot) vs isaacsim_5_1 env (Isaac Sim/Lab)? | `echo $CONDA_PREFIX` — Isaac Sim RL은 isaacsim_5_1 |
| torch sm_100 alive? | `python -c "import torch; print('sm_100' in torch.cuda.get_arch_list())"` |
| Isaac Lab importable? | AppLauncher 먼저 init하지 않으면 pxr/carb/omni FAIL — 정상 |
| EULA accept env var? | `export OMNI_KIT_ACCEPT_EULA=YES` 매번 필요 |

---

## 시간 소요 (실제)

| Step | Time |
|---|---|
| Step 1: SSH + env verify | ~3 min |
| Step 2: 기존 isaacsim_5_1 env 발견 + torch cu128 install | ~5 min |
| Step 3: deps pin 복원 + verify | ~3 min |
| Step 4: V1.0 minimal launch test | ~5 min (Kit init 40s 포함) |
| Step 5: V1.1 PhysX 100step test | ~3 min |
| Step 6: Isaac Lab clone + isaaclab.sh --install | ~10 min |
| Step 7: flatdict bypass + isaaclab core retry | ~5 min |
| Step 8: V1.2 AppLauncher + full stack import | ~5 min |
| **Total Phase 0** | **~40 min** (estimate Day 0-1 = 1 day. **27× faster**) |

→ Phase 1 entry 대기 상태.
