# Phase 1 Steps A-C COMPLETE — RL framework + URDF→USD + spawn verify

**Date**: 2026-05-07 late-night (continuation from `phase0_b200_physics_only_rl_setup_20260507.md`)
**HARD RULE**: #26 (B200 physics-only Isaac Sim RL priority, 5/19 deadline)
**Status**: ✅ Steps A-C complete. Step D (RoArmPickEnv design) ready for next session.

---

## TL;DR

- ✅ Step A: rsl_rl 3.1.2 installed (rsl_rl 단독, not all — see comparison)
- ✅ Step B: URDF→USD conversion (8 STL meshes, 6 revolute joints preserved)
- ✅ Step C: USD spawn + 300step PhysX HOME hold (drift 0.118° steady-state)
- 🔓 RL step loop pattern validated: `set_joint_position_target` → `write_data_to_sim` → `sim.step` → `update`

---

## (1) rsl_rl vs all comparison — DECISION: rsl_rl 단독

| 기준 | **rsl_rl 단독 (선택)** | all (rsl_rl + rl_games + skrl + sb3) |
|---|---|---|
| 패키지 수 | 1 | 4 |
| Isaac Lab 공식 우선 예제 | ✅ Unitree H1 4096env tutorial | mixed |
| Massive parallel (4096+) | ✅ RSL/ETH Anymal 검증 | sb3 단일 env 설계 |
| gymnasium 1.2.1 compat | ✅ native | sb3 종종 older gym 요구 |
| Deps 충돌 (B200 8 pin) | 최소 (3 추가: rsl-rl-lib, onnxscript, tensordict) | sb3 numpy/torch 강제 |
| 설치 시간 | ~3min | ~15min |

**선택 사유**: 5/19 deadline 12일 → 한 framework 단단히. Phase 0 4 trap 거치며 deps 8개 pin 회복한 환경. sb3는 numpy 강제 upgrade risk로 pin 깨질 가능. rl_games는 `gym` 가져와 gymnasium과 충돌 위험. NVIDIA Isaac Lab 공식 4096 parallel example이 rsl_rl. 5/19 이후 비교 필요 시 `--install rl_games` 등 incremental 추가 가능.

---

## (2) Phase 1 curriculum — 1-cube pick → # tower stacking

**확정** (사용자 결정 5/07 late-night):
- Phase 1.A: 1-sponge pick (sponge 제대로 잡는지 검증)
- Phase 1.B: # 우물정자 4-sponge stacking (HARD RULE #20)

---

## (3) URDF source — 결정: 기존 isaac_roarm_m3 사용

| 후보 | 결정 |
|---|---|
| `/home/cgxr/Documents/Robotics/isaac_roarm_m3/.../roarm_m3.urdf` (6862B, 4/14 modified, meshes 8 STL) | ✅ **사용** |
| `sim_renders_v5/` (2.2GB rendered PNG 30+ episodes) | ❌ Phase 1 = state-only RL, rendering 불필요 (HARD RULE #17) |
| `sim_scripts/{stacking_scene_v3,generate_stacking_demos_v3}.py` | ✅ **geometry constants 재활용** |

**Pure sim 진행 방식**: 이미지 0회. URDF→USD 1회 변환. DirectRLEnv subclass로 obs=state-only.

---

## Step A — rsl_rl install on B200

### Pre-install snapshot (모두 보존)
- torch 2.7.0+cu128, sm_100 alive
- numpy 1.26.0, Pillow 11.3.0, packaging 23.0
- 8 pin: typing_extensions 4.12.2, filelock 3.13.1, fsspec 2024.6.1, markupsafe 2.1.3, networkx 3.3, sympy 1.13.3, setuptools 69.5.1, packaging 23.0

### Install (B200, isaacsim_5_1 env)
```bash
ssh JHPark "set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
micromamba activate \$ROARM_B200_ROOT/envs/isaacsim_5_1
cd \$ROARM_B200_ROOT/code/IsaacLab
./isaaclab.sh --install rsl_rl
"
```

### 추가된 패키지 (해당 install로만)
- rsl-rl-lib==3.1.2 (PPO + OnPolicyRunner)
- onnxscript==0.7.0
- onnx_ir==0.2.1
- tensordict==0.12.2
- GitPython 3.1.50, gitdb 4.0.12, smmap 5.0.3, importlib_metadata 9.0.0, zipp 3.23.1, pyvers 0.2.2, orjson 3.11.9

### Post-install verify (모두 PASS)
- torch 2.7.0+cu128 ✓ (`ensure_cuda_torch` skipped — version match)
- 8 pin 모두 unchanged
- `rsl_rl.algorithms.PPO` import OK
- `rsl_rl.runners.OnPolicyRunner` import OK
- B200 matmul bf16 PASS (sm_100)

### Notes
- isaaclab.sh에 `ensure_cuda_torch()` 함수 있음 — current torch == `${torch_ver}+cu${cuda_ver}` 일치 시 skip. 우리 2.7.0+cu128 매칭 → skip. **HARD RULE #15 호환**.
- Warning "isaaclab-mimic 1.0.16 does not provide the extra 'rsl-rl'" — non-blocking, mimic 안 씀.

---

## Step B — URDF → USD on B200

### URDF analysis
- Path: `$ROARM_B200_ROOT/assets/roarm_m3/urdf/roarm_m3.urdf` (6862B)
- 9 links: world, base_link, link1-5, gripper_link, hand_tcp
- 8 joints: 2 fixed (world↔base, link5↔hand_tcp) + **6 revolute** (matches RoArm M3 6-DOF)
- Joint limits (rad): base ±π, shoulder ±π/2, **elbow [-1.0, 2.95]** (asymmetric ✓ HW spec), wrist_p ±1.92, wrist_r ±π, gripper [0, 1.571]
- Meshes 2.8MB (8 STL)
- ⚠️ **Gripper = single link** (no parallel-finger pair) — Phase 1.B grasp는 attach-pattern 사용 (RigidObject reattach)

### Transfer (local → B200)
```bash
# local
cd /home/cgxr/Documents/Robotics
tar czf /tmp/roarm_m3_urdf.tgz -C isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3 urdf/
scp /tmp/roarm_m3_urdf.tgz JHPark:/tmp/
# B200
ssh JHPark
source $ROARM_B200_ROOT/env.sh
mkdir -p $ROARM_B200_ROOT/assets/roarm_m3
cd $ROARM_B200_ROOT/assets/roarm_m3
tar xzf /tmp/roarm_m3_urdf.tgz
```

### URDF → USD convert
```bash
ssh JHPark "
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
micromamba activate \$ROARM_B200_ROOT/envs/isaacsim_5_1
export OMNI_KIT_ACCEPT_EULA=YES
cd \$ROARM_B200_ROOT/code/IsaacLab
python scripts/tools/convert_urdf.py \
    \$ROARM_B200_ROOT/assets/roarm_m3/urdf/roarm_m3.urdf \
    \$ROARM_B200_ROOT/assets/roarm_m3/usd/roarm_m3.usd \
    --fix-base --merge-joints \
    --joint-target-type position \
    --headless
"
```

### USD output (paths to track)
- `$ROARM_B200_ROOT/assets/roarm_m3/usd/roarm_m3.usd` (1457B, USD references)
- `$ROARM_B200_ROOT/assets/roarm_m3/usd/configuration/` (mesh data)
- `$ROARM_B200_ROOT/assets/roarm_m3/usd/config.yaml` (663B, converter config)
- `$ROARM_B200_ROOT/assets/roarm_m3/usd/.asset_hash` (32B)

### Warnings observed (모두 non-blocking)
- `link hand_tcp ... merged into link5` (--merge-joints 결과, fixed joint 정리)
- `link base_link ... merged into world` (fix-base + merge-joints)
- NVML mismatch / "CUDA bad state" — Phase 0 known. PhysX는 PyTorch+Warp CUDA 직접 사용 → 영향 없음.

---

## Step C — USD spawn + 300 step PhysX verify

### V2.0 (initial, 100 step) — drift 0.0554 rad
- 결과: false-fail. arm은 시작점 0에서 HOME π/2로 정확히 수렴 중. dt=1/200s, 100step=0.5s, vel_limit 3.14 → max travel 1.57 rad. 시간 부족 단순.
- 추가 발견: `InitialStateCfg.joint_pos`가 spawn 시 적용 안 됨.

### V2.1 (write_joint_state_to_sim + 300 step) — drift 1.57 rad (FAIL)
- HOME 초기상태 적용 ✓ (Stage 6 pos[2]=1.5661)
- 그러나 step 50/100/200 진행하며 q=0으로 settle
- **Bug 진단**: `set_joint_position_target(home)` 호출했으나 PhysX로 전파 안 됨. controller가 default target=0 그대로 사용 → arm을 q=0으로 능동 pull

### V2.2 (FIX: write_data_to_sim each step) — ✅ PASS
**Fix**: 매 step `arm.set_joint_position_target` 호출 후 **`arm.write_data_to_sim()` 명시 호출**해야 buffer가 PhysX에 propagate.

```python
for i in range(300):
    arm.set_joint_position_target(home_t)
    arm.write_data_to_sim()         # ← CRITICAL — 누락 시 default target 사용됨
    sim.step()
    arm.update(sim_cfg.dt)
```

**결과**:
- step 0 drift=0.00018 rad (HOME 직후)
- step 50/100/200/299 drift=0.00206 rad (steady-state, 0.118°)
- vel_max 0.032 rad/s (gravity vs control 균형)
- drift_max 0.00206 < tolerance 0.01 ✓

### Verify scripts (saved)
- Local: `/tmp/test_usd_spawn_physx_v2p2.py` (final, V2.2)
- B200: `/tmp/test_usd_spawn_physx_v2p2.py`
- B200 logs: `$ROARM_B200_ROOT/logs/phase1/{v2_usd_spawn,v2p1_usd_spawn,v2p2_usd_spawn}.log`

---

## CRITICAL Lessons Learned (RL env 설계 직접 영향)

### Lesson 1 — RL step loop pattern (Isaac Lab DirectRLEnv 필수)
```python
def step(self, action):
    target = clip_to_limits(action)
    self._arm.set_joint_position_target(target)
    self._arm.write_data_to_sim()    # ← 빠지면 default target 0 사용됨
    self._sim.step()
    self._arm.update(self._dt)
```

### Lesson 2 — Episode reset (HOME 적용 패턴)
```python
def _reset_idx(self, env_ids):
    home = ...  # (num_envs, 6)
    self._arm.write_joint_state_to_sim(home, zeros, env_ids=env_ids)
    self._arm.set_joint_position_target(home, env_ids=env_ids)
    self._arm.write_data_to_sim()
    # NO sim.step() needed here; first env step will progress
```

### Lesson 3 — Mesh cache cold start
- 첫 USD load 시 PhysX collision mesh decomposition + tessellation cache 생성 (~5-10분)
- 두 번째부터 즉시 (V2.0 14분 → V2.2 ~30s)
- B200 NVML mismatch 환경에서 GPU mesh build이 CPU fallback될 수 있음

### Lesson 4 — Actuator config (RoArm M3 검증값)
- ImplicitActuatorCfg(stiffness=80, damping=4, effort_limit_sim=2.5, velocity_limit_sim=3.14)
- HOME hold 시 drift 0.118° steady-state (충분)
- RL training 시 stiffness 100-200, damping 10-20 정도로 강하게 튜닝 가능 (현재 약함)

### Lesson 5 — Initial state caveat
- `ArticulationCfg.InitialStateCfg.joint_pos` 는 USD spawn 시점에 강하게 적용되지 않음
- 명시적으로 `arm.write_joint_state_to_sim(home, zeros)` 호출 필요

---

## Step D — RoArmPickEnv design draft (next session 진입점)

### v3 geometry constants (HARD RULE #19/#20 lock-in)
```python
TABLE_Z = -0.012117                  # 4/24 calib RMSE 1.24 mm
SPONGE_HEIGHT_EDGE = 0.047           # edge-stand vertical
SPONGE_LEN_LONG = 0.125              # length on table
SPONGE_WIDTH = 0.022                 # width — gripper closes on this
Z_TCP_GRASP_L1 = 0.033               # v6 mean +36.8mm match
Z_APPROACH = 0.040                   # hover above grasp
Z_TRANSIT = 0.150                    # v6 p50 166mm
SAFETY_Z_MAX_TRAIN = 0.155           # train hard ceiling
HOME_RAD = [0, 0, math.pi/2, 0, 0, 0]  # HARD RULE #1
```

### Phase 1.A RoArmPickEnv (1-cube pick)
- **Obs (22-dim, state-only)**: joint_pos[6] + joint_vel[6] + sponge_pos[3] + sponge_quat[4] + tcp_to_sponge_vec[3]
- **Action (6-dim)**: joint position target rad, scaled to per-joint limits
- **Spawn**: edge-stand sponge at uniform [+0.150,+0.430]×[-0.220,+0.220] (R1-R4 union from v3)
- **Reward (shaped, curriculum)**:
  - P1 (10K steps): −‖tcp − sponge‖ (reach only)
  - P2 (20K): + λ₁·sponge_z_lift
  - P3 (30K): + λ₂·grasp_bonus(joint5 > 0.4 ∧ tcp within 25mm) − λ₃·action_rate
- **Termination**: sponge_z > +0.10m for ≥50 steps = success / 200 step timeout
- **Grasp impl**: Isaac Lab `RigidObject` + frame-attach when condition met (no parallel-finger)

### Phase 1.B RoArmStackingEnv (# tower)
- 4 sponge spawn at random R1-R4 regions
- Targets: DST_L1_SP1=(+0.280,-0.0435), DST_L1_SP2=(+0.280,+0.0435), DST_L2_SP3=(+0.2465,0), DST_L2_SP4=(+0.3135,0)
- L1 wrist_r=0°, L2 wrist_r=+90°
- 4-step pick-place sequence reward

---

## Path Inventory (모든 산출물)

### Local
| 종류 | 경로 | 비고 |
|---|---|---|
| Phase 0 결과 doc | `claudedocs/phase0_b200_physics_only_rl_setup_20260507.md` | 5/07 night |
| **Phase 1 A-C 결과 doc (이 파일)** | `claudedocs/phase1_step_abc_complete_20260507.md` | 5/07 late-night |
| URDF transfer 패키지 (1회용) | `/tmp/roarm_m3_urdf.tgz` (689KB) | scp 완료 후 삭제 가능 |
| Verify 스크립트 v1 (deprecated) | `/tmp/test_usd_spawn_physx.py` | drift bug |
| Verify 스크립트 v2.1 (deprecated) | `/tmp/test_usd_spawn_physx_v2.py` | propagate bug |
| Verify 스크립트 v2.2 (FINAL) | `/tmp/test_usd_spawn_physx_v2p2.py` | PASS |
| RoArm M3 URDF 원본 | `/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf` | 4/14 modified |

### B200 (`$ROARM_B200_ROOT = /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200`)
| 종류 | 경로 |
|---|---|
| Conda env (Isaac Sim/Lab) | `$ROARM_B200_ROOT/envs/isaacsim_5_1/` |
| Conda env (lerobot, 별도) | `$ROARM_B200_ROOT/envs/roarm_b200/` |
| Isaac Lab repo | `$ROARM_B200_ROOT/code/IsaacLab/` (v2.3.2) |
| URDF + 8 STL meshes | `$ROARM_B200_ROOT/assets/roarm_m3/urdf/` |
| **USD (RL env load 대상)** | `$ROARM_B200_ROOT/assets/roarm_m3/usd/roarm_m3.usd` |
| Verify 스크립트 (final) | `/tmp/test_usd_spawn_physx_v2p2.py` |
| Verify 로그 v1/v2.1/**v2.2 PASS** | `$ROARM_B200_ROOT/logs/phase1/{v2,v2p1,v2p2}_usd_spawn.log` |

---

## 환각 방지 — 다음 세션 진입 시 반드시 확인

| 확인 | How |
|---|---|
| isaacsim_5_1 env (lerobot은 roarm_b200 별도) | `echo $CONDA_PREFIX` 후 `/envs/isaacsim_5_1` 끝 매칭 |
| torch sm_100 alive? | `python -c "import torch; print('sm_100' in torch.cuda.get_arch_list())"` |
| rsl_rl alive? | `python -c "from rsl_rl.algorithms import PPO; print('OK')"` |
| USD path alive? | `ls $ROARM_B200_ROOT/assets/roarm_m3/usd/roarm_m3.usd` |
| EULA env? | `export OMNI_KIT_ACCEPT_EULA=YES` 매번 |

---

## Time consumed
| Step | Time |
|---|---|
| (1) rsl_rl 결정 + Isaac Lab docs research | ~5 min |
| (3) URDF source 결정 + v3 geometry 추출 | ~5 min |
| Step A: rsl_rl install + verify | ~5 min |
| Step B: URDF transfer + USD convert | ~10 min (cache cold) |
| Step C: V2.0 (false-fail) + V2.1 (bug) + V2.2 (PASS) | ~25 min |
| **Total Phase 1 A-C** | **~50 min** |

→ Step D (RoArmPickEnv) 진입 대기 상태.
