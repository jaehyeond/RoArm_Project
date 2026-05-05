# Session 2026-05-05 night-2 — Phase ST-D Day 1 진입 + Plan v2.1 정정 + B200 Vulkan 재검증

## 핵심 정정 (사용자 명시 정정 2건, HARD RULE #18)

### 정정 #1: Conda env 신규 생성 거부 → `isaaclab` 재사용
- **이전 (잘못)**: Plan v2 = `roarm_rl` 신규 conda env 생성 + ManiSkill3 install
- **사용자 정정**: "왜 새로 만드는거지? B200 셋업 다 했는데"
- **검증**: `conda run -n isaaclab pip list` →
  - isaacsim 5.1.0 + isaaclab 2.3.0 + isaaclab_rl 0.4.7 + isaaclab_tasks 0.11.13 + isaaclab_mimic 1.0.16
  - **isaac-roarm-m3 0.1.0** (이미 통합됨)
  - rl_games 1.6.1 + rsl-rl-lib 3.0.1 + skrl 1.4.3 + stable_baselines3 2.7.1 (모든 RL backend)
  - gymnasium 1.2.0
- **확정**: `isaaclab` env 그대로 사용. 새 env 생성 불필요.

### 정정 #2: ManiSkill3 거부 → Isaac Lab + 기존 sim env 재활용
- **이전 (잘못)**: ManiSkill3 install + StackCube fork → RoArmStackTask
- **사용자 정정**: "이미 sim env 잘 구축했잖아 (사진), 재활용하면 될 것 같은데"
- **검증**:
  - `sim_renders_v5/stacking_initial_seed0_v3.png` 존재 (사용자 사진과 픽셀 매치)
  - `stacking_scene_v3.py` `conda run -n isaaclab` 9.9s exit 0 재실행 PASS
  - HARD RULE #19/#20 (edge-stand 47mm + # tower) 이미 v3 코드에 구현
  - sim_renders_v5/에 50 episodes (sim_demos_v3) 데이터 존재
- **확정**: stacking_scene_v3.py + generate_stacking_demos_v3.py + render_stacking_demos_v3.py 재활용.

## Plan v2.1 (정정 반영)

| 항목 | Plan v2 (잘못) | **Plan v2.1 (정정)** |
|---|---|---|
| Conda env | roarm_rl 신규 | **isaaclab 재사용** |
| Sim env | ManiSkill3 fork | **stacking_scene_v3.py 재활용** |
| RL framework | ManiSkill3 PPO | **isaaclab_rl (rl_games/rsl_rl/skrl/sb3)** |
| RoArm 통합 | URDF 새 import | **isaac-roarm-m3 0.1.0 + ROARM_M3_CFG 재사용** |
| Day 1 작업 | install 단계 포함 | **Day 1 50% 단축, task class scaffolding 직진** |

## Day 1 진행 결과

### Step 1-4 PASS
1. **Sim env 재실행**: stacking_scene_v3.py 9.9s exit 0, 두 PNG 픽셀 매치
2. **Isaac Lab stack 구조 분석**: StackEnvCfg base + Franka/Galbot/UR10 config 패턴 → roarm_m3 추가 가능
3. **ROARM_M3_CFG 검증**: actuator stiffness/damping mass-calibrated, init_state v6 dataset mean, arm/gripper 그룹 분리. **Day 1 80% 절약**
4. **Franka stack pattern**:
   - line 24: `from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG`
   - line 72: `self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path=...)`
   - line 82: `JointPositionActionCfg(...)` + line 85: `BinaryJointPositionActionCfg(...)`
   - line 107~136: 3 cubes deterministic spawn

### Step 5 PARTIAL — RL reward reference 발견
- **Stack task에는 rewards = None** (BC focus, robomimic only)
- **Lift task가 RL 활성 reference**: rl_games_ppo_cfg.yaml + rsl_rl_ppo_cfg.py + sb3_ppo_cfg.yaml + skrl_ppo_cfg.yaml + mdp/rewards.py 모두 존재
- Lift 패턴: `reaching_object` (dense, std=0.1) + `lifting_object` (sparse +15) + `object_goal_tracking` (dense + fine) + action_rate/joint_vel penalty + curriculum modify_reward_weight

### Step 6-9 미진행 (B200 Vulkan 검증으로 분기)
- (보류) RoArmStackEnvCfg class 작성
- (보류) mdp/rewards.py 작성
- (보류) agents/rl_games_ppo_cfg.yaml + skrl_ppo_cfg.yaml 작성
- (보류) gym register + 4090 PPO smoke test

## B200 Vulkan ICD 재검증 결과 (사용자 의심 정당성 확인)

### 검증된 사실
1. **AppLauncher kit 선택 로직** (app_launcher.py:216-219):
   - headless=True + enable_cameras=False → `isaaclab.python.headless.kit` (no scene delegate, no render)
   - → **state-based RL은 이 kit 사용 (Reach/Lift/Stack 전부 RGB obs 없음 검증)**

2. **그러나 모든 .kit 파일 `app.vulkan = true`**:
   - `isaaclab.python.headless.kit:89` line: `app.vulkan = true`
   - 코멘트: "Enable Vulkan - avoids torch+cu12 error on windows"
   - Linux + ICD 부재 시 동작은 unspecified

3. **NVIDIA forum 실 사례** (B200 막힘 가능성 ≥50%):
   - "Isaac Sim Container runheadless got stuck" — headless container + ICD 부재 stuck
   - "Isaac Gym Black window missing nvidia_icd.json Segmentation fault"
   - "ERROR_INCOMPATIBLE_DRIVER" 2025-12 보고
   - WSL2 + missing ICD Vulkan init failure 다수

4. **HARD RULE #17 정정 필요**:
   - 4/28 V7 재해석 ("PhysX RL = OK")은 **검증 안 된 추정**
   - 실 검증 (B200에 install + headless smoke test) **미진행**

### 옵션 3개 (사용자 confirm 대기)

| 옵션 | 비용 | 시간 | 기대 성공률 |
|---|---|---|---|
| **A. B200 ICD 우회 시도** (user-local nvidia_icd.json + VK_ICD_FILENAMES) | 무료 | 1-2h | 50% |
| **B. 운영팀 문의** (단톡방 문의 #3 미답, vulkan-tools/libnvidia-gl 설치) | 무료 | 1-3일 대기 | 70% (RGB까지 가능) |
| **C. RunPod 추가 임대** (A6000 48GB ~$165 / 14일) | $165~$835 | 즉시 | 99% |

### 옵션 A 명령 (사용자 ssh로 B200 진입 시 실행)
```bash
ssh JHPark
set -e
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && exit 1

# 1. NVIDIA driver lib 위치 찾기
find /usr/lib /usr/lib64 /opt -name "libGLX_nvidia.so*" 2>/dev/null
find /usr/lib /usr/lib64 /opt -name "libnvidia-glcore*" 2>/dev/null
find / -path /proc -prune -o -name "nvidia_icd.json" -print 2>/dev/null | head

# 2. user-local ICD JSON 생성 (lib path는 1단계에서 찾은 결과로 대체)
mkdir -p $ROARM_B200_ROOT/.local/vulkan/icd.d
cat > $ROARM_B200_ROOT/.local/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
  "file_format_version" : "1.0.0",
  "ICD": {
    "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
    "api_version" : "1.3.0"
  }
}
EOF
export VK_ICD_FILENAMES=$ROARM_B200_ROOT/.local/vulkan/icd.d/nvidia_icd.json

# 3. vulkaninfo로 detect 확인 (vulkan-tools 없으면 skip → 4번 직접)
which vulkaninfo && vulkaninfo --summary 2>&1 | head -30

# 4. Isaac Sim headless smoke test (5분)
python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app
print('AppLauncher OK')
import isaacsim
print('isaacsim import OK')
app.close()
"
```

성공 기준: 4단계 명령 exit 0 + "AppLauncher OK" + "isaacsim import OK" 출력.
실패 기준: hang (>5min) 또는 segfault 또는 vulkan error.

## 다음 세션 즉시 진입 (Continuation Prompt)

```
RoArm M3 SmolVLA Phase ST-D Day 1 (재진입). 5/05 night-2 lock-in.

Plan v2.1 확정 (이전 세션 정정 2건):
- Conda env: isaaclab (신규 X) + isaac-roarm-m3 0.1.0 재사용
- Sim env: stacking_scene_v3.py + generate_stacking_demos_v3.py 재활용 (ManiSkill3 X)
- RL framework: isaaclab_rl (rl_games/rsl_rl/skrl/sb3 4 backend 모두 설치됨)
- ROARM_M3_CFG ArticulationCfg 이미 완성 (Day 1 80% 절약)

이전 세션 진행도:
- Day 1 Step 1-4 PASS (sim 재실행 + Isaac Lab stack 분석 + ROARM_M3_CFG 검증 + Franka pattern)
- Step 5 PARTIAL (Lift task가 RL reward reference: rl_games_ppo_cfg.yaml + mdp/rewards.py)
- Step 6-9 미진행 (B200 Vulkan 검증 분기)

🚨 B200 Vulkan ICD 의심 — 사용자 결정 대기:
- HARD RULE #17 ("B200 visual sim 막힘") = 검증 안 된 추정 (4/28 V7 재해석)
- NVIDIA forum 실 사례: headless container도 ICD 부재 시 startup hang/segfault
- AppLauncher headless.kit도 app.vulkan = true (Linux + ICD 부재 동작 unspecified)
- 옵션 3개:
  A. B200 ICD 우회 (user-local nvidia_icd.json + VK_ICD_FILENAMES, 무료, 1-2h, 50%)
  B. 운영팀 문의 (단톡방 #3 미답, 1-3일, 70%)
  C. RunPod A6000 48GB (~$165/14일, 99%)

이전 세션 추천: A 시도 → 실패 시 C, B는 병행.

다음 step (사용자 옵션 선택 후):
- 옵션 A 선택: ssh JHPark에서 4단계 명령 실행
  (claudedocs/session_20260505_night2_st_d_day1_plan_v2_1_pivot.md 옵션 A 섹션 참조)
- 옵션 C 선택: RunPod 결제 + Pod 셋업 + isaaclab env 동등 install
- 옵션 B만: A/C 결정 + 발송 사용자 작업

진입 시 step-by-step + 교차 검증 (사용자 강조).
HARD RULE #11 (/half-clone X) + #16 train_config + #17 정정 필요 + #18 + #21 준수.

핵심 파일:
- claudedocs/session_20260505_night2_st_d_day1_plan_v2_1_pivot.md (본 세션 상세)
- memory/project_research_3way_pivot_20260505.md (3-way lock-in)
- claudedocs/research_plan_v2_3way_pivot.md (Day 1-14 task)
- memory/tech_b200_server_setup.md (V7-V8 Vulkan ICD)

코드 진입 지점:
- /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/stacking_scene_v3.py (재실행 PASS)
- /home/cgxr/Documents/Robotics/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/ (env_cfg base)
- /home/cgxr/Documents/Robotics/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/ (RL config reference)
- /home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/roarm_m3.py (ROARM_M3_CFG)
- /home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/tasks/reach/reach_env_cfg.py (Reach task reference)
```

## 미해결 / 다음 세션 작업

1. 사용자 옵션 A/B/C 결정
2. (옵션 A) B200 ICD 우회 시도 + Isaac Sim headless smoke test
3. (옵션 C) RunPod A6000 결제 + Pod 셋업
4. RoArmStackEnvCfg class 작성 (Step 6)
5. mdp/rewards.py 작성 (Step 7)
6. agents/rl_games_ppo_cfg.yaml + skrl_ppo_cfg.yaml 작성 (Step 8)
7. gym register + 4090 PPO smoke test (Step 9)

## HARD RULE 준수
- #11 /half-clone X (Stop hook 85% 거부 1회, 본 세션 종료 프로세스 적용)
- #17 정정 필요 ("B200 visual sim 막힘" → "B200 Vulkan 동작 미검증, ICD 우회 시도 가치 있음")
- #18 사용자 명시 정정 우선 (Plan v2 → Plan v2.1)
- #21 3-way 비교 lock-in (변경 X)
