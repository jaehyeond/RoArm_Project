# P7 Branch B Sim/Lab 데이터셋 스키마 메모 - 2026-05-20

## 범위

Track A P7/Branch B normalized 2cm cube grasp만 다룬다.

이 문서는 교수님 질문에 답하기 위한 데이터셋 구조 설명이다. 여기서
말하는 schema는 실제로 이미 생성된 cube dataset 파일이 아니라, 앞으로
생성해야 할 sim/lab dataset writer의 제안 구조다.

이 문서를 작성하는 동안 Isaac 실행, 학습, constraint 삽입,
SurfaceGripper 부착, transport, release, gate tuning은 하지 않았다.

## 짧은 답

최종 sim/lab dataset은 object pose 변수를 샘플링하는 구조가 맞다.

- `x`: robot/world frame에서 object center의 x 위치.
- `y`: robot/world frame에서 object center의 y 위치.
- `z`: object center 높이 또는 stacking layer 높이.
- `yaw`: vertical axis 기준 object 회전.

물체 밑변까지 일반화하면 밑변 크기도 명시 변수로 들어가는 게 맞다.

- `a`: object-frame x 방향 밑변 길이.
- `b`: object-frame y 방향 밑변 길이.
- `h`: object 높이.

현재 canonical cube에서는 고정값이다.

```text
object_size = (a, b, h) = (0.020, 0.020, 0.020)m
```

현재 grasp target 계산 구조는 다음과 같다.

```text
world_grasp = object_center + RotZ(yaw) @ (normalized_grasp * object_size)
```

따라서 `a,b,h`는 단순 메모가 아니다. object가 `2cm x 2cm x 2cm`
canonical cube가 아니게 되면 world TCP target 자체를 바꾸는 핵심
condition이다.

## 현재 코드 근거

### Object Pose / Size 변수

`sim_scripts/p7_branch_b_cube2cm_local_grasp_close_sweep_probe.py`:

- `--object_size_m`, `--object_xy`, `--yaw_deg`, `--normalized_grasp`를 받는다.
- `_build_plan()`은 `x/y`와 `TABLE_Z + object_size[2] / 2`로 object center를 만든다.
- `_build_plan_from_center()`는 `RotZ(yaw)`와 `normalized_grasp * object_size`로 `world_grasp`를 계산한다.

`sim_scripts/p7_branch_b_cube2cm_normalized_grasp_static_probe.py`:

- source region 위에 workspace `x/y` grid를 샘플링한다.
- yaw 기본 리스트는 `[0, 45, 90, 135]`다.
- `object_size_m`과 normalized grasp spec을 사용해서 static reachability를 감사한다.

### Isaac Lab Runtime에서 내부적으로 제공되는 값

`roarm_rl/roarm_stack_env.py`의 policy observation은 28차원이다.

```text
joint_pos[6]
joint_vel[6]
sponge_pos[3]
sponge_quat[4]
tcp_to_sponge[3]
target_pos_local[3]
sponge_to_target[3]
```

환경 내부에서 접근 가능한 값은 다음과 같다.

- 실제 joint position: `_robot.data.joint_pos`
- 실제 joint velocity: `_robot.data.joint_vel`
- object position: `_sponge.data.root_pos_w`
- object orientation: `_sponge.data.root_quat_w`
- TCP position: `link5` pose와 `_tcp_local` offset으로 계산
- 상대 벡터: TCP-to-object, object-to-target 등

중요한 점: 이 값들은 sim/lab runtime에서 접근 가능하지만, dataset writer가
명시적으로 기록하지 않으면 자동으로 dataset 파일에 저장되지 않는다.

### 현재 real/LeRobot 변환 데이터셋에 저장되는 값

현재 real collection 경로는 raw episode에 다음을 저장한다.

- RGB frame
- depth `.npy`
- follower 또는 single-arm joint angles: `angles`
- L-F mode일 때 leader joint angles: `leader_angles`
- raw metadata 안의 optional SDK pose: `[x_mm, y_mm, z_mm]`

현재 LeRobot 변환 결과는 다음 feature만 저장한다.

- `observation.images.top`
- `observation.state` = 6개 joint angle
- `action` = 6개 joint angle

즉 현재 변환 dataset에는 TCP pose, object pose, torque, acceleration,
`(x,y,z,yaw,a,b,h)` condition 변수가 first-class feature로 저장되어 있지 않다.

## 현재 파일 / 경로 확인 결과

### schema note가 저장된 경로

영문판:

```text
claudedocs/p7_branch_b_sim_lab_dataset_schema_note_20260520.md
```

국문판:

```text
claudedocs/p7_branch_b_sim_lab_dataset_schema_note_20260520_ko.md
```

다시 강조: 이 파일들은 실제 dataset 산출물이 아니라, dataset을 어떻게
저장할지에 대한 설계 문서다.

### 로컬 workspace에 존재하는 기존 dataset/demo 디렉터리

```text
collected_data/
collected_data_v2_backup/
collected_data_v5/
collected_data_v6/
lerobot_dataset_v3/
lerobot_dataset_v4/
lerobot_dataset_v5/
lerobot_dataset_v6/
lerobot_dataset_stacking_v1/
lerobot_dataset_stacking_v2/
lerobot_dataset_stacking_v3/
sim_demos_v1/
sim_demos_v2/
sim_demos_v3/
```

현재 LeRobot dataset 예시 구조:

```text
lerobot_dataset_v3/data/chunk-000/file-000.parquet
lerobot_dataset_v3/meta/info.json
lerobot_dataset_v3/meta/stats.json
lerobot_dataset_v3/meta/tasks.parquet
```

예전 procedural sim demo 예시 구조:

```text
sim_demos_v1/demo_0000_trajectory.csv
sim_demos_v1/demo_0000_anchors.csv
```

이 예전 CSV는 6개 joint column만 갖는다.

```text
base,shoulder,elbow,wrist_p,wrist_r,gripper
```

따라서 예전 sim demo format은 이번 normalized cube dataset 질문에 충분하지
않다. object pose, object size, TCP, object state, contact telemetry가 없기
때문이다.

### B200 workspace 확인 결과

B200 code root:

```text
/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code
```

확인된 dataset/log-like directory:

```text
./logs
./IsaacLab/scripts/demos
./lerobot/examples/dataset
./lerobot/examples/port_datasets
./lerobot/src/lerobot/datasets
./lerobot/tests/artifacts/datasets
./lerobot/tests/datasets
```

`logs` 아래에서 확인된 project log file:

```text
logs/phase1Balpha/train_p6v14c.out
logs/phase1Balpha/train_p6v14c.err
```

결론: 새 P7/Branch B normalized cube sim/lab dataset은 현재 B200의 dataset
directory 형태로 존재하지 않는다. 최신 v4/v5 cube 관련 B200 증거는 dataset
폴더가 아니라 `/tmp`의 로그와 생성 asset에 있다.

중요한 B200 runtime evidence 경로:

```text
/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/
/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_collision_usd/
/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_close26_hold_lift_b200.out
/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_close26_hold_lift_b200.out
```

## 제안하는 Cube Sim/Lab Dataset Schema

다음 dataset writer는 episode 단위 condition metadata와 frame 단위 runtime
data를 분리해서 저장하는 것이 좋다.

### Episode Metadata

episode 또는 generated trajectory당 1개 record:

```text
episode_id
seed
variant
object_size_m = [a, b, h]
object_pose_initial = [x, y, z, yaw]
normalized_grasp = [gx, gy, gz]
world_grasp
target_tcp_path = [approach_tcp, descend_tcp, close_tcp, hold_tcp, lift_tcp]
close_deg
asset_usd_path
asset_hash
success_verdict
failure_reason
```

메모:

- `object_pose_initial`은 샘플링된 condition이다.
- `object_size_m`은 현재 canonical cube에서는 `[0.020,0.020,0.020]`으로 고정이지만, future `a,b,h` 일반화를 위해 명시적으로 저장해야 한다.
- `normalized_grasp`는 object frame 기준, object size로 normalize된 grasp 위치다.
- `world_grasp`와 `target_tcp_path`는 재현성과 사후 감사용 derived field로 저장해야 한다.

### Frame Data

sim/lab step당 1개 row:

```text
episode_id
frame_index
phase
sim_time_s
joint_pos_rad[6]
joint_vel_rad_s[6]
action_joint_target_rad[6]
actual_tcp_pos_m[3]
target_tcp_pos_m[3]
tcp_target_error_m
object_pos_m[3]
object_quat_wxyz[4]
object_linvel_m_s[3]
object_angvel_rad_s[3]
tcp_to_object_m[3]
object_to_target_m[3]
gripper_q_rad
grasped_marker
posewrite_calls_total
attach_calls_total
```

현재 cube grasp 실패 원인 분석을 위해 추가하면 좋은 diagnostic field:

```text
moving_jaw_center_obj_m[3]
counter_jaw_center_obj_m[3]
moving_overlap_obj_m[3]
counter_overlap_obj_m[3]
moving_gap_obj_m[3]
counter_gap_obj_m[3]
moving_contact
counter_contact
one_sided_push
object_drift_m
object_speed_mps
tilt_deg
upright_z
```

### Torque / Acceleration

현재 저장된다고 말하면 안 된다.

- torque/effort limit은 sim config에 존재하지만, measured torque는 현재 dataset schema에 저장되지 않는다.
- joint acceleration도 직접 저장되지 않는다. 필요하면 joint velocity finite difference로 유도하거나, 신뢰 가능한 simulator API를 정해서 명시적으로 기록해야 한다.

## 현재 상태 / 다음 단계

지금 바로 dataset generation으로 가면 안 된다.

현재 v4/v5 B200 physics log는 close/latch를 아직 실패한다. 대규모 dataset을
생성하기 전에 runtime jaw telemetry diagnostic으로 close 중 실제 jaw/object
기하가 static fixture 가정에서 어디서 벗어나는지 확인해야 한다.

canonical cube grasp primitive가 reached/hold/lift-follow gate를 통과하고,
hidden pose-write artifact가 없다는 것이 확인된 뒤에 dataset writer를
구현하는 것이 맞다.
