# 2026-08-10 (49th) — G0b 부트 + **48th 헤드라인 적대 재검증**: 인과는 살아남되 이유의 절반이 다시 쓰인다 — ★ 켜기만 해서는 원하는 값이 안 나온다(문턱 1.0 N · 마찰/관통깊이 미노출)

읽기 전용 세션. Isaac 0 · 로봇 0 · 물리 재실행 0 · 신규 스크립트 0 · 덮어쓰기 0 ·
git commit/push 0. **신규 패널 1**(`wf_7fd00a4d-284`, 14 에이전트 = 검증 7 + 반증 7,
에러 0, 1.99M tok, 626 tool calls, 1,793 s).

이번 case의 신규 변수: **0** (사다리 전진 0, 재검증·문서 정정만)

---

## §1 부트 (지시 순서 고정)

1. `AGENTS.md` → 2. `START_HERE.md` 48th판(**실측 383줄**) → 3. 48th doc(§4 → §3 → §5 → §6)
4. `DECISIONS.md` D436-R3(`:26567-26673`) + D436-R2/R1/D436/D435/D434-R1~D427 헤더
5. `EXPERIMENT_LEDGER.md` **좁은 grep만**(`grep -n '48th'` → `:509`), tail 미사용
6. `git status --short` + `git rev-parse HEAD origin/master` 7. 핀 재해시 sha256 앞 16자

### 1-1 핀 재해시 — **9/9 정확 일치**

`t3d_perstep_results.json` `7d9d05ba95eb8855`/12,802 · `t3d_perstep_curves.csv`
`cf0aef5b84c22fcd`/47,497 · `t3d_perstep_script.py.txt` `707f54372985e585`/45,180 ·
**`t3d_prereg.md` `71184e4670471090`/10,451** · `t3t_prereg.md` `76c9a80435cf574a`/15,307 ·
`t3t_grasp3_results.json` `f6ca15ae8dd9f662`/14,895 · `t3r_n10_ctq5_results.json`
`236243d4cfaa58ae`/1,437,511 · `t3d_panel_wffb2fba84_findings_raw.json` `d26650e6cbed4e4b`/242,253 ·
`t3d_panel_wffb2fba84_final_result.txt` `3ea03e8ecacffa95`/150,040.

⇒ **49th 부트 프롬프트의 핀 표는 D436-R3 ③ 정정이 이미 반영된 판본**이다(48th가 물려받았던
`t3t_prereg.md` 해시 오라벨 없음). 48th doc `:115`의 "부트 프롬프트도 이 오류를 물려받았다"는
**48th 당시 프롬프트 한정**으로 읽어야 한다.

### 1-2 git — ⚠️ **세 번째 판: 16항목**

`HEAD` == `origin/master` == **`332daab6e30269d0419273df85cc497ce3aec52e`** ✅.
실측 = 수정 **3** + 미추적 **13** = **16**.
- 프롬프트 "수정 4 + 미추적 11 = 15" → **분해가 틀림**(수정된 추적 파일은 3개뿐).
- `START_HERE.md:359` / D436-R3 (4) / LEDGER `:509` "수정 3 + 미추적 12 = 15" →
  **48th가 자기 세션 문서를 쓰기 전에 센 값**. 13번째 미추적 = `session_20260810_48th_...md` 자신.
- ⇒ 이 숫자는 13 → 15 → 16으로 **세 번째 판**이고 원인은 매번 **세는 시점**이다.
  교훈: **git dirty 개수는 상태 문서에 적을 값이 아니다**(자기 갱신이 자기 값을 무효화한다).

---

## §2 방법 — 적대 패널 `wf_7fd00a4d-284`

48th 헤드라인은 "43rd에서 접촉력이 0인 원인 = `roarm_stack_env.py:150`"이라는 **인과 주장**인데,
48th는 **그 파일이 43rd 실행 경로에 실제로 들어갔는지**를 확인하지 않았다. D428 #29/#30/#46이
세 번 규칙화한 실패 유형("파이프라인이 실제 소비하는 객체를 재라")이 정확히 이 지점에 걸린다.

7축 × (검증 → 반증) 파이프라인. 전 에이전트 읽기 전용 강제(파일 수정·git write·Isaac 실행 금지,
임시 파일은 scratchpad 한정).

| 축 | 대상 |
|---|---|
| V1 env-path | probe가 env를 인스턴스화하는가 / 43rd 물체는 어디서 spawn되는가 |
| V2 friction | 마찰 material 귀속 + "0.40/0.30" 출처 추적 |
| V3 isaaclab-doc | **NVIDIA 공식 출처 검증**(AGENTS.md 규칙) — `activate_contact_sensors` 시맨틱 |
| V4 numbers | `curves.csv` 396행에서 48th 수치 11군 독립 재도출 |
| V5 a4-schema | A-4가 스키마 결함인가 / FK 복원 가능한가 |
| V6 defects | 계측 결함 4종 줄번호 실재 + D433 전복 가능성 |
| V7 docsync | 상태 문서 4종 정합 |

**반증 단계에서 10건이 무너졌다** — 그중 4건은 검증 에이전트 자신의 과잉 주장이었고,
6건은 48th/기존 문서의 결함이었다.

영속화 3건(`g0b_d420/`):
- `t3d_panel_wf7fd00a4d_findings_raw.json` sha **`eb90e1444e7a7d71`** / 260,659 B
- `t3d_panel_wf7fd00a4d_refutations.txt` sha **`8c53bedd3d6d1c06`** / 76,810 B
- `t3d_panel_wf7fd00a4d_journal.jsonl` sha **`b9f60b465e27de5d`** / 223,804 B

---

## §3 재현 PASS — D436 계열 수치 100%, 그리고 **더 강해졌다**

`t3d_perstep_curves.csv`(`cf0aef5b84c22fcd`, 396행) stdlib csv 독립 재계산, **불일치 0건**:
비양수 98/116·111/131·114/149 · 평탄 밴드 [−0.035,−0.033] **69/72/72** ·
최소 −0.034773/−0.034743/−0.034743 **전부 step 428** · M-B 비양수 101/114/119 ·
leg3 step 400~480 mean −0.033270 / pstd **0.000859**(종합자 0.000838 여전히 재현 불가) ·
P-a M-A (391,392)/M-B (388,389) · P-b leg1 (489,490) · P-c leg3 (500,501)+(520,521) ·
lift tail 고정 +0.5385→+7.4668 / 이동 14/15 음수(gap 520) ·
`moving_argmin_part` **10종** 분포 개수 정확 일치 · 열별 첫 분기 387/393/501.

### 3-1 ★★ 신규 — leg 2와 leg 3은 **step 0~500이 비트 동일**하다 (49th가 직접 재확인)

상류 물리 로그 실측:

| 파일 | sha256(16) | 행 |
|---|---|---|
| `t3t_grasp2_steps.csv` | `58e87e2395279600` | 517 |
| `t3t_grasp3_steps.csv` | `20286054c90ac286` | 535 |

헤더 24열 동일, **공통 517스텝(0~500) 전 24열이 `sim_time_s`까지 비트 동일**, 상이 16행,
**첫 상이 = step 501**.

⇒ leg 2·3은 "우연히 일치하는 두 시행"이 아니라 **한 시행과 그 심화 연장**이다.
leg 1까지도 P-a 판정창에서 동일(391 **+0.001509** / 392 **−0.006272**, 3 leg 6자리 일치).
⇒ **"P-a는 n=1"은 D436-R2 (3)이 말한 것보다 강하다.** 독립 복제 수는 사실상 **1**이고,
"3 leg"가 받치는 것은 **C5(부품 쌍)·C6(접촉 유무)뿐**이다.
파생 CSV의 동일성은 계산 아티팩트가 아니라 **물리 로그에서 상속**된 것이다.

---

## §4 ★★ 48th 헤드라인 = **(b) 부분 성립** — 정정 4건

### 4-1 살아남는 것 (49th 직접 확인)

`p10_...probe.py:1214` `cfg = RoArmStackEnvCfg()` → `:1265`
`env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)` → `roarm_rl/__init__.py:19-25` 등록 →
`roarm_stack_env.py:393` `class RoArmStackEnv(DirectRLEnv)` → `:466-470` `_setup_scene`
`Articulation(self.cfg.robot)` / `RigidObject(self.cfg.sponge)`.
probe에 독립 `SimulationContext`·`InteractiveScene`·`Articulation(`·`DirectRLEnv` 서브클래싱
**0건**. 런타임 확증 `t3t_grasp_stdout.log:11` `usd_effective PASS`.

⇒ **`:150 activate_contact_sensors=False`는 43rd 실행에 확실히 들어갔다.** ✅
`ContactSensor`/`ContactSensorCfg`/`net_forces_w`/`force_matrix_w`/`contact_forces`
심볼은 probe·동결본 **0건**.

### 4-2 무너지는 것 4건

| # | 48th/D436-R3 기재 | 정합값 |
|---|---|---|
| ① | 물체 spawn = `roarm_stack_env.py:194 CuboidCfg`(플래그 부재) | ⛔ **43rd에서 실행되지 않은 코드**. 실제 = **`p10:1241`이 `cfg.sponge.spawn`을 `CylinderCfg`로 통째 교체**(`:1241-1261`). 런타임 확증 `t3t_grasp_stdout.log:8` `object cylinder D=29.0mm H=50.0mm`. ⇒ 결론(플래그 부재 → 상속 기본 False, `spawner_cfg.py:92`)은 대상만 바뀐 채 성립하므로 **판정 영향 0**, 인용만 정정 |
| ② | "플래그 False라서 export를 안 했다" | **필요조건일 뿐**. `ContactSensor` 0건이 **충분조건** — 플래그만 True로 해도 센서가 없으면 출력은 여전히 0 |
| ③ | "값이 **어디에도** 없었다 — 엔진이 export 자체를 안 함"(48th doc `:206`) | ⛔ **과장 ⇒ PARTIAL**. `ArticulationData.body_incoming_joint_wrench_b`(`articulation_data.py:706-719`, `:717` `get_link_incoming_joint_force()`)는 contact report API와 **무관하게 상시 제공**되고, 43rd 스크립트는 이미 같은 `base_env._robot.data`를 쓴다(`p10:1258,1289-1290,1305`). ⚠️ 단 이건 접촉력이 아니라 **접촉 하중을 포함하는 관절 반력 대리지표**이고, 접촉 성분 분리에는 **무하중 대조 실행**이 추가로 필요하다 |
| ④ | "켜면 법선력·마찰력·접촉점·관통깊이가 나온다"(48th doc `:248`) | ⛔ **부정확**. 43rd가 로드한 트리(Kit 로그 `kit_20260810_151852.log:193` = **pip 번들 isaaclab ext 0.47.2 / 2.3.0**)의 `ContactSensor`는 **순 법선력·필터쌍 법선력·접촉점 평균 위치·air/contact time**만 노출. **마찰력·관통깊이(separation)는 버린다**(`contact_sensor_data.py` 필드 11개, `grep -rn friction .../contact_sensor/` = **0 hit**) |

### 4-3 ★★★ 재실행 설계를 바꾸는 **신규 소스 결함 2건**

48th의 "플래그 + 센서 + 재실행" 견적은 **그대로는 실패할 수 있다.**

**결함 1 — 켜도 문턱이 0이 아니라 1.0 N이 된다.**
`isaaclab/sim/utils.py:290-291`이 bool을
`schemas.activate_contact_sensors(prim_path, threshold: float = 0.0)`의 **threshold 인자 자리**에
그대로 넘겨 `True → 1.0`으로 형변환된다. 정작 `schemas.py:528` 주석은
`# add contact report API with threshold of zero`, `:535`는 `# set threshold to zero`.
**43rd 물체 = 0.02483 kg ≈ 0.244 N**(`t3t_grasp3_results.json /object/mass_kg`)
⇒ **문턱 1.0 N은 물체 무게의 약 4배** ⇒ 그냥 켜면 **접촉이 통째로 보고되지 않을 수 있다.**
범위: **pip isaaclab 2.3.0(ext 0.47.2) 한정** — 로컬 클론 2.3.2(`sim/utils/prims.py:738`)는
인자를 넘기지 않아 이미 0.0. ⇒ 대책 2갈래(스폰 후 threshold 명시 0 재설정 **또는** 2.3.2 상향).
⚠️ 문턱이 텐서 API 경로(`get_net_contact_forces`)까지 게이팅하는지는 **소스로 확인 불가**.

**결함 2 — sleep threshold 0이 자식 강체가 아니라 루트 prim에 걸린다.**
`schemas.py:516` `all_prims = [prim]` · `:519` `child_prim = all_prims.pop(0)` ·
`:524` `if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):` · `:526`
`rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, **prim**.GetPrimPath())` — 루프 변수는
`child_prim`인데 `:526`만 바깥 `prim`을 쓰고 `Apply`가 아니라 `Get`을 쓴다.
**2.3.2 클론에도 동일** = 업스트림 미수정. 대조 구현: `isaacsim/core/prims/impl/rigid_prim.py:2289-2292`
(`# sleeping bodies don't get contact reports`).

### 4-4 "환경 파일 수정 주의" 경고는 **절반만 유효**

- **무효**: §4-6 표의 "`roarm_stack_env.py:150`을 True로" = **파일 직접 수정 불필요**.
  선례 `sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:331`
  `env_cfg.robot.spawn.activate_contact_sensors = True` (cfg 인스턴스에서 뒤집음, 산출물 실재
  `g0a_d330/`). 그 env는 `roarm_cube_push_env.py:69`가 `RoArmStackEnvCfg`를 상속 = **동일 필드**.
  ContactSensor 부착 선례도 3종(`...d362_...physx_contact_motion.py:1137,1178-1191`,
  `d333:89,121,132`, `d328:87-95`).
- **유효**: **재현성 기준선 선결은 유지된다.** 경고의 실체는 "어느 파일을 고치느냐"가 아니라
  **"플래그를 켜느냐"**다 — ON은 설정 위치와 무관하게 강체마다 `PhysxContactReportAPI` Apply +
  sleepThreshold 0 저작(`schemas.py:524-536`)을 일으켜 43rd와의 비트 비교를 깰 수 있다.
  ⚠️ 또한 ContactSensor 부착은 **env 서브클래스 + `_setup_scene` 오버라이드**를 요구한다(선례 3/3).

---

## §5 마찰 — 48th `[미확인]`은 해소가 아니라 **철회** 대상

`t3t_grasp3_results.json`(`f6ca15ae8dd9f662`) `/materials_runtime` = probe가 **런타임 USD에서
직접 읽어 기록한** material 전량(3개):

| prim 경로 | μ_s / μ_d | combine |
|---|---|---|
| `/physicsScene/defaultMaterial` | 1.0 / 1.0 | multiply |
| `/World/envs/env_0/Sponge/geometry/material` | **0.4000000059604645 / 0.30000001192092896** | average |
| `/World/ground/terrain/physicsMaterial` | 1.0 / 1.0 | multiply |

- 출처 사슬: `p10:824-825` argparse 기본값 `--static_friction 0.40` / `--dynamic_friction 0.30`
  → `:1256-1258` 원통 material 주입 → `:804-806` 런타임 되읽기.
  `/object/friction_provenance` = `"preregistered assumption (t3_mass_friction_contract.md), NOT measured"`.
- **로봇에는 physics material이 없다** — pxr 순회(Isaac 미실행, `isaacsim/extscache/omni.usd.libs.../pxr`
  경유)로 attempt3 `roarm_m3.usd` 249 prim 확인: `UsdPhysics.MaterialAPI` **0개** /
  `physics:*friction*` 속성 **0개** / `material:binding:physics` 관계 **0개**.
  (UsdShade 렌더 material 21건은 `PhysicsMaterialAPI` 미적용 = 물리 무관.)
  ⇒ 그리퍼 조는 `defaultMaterial` **1.0/1.0**을 상속한다.
- PhysX 쌍 결합 = 두 결합모드의 **최댓값**(eAVERAGE=0 < eMULTIPLY=2) ⇒ 유효
  **μ_s = 1.0 × 0.40 = 0.40** / **μ_d = 1.0 × 0.30 = 0.30**.
- 최초 출처는 코드가 아니라 **19th 세션 저작 `t3_mass_friction_contract.md:26`**(2026-08-05),
  실측이 아니라 목재-금속 건식 밴드(≈0.2~0.6) 중앙값 채택.
- 사전등록 **감도 leg(0.25/0.19 · 0.60/0.45)는 실행 이력 0건**(g0b_d420 JSON 57개 전수 walk).

⇒ **`START_HERE.md:383`의 마찰 행과 D436-R3 (11)은 정정이 아니라 철회 대상.**
답은 Must Read 8번에 **이미 핀돼 있던 동결 산출물** 안에 있었다.
⚠️ 48th의 **본질적** 지적은 불변: 계수는 **입력**이고, 실제 발생한 **마찰 임펄스는 여전히 미기록**.
⛔ `roarm_stack_env.py:206-207`의 1.5/1.2는 **p10이 교체하는 CuboidCfg 경로**라 이 실행에 무관 —
이 값을 "물체 마찰"로 인용하면 D428 #29 재발이다(49th 패널 내부에서 실제로 1회 발생, 반증됨).

---

## §6 A-4 — CONFIRMED, 그리고 우회로가 하나 더 막혔다

- `t3t_grasp3_steps.csv` 24열에 **팔 관절 q0~q4가 0개**(q5만) ⇒ **FK 복원 불가**.
  `quat_*`가 물체 자세임도 코드 확증(`p10:1336-1337` `object_quat()` = `_sponge.data.root_quat_w`,
  `:1403`에서 CSV로). `tilt_deg`/`upright_z`도 **물체** 지표(`p10:540-553`).
- **손목 롤은 TCP를 정확히 못 바꾼다** — `roarm_m3.urdf:216-220` joint4 axis=(0,0,1),
  TCP 오프셋이 link5 **+z**(`:235` `xyz="0 0 0.115428"`) ⇒ `Rz(q4)(0,0,L)=(0,0,L)`.
  **D436-R1 FATAL 전제를 독립 FK로 자리수까지 재현**(2-파라미터 족).
- ★ **신규 — close 구간의 실제 팔 관절각은 어디에도 기록되지 않는다**:
  `p10:1721` `q_close = current_seed_q.copy()`로 close는 descend **wp006의 IK 해**를 유지하는데,
  그 해는 seed(wp005)로부터 `seed_dev_deg=1.718` 떨어진 별개 해이고 로그에 없다
  ⇒ **"results.json 앵커 3점으로 복원" 우회로도 성립하지 않는다.**
- ⚠️ **"비용 = 코드 2줄"은 D436-R3 (9)에 없다** — `START_HERE.md:115`·`:196` 문구다.
  생산 측 최소 2줄은 정확하나(정의 `:1365-1379` / `csv_rows.append` `:1395` / 호출 `:1459,:1550` /
  헤더 `:1910-1916` / writerows `:1918`), **P-a 판정 가능**까지 가려면 소비 측
  `R_TOOL_MA`/`R_tool` 참조 **8개 지점**(`...readonly_audit.py:331,332,406,439,440-442,611,663`)
  + `p11_*` 파생 + D341 계약이 함께 필요 ⇒ **과소 견적**.

---

## §7 계측 결함 4종

- `:1496`·`:1539`(세그먼트 리셋 drift) · `:1781`·`:1784`(마지막 레코드 집계) ·
  `:1531-1536`(stall 본체) — **줄번호 verbatim 정확, 결함 실재**. 파일 sha `63c6b2127d969e32` 일치.
- ⛔ **`:915`는 오인용** — 실제 stall 임계는 **`:927-928`**
  (`--gripper_stall_rate_deg_per_step 0.02` / `--gripper_stall_min_steps 5`).
  `:915`는 `--settle_steps default=2`. 동결 leg1본에서는 `:913`/`:914`.
- ⛔ **"309배"는 leg 1 값**: leg1 `t3t_grasp_results.json`(`06ddff578b565e3a`)
  2.45757375578274e−06 vs `close_records[1]` 7.584976948429992e−04 = **308.6368**.
  **leg2 = 24.3054 · leg3 = 2.2789**. D434-R1 원문은 "leg 1"로 정확하나
  `START_HERE.md:265`가 leg 표기 없이 인용한다.
- stall 검출기 3 leg **발화 0회 경험적 확증**. 단 **"구조적 발화 불가"는 자유 슬루 체제 한정** —
  요구 비율 0.02/0.75 = 0.026667 vs leg3 close 실측 `|Δq5|/err_prev` **[0.067861, 0.180665]**
  median 0.180600 (n=98) ⇒ 상호 배타. 경성 차단 시엔 발화 가능.
- ★★ **신규 계통 오차**: `gripper_error_gate_deg=0.75°`(`:911`) + `settle_steps=2`(`:915`)
  탈출 규칙 때문에 조는 **모든** close 각도에서 지령보다 **0.50~0.62° 열린 채** 종료한다
  (leg3 **16/16**, leg1 **9/10**, 최대 **0.6176° @ close_19.00deg**;
  leg3 최종 실측 `q5_deg=17.5174` vs 지령 17.0).
  ⇒ **`close_records[*].angle_deg`는 지령값이지 실제 관절각이 아니다.**
  ⇒ (i) 잔여 ≤0.75°를 남기는 **부분** 차단은 `reached=True, gripper_stalled=False`로 보이지 않고
  (ii) **완전** 차단은 err≈1.0°>0.75라 오히려 stall 분기로 들어간다.
  ⚠️ `START_HERE.md:205`의 "19.0은 명령값 라벨" 주의와 같은 계열이나,
  편차가 말단이 아니라 **전 각도 계통**임이 새롭다.
- ✅ **4종 중 D433(`LIFT_FAIL` ×3)을 뒤집을 수 있는 것은 0건**:
  `aggregate.lift_follow_delta_m` = **−3.726594150066376e−04 / −3.559030592441559e−04 /
  −2.635996788740158e−04 m** vs `min_lift_follow_m` **+0.006**(`:914`) ⇒ 부족분 **6.26~6.37 mm**,
  3/3 `verdict=LIFT_FAIL`. leg3는 CSV 독립 재계산으로 확증(hold 종료 step 519 obj_z 0.025766 →
  마지막 lift step 534 obj_z 0.025502 = −0.000264).
  ⚠️ 범위 한정: 이 결론은 **기존 로그의 재채점**에 대해 성립한다. "고쳐 재실행해도 실패한다"는 뜻이 아니다.
- ★ 부수 해소: **leg2/leg3 생산 리비전** — `allow_closing_interference` 키가
  `t3t_grasp{2,3}_results.json`에는 있고 leg1에는 **없다**. 그 키를 방출하는 코드는 현 리비전
  (`63c6b2127d969e32`)에만 있고 동결 leg1본(`6861c35f94ed6427`)에는 없다(diff 44줄)
  ⇒ **동일 계열 리비전 확인, 비트 동일은 미확인**.
- ★ 부수 해소: **D434-R1 ⓔ "leg 3의 149 스텝쌍"** = **descend + close, label 분리**
  (34 + 115 = 149). 다만 **"0.0919°"는 전 phase·label 미분리(534쌍)에서만** 나온다 —
  동일 표본(149쌍)으로 통일하면 **0.0006°**가 되어 ⓔ 결론(구조적 도달 불가)이 **더 강해진다**.
  `err>0.75`인 스텝의 최소 `|Δq5|` = **0.1583°**는 부분집합 선택과 무관하게 불변.

---

## §8 문서 정합 — 사실은 6/6 정확, **본문 반영은 3/6**

D436-R3 정정 6건은 **사실로서 전부 정확**하다(동결 CSV·sha256·env 소스로 재현). 그러나
`START_HERE.md` **본문** 반영은 ①②④뿐이고 나머지는 파일 맨 끝 정정표(`:376-383`)에만 있다.

| # | 결함 | 실측 |
|---|---|---|
| 1 | ⛔ **`:75` "이동측 = `part_063`(정점 8개)"에 한정어 없음** | `## ⚡ 현재 진실`(`:65`~`:159`) **권위 절 안**. 정정은 `:381`에만. **인용 위험 최상** |
| 2 | ⛔ `/half-clone` 카운터 불일치 | `:317`=**54** / `:320`=56 / `:368`=56 |
| 3 | ⛔ LEDGER 46th 행은 **51**이지 52 아님 | `EXPERIMENT_LEDGER.md:506`=51 / `:507`=52 / `:508`=53. `START_HERE:319,369` + 46th doc `:330`이 승계한 오류 |
| 4 | ⛔ `START_HERE.md` 실측 **383줄** | 48th 기재 344는 자기 편집 전 상태 ⇒ 규약 ~120줄 대비 **+39줄 악화** |
| 5 | ⛔ `## Must Read First`(`:324-347`)가 **48th 문서도 D436-R1/R2/R3도 안 가리킴** | 1순위 = 46th 문서인데 **그 항목만 정정 병기 경고 없음**(`:326-327`), 그런데 지시하는 §5-4·§6-1이 정확히 정정 대상 두 절 |
| 6 | ⛔ DECISIONS 독서 순서 `:334-335`에 R1/R2/R3 부재 | D436 절 본문(`:26272-26380`)에 후속 리비전 **전방 포인터 0건**. `:26320`의 −0.1568은 append-only라 정상 잔존이나 **인라인 병기 없음** |
| 7 | ⚠️ Must Read 8번 10항목 중 **sha 미병기 3건** | `t3d_prereg.md`(`:336`) · `t3d_perstep_contactband.png`(`:337`) · `t3t_grasp{,2,3}_steps.csv`(`:339`). ③ 정정 핀은 `:378`에만 = **소비 지점 미설치**. D436-R3 (3)의 "`START_HERE.md:318`" 참조도 **`:341`로 갱신 필요** |
| 8 | ⚠️ D436-R3 (7) 정밀화(393) 본문 미반영 | `:257-258`은 여전히 "leg1은 **387**". 393은 `:382`에만. ⚠️ 단 `:257`의 387은 **오류가 아니다** — `q5_deg`·`clr_moving_*` 열 기준으로는 참. 결함의 성격은 **압축 과정에서 판정 열 근거 탈락** |

✅ LEDGER `:509`(48th 행) vs D436-R3 본문 — **불일치 0건**.

---

## §9 주장하지 않는 것 / 해소하지 못한 것

- **접촉력 여전히 0** — 49th도 물리를 안 돌렸다. 바뀐 것은 **"켜면 무엇이 나오는가"의 정밀화**뿐.
- **`True`로 켰을 때 유의미한 보고가 나오는지 여전히 미검증** — 오히려 §4-3 결함 1 때문에
  **위험이 커졌다**(문턱 1.0 N > 물체 무게 0.244 N).
- 문턱이 **텐서 API 경로까지 게이팅하는지 미확인** / `schemas.py:526` 오류의 **런타임 실효 미확인**.
- attempt3 USD 4개 crate 레이어에 `PhysxContactReportAPI` **선재 여부 미확인**(바이너리 미개봉).
  ⚠️ 다만 선재해도 `ContactSensor` 0건이라 **43rd 출력은 불변**.
- `body_incoming_joint_wrench_b`가 **이 자산·이 솔버에서 유의미한 크기를 내는지 미검증**
  (소스상 상시 가용은 확정, 값의 유용성은 실행 필요). 접촉 성분 분리에 **무하중 대조 실행 필요**.
- 46th §5-4 M-B 값의 **출처 미규명** / 깊이 33 µm 물리적 의미 / `part_029` vs `part_030` 미분해 /
  **측정 자세 변환 게이트 0개** / θ=0 대조 미실시 / θ=29 단일 · 스폰 `seed0_S1` 1점 /
  q5 20.50~27.70° 자산 오차 미측정 / **ρ_real 여전히 모델**.
- **D427·D429·D430·D431·D432·D433·D434·D434-R1·D435·D436·D436-R1·D436-R2·D436-R3 헤드라인
  전부 불변**(재실행 0, 재판정 0) · **`g0a_pass=false` 불변** · **"T1이 파지력 증명" 금지** 불변.
- **D324/D341 비해당** — 신규 산출물 = 패널 전사 3건뿐, 신규 과학 verdict 0, 공간·시간 판정 0.
  지위 = **부트 재검증 + 적대 감사 + 문서 무결성 보고**(D341 "순수 파일/해시/감사" 예외).
- ⚠️ **`AGENTS.md ## Session progress rule` — 물리 미실행 5세션째**(44th~49th).
  D436-R3 (3)의 자진 비판이 연장된다. ⚠️ **다만 이번엔 처음으로 "게이트를 열 준비"에 기여했다** —
  §4-3 결함 2건과 §4-4 선례는 재실행 설계를 실제로 바꾼다.
- ⚠️ **`/half-clone` 요구 1회 · 거부 1회**(context **114%**, stop-hook) ⇒ 카운터 **56 → 57**.

---

## §10 다음 — 순서 고정 (전부 사용자 결정 대기, Claude 단독 진행 금지)

1. 🔴 **접촉 위상 결정** — ② 옆면 파지(교수님) / ① D≤16 mm(사용자, 전복 악화) / ⓓ 깊이 재타깃.
   **49th가 이 순서 권고를 강화한다** — 재실행 준비 항목이 2개 늘었으므로(문턱 명시 0,
   마찰·separation 별도 API) 위상이 바뀌면 낭비가 더 커진다.
2. 🔵 **문서 정정 = D436-R4 append + `START_HERE.md` 수정** (§4-2 4건 + §4-3 2건 + §5 철회 +
   §7 3건 + §8 8건 + §3-1). ⚠️ `START_HERE.md`는 overwrite라 **승인 후 별도 turn**.
3. 🔵 **널 공간 포락 스윕**(D436-R2 (3)) — Isaac 0·물리 0·수 분 CPU. 신규 파생 태그 + 사전등록 + 승인.
4. 🔵 **ⓕ+ⓔ 통합 재실행 준비** — `p11_*` 저작 + 계측 결함 4종(`p10:1496,1539,1531-1536,1781,1784`
   + **`:927-928`**) + `l5_q{w,x,y,z}` 4컬럼 + **cfg-side `activate_contact_sensors=True`
   (env 파일 무수정, d330:331 선례)** + **threshold 명시 0 재설정** + `ContactSensor`
   (env 서브클래스 + `_setup_scene` 오버라이드) + **마찰은 `get_friction_data(dt)` 또는 2.3.2 상향** +
   측정 경로 독립 구현 대조 게이트 1건 + 사전등록. **저작까지만, 실행 직전 재승인.**
5. 🔵 θ=0 대조(별건 승인) / 🔵 40th·41st·42nd 적대 검증 부채(D430·D431·D432).
6. ⛔ **재개 금지 불변** — Gate-0 재실행 · 완전 수직 T2/T2b · 38th §11 리드 · 86.4 신규 가설 ·
   G0a/G0b 실물 테스트 · `t3r_*`/`t3t_*`/`t3d_*` 덮어쓰기.
7. **별건**: `MEMORY.md` 사용자 보류 유지 · `START_HERE.md` 압축(승인 후).

**상태**: DECISIONS append **D436-R4** / LEDGER 49th 행 append / `START_HERE.md` 49th판 갱신 /
`MEMORY.md` 미갱신(사용자 보류) / git commit **0** · push **0** / Isaac **0** / 로봇 **0** /
신규 패널 **1**(`wf_7fd00a4d-284`) / `/half-clone` 거부 **57회**. 프로포절 2026-08-20까지 **10일**.
