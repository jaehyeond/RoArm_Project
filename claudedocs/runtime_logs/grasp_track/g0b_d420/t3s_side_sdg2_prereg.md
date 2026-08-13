# `t3s_side_sdg2` preregistration — reactive D29×H50 side-midpoint SDG proposals

- Date: 2026-08-11 KST
- Scope: **instrumentation-only candidate generation; no physics and no grasp verdict**
- User authority: sim-only D419 exception approved in chat: top-centre → upright
  cylinder side-midpoint for this case only. D29×H50 / 24.83 g stays fixed.
  Friction is not sampled, measured, or claimed here.
- 이번 reactive case의 신규 변수: **SDG surface samples `1024 → 4096` only**.
  Seed, mesh, object pose, orientation count, frame mapping, all safety filters,
  and canonical candidate count remain fixed.

## 0. Frozen predecessor failure and why this tag exists

`t3s_side_sdg1` is retired and must not be resumed or overwritten. Its first
real run used source SHA256
`8fefc670d483f740b956649791db9650770e973263207fec52eb184370471935`
and preregistration SHA256
`83b3af3af1d49d67f299cdd0dbfce46c5b547e66542ce90cf589879fb0cbad13`.
The preserved evidence is:

- `t3s_side_sdg1_failed_script.py.txt` — identical source bytes above;
- `t3s_side_sdg1_stdout.log` — SHA256
  `8752695c83c5810d4655b15f77d608a70669fda82281b73468c0d4c49ef9aef9`;
- `t3s_side_sdg1_failure.json` — SHA256
  `8a6d753e2ab8c962f7d627ecbe80b34e4ac5d4344a659c969f48a406210c49cc`;
- `t3s_side_sdg1_argv.txt` — SHA256
  `adb16cd1d3b00c1573d1cd3f6155017d5a93a7afff44e93bd62221f6e03d9663`;
- `t3s_side_sdg1_pid.txt`;
- `t3s_side_sdg1_prereg.md`.

The run performed zero physics steps and emitted no canonical candidate or
visualization artifacts. The preserved failure JSON records a read-only
diagnostic that intercepted the exception before Kit terminal close and
recovered the exact failure:
`SIDE_FILTER_TOO_FEW expected_at_least=8 actual=6`. Thus the failure was not a
frame/safety-gate failure; the fixed 1024 surface samples yielded only six rows
after the already-audited 1° vertical safety gates. This reactive retry changes
only sample count and uses a new forward-only tag.

## 1. Decision question and non-claims

This stage asks only whether NVIDIA's installed Grasping SDG antipodal sampler
can emit a deterministic, frame-explicit set of at least eight side-midpoint
*proposals* after increasing mesh surface samples from 1024 to 4096.

It does **not** test or claim RoArm IK, JOINT_LIMITS feasibility, robot/object or
desk collision freedom, jaw contact, force closure, lift, physical grasp
success, friction realism, a real-robot result, or a training sample. PASS only
authorizes a separately preregistered p16 fixed-base PhysX test.

Official version-matched reference: NVIDIA, **Replicator Grasping Synthetic
Data Generation**, Isaac Sim 5.1:
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/synthetic_data_generation/tutorial_replicator_grasping_sdg.html>.

## 2. Installed authority and immutable pins

| Item | Pin |
|---|---|
| Isaac Sim / Kit / Isaac Lab | `5.1.0.0` / `107.3` / `2.3.0` |
| `isaacsim.replicator.grasping` | `1.0.9` |
| NumPy / psutil | `1.26.0` / `5.9.8` |
| SciPy / trimesh / rtree | `1.15.3` / `4.5.1` / `1.3.0` |
| Rerun SDK/CLI | `0.34.1` |
| extension manifest | SHA256 `5e599aafec0d1c66776c70318535faeffc539e66070d64bf5ca15f6c5e21393a` |
| installed sampler source | SHA256 `613d3b41cbe0577d81bdd15a0b620a52c2516e54d80da11b6e45d1228eb1e925` |
| frozen jaw extractor | SHA256 `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3` |
| Rerun validator | SHA256 `aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e` |

The attempt3 root/base/physics/robot/sensor layer hashes remain, respectively,
`a4be58e87b1f9790...`, `ea0ee8f258e93579...`, `043a5d35aa425c25...`,
`2227536fcb8c9dae...`, and `3f44081f42b452bc...`; p15 hard-pins the full
hashes. Headless Kit is launched only to register the installed extension and
read the frozen USD composition. The runner creates no `SimulationContext`, no
physics scene, and advances zero physics steps.

## 3. Object authority and candidate-only mesh

The p16 physics authority remains an analytic upright cylinder:

- diameter `0.029 m`, height `0.050 m`, mass `0.02483 kg`;
- exact `seed0_S4` base-frame centre
  `[0.4235072423787768, 0.17237803311822986, 0.025] m`;
- yaw `0°`, support plane `z=0`.

The SDG sampler receives only the deterministic closed 256-segment proxy:

- radius `0.0145 m`, bottom/top `z=-0.025/+0.025 m`;
- 514 Float64 vertices and 1024 Int64 triangles;
- vertex SHA256 `6cffe59dfe701358dabbddc05d04a34016b674763b761b05c7795455b0512fcb`;
- face SHA256 `f40e9f9fe40a882c616930a6c6436ce4d07c949367e24a31ab58c05fd5ced23b`;
- combined SHA256 `871efea113d4fb4b55b33bcb87afd3b9173eed872fc39037b6a80971d9a4ae4f`.

Exact D29×H50 extents, finite indices, watertightness, consistent winding, and
positive volume are fatal gates. The proxy has no mass, material, collider, or
physics-shape authority.

## 4. Exact reactive sampler configuration

```json
{
  "sampler_type": "antipodal",
  "num_candidates": 65536,
  "num_orientations": 16,
  "gripper_maximum_aperture": 0.035,
  "gripper_standoff_fingertips": 0.040,
  "gripper_approach_direction": [0.0, 0.0, 1.0],
  "grasp_align_axis": [1.0, 0.0, 0.0],
  "orientation_sample_axis": [1.0, 0.0, 0.0],
  "lateral_sigma": 0.0,
  "random_seed": 42015,
  "verbose": false
}
```

`65536 // 16 = 4096` surface samples. Relative to the failed predecessor, only
`num_candidates` changes from 16384 to 65536; `num_orientations=16`, seed
`42015`, and every remaining key are byte-for-value unchanged. Observed raw and
filter-pass counts are data, not preregistered success numbers.

The sampler is called twice in one process with the identical mesh/config.
Shape, Float64 values, and ordering must be bit-identical. Duplicate raw
transforms are fatal.

## 5. Frame and geometry-mapped TCP contract

`T_A_B` maps B-frame coordinates into A. Quaternions are active `[w,x,y,z]`.
The proxy frame is the cylinder geometric centre; the support-object frame is
its bottom centre; the base frame is fixed RoArm base/world.

Candidate rotation is interpreted as proposed link5 orientation:

- local `+X`: antipodal jaw-closure line, desired horizontal tangent;
- local `+Y`: q5 joint axis, desired world up;
- local `+Z`: tool approach, desired horizontal base→object radial direction;
- right handed: `+X × +Y = +Z`.

The installed sampler returns
`T_proxy_sdg_gripper = T_midpoint · R · T(-approach*0.040)`. This is a synthetic
flying-gripper root, not a RoArm prim or TCP. `gripper_frame_prim` and
`T_sdg_gripper_link5` remain `null`. Since `lateral_sigma=0`, p15 recovers the
antipodal midpoint as `p_raw + R@[0,0,0.040]`; the sampler does not expose its
two original surface points or axis length, so those fields remain `null`.

Separately, p15 re-derives the asymmetric midpoint→TCP position mapping from
the pinned attempt3 64+64 convex-hull geometry:

1. require 64 active `convexHull` parts per jaw, no hull fallback, one disabled
   legacy collider per jaw, exact five-layer hashes, and 0.5 mm surface pitch;
2. in fixed link5 coordinates select
   `|z-0.115428|≤0.00025 m` and `|y|≤0.025 m`;
3. require the inner +X boundary within 0.5 mm of
   `-10.02584956586361 mm`;
4. derive `x_offset=x_inner+14.5 mm=+4.47415043413639 mm` rather than treating
   the expected number as the measured result;
5. map
   `p_TCP=p_antipodal_midpoint-R_base_link5@[x_offset,0,0]`.

Each selected row records `R_base_link5_proposal`, exact +X/+Y/+Z axes,
midpoint and side-surface points, the derived TCP/link5-origin target, source
raw index/hash, and null `q5_control`. This mapping is not IK, collision, or
contact evidence.

## 6. Safety filter, ordering, and canonical count

All failed-predecessor safety gates remain unchanged:

- midpoint: `|z_proxy|≤0.0025 m`, `sqrt(x²+y²)≤0.00025 m`;
- +X closure vertical error `≤1°`;
- +X closure tangential unsigned error `≤20°`;
- +Y world-up error `≤1°`;
- +Z approach vertical error `≤1°`;
- +Z signed radial-azimuth error `≤12°`;
- rotation orthonormal maximum error `≤1e-10` and determinant error `≤1e-10`.

Rows are ordered by absolute midpoint-height error, absolute signed closure
tangential error, absolute signed approach radial-azimuth error, midpoint
centreline offset, then raw sampler index. The canonical output contains exactly
the first **8** passing rows. Fewer than 8 remains a hard failure; this retry may
not relax the 1° gates or reduce the required count.

## 7. Forward-only outputs, early freeze, and failure evidence

Only new prefix `g0b_d420/t3s_side_sdg2_*` is writable. The runner accepts only
`--run_label side_sdg2`; `side_sdg1` is retired. Expected success artifacts:

- `t3s_side_sdg2_config.json`
- `t3s_side_sdg2_mesh_proxy.json`
- `t3s_side_sdg2_raw_candidates.json`
- `t3s_side_sdg2_candidates.json`
- `t3s_side_sdg2_timeline.rrd`
- `t3s_side_sdg2_timeline.rbl`
- `t3s_side_sdg2_rerun_validation.json`
- `t3s_side_sdg2_inspection.png`
- `t3s_side_sdg2_script.py.txt`
- `t3s_side_sdg2_argv.txt`

`t3s_side_sdg2_failure.json` is written only on failure after preflight and
before Kit terminal close. G0 checks every success/failure path and aborts
before Kit if any exists. Source and argv are frozen immediately after static
hash/package/mesh gates and before Kit launch, so a sampling failure still has
an exact executable source witness. The failure marker records stage, exception
type/message/traceback, source/prereg hashes, `physics_steps=0`, and hashes of
already-written tag artifacts; it is flushed before `simulation_app.close()`.

The source is re-read after sampling and again after Rerun emission. Any byte
change is fatal. No old tag is moved, renamed, deleted, or overwritten.

PASS means only: deterministic pinned sampling, exact mesh/frame/filter
contracts, eight canonical proposals, and technical D341 artifact gates. FAIL
means p16 must not consume this tag. Neither is a grasp verdict.

## 8. D341 observability contract

Float64 JSON/hashes remain numerical authority. Rerun `0.34.1` is an inspection
copy and must include separate analytic-cylinder/proxy entities, fixed S4 and
support plane, desired radial/tangent/up/down axes, deterministic raw pass/fail
frames, all eight accepted midpoint/side/TCP targets and +X/+Y/+Z axes,
rejection counts, and decision scalars.

Required technical completion: footer-enabled `rrd verify`, exact non-system
entities/timelines/components, embedded fixed blueprint, verified `.rbl`,
2400×1400 headless PNG, and `rerun_validation.json pass=true`. Recording/app
identities must derive from the active `t3s_side_sdg2` prefix, never retain the
retired `side_sdg1` identifier. Screenshot generation is not visual inspection;
a root agent must open the PNG and record observations before D341 is complete.
