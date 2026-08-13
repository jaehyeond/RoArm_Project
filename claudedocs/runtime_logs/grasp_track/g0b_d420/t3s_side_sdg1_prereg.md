# `t3s_side_sdg1` preregistration — D29×H50 side-midpoint SDG proposals

- Date: 2026-08-11 KST
- Scope: **instrumentation-only candidate generation; no physics and no grasp verdict**
- User authority: sim-only D419 exception approved in chat: top-centre → cylinder
  side-midpoint for this case only. D29×H50 / 24.83 g stays fixed. Friction is
  not sampled, measured, or claimed here.
- 이번 case의 신규 변수: **grasp point = upright-cylinder side midpoint**.
  Object pose, mesh tessellation, sampler seed/configuration, and filters below
  are fixed controls, not sweep variables.

## 1. Decision question and non-claims

This stage asks only whether NVIDIA's installed Grasping SDG antipodal sampler
can emit a deterministic, frame-explicit set of side-midpoint *proposals* for
the next fixed-base RoArm PhysX stage.

It does **not** test or claim any of the following: RoArm IK, JOINT_LIMITS,
collision freedom, desk clearance, jaw contact, force closure, lift, physical
grasp success, material/friction realism, a real-robot result, or a training
sample. A sampler PASS authorizes only consumption by the separately gated
fixed-base physics harness.

The official version-matched reference is NVIDIA, **Replicator Grasping
Synthetic Data Generation**, Isaac Sim 5.1:
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/synthetic_data_generation/tutorial_replicator_grasping_sdg.html>.

## 2. Installed authority and source pins

| Item | Pin |
|---|---|
| Isaac Sim | `5.1.0.0` |
| Kit | `107.3` |
| Isaac Lab | `2.3.0` |
| `isaacsim.replicator.grasping` | `1.0.9` |
| NumPy | `1.26.0` |
| psutil | `5.9.8` |
| SciPy | `1.15.3` |
| trimesh | `4.5.1` |
| rtree | `1.3.0` |
| Rerun SDK/CLI | `0.34.1` |
| extension manifest | `.../isaacsim.replicator.grasping/config/extension.toml`, SHA256 `5e599aafec0d1c66776c70318535faeffc539e66070d64bf5ca15f6c5e21393a` |
| installed sampler | `.../isaacsim.replicator.grasping/isaacsim/replicator/grasping/sampler_utils.py`, SHA256 `613d3b41cbe0577d81bdd15a0b620a52c2516e54d80da11b6e45d1228eb1e925` |
| frozen jaw extractor | `sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py`, SHA256 `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3` |
| attempt3 root/base/physics/robot/sensor | `a4be58e87b1f9790` / `ea0ee8f258e93579` / `043a5d35aa425c25` / `2227536fcb8c9dae` / `3f44081f42b452bc` (full hashes are hard-pinned in p15) |

The runner must launch headless Kit only to register the installed extension
and call its `sample_antipodal` function. It must create no
`SimulationContext`, no physics scene, and execute zero physics steps. Runtime
extension version/path/hash must equal the pins above or the run aborts.

## 3. Physical authority versus sampler proxy

The next physics stage's sole object authority remains an analytic upright
cylinder:

- diameter `0.029 m`
- height `0.050 m`
- mass `0.02483 kg`
- centre in robot-base coordinates, exact existing `seed0_S4`:
  `[0.4235072423787768, 0.17237803311822986, 0.025] m`
- yaw `0°`, support plane `z=0`

SDG 1.0.9 samples a triangle mesh, so this stage constructs a candidate-only
closed proxy in a geometric-centre frame:

- radius `0.0145 m`, bottom/top `z=-0.025/+0.025 m`
- `256` radial segments
- `514` vertices (`2×256` ring vertices + two cap centres)
- `1024` triangles (`2×256` side + `256` bottom + `256` top)
- vertices use little-endian Float64 and faces little-endian Int64 for hashing
- canonical vertex SHA256 `6cffe59dfe701358dabbddc05d04a34016b674763b761b05c7795455b0512fcb`,
  face SHA256 `f40e9f9fe40a882c616930a6c6436ce4d07c949367e24a31ab58c05fd5ced23b`,
  combined SHA256 `871efea113d4fb4b55b33bcb87afd3b9173eed872fc39037b6a80971d9a4ae4f`
- exact extents must be D29×H50; closed/watertight, consistent winding, positive
  volume, finite coordinates, indices in range
- the regular polygon is an inscribed approximation. Its maximum radial chord
  sagitta is about `0.001092 mm`; it is **not** a replacement for the analytic
  PhysX shape and carries no mass/material/collider authority.

The proxy arrays and their canonical SHA256 values are written to
`t3s_side_sdg1_mesh_proxy.json`.

## 4. Exact SDG sampler configuration

```json
{
  "sampler_type": "antipodal",
  "num_candidates": 16384,
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

`num_candidates // num_orientations = 1024` mesh surface samples. The installed
sampler may return fewer than 16,384 transforms after its aperture/ray rejection;
the observed count is data, not a preregistered success number.

The sampler is called twice from the same mesh/config in one process. Array
shape, Float64 values, and ordering must be bit-identical. This reproducibility
gate is conditional on the pinned NumPy/trimesh/installed sampler stack; it is
not an engine-wide guarantee.

## 5. Frame contract — raw root must never be called RoArm TCP

Matrix notation: `T_A_B` maps coordinates expressed in frame B into frame A.
Quaternions are `[w,x,y,z]`, active rotations from candidate-local axes to the
named parent frame.

- `proxy`: cylinder geometric-centre frame, +Z up.
- `support_object`: bottom-centre frame. A proxy midpoint at local `z=0` is
  `z=H/2=0.025 m` in this frame.
- `base`: fixed RoArm base/world frame used by the next harness.
- SDG/candidate rotation is interpreted as a proposed **link5 orientation**:
  `+X = antipodal/jaw-closure line`, `+Y = joint axis and desired world-up`,
  `+Z = tool approach` (gripper toward object). This is the actual RoArm frame
  convention: q5 rotates about link5 local +Y and the primary jaw separation is
  along link5 local +X. The earlier +Y-closure draft was corrected before any run.

The installed sampler returns
`T_proxy_sdg_gripper = T_midpoint · R · T(-approach*standoff)`. Its origin is a
synthetic flying-gripper root created solely by the sampler config. For this
case both raw-root calibration fields are deliberately `null`:
`gripper_frame_prim`, `T_sdg_gripper_link5`. Therefore neither the raw
translation nor the recovered midpoint may be silently re-labelled a RoArm TCP
target. The separately geometry-derived `T_link5_tcp` below does not calibrate
that synthetic root.

Because lateral sigma is zero, recover the sampler's antipodal midpoint exactly
from each transform as
`p_mid = p_raw + R @ ([0,0,1] * 0.040)`. The sampler public return value does
not expose its original two surface-hit endpoints or axis length, so those
fields remain `null`; they must not be reconstructed and called raw SDG data.

The raw SDG root still has no RoArm calibration. Separately, p15 must derive a
midpoint-to-RoArm-TCP **position** calibration from the frozen attempt3 geometry;
it is not allowed to use a chat literal as authority:

1. pin/read
   `sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py`
   SHA256 `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3`;
2. open the exact attempt3 five-layer composition pinned by p14 and call the
   extractor's read-only `extract_asset()` plus 0.5 mm convex-hull surface
   sampler;
3. require 64+64 active convexHull parts, no hull fallback, one disabled legacy
   collider per jaw, and exact layer hashes;
4. in the fixed link5 cloud select the finite-cylinder slab
   `|z-0.115428| <= 0.00025 m` and `|y| <= 0.025 m`, then measure the inner +X
   boundary;
5. require it to agree within the declared 0.5 mm sampling tolerance with
   `x_inner=-10.02584956586361 mm`, then derive rather than assume
   `x_offset = x_inner + R = +4.47415043413639 mm`.

For a proposed link5 orientation `R_base_link5`, the geometry-derived positional
mapping is
`p_object_midpoint = p_TCP + R_base_link5@[x_offset,0,0]`, hence
`p_TCP = p_object_midpoint - R_base_link5@[x_offset,0,0]`. The result must expose
the sampled boundary, slab population, tolerance, source/layer hashes, derived
offset, and mapped TCP target. It does not calibrate the raw SDG flying root and
does not by itself prove IK/collision/contact.

Each selected row names these fields explicitly: `R_base_link5_proposal`,
`axes_base.jaw_closure_x`, `axes_base.vertical_up_y`,
`axes_base.tool_approach_z`, and `geometry_mapped_roarm_targets`. The latter
contains the derived `tcp_target_base_m`, the corresponding link5-origin target,
and `T_link5_tcp` with translation `[0,0,0.115428] m`. The mixed
`T_base_candidate_midpoint` uses the antipodal midpoint as its position and the
proposed link5 rotation as its axes; it is explicitly not a rigid-body prim
pose. p16 must reject the raw flying-root pose and must still establish
parsed-URDF-limit IK, collision freedom, contact, and lift independently.

## 6. Side-midpoint filter and deterministic ordering

For the fixed S4 object pose:

- desired approach `r_hat` is the horizontal unit vector from robot base to the
  object centre; +Z must point along `r_hat`. Pregrasp therefore lies toward the
  base, opposite +Z.
- desired local +X jaw-closure direction is horizontal tangent
  `t_hat=[-r_hat_y,r_hat_x,0]`. Its sign is physically symmetric for the sampler
  filter, but the raw signed orientation is preserved because the real jaw is
  asymmetric. Desired link5 local +Y is world up `[0,0,1]`.
- recovered midpoint height: `|z_proxy| ≤ 0.0025 m`
- recovered midpoint centreline offset: `sqrt(x²+y²) ≤ 0.00025 m`
- jaw-closure vertical error: `≤1°`
- jaw-closure tangential unsigned error: `≤20°`
- link5 +Y world-up error: `≤1°`
- approach vertical error: `≤1°`
- signed approach azimuth error relative to `r_hat`: `|error|≤12°`
- rotation orthonormal error `≤1e-10`, determinant positive within `1e-10`

Every raw row records each individual Boolean, rejection reasons, signed
tangential/radial/vertical errors, midpoint height/radial offsets, both frame
transforms, and raw index. Accepted rows are sorted by:

1. absolute midpoint height error,
2. absolute signed jaw tangential error,
3. absolute signed approach radial-azimuth error,
4. midpoint centreline offset,
5. raw sampler index.

Canonical output contains exactly the first **8** candidates. If fewer than 8
pass, the canonical run aborts; the observed pass count itself is not
preclaimed. Duplicate raw transforms are a fatal gate rather than silently
deduplicated.

For each selected row, the runner records:

- raw `T_proxy_sdg_gripper` and its base-frame origin (provenance only),
- recovered antipodal-midpoint candidate frame and `[w,x,y,z]` orientation,
- +X closure/+Y up/+Z approach axes in proxy and base,
- midpoint in proxy, support-object, and base frames,
- the near-side D419 lateral-surface midpoint obtained by intersecting the
  backwards approach ray with radius 14.5 mm,
- a reference pregrasp point 40 mm outside that side surface,
- all filter metrics and the raw row index,
- geometry-derived RoArm TCP target position, kept distinct from the raw SDG root,
- `q5_control = null/unassigned`; p16 owns open/close commands and p15 must not
  present those later controls as NVIDIA sampler output.

## 7. Output/G0/source-freeze contract

New forward-only prefix: `g0b_d420/t3s_side_sdg1_*`.

Expected outputs:

- `t3s_side_sdg1_config.json`
- `t3s_side_sdg1_mesh_proxy.json`
- `t3s_side_sdg1_raw_candidates.json`
- `t3s_side_sdg1_candidates.json` — canonical p16 handoff
- `t3s_side_sdg1_timeline.rrd`
- `t3s_side_sdg1_timeline.rbl`
- `t3s_side_sdg1_rerun_validation.json`
- `t3s_side_sdg1_inspection.png`
- `t3s_side_sdg1_script.py.txt`
- `t3s_side_sdg1_argv.txt`

If any expected output already exists, exit before Kit launch. The preregistration
hash, installed files, executed source, object constants, and environment pins
are checked before sampling. Executed source bytes are checked again after
sampling and then frozen to `script.py.txt`; changing the script during the run
is fatal. JSON uses finite values only and deterministic sorted-key encoding.
The canonical handoff records SHA256 for preregistration, frozen source, mesh,
and raw candidates.

PASS means only: pinned sampler ran deterministically, mesh/object/frame/filter
contracts passed, and exactly 8 frame-explicit proposals were emitted.
FAIL/ABORT means p16 must not consume this tag. Neither outcome is a physics or
grasp verdict.

## 8. D341 observability contract

Although JSON/Float64 bytes remain the numerical authority, this frame decision
is spatial and therefore also requires Rerun `0.34.1` completion before p16 may
consume it. The fixed RRD contains:

- an analytic-cylinder outline and the exact triangle proxy as separate entities,
- the fixed S4 base position and support plane,
- desired radial, tangential, world-up, and down axes,
- a deterministic raw-SDG frame subset with pass/fail colors,
- all eight accepted midpoint/side-surface frames and +X closure/+Y up/+Z approach axes,
- filter/rejection counts and the authoritative/non-authoritative distinction,
- scalar plots for midpoint height, tangential error, approach error, and pass.

Required completion: footer-enabled `rrd verify`, exact non-system entities and
timeline names, required components, fixed embedded blueprint, verified `.rbl`,
2400×1400 headless screenshot, and `rerun_validation.json pass=true`. Screenshot
generation alone is not visual inspection. A root agent must actually open the
PNG after the run and record what was seen before the overall D341 contract can
be called complete. The script records that human/agent review as pending.
