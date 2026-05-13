"""Boot Isaac Sim, import RoArm M3 URDF, print full prim tree to diagnose mesh prim type.

Purpose: find why `prim.GetTypeName() == "Mesh"` matches 0 prims in render_p6v12_trajectory_replay.py.
"""
from __future__ import annotations
import sys
from isaacsim import SimulationApp

URDF = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"

app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")

_, ic = omni.kit.commands.execute("URDFCreateImportConfig")
ic.set_fix_base(True)
ic.set_make_default_prim(True)
ic.set_distance_scale(1.0)
ic.set_create_physics_scene(True)
ok, model = omni.kit.commands.execute("URDFParseFile", urdf_path=URDF, import_config=ic)
if not ok:
    app.close(); sys.exit("URDF parse fail")
omni.kit.commands.execute(
    "URDFImportRobot", urdf_path=URDF, urdf_robot=model, import_config=ic, dest_path=""
)
for _ in range(10):
    app.update()

root = stage.GetPrimAtPath("/roarm_m3")
print(f"\n=== PRIM TREE under /roarm_m3 (valid={root.IsValid()}) ===", flush=True)
type_counts = {}
mesh_like = []
# Use AllPrimsPredicate to include inactive/over/instance proxy prims
predicate = Usd.PrimAllPrimsPredicate
for prim in Usd.PrimRange(root, predicate):
    tn = prim.GetTypeName() or "<None>"
    type_counts[tn] = type_counts.get(tn, 0) + 1
    p = str(prim.GetPath())
    is_mesh_geom = prim.IsA(UsdGeom.Mesh)
    if is_mesh_geom or "mesh" in tn.lower() or "visual" in p.lower():
        mesh_like.append((p, tn, is_mesh_geom))

# Also TraverseAll to catch payloads/references
print(f"\n=== TraverseAll FULL (Mesh + Xform under /roarm_m3) ===", flush=True)
all_meshes = []
all_materials = []
all_shaders = []
for prim in stage.TraverseAll():
    p = str(prim.GetPath())
    if not p.startswith("/roarm_m3"):
        continue
    if prim.IsA(UsdGeom.Mesh):
        all_meshes.append(p)
    tn = prim.GetTypeName() or ""
    if tn == "Material":
        all_materials.append(p)
    if tn == "Shader":
        all_shaders.append(p)
print(f"  Total Mesh prims (TraverseAll, IsA(UsdGeomMesh)): {len(all_meshes)}", flush=True)
for p in all_meshes[:20]:
    print(f"    {p}", flush=True)
print(f"  Materials: {all_materials}", flush=True)
print(f"  Shaders: {all_shaders}", flush=True)

# Inspect existing shaders attrs (diffuseColor etc)
from pxr import UsdShade, Sdf
for sp in all_shaders:
    sh_prim = stage.GetPrimAtPath(sp)
    sh = UsdShade.Shader(sh_prim)
    print(f"\n  Shader {sp}:", flush=True)
    for inp in sh.GetInputs():
        try:
            val = inp.Get()
        except Exception as e:
            val = f"<err:{e}>"
        print(f"    {inp.GetBaseName()} = {val}", flush=True)
    print(f"    impl_id = {sh.GetIdAttr().Get()}", flush=True)

# Probe one visual Xform for binding
print(f"\n=== /roarm_m3/link5/visuals descendants (predicate=All) ===", flush=True)
v5 = stage.GetPrimAtPath("/roarm_m3/link5/visuals")
if v5.IsValid():
    for prim in Usd.PrimRange(v5, Usd.PrimAllPrimsPredicate):
        print(f"  [{prim.GetTypeName() or '<None>':20s}] active={prim.IsActive()} defined={prim.IsDefined()} | {prim.GetPath()}", flush=True)

print(f"\n=== TYPE COUNTS ===", flush=True)
for tn, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {tn}: {cnt}", flush=True)

print(f"\n=== MESH-LIKE PRIMS (first 30) ===", flush=True)
for p, tn, is_g in mesh_like[:30]:
    print(f"  [{tn:20s}] is_Mesh={is_g} | {p}", flush=True)
print(f"\nTotal mesh-like: {len(mesh_like)}", flush=True)

# Also enum link5 explicitly (we saw STL_BINARY_ warning there)
link5 = stage.GetPrimAtPath("/roarm_m3/link5")
if link5.IsValid():
    print(f"\n=== /roarm_m3/link5 descendants ===", flush=True)
    for prim in Usd.PrimRange(link5):
        print(f"  [{prim.GetTypeName() or '<None>':20s}] {prim.GetPath()}", flush=True)

app.close()
