# b601_asset — upstream provenance (pinned copy)

- Source repo: https://github.com/Seeed-Projects/reBot-Isaacsim
- Upstream commit: `cb824be157fdd5db7d6153b644b9b8ce85775bef`
  (2026-08-10 07:19:39 +0800, "Merge pull request #23 from johnnynunez/update/mjcf-from-menagerie")
- Copied subtree: `usd/reBot_B601_DM/` (9 files, verbatim) + repo `LICENSE` → `LICENSE_upstream`
- License: CERN-OHL-W-2.0 (hardware) / Apache-2.0 (software) per upstream repo
- Copied on: 2026-08-13 KST (59th session, case `g0c_d446` open)
- Provenance note: the USD was generated upstream by "URDF USD Converter v0.1.3"
  from `urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf` (root-layer doc string).
  Physics variant default = `physx` (sublayers `physics.usda`).

SHA-256 pins (authority; drift = fatal for any g0c run consuming these):

```
e8a217cb3cfe56129b25e00c8f2171e9ba0f5c6145651f4340dba5707999bdc9  payloads/base.usda
4ead3b7d29627101085634014893b680ad148c7e87b33325bc8a525ba836ace6  payloads/geometries.usd
b2209f637eec1831a59e95e862768313aaeae77ba460eabf2ec9ddb3143833d6  payloads/instances.usda
0f5b1bb484ce696b4dc987321bf267c8e5b484417ba53344df912f55fe537b05  payloads/materials.usda
ecff1ef3aa0e7daa7e8402bfecec7bd21fd7601b53351eee971e64a57ebaf134  payloads/Physics/mujoco.usda
131e9e667403adf10bfdd641ebbb66ab49af55befa0d598f6efeefcfec8af4a2  payloads/Physics/physics.usda
58038daab9219f0a7809a868fe3ce3f491f0387f9e41272ab6fa6f8211e4048f  payloads/Physics/physx.usda
497e66972d6f4bdfca9dc3592601d9843a125de8d654f646c97c38b0f298102b  payloads/robot.usda
6b9d39de1200732c581c91e895bee412844e101006fb0c3df54259d81ee28e84  reBot_B601_DM.usda
3c46ce472bd9cd9c419bae897f4ced4ae73691c41fdd8eafa8a4b673725664e8  LICENSE_upstream
```
