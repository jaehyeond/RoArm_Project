# Root actual inspection — synthetic whole-clump boundary

2026-09-13 15:20 KST. Root opened the actual local `decision_snapshot.png` with the image-view tool, then read `fixture.json`, `INSPECTION.md`, `MANIFEST.json`, and the relevant validation report fields. No physics, Isaac or new image generation was run by root.

Audit directory: `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/whole_clump_boundary_03/`.

## Observed

- The title visibly says `SYNTHETIC / NO PHYSICS`.
- Seven solid blue projected spheres and seven dashed red target spheres, their distinct owner centers, the black physical wall and the orange margin plane are visible. The dotted red owner-center limit includes the directional clump support.
- The actual blue center lies left of the orange line, but the outer blue sphere crosses into the shaded required-margin band. It remains left of the black physical wall.
- This is a false positive for **whole-sphere-plus-margin containment**, not a wall penetration, physical collision or failed real delivery.
- The printed numbers agree with fixture JSON: center is1.000mm inside the margin plane; directional support2.071254206989396mm; outer sphere exceeds the margin plane by1.071254206989396mm. Margin stays2.5mm; yaw37 degrees is a synthetic observation fixture, not a new physical case variable.

## Root-verified hashes

- decision_snapshot.png: `c622ece9e9e64cab3089117cc8660a967fdb0c2ac77a8574b79bf4d941c99a3e`
- fixture.json: `d33bc1594d244f043b1915e995aaa5d68aa64db6704c5d655e1b9304f1775c36`
- whole_clump_boundary.rrd: `8d2554201f90f39cec0db014491f5cd8bea45a2c76f12eaa71b41d447f58bee7`

## Acceptance boundary

CPU diagnostic and actual PNG inspection accepted as partial evidence. The image is a Matplotlib diagnostic, **not** a Rerun-rendered screenshot. Existing RRD footer/RBL checks are reported PASS, but `exact_entity_paths=null`, `exact_timeline_names=null`, empty required-component checks, and `headless_render.attempted=false` do not complete AGENTS D341. Root msg_e336425ea470 requests supplemental exact CPU validation and an exact bounded screenshot plan, preserving existing artifacts. No GPU HOLD or production NO-GO is lifted.
# 실제 Rerun 화면 추가 검수 — 2026-09-13 15:49 KST

root가 `audit/whole_clump_boundary_03/rerun_headless_inspection_01.png`를 실제로 열었다. Rerun viewer의 경계 평면들과 구 표면, 우측 하단 `SYNTHETIC / NO PHYSICS: center-only PASS, full oriented-sphere-plus-margin FAIL` 이벤트를 확인했다. 화면은 구 하나가 크게 겹쳐 보이고 라벨이 중첩되어 7개 구체 전수 확인이나 수치 판독용으로는 부족하다. 이 한계를 숨기지 않으며, 앞서 검수한 `decision_snapshot.png`의 전체 구체 투영과 canonical JSON/NPZ 산술을 함께 근거로 쓴다. 표시 개선을 위한 추가 실행은 요구하지 않는다.

`HEADLESS_RECEIPT_01.json`의 실제 argv는 제한적 GO와 일치한다. 실행1회·rc0·timeout false·0.8186779789975844초, 생산 물리/Isaac 실행이 아니다. 원래 RRD/RBL 및 새 PNG를 root가 SHA256으로 재대조했다: PNG `2b072344015322a528fe4cfdaf828edfb4934a454a0f07292a2d3ca8bdcafb9b`, RRD `8d2554201f90f39cec0db014491f5cd8bea45a2c76f12eaa71b41d447f58bee7`, RBL `87381ba99e0c0fa0c5994678e16a6014954442e8936f4f1d773bf8d7a9f26a35`. 감사의 정확 entity/component21/21·타임라인·footer/RBL 검증과 이번 실제 화면 확인을 합쳐 이 **합성 경계 진단**의 관찰 산출 요구를 인수한다. 생산 RRD/Isaac/full-cycle 검수나 기술 GO로 확대하지 않는다.
