# Session 2026-07-27 (2nd) — Option A(SDF) 확정 후 착수 전 정찰 + 브리핑 (read-only, 실험 0)

> Append-only session log. 이 세션은 실험을 실행하지 않았다 (Session progress rule
> 명시 정당화: 사용자 지시가 "실행하지 말고 현상파악 먼저 한 후 브리핑"이었고,
> D398 frozen FAIL_STOP 상태에서 승인된 case가 없음). 코드/USD/Isaac/PhysX/물리/
> 커밋/푸시 0. repo 수정은 상태 문서(START_HERE/LEDGER/본 파일)와 auto-memory뿐.
> Stop hook의 /half-clone 지시는 HARD RULE #11로 거부하고 파일 기반 종료 수행.

## 유저 결정

- **방향 A 선택**: SDF collider 표현 재평가 케이스로 간다 (BACKLOG
  `sdf_collider_representation_reeval`). D399 라벨수리(B)는 선택하지 않음.
- 단 "실행은 하지 말고 현상파악 → 브리핑 먼저" — 이 세션은 그 정찰 결과다.
- 착수는 여전히 preregistration 제시 + 별도 승인 후에만 가능 (미승인 상태 유지).

## 하우스키핑

- MEMORY 회전 1건 (HARD RULE #8): 최고(最古) bullet(7/12 D336 discriminator,
  1,446자)을 `MEMORY_archive_20260712.md` 최상단으로 verbatim 이동, Recent
  Sessions 4건으로 축소. 병합돼 있던 `## Topic Files` 헤딩 줄바꿈 복구.
- git: worktree clean, HEAD `4c88865 "현재 D398분석"` (부트 프롬프트의 3e2839b
  위에 사용자 커밋 1개 추가 확인). commit/push 미승인 유지.

## 정찰 방법

4개 병렬 read-only 에이전트 (Workflow `sdf-case-recon`, 총 597k tokens, 19분).
전문: scratchpad `recon_{d362,assets,nvidia,harness}.md` (세션 한정 임시 산출물).

## 결과 1 — D362 기준선의 실체 (재측정 대상)

- 프로토콜: OPEN baseline 200 + q5 `1.5413000583648682`→`0.0` 1회 전환 + 300 =
  500 step, dt=0.005s, seed 33201, PD 80/4/2.5Nm, 접근 동작 없음
  (`DECISIONS.md:20757-20759`, D362 세션 doc :52-62).
- 물체: 해석적 CylinderCfg r17/h90mm, 0.72kg·마찰 1.5/1.2 = 미실측 placeholder
  (`sim_scripts/cyl34_top_view_d332_...py:61-65,487-501`).
- collider: A64 = `g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd`.
  **오늘 재해싱 → d362_preregistration.json pin과 bit-identical**
  (`a4be58...e46fff`), URDF 해시 일치, `local_assets/` 변경 0건 → A/B 기반 무결.
- 전도 원자료 재검증: moving jaw onset closure 31/32 (2.869N, 접촉 z=top rim
  0.098mm 아래), link5 14-step 지연, jaw 높이차 28.7644mm(trace 재계산 일치),
  최종 XY 60.619mm / tilt 89.998° (`g0a_d362/d362_physics_trace.json`).
- 모멘트 비 1.37 = 수평력만의 준정적 진단값 (수직 성분 미포함).

## 결과 2 — SDF 실행 가능성 (NVIDIA 1차 소스, 설치 버전 일치)

설치: omni.physx **107.3.26** / Isaac Sim 5.1.0 / Isaac Lab 2.3.0 /
PhysX 5.6.1 [INFERENCE: 바이너리 문자열]. 근거 경로는 recon_nvidia.md PATHS 참조.

- `PhysxSDFMeshCollisionAPI` 속성 7개 전문 확보 (`schema.usda:1043-1141`,
  isaaclab env extscache): sdfResolution=256, subgrid 6, BitsPerPixel16,
  narrowBand 0.01, margin 0.01, remeshing off, triangleReduction 1.0.
- 엔진 쿠킹 하드캡 sdfResolution **1250** (`libomni.physx.cooking.plugin.so`
  "Limited the SDF resolution to 1250"; UI 1..1250) — schema 기본 256과 별개 범주.
- **SDF는 GPU-only 아님**: 107.3 Rigid Body Collider Compatibility 표에
  "SDF Mesh CPU" 열 존재; GPU-only 경고 목록(Particles/Deformable)에 SDF 없음.
  (doc: "Rigid Bodies" 107.3, rigid_bodies.html#rigid-body-collider-compatibility)
- 충돌쌍: SDF↔convex ✓, SDF↔SDF ✓, SDF↔plane ✓, **SDF↔Convex Core Geometry
  Cylinder ✓** → 계획 조합(SDF 그리퍼 + 해석 원통) 표상 지원 (Fig.5 직접 판독).
- 공식 레시피: CollisionAPI + `approximation="sdf"` 토큰 + PhysxSDFMeshCollisionAPI
  3종 세트 (doc: "Collision Setup" 107.3 collision.html 'Create an SDF Collider').
- 얇은 벽 가이드: "두께 < 격자 간격 약 2배면 부정확", "해상도 ~250이면 대부분
  충분" (doc: "Collider Simulation Stability Guide" 107.3 collision_guide.html).

## 결과 3 — 함정 4개 (preregistration 필수 반영)

1. **Articulation link 위 SDF = 공식 NOT STATED** — 107.3 Articulation
   Limitations와 PhysX 5.6.1 Articulations 문서 모두 지원/금지 문구 부재.
   그리퍼는 articulation link → 케이스 결정적 리스크, 첫 in-run 프로브
   (파서 경고/접촉 발생) 필수.
2. **Isaac Lab 배선 부재** — `SDFMeshPropertiesCfg`(schemas_cfg.py:576) 소비처는
   MeshConverter뿐, 스포너에 mesh_collision_props 없음, 해당 경로는
   approximation 토큰 미설정 [INFERENCE: schemas.py:990-993] → **pxr 직접 적용 +
   "sdf" 토큰 수동 설정** 필요 (D338/D373 패턴과 동일 방식).
3. **원통 표현 미감사** — D362 원통이 exact인지 convex 근사인지 미확인
   (D336 감사 :50-51 지적 잔존). Isaac Lab 2.3.0의 carb 키
   `/physics/collisionCylinderCustomGeometry`는 107.3.26에서 deprecated 플래그
   (asset_validator) → no-op 가능성 [INFERENCE]; 유효 키
   `/physics/collisionApproximateCylinders`(기본 off=해석적). PhysX 5.6.1:
   구식 PxCustomGeometry는 SDF와 TriangleMesh 폴백 경고 → 107.3 원통이
   PxConvexCoreCylinder 경로인지 런타임 확인 필요.
4. **메시 품질** (trimesh 4.11.5 오프라인 실측, 물리 아님):
   gripper_link.stl = 수밀 단일 컴포넌트, 최박벽 1.487mm → 256 해상도 복셀
   0.304mm = **4.9복셀 충분**; link5.stl = **비수밀 114-컴포넌트 soup**, 벽
   1.0mm ≈ 2.1복셀(경계선) → link5가 최대 쿠킹 리스크 (512 해상도 또는 수리
   검토; 엔진 내장 non-watertight 경고 존재).
- 부가: contact capacity 33,280은 D362 exact inventory(1/1/1/64/64) 전용 →
  SDF inventory에서 재산정 필수 (SDF는 "접촉 다수 생성" 공식 문구).

## 결과 4 — Variable Ladder 충돌과 권고

- START_HERE 제안문(SDF + custom-geometry **29×50**)은 D362 대비 신규 변수 3개
  (그리퍼 표현/물체 치수/물체 충돌 표현) → 사다리(1-2개) 위반 + A/B 오염.
  29×50은 코드 상수 0건(문서만), 질량/마찰/COM 전부 미측정.
- **권고 = 2단계 분리**: ① 다음 case = 신규 변수 1개(A64→SDF), 물체는 역사적
  34×90/0.72kg 및 D362 동결 계약 전부 유지 → 깨끗한 A/B 전도 재측정.
  ② 이후 case = 29×50 rebase (실측 선행). ③ (선택) 사전 read-only 원통 표현
  감사로 변수 카운트 확정.

## 유저 결정 대기 3건 (preregistration 차단 요인)

1. 1차 물리의 물체: 34×90 A/B (권장) vs 29×50 직행.
2. case 번호: SDF를 D399로 vs D399 라벨수리 예약 유지 후 D400.
3. 사전 원통-표현 read-only 감사 포함 여부 (권장 yes).

## 승인 경계 (불변)

- 아무것도 실행/승인되지 않음. D389–D398 동결, D334 사이드카 불가침.
- 다음 단계 = 유저가 3건 답변 → preregistration(신규 변수/gate/산출물 경로
  `claudedocs/runtime_logs/grasp_track/g0a_dNNN/`) 문서 제시 → 별도 승인 → 실행.
