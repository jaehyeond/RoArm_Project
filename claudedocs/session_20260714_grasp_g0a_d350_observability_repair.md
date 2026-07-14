# D350 — attempt2 reactive observability repair

날짜: 2026-07-14 KST  
상태: attempt1 실제 실패 관찰 후 사전등록 / repair 실행 전  
이번 repair의 신규 변수: `[]`  
신규 물리 변수: `[]`

## 1. 왜 repair가 필요한가

D350 attempt1은 실제 `headless=False` Isaac Viewer를 띄우고 동결 D349 자세에서
물리 step 없이 고정-jaw를 측정했다. 과학 결과는 다음과 같이 성공했다.

- fixed-jaw semantic binding PASS
- geometry measurement PASS (`MEASURED`, `aligned_pass=null`)
- D349 raw/live 거리 네 값 exact 재현
- final joint/object Float32 bits exact
- simulation counter `0`, initial timeline pause intervention `1`, interactive
  Viewer hold intervention `0`, timeline time unchanged, state delta `0`
- source/asset hash와 외부 사용자 파일 불변

하지만 종합 자동 gate는 관찰성 때문에 FAIL했다. 원본 실패는 절대 덮어쓰지 않는다.

1. Viewer capture token은 6/6 성공했지만, 비동기 PNG sink 완료 직후 단 두 번 UI
   update하고 stat하여 세 파일을 아직 0-byte로 기록했다. 180초 Viewer hold와 process
   종료 뒤에는 여섯 파일 모두 정상 생성돼 있었다.
2. 130개 mesh를 `static=True`로 Rerun에 기록했으므로 `part_idx` timeline은 archive에
   생기지 않는다. 각 part는 이미 `part_000..063`의 고유 entity path로 식별되며,
   실제 archive는 footer/RBL/296 entities가 남아 있다. attempt1의 component
   contract는 checks가 비어 있었으므로 component 보존은 repair에서 original
   `297` recording chunks와 `12` embedded-blueprint chunks의 Arrow payload exact
   equality 및 새 archive의 `296/296` component contract로 독립 증명한다.
3. immutability의 `asset_write=false`는 성공 조건인데 `all(dict.values())`에 직접 넣어
   aggregate가 false가 됐다.

이는 결과를 본 뒤 과학 허용값을 바꾸는 일이 아니다. 실제 평가 실패에 반응해 Rerun
기록 구현과 비동기 PNG 검증 구현만 수리하는 reactive observability implementation
repair다.

## 2. 동결 입력과 출력

- base Git HEAD: `647dfe6ba8e13c781b39850bf7228010fd1683b4`
- attempt1 root:
  `claudedocs/runtime_logs/grasp_track/g0a_d350/`
- repair output:
  `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/`
- attempt1 harness SHA-256:
  `99a9b558754c9c4ebf83b265e4bcc70744e1981786066d1343c96cd046d4c538`
- RRD SHA-256:
  `3d0b978d86e7ccff0f02bdadb41ce0f9c09ba24eee05fb489e479f4d6f95ef52`
- RBL SHA-256:
  `7256c3a04655b7665ba423b940073d28fec06c9a3391d472492901f2a23f0576`
- measurement SHA-256:
  `4fe91e4cd37f5b0f064c7e9c91480881973ca51e651132af2c8feb57750e8446`
- binding SHA-256:
  `1ec1c309461357eeae89204fa55a498b64d2d216708ab6e6c7dfdd3d0b878c12`
- capture V1 SHA-256:
  `9e43b105cb5b12635a28a9fdfd2748a7d07de81c41a35120b6b8989ab257e9b6`
- Rerun validation V1 SHA-256:
  `e2ecbb9715189d18c289523265a9291c24f2df06c8b0a3db8ae0a88f353a3751`
- automated summary V1 SHA-256:
  `a79b26ebcfde9590788e11c59f61cfe27b6d47ececf5b48977f4af140fc49048`

Repair는 위 원 경로와 attempt1 top-level 파일을 수정·덮어쓰지 않는다. 단 등록된
attempt2 출력 경로에 원 RRD를 full merge source로 포함하고 RBL bytes를 exact
forward-copy하는 것은 허용한다.

## 3. 사전등록된 repair 계약

### Viewer post-close revalidation

- original V1의 expected path set과 capture token은 정확히 6/6이어야 한다.
- 즉시 실패 집합은 정확히
  `[tool_oblique_physx, whole_oblique_colored, tool_side]`여야 한다.
- process 종료 뒤 3회 독립 read에서 size/mtime_ns/SHA-256이 exact stable해야 한다.
- 여섯 파일 모두 PNG CRC/decode/load PASS, `1280x720 RGBA`, nonzero여야 한다.
- 여섯 SHA는 모두 달라야 하고, V1 즉시 성공 세 파일은 V1 hash/size/dimension과
  post-close 값이 exact해야 한다.
- PNG mtime은 capture V1/automated summary 작성보다 앞서야 한다.
- target/state/timeline/counter/setting/session-layer guard는 원 V1에서 모두 PASS여야 한다.

### Rerun original-contract repair와 revalidation

- attempt1 RRD/RBL bytes와 원 FAIL은 hash-exact로 보존한다.
- 원 계약 digest
  `a2b1ddb4a6fa55b30b7b277c7a1f37fa4b1d01995fa25b4e3385105442aa98de`와
  exact timelines
  `[blueprint, event_idx, log_time, measurement_idx, part_idx]` 다섯 개를 완화 없이
  그대로 유지한다.
- attempt1 full RRD를 새 forward-only RRD에 merge하고, 기존 130개 mesh metadata
  path에 대응 원 TextDocument payload를 `static=False`로 다시 기록한다.
- `part_idx` 매핑은 `0..63=link5`, `64..127=gripper_link`, `128=fixed-jaw`,
  `129=cylinder`로 exact하며 누락·중복·추가 path는 `0`이어야 한다.
- 원 recording chunk ID `297`개와 static Mesh3D `130`개를 전부 보존하고, 추가되는
  chunk는 기존 metadata path의 TextDocument `130`개뿐이어야 한다. Mesh3D 재구성,
  semantic binding/PCA/distance 함수 재호출은 금지한다.
- 새 RBL은 attempt1 RBL의 byte-exact forward copy다. 새 full RRD/RBL은 exact
  non-system entities `296/296`, required components exact, original five timelines,
  footer, embedded blueprint, headless screenshot을 모두 PASS해야 한다.

### Boolean aggregation repair

`asset_write=false` 자체를 성공 조건으로 검사한다. 다른 세 immutability field가 true이고
원 aggregate만 false였음을 증명해야 한다. 원 V1 값을 수정하지 않는다.

## 4. 금지 경계

- Isaac/SimulationApp launch `0`
- physics step/forward write/target write `0`
- geometry/raw-live distance 재측정 `0`
- threshold/asset/decomposition/material/actuator/physics 변경 `0`
- attempt1 artifact overwrite `0`
- settle/10-trial/G0b/RL/PPO/ladder `0`
- `g0a_pass=false`, commit/push `0`

## 5. 실행 순서

1. `prepare`: attempt1 전체 SHA manifest, repair 코드/state/hash/금지 범위 동결
2. `repair`: 기존 6 PNG post-close 검증 + immutable full RRD에 exact per-mesh
   metadata temporal binding을 추가한 새 full RRD/RBL 생성·원 5-timeline 계약 검증
3. 6 Viewer PNG와 새 Rerun PNG를 `view_image detail=original`로 육안 검사
4. manual JSON/MD를 read-only 선검증
5. `finalize`: attempt1 scientific PASS와 attempt2 observability PASS를 결합

이번 repair는 자체적으로 실패 가능하다. PNG 손상/불안정, path swap, RRD entity/component
누락, screenshot 실패 중 하나라도 있으면 D350은 observability FAIL_STOP으로 남고 재GUI
필요 여부를 다시 판단한다. RL/perturbation을 실행하지 않는 이유는 관찰 실패의 입력이
이미 생성된 정적 RRD/PNG이고, 물리 재실행은 과학 자세를 불필요하게 다시 실행하며
사용자 승인 범위를 넘기 때문이다.

## 6. 실행 결과

### 6.1 prepare / repair

`isaaclab` 환경에서 exact Python, Rerun SDK/CLI `0.34.1`, `numpy==1.26.0`,
`psutil==5.9.8`을 확인한 뒤 `prepare`가 통과했다. 실행 harness SHA-256은
`da614e7683a5541a14db1389694d6df785a67e3f77203580c28c2cb8f9473c36`였다.

`repair`는 Isaac 또는 SimulationApp을 띄우지 않고 다음을 수행했다.

- 여섯 PNG를 process close 뒤 세 번 독립 표본으로 읽어 size/mtime/SHA 안정성을
  확인했다. 전부 `1280x720 RGBA`, Pillow verify/load 및 production-path 음성 대조군을
  통과했다.
- 원 RRD recording payload `297/297`와 embedded blueprint `12/12`를 Arrow
  RecordBatch equality로 exact 보존했다.
- 원 static chunks `279`, Mesh3D chunks `130`, original five timelines
  `[blueprint,event_idx,log_time,measurement_idx,part_idx]`를 그대로 유지했다.
- 기존 130 metadata entity에 원 TextDocument payload를 exact하게 non-static으로
  기록해 `part_idx=0..129` witness만 추가했다. recording chunks는 `297->427`, 즉
  정확히 `+130`이었다.
- repaired archive의 non-system entities/components는 `296/296`, failed structural
  checks는 `0`이었다. RBL은 원본의 byte-exact forward copy다.

주요 산출물 SHA-256:

- repaired RRD: `47fed7d82334195d1e45e58c476ca5bb120df6692acab46f989efb5ee8502813`
- repaired RBL: `7256c3a04655b7665ba423b940073d28fec06c9a3391d472492901f2a23f0576`
- temporal witness RRD:
  `a43e8c379ace100ce0f143df4b6a492829278386266df54dd7e296003fcec43a`
- headless screenshot (`4800x2800 RGBA`):
  `e4138bb1f507a746186b0f20a339571ef60aab7ced06e3846c23010dd6bb1d77`

### 6.2 실제 육안 검사

여섯 Isaac Viewer PNG와 repaired Rerun screenshot을 모두
`view_image detail=original`로 열었다. whole-oblique 화면에서 조립된 전체 robot/tool/
cylinder, PhysX collider 표시와 colored family를 확인했고, tool top/side/oblique에서
jaw와 cylinder의 상대 배치를 확인했다. Rerun 화면에서는 두 body의 colored parts,
actual fixed-jaw component, cylinder, frame 및 centerline/radial/tangent/normal/gap
witness를 확인했다.

Rerun 하단 Float64 metrics panel은 존재하지만 선택된 headless `part_idx`에서 개별 숫자
trace가 읽기 좋지 않았다. 이를 screenshot 숫자 판독으로 보완하지 않았고, Float64 JSON을
계속 수치 권위로 유지했다. exact 64+64도 육안 count가 아니라 기계 계약으로 확인했다.

수동 검사 파일:

- `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_manual_visual_inspection.md`

### 6.3 finalize와 범위 감사

읽기 전용 사전 finalize 검사에서 attempt1 hashes, capture, Rerun, manual, state hashes가
모두 PASS했다. 이어 `--stage finalize`가 다음을 반환했다.

`D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_AND_VIEWER_SUPPORTED`

- `completion_pass=true`
- `aligned_pass=null`
- `g0a_pass=false`
- `settle_authorized=false`
- repair Isaac launches / physics steps / target writes = `0 / 0 / 0`
- settle / ten-trial / G0b / RL / PPO / ladder = 모두 미실행
- commit/push = `false`

사전 수동검사 전 repair summary의 `MANUAL_PENDING` 문자열은 immutable 단계 snapshot으로
보존했다. 최종 manual PASS 및 통합 판정의 권위는 completion summary다.

최종 파일:

- `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_report.md`
- completion summary SHA-256:
  `7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb`

## 7. 일상어 판정과 다음 경계

D350은 실제 고정 손가락 표면을 찾아 재고, 실제 Isaac Viewer와 64+64 collider를 사용자
관찰 가능한 형태로 보여 주는 데 성공했다. 그러나 손가락 축이 radial 방향과 가깝다는 것과
그 중심선이 원통 중심을 지난다는 것은 같은 말이 아니다. 실제 중심선은 cylinder center
대비 tangent `-24.055688mm`, height `-20.360856mm`였다. 새 허용값이 없으므로 이를
`ALIGNED_FAIL`이나 `ALIGNED_PASS`로 이름 붙이지 않고 `MEASURED`로 멈춘다.

따라서 다음은 settle이 아니라 actual connected jaw surface를 권위로 쓰는 별도 target/IK
geometry-repair 승인 경계다. 승인 전에는 새 case ID, 코드, 출력 경로, target/IK/path,
physics step을 만들거나 실행하지 않는다.
