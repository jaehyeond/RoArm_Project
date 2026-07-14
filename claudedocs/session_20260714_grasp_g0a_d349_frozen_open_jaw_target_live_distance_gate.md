# D349 — 동결 열린 턱 목표의 raw/live 거리 게이트

- 상태: **완료 — `D349_FROZEN_OPEN_JAW_TARGET_LIVE_DISTANCE_SUPPORTED`**
- 사용자 별도 승인: 2026-07-14, D349를 step-by-step으로 진행
- 이번 case의 신규 변수: `[frozen_open_jaw_target_live_distance_gate]`
- 신규 물리 변수: `[]`
- 출력: `claudedocs/runtime_logs/grasp_track/g0a_d349/`
- 현재 `g0a_pass=false`

## 1. 무엇을 왜 재는가

D348은 D347의 `127/128` 실패가 충돌 조각 결함이 아니라, callback이
제공한 면 연결을 버리고 꼭짓점으로 새 볼록껍질을 만든 비교 오류였음을
증명했다. callback의 `polygons + indices`를 그대로 쓰면 PhysX 속성 부피
비교는 `256/256`, 조각은 `128/128` PASS다.

실행 전에는 동결 목표 `(radial,tangent)=(7,11)mm`, q5 `1.5413rad` OPEN에서
raw mesh와 D347 callback-face topology로 재구성하고 D348이 인증한
active-collider surface proxy의 거리를 아직 재지 않았다. D349는 그 한 가지만
물리 step 전에 재는 실패 가능한 평가로 등록했다.

## 2. 동결 계약

- 자산: D344 attempt3 그대로
- 분해: body당 64개, 합계 128개 그대로
- 목표: `(7,11)mm`, tangent sign `-1`, q5 `1.5413rad`
- IK: seed `33201`, HOME-seeded position-only
- 허용값: raw/live 각각 `>= +0.1mm`, `abs(raw-live) <= 0.5mm`
- 진단 거리 참고값: D337의 기존 `0.05mm` (D349 PASS/STOP 판정권 없음)
- 재질·질량·구동기·물리 설정: 변경 `0`
- 자산 쓰기·cook callback·PhysX 속성 재조회: `0`
- controlled physics step: `0`
- settle·10-trial·G0b·RL·ladder: 금지

## 3. 거리 의미와 재구성 계약

판정 권위 live 형상은 D347 fresh PhysX callback의 꼭짓점과 D348이 검증한
polygon topology를 그대로 삼각화해 `BVHModelOBBRSS`에 넣은 body당 64개
표면의 합집합이다. 이 경로는 callback이 준 면을 그대로 거리 질의한다.

`hppfcl.Convex(points, topology_triangles)`는 생성 직후 `computeVolume()`이
D348 topology 부피와 같은지만 확인하는 **부피 결합 증거**다. Convex의 GJK
거리는 꼭짓점 support mapping(방향별 가장 바깥점을 찾는 계산)에 의존하므로
callback 면 거리의 권위값으로 쓰지 않는다. 꼭짓점만 다시 Qhull한 과거 외피도
동일하게 진단값일 뿐이다. 두 진단 경로는 값·상태·권위 BVH와의 차이를 기록하지만
D349 PASS/STOP을 거부하거나 통과시킬 수 없다.

이 값은 PhysX 내부 narrowphase의 직접 거리 API가 아니다. 정확한 표현은
**D347 live callback 면 위상으로 재구성한 active-collider 표면 proxy의
0-step 거리**다. D348의 `128/128` 부피·소유·활성 결합 증거와 함께 범위를
제한해 해석한다.

## 4. 순차적 실행 계획

1. 현재 HEAD, 환경 pin, D344/D337/D347/D348 해시, 저장공간 세션 dirty baseline을 봉인한다.
2. 새 Isaac process에서 D344 attempt3를 읽고 reset 직후 절대 joint를 저장한다.
3. HOME-near `±0.02rad`, q5 `0` CLOSED임을 기록한다. 정확 HOME라고 부르지 않는다.
4. stage/sensor/unit, active `64+64`, owner, prim→body identity, raw source를 검증한다.
5. D337 동결 controls를 통과해야만 목표로 간다.
6. 동결 OPEN 목표를 exact-write한다. 이는 `sim.forward` + `dt=0` update이며 물리 이동이 아니다.
7. raw mesh 거리를 먼저 저장한다.
8. 실패 화면의 exact witness를 위해 같은 raw 질의를 한 번 반복하고 첫 값과 exact 비교한다.
9. 상태를 바꾸지 않고 callback-topology BVH 합집 권위 거리를 저장한다.
10. Convex/Qhull 진단값을 비권위 채널에 저장한다. 실패·불일치도 판정을 바꾸지 않는다.
11. counter `0→0`, 질의 사이 pose bit-exact, 입력 불변, 거리 게이트를 판정한다.
12. RRD/RBL을 종료·검증하고 headless 화면과 decision PNG를 원본 해상도로 직접 검사한다.

## 5. PASS/STOP 경계

두 body 모두에서 다음이 참이어야 수치 PASS다.

- raw/live 값이 유한하고 질의 상태가 일관됨
- raw/live 모두 collision 없음
- raw/live 각각 `>= +0.1mm`
- `abs(raw-live) <= 0.5mm`
- raw 재질의 증거가 첫 raw 질의와 exact 일치
- raw와 callback-topology BVH 질의의 object/body pose stream이 exact 일치
- D337 raw anchor link5 `4.2726455336106985mm`, gripper
  `11.175088374613944mm` (`±0.05mm`)
- actual target joint/object 상태와 q5가 동결 명령을 따름
- global simulation counter 불변

어느 하나라도 실패하면 즉시 STOP이다. 수치가 PASS해도 Rerun 기계 계약과
실제 육안 검사 전에는 완료라고 하지 않는다.

## 6. Rerun 사전 계약

D347/D344의 검증된 8-panel collision contract를 기반으로 한다.

- frames `6`, coordinate frames `2`, meshes `522`, Float64 scalar rows `1040`
- measured-authority profile: point entities `4`, arrow entities `4`, event rows `136`,
  exact non-system entities `2112`
- timelines exact: `[blueprint,event_idx,log_time,part_idx]`
- raw/live x link5/gripper 네 witness endpoint/vector
- 네 개의 짧은 ASCII static summary 행에 q5 OPEN, `(7,11)mm`, 네 거리,
  두 delta, `0.1/0.5mm`, `0 steps`, `G0a=false`
- clear면 GJK nearest-point separation을, overlap이면 EPA 최대 침투 접점·법선·깊이를
  endpoint/vector로 표시해 실패 화면을 clear라고 잘못 부르지 않는다.
- Float32 Rerun 메시/표시값은 관찰용이며 판정·해시 권위가 아님

## 7. 외부 dirty baseline 보존

설계 검토 중 별도 저장공간 세션이 완료되어 다음 변경을 남겼다. D349는
이 네 파일을 수정·삭제·add·commit하지 않고 실행 전후 hash/bytes/status가
같은지 확인한다.

- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/ARCHIVE_PLAN.md`
- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_local_cleanup_receipt_20260714.json`
- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_predelete_manifest_20260714.tsv`
- `claudedocs/session_20260714_cube10cm_0_999_windows_archive_local_raw_cleanup.md`

## 8. 연구 세션 규칙 충족

D349 거리 게이트는 callback-topology 권위 결합, clear `0.1mm`, raw/live `0.5mm`,
동결 anchor, exact state, 0-step 계약 중 하나라도 실패하면 STOP하는 실패
가능한 perturbation evaluation이다. 다만 현재 case의 판정 대상이 사전 거리이므로
settle·10-trial·RL을 실행하지 않는다.

## 9. 완료 경계

D349가 최종 PASS해도 `g0a_pass=false`다. 열리는 것은 사용자가 다시 승인해야
하는 별도 settle case의 **자격**뿐이다. 자동 settle, G0b, RL, ladder 승격은 없다.
또한 PASS 범위는 D347 callback-face surface proxy의 0-step 거리이며 direct PhysX
narrowphase 거리나 물리 settle 결과로 확대 해석하지 않는다.

## 10. 실제 실행 순서

1. `AGENTS.md`의 Current-State Protocol에 따라 `START_HERE.md`,
   `DECISIONS.md`, `EXPERIMENT_LEDGER.md`, D348 session/completion과 D347
   provenance 문서를 읽고 `git status --short`를 저장했다.
2. 외부 저장공간 세션의 네 dirty-baseline 파일은 status, bytes, SHA-256을
   봉인했다. D349는 이 파일을 수정·삭제·add·commit하지 않았다.
3. D349 하네스와 사전등록 문서를 작성한 뒤 세 개의 독립 read-only 검토에서
   과학 권위, Rerun exact 계약, 실행/금지 경계를 확인했다. 세 검토 모두
   `NO_BLOCK`이었다.
4. `--stage prepare`에서 입력 해시, HEAD, 환경 pin, frozen parameter audit,
   D337/D348 계약, preregistration을 확인했다. `numpy==1.26.0`,
   `psutil==5.9.8`, `rerun-sdk==0.34.1`을 유지했다.
5. 새 Isaac/RTX4090 프로세스에서 유효 `--stage validate`를 정확히 한 번
   실행했다. preflight가 통과한 뒤 reset 상태를 기록하고, D337 controls와
   D348 corrected `128/128`을 확인했다.
6. 동결 OPEN 목표를 `sim.forward`와 zero-time update로 exact-write했다.
   raw mesh 거리, 같은 raw witness 반복, callback-topology BVH live 거리,
   비권위 Convex/Qhull 진단 순으로 측정했다.
7. 여덟 실행 phase 모두 global simulation counter가 `0`인지 확인했다.
   물리 step, asset write, cook callback, PhysX property query는 모두 0회였다.
8. RRD를 먼저 종료한 뒤 footer/entity/component/timeline/count를 검증하고,
   고정 blueprint와 RBL export 및 headless screenshot을 만들었다.
9. main Rerun screenshot과 decision PNG를 원본 해상도로 실제 열어 검사했다.
   main event viewport에 네 static summary가 보이지 않아 실패한 보조 표시
   attempt 두 개를 보존하고, main evidence hash에 묶인 비권위 text-only RRD로
   동일 네 문자열의 가독성만 추가 확인했다.
10. 수동 검사 JSON/Markdown을 작성한 뒤 `--stage finalize`를 실행했다.
    모든 completion input check가 통과했고 최종 판정을 봉인했다.

유효 validate는 exit code `0`이었다. Isaac 로그에
`[Error] Failed to clone in Fabric` 한 줄이 있었지만 preflight, runtime,
measurement, output, finalize의 등록된 모든 gate가 PASS했고 재실행하지 않았다.
이 로그 한 줄을 과학 실패나 재시도 근거로 확대 해석하지 않는다.

## 11. 시작 자세와 동결 목표 확인

Reset 직후 actual Float32 joint radians는 다음이었다.

```text
[0.01896364986896515,
 0.019351154565811157,
 1.5649892091751099,
-0.013456540182232857,
-0.014788953587412834,
 0.0]
```

이는 nominal HOME `[0,0,90,0,0,0]deg` 근방의 deterministic
`±0.02rad` jitter이고 q5 `0` CLOSED다. exact HOME가 아니다.

동결 목표 commanded/actual Float32 joint radians는 bit-exact하게 같았다.

```text
[0.03750238195061684,
 0.542945146560669,
 1.9687392711639404,
 0.18299327790737152,
 0.0,
 1.5413000583648682]
```

q5는 OPEN이고 object position은
`[0.30000001192092896,0.0,0.03288299962878227]m`, quaternion은
`[1,0,0,0]`으로 동결값과 exact였다. 실제 로봇이 HOME에서 목표까지 물리적으로
움직인 것이 아니라, 물리 step 전의 zero-time exact-write 상태다.

## 12. 정량 결과

| body | raw mesh (mm) | live topology proxy (mm) | absolute delta (mm) | 판정 |
|---|---:|---:|---:|---|
| link5 | 4.2726455336106985 | 4.272736580324082 | 0.00009104671338366899 | PASS |
| gripper_link | 11.175088374613944 | 11.340262326338637 | 0.16517395172469307 | PASS |

두 body 모두 raw/live 값이 finite이고 collision이 아니며 각각 `>=0.1mm`였다.
raw/live 차이는 `<=0.5mm`, live part 수는 body당 64, raw witness 반복은 exact였다.
D337 raw anchor `±0.05mm`, target state, authoritative pose streams,
D337 controls, D348 corrected audit, runtime binding도 모두 PASS했다.

Convex-support와 vertex-only Qhull 진단값은 기록했지만 판정권이 없다.
link5 진단 거리 두 값은 `4.272736580342085mm`, gripper_link는 각각
`11.340352212329284mm` / `11.34035221232928mm`였다. 이 일치는 권위 경로를
바꾸지 않는다.

## 13. Rerun 기계·육안 계약

- profile: `MEASURED_AUTHORITY`
- frames `6`, coordinate frames `2`, meshes `522`, points `4`, arrows `4`
- Float64 scalar rows `1,040`, events `136`, non-system entities `2,112`
- exact timelines: `[blueprint,event_idx,log_time,part_idx]`
- contract digest:
  `01a37bb21b626a6944fdf43de511e9891af7a0683d4e2587ceff87fe9f75513c`
- RRD SHA-256:
  `33ff02e7de2a979eb22b8cf07aac3fc45f564adfa2e604a79a012c015c8a7633`
- RBL SHA-256:
  `517c9be6e98803173d0305c313eb9eaa8cf8dea4c5a9d3647b6337ce844f72a9`
- main screenshot SHA-256:
  `4bda1947771a1ff39d8836adb426c9f4d0c74ad01069a541969ab9ccc516ff7f`
- decision PNG SHA-256:
  `f51ddfd39fd0400d9ff82b6e0627935b6eb2889dc9c1435f01fe62d551397944`

원본 inspection에서 link5/gripper 각각 source white, live blue, prototype
magenta, candidate green의 여덟 panel이 비어 있지 않았고, raw/live endpoint와
vector, 목표 원통, target/commanded/actual frame을 확인했다. main screenshot의
viewer notice는 우상단 일부를 가렸지만 판정 대상을 가리지 않았다.

main RRD 안의 네 exact TextLog는 CLI로 직접 확인했다.

```text
OPEN q5=1.5413 | target=(7,11)mm | steps=0
L5 raw/live/delta=4.2726/4.2727/0.0001mm
GR raw/live/delta=11.1751/11.3403/0.1652mm
gates=0.1/0.5mm | G0a=false | settle=separate
```

embedded viewport가 이 네 줄을 화면에 올리지 못해 `summary_inspection`과
`attempt2`가 각각 실패했다. 둘은 삭제·덮어쓰기 없이 보존했다. 마지막
`d349_summary_text_only.*`는 네 문자열과 main RRD/measurement source hash의
가독성만 검사하는 비권위 사본이며 main RRD, 원본 JSON, Float64 수치의 대체물이
아니다.

## 14. 최종 판정과 범위

최종 판정은
`D349_FROZEN_OPEN_JAW_TARGET_LIVE_DISTANCE_SUPPORTED`다. completion contract,
기계 Rerun contract, 실제 시각 검사 모두 PASS했다.

일상어로 말하면, 열린 그리퍼의 동결 목표에서 원래 로봇 mesh와 D348 방식으로
복원한 실제 callback 면은 둘 다 원통을 뚫지 않았고, 서로의 거리도 동결 허용값
안에서 일치했다. 그러나 live 값은 PhysX 내부 narrowphase API를 직접 부른 값이
아니라 **D347 callback-face surface proxy**다. 또 물리를 한 step도 진행하지
않았으므로 안정화, 접촉, 파지 성공은 전혀 증명하지 않았다.

- `g0a_pass=false`
- controlled physics steps `0`
- settle/10-trial/G0b/RL/ladder 미실행
- 별도 settle evaluation 자격만 `true`
- 다음 case는 새 사용자 승인과 새 forward-only path가 필요

## 15. 주요 증거와 해시

- measurement:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json`
  SHA-256 `5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300`
- home:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_home_start_contract.json`
- D348 corrected audit:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_d348_corrected_live_topology_audit.json`
- live runtime binding:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_live_topology_runtime_binding.json`
- Rerun validation:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_rerun_validation.json`
  SHA-256 `532c6d7064c3c618ed9c35b6013d55ddbd8b151bf29d03faa6411c4baaf5b6c1`
- manual inspection:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_manual_visual_inspection.json`
  SHA-256 `96b1e72b725154388ba11ff2d6093d9065aaa238bd041a196af1460f5268c456`
- completion:
  `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_completion_summary.json`
  SHA-256 `6ec883c4ebf4dd25aa2795006699b1d09e3b554412e2dcfa86277de541bd677e`
- harness:
  `sim_scripts/cyl34_top_view_d349_grasp_g0a_frozen_open_jaw_target_live_distance_gate.py`
  preregistered SHA-256
  `33a9743337fa269b71e4da3ccccfabc1d746ee29e1582a3d0f8c4764f42d68b9`

Prepare부터 finalize까지 preregistered critical hashes와 외부 dirty baseline은
exact였다. Finalize 뒤 이 session doc과 rolling `START_HERE.md`를 결과로 갱신한
것은 Current-State Protocol의 정상 종료 작업이며 과학 입력을 소급 변경하지 않는다.
commit/push는 하지 않았다.

## 16. 연구 세션 진행 규칙 판정

D349는 raw/live clear gate, raw/live agreement, D337 anchor, callback-topology
binding, exact target/pose stream, zero-counter 중 하나라도 실패하면 STOP하는
실패 가능한 perturbation evaluation을 실제로 실행했다. 따라서 같은 세션에서
settle, 10-trial 또는 RL을 추가하지 않은 이유는 검증이 약해서가 아니라 D349의
명시적 authorization boundary가 “물리 step 전 거리”에서 끝났기 때문이다.
