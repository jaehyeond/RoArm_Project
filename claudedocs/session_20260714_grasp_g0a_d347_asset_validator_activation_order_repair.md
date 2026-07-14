# D347 — PhysX 충돌 형상 검사기의 활성화 순서 수리

날짜: 2026-07-14 KST  
상태: 사전등록 / 실행 전  
이번 case의 신규 변수: `[physx_asset_validator_activation_order]`
(측정 준비 순서 1건; 신규 물리 변수 0건)

## 1. 무엇을 왜 확인하는가

D346은 D344 attempt3 충돌 자산의 128개 조각을 검사하려 했지만, 첫 callback을
요청하기 전에 `omni.physxassetvalidator` 모듈 import가 실패했다. 여기서 callback은
PhysX에 “실제로 사용할 convex 충돌 형상을 돌려 달라”고 요청했을 때 결과를 받는
함수 호출이다. D346의 실제 수치는 callback `0/256`, 분류 조각 `0/128`이다.

로컬 설치에는 검사 확장 기능 `omni.physx.asset.validator` v`107.3.26`과 그 확장
기능이 제공하는 Python 모듈 `omni.physxassetvalidator`가 모두 존재한다. 실패 원인은
설치 누락이 아니라 D340 재사용 함수가 다음의 잘못된 순서를 사용한 것이다.

1. 확장 기능이 아직 꺼진 상태에서 Python 모듈 import
2. 그 뒤에 확장 기능을 켜려고 함

새로운 최소 headless Isaac 프로세스에서는 1번에서 즉시 예외가 나므로 2번에 도달하지
못한다. D347의 질문은 다음 하나다.

> 정확한 검사 확장 기능을 먼저 켜고 켜졌는지 확인한 뒤 공개 Python 모듈과 API를
> 불러오면, 같은 attempt3 자산의 256 callback·128조각·고정 목표·Rerun 계약을 처음부터
> 끝까지 실행할 수 있는가?

사용자는 현재 턴에서 D347 진행을 명시 승인했다. D339, D340, D344, D346의 코드와
결과는 수정하거나 재실행하지 않는다.

## 2. 용어를 일상어로 풀어 쓰기

- **확장 기능(extension)**: Isaac Sim에서 필요한 기능을 필요할 때 켜는 플러그인
  묶음이다. 파일이 설치돼 있어도 확장 기능이 꺼져 있으면 그 안의 Python 모듈을
  불러오지 못할 수 있다.
- **모듈 import**: Python 코드에서 기능을 사용할 수 있도록 이름 공간을 불러오는
  단계다. 이번에는 반드시 확장 기능 활성화 확인 뒤에 수행한다.
- **공개 API**: 비공개 `.so` 파일을 직접 여는 우회가 아니라 확장 기능이 공식적으로
  내보낸 `get_physx_asset_validator_interface()` 함수와 인터페이스를 뜻한다.
- **convex 조각**: PhysX 충돌 계산에 사용하는 볼록한 작은 입체다. link5 64개와
  gripper_link 64개, 총 128개다.
- **prototype / instance**: 원본 설계 경로와 실제 장면 복사본 경로다. 각 조각을 두
  경로에서 독립 요청하므로 callback은 `128 x 2 = 256`개다.
- **Rerun**: 숫자를 결정하는 판정기가 아니라, 충돌 형상과 좌표축·원통·수치표를
  사람이 다시 볼 수 있게 저장하는 관찰 도구다. callback JSON과 Float64 거리값이
  과학 판정 권위이고 Rerun은 화면과 배치가 맞는지 확인한다.

## 3. 불변 입력과 승인 경계

- Base Git HEAD: `d9d224be7793c02754992401a06c3b5eb94826fa` (`D346`)
- D346 최종 판정:
  `D346_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`
- D346 completion SHA-256:
  `98a0c126824a27e7651ea2fe352394eb8829a4bf1137532e180ed7ae5629bece`
- D346 원인 감사 SHA-256:
  `3e53ee4415446df8abe891bc3f541b89a671d36803ad49e0bb87b006e813f23a`
- D346 harness SHA-256:
  `004fe0989f004ee554d8eea81ad8a28c2817a0b39af7b61dd43139c4488e8d4f`
- D346 로컬 출력: 26파일, 전체 inventory digest
  `dde081a24d4c3e49503819997db5f402484a1c4526ae82e54c15b0d046fa5aa7`
  (Git에서 무시되는 decision PNG와 Rerun 화면 PNG도 포함)
- D344 attempt3: 9파일 inventory digest
  `ea6965199ff1f195a6d19d9c55febfe44cc9838f12651570c80d5bb97fa6caf1`
- attempt3 root USD SHA-256:
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`
- attempt3 physics USD SHA-256:
  `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`

새 결과는
`claudedocs/runtime_logs/grasp_track/g0a_d347/`에만 생성한다. 이 폴더가 이미 있으면
prepare를 거부한다. D346 폴더와 attempt3 자산에는 쓰지 않는다.

## 4. 변수와 파라미터 동결

| 구분 | D347 고정값 | D346 대비 |
|---|---:|---:|
| 신규 측정 변수 | 활성화 순서 1건 | 신규 1 |
| 신규 물리 변수 | 없음 | 0 |
| hull vertex limit | 64 | 변경 0 |
| max convex hulls | 64 | 변경 0 |
| voxel resolution | 1,000,000 | 변경 0 |
| error percentage | 1.0 | 변경 0 |
| min thickness | 0.0001m / `0x38d1b717` | 변경 0 |
| shrink wrap | true | 변경 0 |
| 표면 오차 허용값 | 0.0001m = 0.1mm | 변경 0 |
| 부피 상대 오차 허용값 | 0.05 = 5% | 변경 0 |
| 목표 최소 이격 | raw/live 모두 +0.1mm 이상 | 변경 0 |
| raw-live 거리 차이 | 0.5mm 이하 | 변경 0 |
| q5 | 1.5413rad 열린 조 | 변경 0 |
| radial / tangent | 7 / 11mm, sign -1 | 변경 0 |
| seed / IK | 33201 / HOME 시작 position-only | 변경 0 |
| controlled physics | 0 step | 변경 0 |

코드는 D346 harness·parameter audit·preregistration의 해시를 먼저 확인한 뒤, D346과
D347에서 실제 import한 상수로 두 계약 지도를 다시 만든다. 분해·목표·장면·허용값·
callback·Rerun 개수를 필드별로 비교해 다음 목록을 자동 산출한다.

- 추가·제거·변경·증가·감소한 파라미터
- 분해·목표·장면 파라미터 변화
- callback 계약과 Rerun 개수 변화
- 허용값 완화

모든 변화 목록이 비어야 prepare가 통과한다. 따라서 “변경 없음”은 수동으로 적은
빈 배열이 아니라 실제 D346↔D347 구조 비교 결과다.

## 5. 활성화 순서 계약

callback 1번 전에 다음 순서를 별도 JSON으로 먼저 저장한다.

1. 로봇·물체 상태, simulation counter, `sys.path`, `PYTHONPATH`를 기록한다.
2. 검사 확장 기능의 초기 상태가 꺼짐이고 공개 모듈이 아직 `sys.modules`에 없는지
   확인한다.
3. D347이 명시적으로 요청하는 확장 ID는
   `omni.physx.asset.validator` 한 개뿐이다.
4. `set_extension_enabled_immediate(..., True)`를 정확히 한 번 호출한다.
5. 활성 상태, version `107.3.26`, 확장 경로를 확인한다.
6. 이 확인 뒤에만 `importlib.import_module("omni.physxassetvalidator")`를 호출한다.
7. 모듈 파일이 정확한 확장 경로 아래인지, 공개 getter와 GPU convex 검사 메서드가
   존재하는지 확인한다.
8. 활성화 전후 로봇·물체 상태와 simulation counter가 변하지 않았는지 확인한다.
9. JSON을 디스크에 확정한 뒤에만 callback 1번을 허용한다.

금지하는 우회는 수동 `PYTHONPATH`, 비공개 `.so` 직접 import, 전체 PhysX bundle,
custom experience, `simulation_app.update()`, fallback, retry, 확장 기능 disable/re-enable다.

## 6. 같은 과학 계약을 한 번 실행하는 순서

1. Isaac 시작 전 승인, HEAD, D344/D345/D346 해시·inventory, 환경 pin, 자동 파라미터
   비교, AppLauncher 인자를 검사한다.
2. prepare와 validate가 서로 다른 PID·nonce의 새 프로세스인지 확인한다.
3. immutable attempt3를 로봇 입력으로 열고 reset 뒤 측정 구간 counter를 기록한다.
4. 장면·센서·`metersPerUnit=1.0`·retained raw source를 확인한다.
5. 5절의 검사기 활성화 순서 계약을 실행하고 JSON을 저장한다.
6. 활성화 계약 PASS 뒤에만 각 조각을 `prototype -> instance` 순서로 요청한다.
   256 witness를 분류 전에 각각 저장한다.
7. 요청마다 callback 1회, `RESULT_VALID(0)`, convex 1개, 직렬화 오류 0, cache 해제와
   임시 설정 원복을 검사한다.
8. link5 64개와 gripper_link 64개, 총 128개를 D342/D343 계약으로 분류한다.
9. 128/128 PASS인 경우에만 D337 대조 검사와 고정 `(7,11)mm`, q5=1.5413rad
   raw/live 거리를 계산한다.
10. simulation counter 변화 0과 입력 inventory 불변을 확인한다.
11. footer가 있는 RRD/RBL과 decision PNG를 만들고 정확한 기계 계약을 검사한다.
12. 자동 결과는 육안 확인 대기로 끝낸다. Rerun 화면과 decision PNG를 실제 원본
    해상도로 연 뒤 별도 수동 기록을 만들고 finalize한다.

## 7. Rerun 완료 조건

- frame `6`
- 몸체 좌표계 `2`
- mesh `522`
- Float64 scalar `1,040`
- event `132`
- 비시스템 entity `2,100`
- 시간축 `blueprint`, `event_idx`, `log_time`, `part_idx`
- `rerun-sdk/CLI==0.34.1`
- footer 검증된 RRD/RBL, 고정 blueprint, headless 화면, 실제 육안 확인

사건 경로는 `events/d347`이므로 개수는 D346과 같아도 exact Rerun 계약 해시는
D347 경로로 새로 계산해 preregistration에 봉인한다. 활성화 JSON의 경로·SHA는 Rerun
recording metadata에 넣되 새 scalar/event entity를 추가하지 않아 기존 개수를 유지한다.

## 8. 실패 가능성과 물리 0회 사유

이 사례는 활성화 순서, 256 callback, 128개 조각, 고정 목표 거리, Rerun 계약 중
어느 하나라도 실패할 수 있는 실제 fresh Isaac 측정이다. D337의 step-0 충격과 D346의
callback 전 중단에 반응하는 reactive 검증이다.

현재 인과 질문은 접촉 뒤 동역학이 아니라 “검사기를 올바른 순서로 켠 뒤 PhysX가 어떤
충돌 형상을 읽는가”다. 물리를 진행하면 형상 표현과 접촉 반응이 섞이므로 controlled
physics를 0으로 고정한다. 이 이유로 이번 세션의 no-training/no-physics 예외를 명시한다.

## 9. 판정과 다음 승인 경계

- 활성화·모듈·API 순서 실패:
  `D347_ASSET_VALIDATOR_ACTIVATION_ORDER_FAIL_STOP`
- 장면·센서·원본 기준 사전조건 실패:
  `D347_G0A_RUNTIME_PREREQUISITE_CONTRACT_FAIL_STOP`
- callback 또는 실제 128조각 실패:
  `D347_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`
- 고정 목표 거리·충실도 실패:
  `D347_G0A_COOKED_TARGET_FIDELITY_FAIL_STOP`
- 과학 수치 PASS, 육안 확인 전:
  `D347_G0A_PREPHYSICS_COLLISION_REPRESENTATION_SUPPORTED_MANUAL_INSPECTION_PENDING`
- Rerun/육안 완료 실패:
  `D347_RERUN_OBSERVABILITY_INCOMPLETE_STOP`
- 모든 계약 PASS:
  `D347_G0A_PREPHYSICS_COLLISION_REPRESENTATION_SUPPORTED`

어떤 결과에서도 `g0a_pass=false`다. D347 전체 PASS는 별도 사용자 승인 fresh settle
사례의 자격만 만든다. settle, 10-trial, G0b, RL, ladder는 D347 범위 밖이다.

## 10. 등록 명령

`validate`는 D346에서 CUDA가 숨겨진 전례 때문에 처음부터 RTX 4090이 보이는 승인된
host 실행으로 호출한다. 일반 관리형 sandbox에서 “시험 실행”하지 않는다. preflight도
D347의 단일 validate 시도 일부이며, 실패 파일을 만든 뒤 재시도하는 경로는 등록하지
않았다.

```bash
conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d347_grasp_g0a_asset_validator_activation_order_repair.py \
  --stage prepare

OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d347_grasp_g0a_asset_validator_activation_order_repair.py \
  --stage validate --headless --livestream 0

conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d347_grasp_g0a_asset_validator_activation_order_repair.py \
  --stage finalize
```

실행 결과는 아직 없다. commit/push는 사용자 요청 전 금지다.

## 11. 실행 결과 — 2026-07-14 KST

위의 “실행 결과는 아직 없다”는 사전등록 시점의 문장이다. 등록 뒤 `prepare`, 승인된
host GPU의 `validate`, 원본 해상도 육안 확인, `finalize`를 순서대로 한 번 수행했다.
`validate` 재시도나 반응형 수정은 없었다.

최종 판정은
`D347_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`이다. 이 판정은 검사기 활성화
순서 수리 실패가 아니다. 활성화 순서는 통과했고, 그 다음 128개 실제 충돌 조각 중
한 조각의 두 부피 판독 채널이 등록된 5% 범위에서 합의하지 않아 정지했다.

## 12. 실제로 수행한 순서와 관측값

1. **자동 파라미터 비교**
   - D346의 해시로 고정된 계약과 D347 실제 상수를 구조적으로 비교했다.
   - 추가·제거·변경·증가·감소, 분해·목표·장면·callback·Rerun 변화, 허용값 완화는
     전부 빈 목록이었다.
   - 신규 변수는 `[physx_asset_validator_activation_order]` 한 건이고 신규 물리 변수는
     0건이다.
2. **검사 확장 기능 활성화**
   - 초기 상태는 exact extension 비활성, 공개 모듈 미등록이었다.
   - `omni.physx.asset_validator`를 정확히 한 번 켠 뒤 ID
     `omni.physx.asset_validator-107.3.26`, version `107.3.26`, 설치 root와 다섯
     파일 해시를 먼저 확인했다.
   - 그 뒤에만 `omni.physxassetvalidator`를 import하고 공개 interface와 GPU convex
     검사 메서드를 획득했다. 활성화 기록은 첫 callback 파일보다 6,000,203ns 먼저
     확정됐다.
   - retry, fallback, 수동 `PYTHONPATH`, private `.so`, 전체 bundle/custom
     experience, `simulation_app.update()`는 모두 0이다.
3. **callback 256회**
   - 128조각 각각을 prototype 뒤 instance 순서로 요청했다.
   - witness `256/256`, 요청 순번 `1..256`, callback 1회/inline, `RESULT_VALID(0)`,
     한 convex, 직렬화 오류 0, 설정 원복·cache 해제 `256/256`이다.
4. **실제 128조각 재분류**
   - gripper_link `64/64`, link5 `63/64`, 전체 `127/128`이다.
   - 표면 오차, 고정점/보존, typed float bits `0x38d1b717`, owner, GPU 호환성은
     각각 `128/128` 통과했다. 부피 교차검사만 `127/128`이다.
5. **유일한 실패를 분리**
   - link5 `part_045`의 prototype과 instance callback geometry는 비트 단위로 같고
     표면 오차는 `0m`다.
   - callback 삼각형 부피는 `5.171636397368745e-7m^3`, PhysX property-query
     부피는 `4.061547542733024e-7m^3`다.
   - property 값을 분모로 한 등록 상대차는
     `0.2733167205248915 = 27.331672%`로 5%를 넘었다. callback 삼각형을 독립
     재계산한 값은 `5.171636397368743e-7m^3`로 JSON과 수치 정밀도 안에서 같았다.
   - 따라서 callback 형상 계산이 틀렸다고 볼 근거도, property-query가 틀렸다고 볼
     근거도 아직 없다. 현재 증명된 것은 두 채널이 같은 부피를 보고하지 않는다는
     사실뿐이다.
6. **조건부 목표와 물리 정지**
   - 128/128 선행조건이 실패했으므로 D337 controls와 raw/live 목표 합집합 거리
     query를 실행하지 않았다. `decision_raw.queries=[]`,
     `decision_live.queries=[]`, body distance는 둘 다 `null`이다.
   - simulation counter `0->0`, controlled physics `0`; settle, 10-trial, G0b, RL,
     ladder는 전부 미실행이다.
7. **Rerun 기계·육안 완료**
   - 기계 계약: frame `6`, 몸체 좌표계 `2`, mesh `522`, Float64 scalar `1,040`,
     event `132`, 비시스템 entity `2,100`으로 모두 exact다. RRD/RBL footer와
     엔티티·컴포넌트·등록 시간축 이름 검사를 통과했다.
   - RRD는 `6,177,066`바이트 / SHA-256 `6d266688eafbb795f04184091f71cf9caeac9ca60b7f93e41701f8441bb73bf1`,
     RBL은 `96,422`바이트 / SHA-256
     `ad2b7f190bacfe51505cf27279ed76d22bb1fa71427d66a1b275b169e928421b`다.
   - `view_image detail=original`로 `4800x2800` Rerun 화면과 `1076x665` decision
     PNG를 직접 열었다. 여덟 패널의 형상, 목표 원통, 등록 frame, 수치·사건 표가
     보였으므로 수동 관찰성은 PASS다.
   - CLI가 사건의 `part_idx` timeline을 비정렬로 표시한 것은 보존할 관찰성
     주의사항이다. `event_idx`와 `log_time`은 정렬됐고 등록된 timeline 이름,
     footer, entity, component 계약은 통과했다. 과학 실패 원인은 아니다.
8. **최종화**
   - `finalize`는 등록된 과학 FAIL을 의미하는 종료 코드 2와 위 판정을 반환했다.
   - source immutability와 모든 finalize 입력 검사는 PASS했다. 전체 완료 계약은
     과학 수치가 127/128이므로 false다.

## 13. 결과 파일과 권위 순서

- 최종 요약:
  `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_completion_summary.json`
  (SHA-256 `93ae7a6daea4d8ba9af6fa09d01deb6c72017925375195a53804b0d55286d65e`)
- 활성화 순서:
  `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_asset_validator_activation_order.json`
- callback manifest:
  `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_validate_cook_witness_manifest.json`
- corrected 128-part authority:
  `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_fresh_live_representation_audit.json`
- 조건부 목표 정지:
  `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_zero_step_representation_gate.json`
- 자동 파라미터 비교:
  `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_parameter_freeze_audit.json`
- Rerun 기계/육안:
  `d347_rerun_validation.json`, `d347_manual_visual_inspection.json/.md`

`d347_raw_live_measurement.json`에는 재사용한 D340 함수의 사전-correction 필드가
포함된다. 이를 새 실패로 인용하지 않는다. corrected 128-part 판정 권위는
`d347_fresh_live_representation_audit.json`이다. 출력 폴더는 finalize 뒤 총 277파일이며
기존 D344/D346 입력은 변경되지 않았다.

## 14. 일상어 판정과 다음 승인 경계

이번 수리는 “검사 도구의 전원을 먼저 켜고 나서 도구를 불러오는 순서”를 제대로
고쳤다. 그래서 지난번의 0/256이 이번에는 256/256으로 회복됐다. 그러나 실제 조각을
끝까지 대조하자 link5의 한 조각에서, 눈에 보이는 표면은 완전히 같은데 두 PhysX
측정 경로가 부피를 27.33% 다르게 말했다. 어느 값을 써야 하는지 증명하지 않은 채
5% 기준을 늘리거나 검사를 삭제하면 안 된다.

권장 다음 후보는 별도 승인 D348 measurement-only
`[physx_property_query_volume_semantics]`다. immutable D339/D347 원문과 통과 조각
대조군을 이용해 callback hull 부피와 property-query 부피가 정말 같은 물리량이어야
하는지부터 확인한다. 자산·분해·목표·허용값·물리는 그대로 둔다. D348 전에는 목표
거리, settle, 10-trial, G0b/RL/ladder로 넘어가지 않는다. commit/push도 사용자 요청
전에는 하지 않는다.
