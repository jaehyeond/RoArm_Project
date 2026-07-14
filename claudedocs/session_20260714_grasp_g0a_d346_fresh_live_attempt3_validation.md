# D346 — 고정된 attempt3를 새 Isaac 프로세스에서 실제 충돌 형상으로 검증

날짜: 2026-07-14 KST  
상태: 완료 / `D346_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`  
이번 case의 신규 변수: `[attempt3_fresh_live_representation_validation]`
(측정 전용 1건; 신규 물리 변수 0건)

## 1. 무엇을 왜 확인하는가

D345에서 끝난 일은 USD 파일의 의미 비교기가 실행 중 메모리 주소 때문에 거짓
차이를 만들었다는 증명이다. D345는 파일 비교를 수리했지만, Isaac/PhysX가 D344
attempt3를 실제로 열었을 때 어떤 convex 충돌 형상을 사용하는지는 확인하지 않았다.

D346의 질문은 하나다.

> 이미 만들어져 고정된 D344 attempt3를 새 Isaac 프로세스가 읽을 때, 128개 활성
> 충돌 조각이 작성 형상에 충실하며 고정된 열린 조 목표 자세에서 원통과 실제로
> 떨어져 있는가?

사용자는 현재 턴에서 `D346진행해`라고 명시 승인했다. D345는 완결된 사례이며,
D344의 과거 FAIL 판정과 attempt3 파일은 수정하지 않는다.

## 2. 용어를 일상어로 풀어 쓰기

- **attempt3**: D339 자산에서 문제였던 13개 조각만 이미 계산된 고정점 형상으로
  바꾼 세 번째 충돌 자산 후보다. D346에서는 이 파일을 만들거나 고치지 않고 읽기만
  한다.
- **실제 충돌 형상(live representation)**: USD 파일에 적힌 점 목록 자체가 아니라,
  PhysX가 실행 중 충돌 판정에 쓰려고 조리해 돌려준 convex 형상이다.
- **callback**: PhysX에 “이 조각의 실제 충돌 형상을 돌려 달라”고 요청했을 때 결과가
  도착하는 호출이다. 128개 조각마다 두 경로에 독립 요청하므로 총 256개다.
- **instance / prototype**: 같은 조각을 장면에 배치된 개별 복사본 경로와 그 복사본의
  원본 설계 경로에서 각각 읽는 두 통로다. 두 결과가 같아야 cache나 경로에 따른
  우연을 배제할 수 있다.
- **Rerun**: 수치 판정기가 아니라 공간 증거를 사람이 다시 볼 수 있게 기록하는
  관찰 도구다. callback 배열·JSON·해시·Float64 거리가 과학 판정의 권위이고,
  Rerun 화면은 형상·좌표축·원통·표·사건 기록이 올바른 위치에 보이는지 확인한다.

## 3. 실행 전 고정한 입력

- Git HEAD: `b09b62e0ffad919b9bdc1bb6155de2f662f2ab5c`
- D345 verdict: `D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS`
- D345 summary SHA-256:
  `d7cd4d4b0cb4c5a010b8673b47cb010103c337642b6e9b33df1fe577ca73bba5`
- D345 evidence SHA-256:
  `68652b51c7a0667a63c5d4b1e812e43868af07a93b4b91838322f81ac4cb4379`
- D344 attempt3 9파일 inventory digest:
  `ea6965199ff1f195a6d19d9c55febfe44cc9838f12651570c80d5bb97fa6caf1`
- attempt3 physics USD SHA-256:
  `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`
- D344 전체 출력 19파일 inventory digest:
  `bd07acc09a8c89f4f8da78da7e9492546a8f2974864e3e42c9069a211195a3c8`

`attempt3/.asset_hash`는 D339에서 복사된 오래된 값이므로 권위로 쓰지 않는다. 실제
9개 파일의 경로·크기·SHA-256 목록을 봉인한다.

## 4. 변수와 파라미터 감사

| 구분 | D346 등록값 | 변경 여부 |
|---|---:|---:|
| 신규 측정 변수 | 새 프로세스 실제 형상 검증 1건 | 신규 1 |
| 신규 물리 변수 | 없음 | 0 |
| hull vertex limit | 64 | 0 |
| max convex hulls | 64 | 0 |
| voxel resolution | 1,000,000 | 0 |
| error percentage | 1.0 | 0 |
| min thickness | 0.0001 m (`float32 0x38d1b717`) | 0 |
| shrink wrap | true | 0 |
| 표면 허용 오차 | 0.0001 m = 0.1 mm | 0 |
| property 부피 상대 오차 | 0.05 = 5% | 0 |
| 목표 최소 이격 | +0.1 mm | 0 |
| raw-live 차이 | 0.5 mm 이하 | 0 |
| 고정점 같은-좌표 비교 | 1e-9 m 이하 | 0 |
| q5 | 1.5413 rad = 열린 조 | 0 |
| radial / tangent | 7 / 11 mm, tangent sign -1 | 0 |
| seed / IK | 33201 / HOME 시작 position-only IK | 0 |
| Isaac 실행 장치 | cuda:0, headless, livestream 0 | 0 |
| Kit 추가 인자 / custom experience / XR | 없음 / 기본 / 꺼짐 | 0 |

callback마다 cache를 비우고 세 cooking 보조 설정을 잠시 끈 뒤 원복하는 것은 분해
파라미터 조정이 아니다. 두 독립 요청이 과거 cache를 재사용하지 않도록 만드는 측정
절차이며, 원복 성공 자체가 필수 검사다.

## 5. 사전등록한 실행 순서

1. Isaac을 시작하기 전에 D345 PASS, D344 역사적 FAIL 유지, attempt3 9파일,
   D344/D345 전체 입력, 코드 해시, 환경 버전을 검사한다.
2. prepare와 validate가 서로 다른 PID와 nonce인 새 프로세스인지 검사한다.
3. attempt3 root USD를 로봇 spawn 입력으로 지정하고 reset 뒤 측정 구간 simulation
   counter 시작값을 기록한다.
4. 몸체별 USD 충돌 목록이 활성 64개 + 비활성 과거 collider 1개인지 확인한다.
5. 각 활성 조각에서 `prototype -> instance` 순서로 독립 요청한다. 256개 witness
   JSON을 판정보다 먼저 저장한다.
6. 요청마다 callback 1회, `RESULT_VALID`, convex 1개, 직렬화 오류 0, cache 해제와
   cooking 설정 원복을 검사한다.
7. D342 좌표 수리와 D343 정확한 float32 비트 계약으로 128개 조각을 다시 분류한다.
8. 128/128과 사전 조건이 모두 통과한 경우에만 고정된 `(7,11)mm`, q5=1.5413rad
   목표에서 raw/live 거리와 차이를 계산한다.
9. reset 이후 측정 구간 simulation counter가 시작값과 같은지 확인한다.
10. 입력 inventory가 실행 전후 같음을 확인한다.
11. decision PNG와 footer가 있는 RRD/RBL을 만들고, 정확한 경로·시간축·구성요소와
    2400x1400 headless 화면 생성을 검사한다.
12. 자동 결과는 `육안 확인 대기`로 끝낸다. 생성된 Rerun 화면과 decision PNG를
    실제로 연 뒤 별도 JSON/Markdown에 관찰을 기록하고 완료 요약을 만든다.

## 6. 실패할 수 있는 필수 게이트

- callback: 256요청 / 256 callback / 요청당 convex 1개
- 실제 조각: link5 64/64 + gripper_link 64/64
- 조각별: 소유 몸체, 활성 상태, GPU 호환성, 최소 두께 비트, instance/prototype 합의,
  작성 형상 대비 표면 오차 0.1mm 이하, property 부피 오차 5% 이하
- 고정 목표: raw와 live 모두 +0.1mm 이상, 차이 0.5mm 이하, D337 raw anchor와 일치
- Rerun: frame 6, body 좌표계 2, mesh 522, scalar 1,040, event 132,
  비시스템 entity 2,100, 시간축 정확히 4개
- 실제 화면: 두 몸체의 source/instance/prototype/candidate 여덟 패널, 원통, 좌표축,
  수치 표와 사건 표가 모두 보이고 필수 정보가 가려지지 않아야 한다.

이번 세션의 실패 가능 실험은 새 Isaac 프로세스에서 얻는 실제 128조각과 고정 목표
거리 검사다. 이 사례는 D337의 step-0 충격과 D339의 13개 실제 형상 실패에 반응해
진행하는 reactive 검증이다. 다만 RL 학습이나 물리 perturbation 평가는 아니다.
그 이유는 현재 인과 질문이 “접촉 뒤 반응”이 아니라 “접촉 전에 PhysX가 어떤 형상을
읽었는가”이기 때문이다. 여기서 물리를 진행하면 충돌 표현과 접촉 반응이 섞여 원인을
분리할 수 없으므로 controlled physics step을 0으로 고정한다. 이 명시적 예외 사유로
Session progress rule을 충족하며, 다음 물리 평가는 D346 전체 PASS 뒤 별도 사례다.

## 7. 즉시 중단 조건과 다음 경계

- 입력·해시·새 프로세스 조건 실패: Isaac 시작 전 중단
- callback 또는 실제 128조각 실패: live union과 목표 판정 금지
- 목표 이격·충실도 실패: 물리 전에 중단
- Rerun 기계 검사 또는 실제 화면 확인 실패: 수치 결과는 보존하되 다음 settle 자격 없음

어떤 결과에서도 D346의 `g0a_pass=false`다. settle, 10-trial, G0b, RL, ladder는
D346 범위 밖이며, D346 전체 PASS 뒤 별도 사용자 승인 사례에서만 진행한다.

## 8. 실제 실행 결과

최종 판정은 `D346_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`이다. 다만 이 말을
“attempt3 충돌 형상이 틀렸다”로 번역하면 안 된다. 정확한 뜻은 다음과 같다.

> 실제 충돌 형상을 요청하는 검사기가 첫 요청 전에 멈춰서, attempt3 형상은 좋고
> 나쁨을 아직 판정하지 못했다.

### 8.1 GPU 격리로 끝난 비과학 실행 1건

최초 validate는 관리형 실행 환경이 CUDA 장치를 숨겨
`RuntimeError: No CUDA GPUs are available`로 `SimulationContext` 생성 중 멈췄다.
환경 객체 반환/reset, callback, 목표 거리, Rerun, controlled physics는 모두 실행 전이었다.

- callback: `0`
- controlled physics: `0`
- 자산 쓰기 / 파라미터 변경: `0 / 0`
- 보존한 preflight SHA-256:
  `e0759345507dbf48df80a490cb4c9da05383658c4903a1a6a07beb940529515f`
- 보존한 exception SHA-256:
  `21ecca9860192601b2b813a2bee43c51552470f490f346a7676b7f6aa4e9b624`

이 시도는 과학 실행으로 세지 않았다. 별도 reactive amendment에서 원본 증거와 모든
입력 inventory를 다시 봉인하고, 물리·자산·목표·기준을 바꾸지 않은 유효 실행 1회만
허용했다. amendment는 RTX 4090(`16376 MiB`, driver `580.159.03`)을 확인했고 PASS했다.

### 8.2 유효 실행에서 처음 실패한 지점

GPU 밖 실행의 사전검사는 전부 통과했다. CUDA 장치는
`NVIDIA GeForce RTX 4090 Laptop GPU`, 장면/센서/단위(`metersPerUnit=1.0`), 원본
기준 형상, AppLauncher 설정, 입력 해시, 파라미터 동결도 모두 PASS였다.

그 뒤 재사용한 D340 함수가 다음 오류로 첫 callback 전에 중단됐다.

```text
ModuleNotFoundError: No module named 'omni.physxassetvalidator'
```

설치 누락은 아니었다. Isaac Sim 5.1 환경에는
`omni.physx.asset_validator` 확장 기능 v`107.3.26`과
`omni.physxassetvalidator` Python 모듈 파일이 모두 있다. 직접 원인은 실행 순서다.

- D340 함수: 모듈 import(1567행) -> 확장 기능 활성화(1570-1575행)
- D339 정상 선례: 확장 기능 활성화/확인(1725-1736행) -> 모듈 import(1738행)

새 headless 프로세스에서는 확장 기능이 기본 비활성이다. 따라서 D340은 모듈을 먼저
찾다가 예외가 나서 활성화 줄까지 도달하지 못했다. D340 capture와 D344는 각각 앞선
게이트에서 멈춰 이 validate 함수가 실제로 실행되지 않았기 때문에 D346에서 처음
드러났다.

### 8.3 필수 수치와 중단 경계

| 항목 | 등록값 | 실제값 | 해석 |
|---|---:|---:|---|
| callback witness | 256 | 0 | 첫 요청 전 import 중단 |
| 실제 충돌 조각 분류 | 128 | 0 | 형상 불합격이 아니라 미측정 |
| D337 대조 검사 | 조건부 실행 | 미실행 | 256/128 선행조건 실패 |
| 고정 목표 raw/live 거리 | 조건부 실행 | 미실행 / `null` | 충돌 여유를 주장할 수 없음 |
| simulation counter | 변화 0 | `0 -> 0` | 물리 진행 없음 |
| controlled physics | 0 | 0 | 계약 준수 |
| 자산 불변성 | PASS 필요 | PASS | D344 attempt3 변화 없음 |
| 파라미터 증가/변경/기준 완화 | `0/0/0` | `0/0/0` | 값 조정 없음 |

판정 PNG의 `tcp error 0.817895mm`, `jaw tangent error 2.148675deg`는 로봇 자세의
운동학적 정렬 오차다. 충돌 여유나 표면 간격이 아니며, 이 숫자로 target clear를
주장하지 않는다.

## 9. Rerun 기계 검사와 실제 화면 확인

Rerun 파일은 생성됐지만 완료 계약은 FAIL이다.

| 항목 | 기대 | 실제 |
|---|---:|---:|
| frame | 6 | 6 |
| body coordinate frame | 2 | 2 |
| mesh | 522 | 266 |
| Float64 scalar row | 1,040 | 1,040 |
| event row | 132 | 132 |
| non-system entity | 2,100 | 1,588 |

누락된 `256` mesh와 `512` entity는 callback에서 얻어야 했던 instance/prototype
형상에 정확히 대응한다. RRD footer, RBL footer, CLI v`0.34.1`, 네 시간축, 화면 생성은
정상이었지만 빈 live 형상을 placeholder scalar/event가 대신할 수 없으므로 전체
기계 계약은 실패했다.

- RRD: `4,554,359` bytes,
  SHA-256 `b9a4f2e7ca6568c7274c6fa9e7f34ffa5868fa2822f9343ea8e1fe3ffc788eec`
- RBL: `96,496` bytes,
  SHA-256 `d19df02e3b698d4629f10afd237b0fc5479afdb657dfa61b62ba1e438968301e`
- Rerun PNG: 실제 `4800x2800`, `9,413,786` bytes,
  SHA-256 `f454d2a3ef36bb088a8c83d859bb28ab87222ec1545a77feaa111d7237225b67`
- decision PNG: `1076x665`, `74,902` bytes,
  SHA-256 `3ac985c06a129f8a87175f0d42a2f08df597c2da17da4305722acd24abdd5b3a`

두 PNG를 `view_image detail=original`로 실제 열었다. 여덟 패널은 있었지만 두 몸체의
live instance/prototype 네 패널에는 callback 조각이 없고 원통과 marker만 보였다.
수치 표는 측정 불가 값, 사건 표는 `live=FAIL` 경고를 표시했으며 viewer 알림이 오른쪽
위 일부를 가렸다. 수동 완료 조건도 정직하게 FAIL로 기록했다.

등록 경로를 잘못 추측해 첫 finalize가 수동 JSON을 찾기 전에 중단된 표시 파일명 실수가
한 번 있었다. 그때 completion 파일은 생성되지 않았다. 등록 권위 파일은
`d346_manual_visual_inspection.json/.md`이고, 먼저 만든
`d346_rerun_manual_visual_inspection.json/.md`는 forward-only 보존하는 비권위
보조본이다. 두 번째 finalize는 기대한 종료코드 `2`와 최종 FAIL 요약을 남겼다.

## 10. 결론과 다음 경계

- 완료 요약 판정: `D346_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`
- completion contract: `false`
- `g0a_pass=false`
- settle, 10-trial, G0b, RL, ladder: 전부 미실행/금지 유지
- D344 과거 FAIL과 D345 comparator PASS: 그대로 유지

다음 권장 선택은 별도 승인 D347
`[physx_asset_validator_activation_order]`(측정 전용 1건)이다. D340/D346을 고치거나
재실행하지 않고 새 wrapper에서 정확한 확장 기능을 먼저 활성화하고 확인한 뒤 모듈과
interface를 불러온다. 그 다음에만 같은 256 callback -> 128조각 -> 조건부 목표 거리 ->
Rerun 계약을 새 출력 경로에서 한 번 실행한다.

금지하는 우회는 수동 `PYTHONPATH` 삽입, private `.so` 직접 import, 전체 PhysX bundle
또는 custom experience 활성화, `simulation_app.update()` 추가다. D347에서도 자산,
분해 설정, 목표 자세, 허용 기준, 물리 변수는 모두 변경 0으로 유지해야 한다.

권위 결과:

- `d346_completion_summary.json` SHA-256
  `98a0c126824a27e7651ea2fe352394eb8829a4bf1137532e180ed7ae5629bece`
- `d346_postrun_root_cause_audit.json` SHA-256
  `3e53ee4415446df8abe891bc3f541b89a671d36803ad49e0bb87b006e813f23a`

`d346_postrun_root_cause_audit.json`은 등록 수동 화면 기록을 만들기 전의 원인 진단
스냅샷이다. 활성화 순서 원인의 권위이지만 최종 수동 완료 상태의 권위는 아니다.
최종 수동 상태는 `d346_manual_visual_inspection.json`과
`d346_completion_summary.json`이 권위다. 완료 요약의
`next_case_requires_separate_approval=null`은 D346 PASS 때만 열리는 settle 자격 필드가
열리지 않았다는 뜻이다. 실패 원인에 반응해 권고한 D347은 postrun 진단에서 새로 정한
별도 승인 사례이므로 이 `null`과 충돌하지 않는다.
