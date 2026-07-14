# D348 — PhysX 속성 조회 부피의 의미 판별

- 날짜: 2026-07-14 KST
- 상태: **완료 — `D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED`**
- Active Case: `g0a_d348`
- 이번 case의 신규 변수:
  `[physx_property_query_volume_semantics, rerun_static_summary_and_hidpi_contract]`
- 물리 변수 변경: **없음**
- 시작 기준: `d452921e04b7d5082c20d4edcfcc44bcefc7c34d` (`D347`)

## 0. Prepare attempt1 reactive 기록

첫 `--stage prepare`는 과학 자료를 판독하기 전에
`git_status_scope_only=false`로 중단됐다. 실제 작업 범위가 넓어진 것이 아니라,
`_git(...).strip()`이 Git porcelain 첫 줄의 선행 공백을 제거해 ` M file`을
`M file`로 바꾼 뒤 원래 상태 코드와 비교한 파서 오류였다.

- 과학 분석 실행: `0회`
- callback 판독: `0/256`
- Rerun/결정 PNG: 생성되지 않음
- attempt1 JSON 4개: `g0a_d348/` 루트에 hash 고정하여 보존
- 삭제/덮어쓰기: 금지
- reactive repair: 상태 코드 두 글자와 경로를 분리 판독
- attempt2 출력: `g0a_d348/attempt2/`

이는 신규 물리 변수나 두 번째 과학 시도가 아니다. 실패가 실제로 관측된 뒤
허용된 준비 제어계약 수리이며, 본 과학 분석은 attempt2 prepare 통과 후 한 번만
실행한다.

## 0.1 Attempt2 과학 PASS와 실제 Rerun 화면 FAIL

attempt2에서 과학 분석을 한 번 실행했고 다음 수치 게이트는 통과했다.

- callback topology ↔ property volume: `256/256 PASS`
- part gate: `128/128 PASS`
- raw instance/prototype pair: `128/128 exact`
- 닫힌·방향 일관 callback: `256/256`
- 최대 상대 오차: `1.362105296456897e-7`
- `link5/part_045` topology 상대 오차: `3.015486183560612e-8`
- 같은 조각 vertex-only Qhull 상대 오차: `0.27331672052498324`
- attempt2 Rerun 기계 계약: `2308 entity / 512 mesh / 1280 scalar / 132 event PASS`

그러나 실제 PNG 두 장을 원본 해상도로 연 뒤 수동 완료 게이트는 FAIL했다.

- Rerun 기본 timeline이 `part_idx`여서 `event_idx` event 패널이 비어 있음
- per-part dataframe도 현재 시점에서 `-`만 보여 5%, 256/256, 128/128을 읽을 수 없음
- 논리 창 2400×1400이 HiDPI에서 실제 4800×2800 raster로 저장됐지만,
  attempt2 checker가 물리 pixel을 2400×1400으로 잘못 고정함

실패 증거는 `attempt2/d348_manual_visual_inspection_fail.json/.md`에 보존한다.
과학 결과는 바꾸거나 다시 계산하지 않는다.

Reactive observability attempt3은 두 번째 신규 변수
`rerun_static_summary_and_hidpi_contract`를 사용한다. 이로써 case 전체 신규 변수는
2개이며 Variable Ladder 상한 이내다. 변경 범위는 다음뿐이다.

1. 기존 512 mesh와 1280 scalar를 불변 evidence에서 다시 기록
2. 모든 timeline에서 보이는 정적 metadata 요약 패널 추가
3. 정적 완료 event 1개 추가 (`133 events`, `2309 entities`)
4. 논리 창과 실제 raster의 균일 DPR을 분리하고 등록된 `1×/2×`만 허용

attempt3의 과학 재계산, PhysX/Isaac 실행, cook, asset write, target query,
physics step은 모두 `0회`다.

실제 화면 검사는 attempt3과 attempt4에서도 관찰 가능한 실패를 잡았다. 둘 다
수치 결과를 바꾸지 않고 forward-only로 보존했다.

- attempt3: 정적 요약은 생겼지만 긴 HOME 한글이 `\\u...`로 보이고 완료문이 잘림
- attempt4: UTF-8 저장과 짧은 문장은 성공했지만 Rerun 0.34.1 내장 글꼴에서 한글이
  누락 글리프 네모로 표시됨
- attempt5: Rerun 기계 화면은 짧은 ASCII 계약으로 고정하고, 같은 뜻의 한국어는
  이 문서와 사용자 브리핑에서 제공. 원본 화면 수동 검사까지 PASS

이 세 시도는 동일한 두 번째 변수인
`rerun_static_summary_and_hidpi_contract`의 반응적 화면 수리다. 신규 과학 변수나
물리 변수를 추가하지 않았다.

## 1. 무엇을 왜 확인하는가

D347은 새 프로세스에서 PhysX 검사 확장을 올바른 순서로 켠 뒤 256개 cook
콜백을 모두 받았다. 그러나 `link5/part_045` 한 조각에서만 다음 두 값이
달라 128/128 표현 게이트가 멈췄다.

- 콜백 꼭짓점으로 SciPy가 새로 만든 볼록껍질 부피:
  `5.171636397368745e-7 m^3`
- PhysX 속성 조회가 반환한 충돌체 부피:
  `4.061547542733024e-7 m^3`
- 기존 상대 차이: `27.33167205248915%` > 동결 허용값 `5%`

D347의 콜백은 꼭짓점뿐 아니라 **각 면이 어느 꼭짓점을 어떤 순서로 잇는지**도
저장했다. 기존 판독기는 이 면 정보를 버리고 꼭짓점만 다시 볼록껍질로 감쌌다.
D348은 PhysX 속성 조회의 `volume`이 어느 기하 표현과 대응하는지 판별한다.

이 case는 자산을 고치거나 물리를 다시 돌리는 case가 아니다. D347의 불변 원자료를
읽는 측정 case이며, 판독법이 틀렸는지 실제 충돌 표현이 틀렸는지를 분리한다.

## 2. 바꾸지 않는 것

- D344 attempt3 충돌 자산 및 모든 USD/STL
- convex decomposition 설정과 조각 수 `128`
- D337 동결 목표 `(radial=7 mm, tangent=11 mm, q5=1.5413 rad)`
- 부피 상대 비교 허용값 `5%`
- 조각별 128/128 통과 요구
- 물리 재질, 질량, 마찰, 감쇠, 구동기, 시간 간격
- `JOINT_LIMITS`, G0a/G0b/RL/ladder 상태

금지 사항:

- `5%`를 올려 실패를 통과시키기
- 문제 조각을 제외하거나 조각별 검사를 없애기
- mesh/asset rewrite 또는 재분해
- 새 cook callback 요청, PhysX step, settle, trial, PPO
- 목표 거리 조회를 D348 안에서 실행하기

## 3. 불변 입력

주요 입력과 실행 전 등록 SHA-256:

- D347 completion summary:
  `93ae7a6daea4d8ba9af6fa09d01deb6c72017925375195a53804b0d55286d65e`
- D347 corrected audit:
  `e652b16063cc0d7f9370df7e597ba6dcff9813f260c897b3b58b8b6c4d1b96ab`
- D347 witness manifest:
  `a57bcd32b60c65ead4313a8914c8c2d61efd3fb7d620b993ba29af6967791438`
- D347 raw measurement:
  `2b2306862b4fc0cb22ffc6ed41c179f542b4a07f7014db17290c2003e99dfb9a`
- D347 parameter audit:
  `417ca99f2c56d276f18ec455e1f3c499b0870796c57fe40379001a696022b669`
- D347 preregistration:
  `ca5f0c31e7974520f21ef51d765c8a78f78f276b720edc0d8019048f8fd50655`
- D347 zero-step gate:
  `ebc6fa2e6ba708721a7ad8d8786f3a8e83fcb81c1f530391bfc37c8f0c7748d9`
- D339 live audit helper:
  `6148252b654a6250faf78a1ebcde4caa57870e800fa1d3c45b93c803fdf882cb`
- D339 cook manifest:
  `7d0a82842af141c1e194ffcb5f9947777b8087c8fd56c72e13f684cf61481e81`
- local PhysX Python declaration:
  `ff13abb83480dcc707ac2ad60062306aef7a33f885d32ed4c8ee6dfea2008e79`
- local PhysX property-query test:
  `4c22d665fef5dce39bec2e1fb06c259c66d70e79764dac5c0ca3fe89fe07f108`

256개 개별 witness 파일은 D347 manifest의 파일명·SHA-256 목록과 다시 대조한다.

## 4. 판독 절차

### 단계 A — 입력·변수 동결

1. HEAD가 D347 commit인지 확인한다.
2. 위 원자료 hash와 256개 witness hash를 전부 재검증한다.
3. `numpy==1.26.0`, `psutil==5.9.8`, `rerun-sdk==0.34.1`,
   Qhull 판독용 `scipy==1.15.3`을 확인한다.
4. 기존 부피 허용값이 정확히 `0.05`인지 원자료에서 확인한다.
5. D347 runtime의 stage 단위가 `metersPerUnit=1`이었는지 zero-step 원자료로
   실제 게이트한다.
6. 자산·분해·목표·허용값·물리값 변경이 0건임을 JSON으로 남긴다.

### 단계 B — 콜백 면 목록 복원

각 instance/prototype witness에서 다음 원자료를 읽는다.

- `vertices`: 꼭짓점 좌표
- `indices`: 모든 면이 참조하는 꼭짓점 번호의 연속 목록
- `polygons[index_base, num_vertices]`: 각 면이 `indices`의 어디부터 몇 개를 쓰는지

각 다각형 면 `(v0, v1, ..., vn)`을
`(v0,v1,v2), (v0,v2,v3), ...`로 나눈다. 이는 새 형상을 추정하는 것이 아니라
콜백이 준 면 경계를 그대로 삼각형으로 표현하는 작업이다.

### 단계 C — 닫힌 입체·방향 검사

모든 삼각형 모서리에 대해 다음을 요구한다.

- 무방향 모서리 하나가 정확히 두 번 등장한다.
- 두 등장은 서로 반대 방향이다.

즉 구멍이나 뒤집힌 면이 없는 닫힌 입체여야 한다. `part_045`에서 면 하나를
메모리상 제거한 음성 대조군은 이 검사에서 반드시 실패해야 한다. 원본 파일은
수정하지 않는다.

### 단계 D — 두 독립 부피 계산

같은 콜백 면 삼각형에 대해 부호 있는 사면체 합으로 부피를 두 번 계산한다.

1. 좌표 원점 기준
2. 모든 꼭짓점을 그 중심만큼 평행이동한 뒤 계산

두 결과가 부동소수점 오차 범위에서 같아야 한다. 이는 계산 결과가 좌표 원점의
우연한 위치에 의존하지 않는지 확인하는 독립성 검사다.

### 단계 E — 128개 전수 및 대조군 비교

각 조각에서 다음을 동시에 기록한다.

- PhysX 속성 조회 부피
- 콜백 면 목록 부피(instance/prototype 각각)
- 기존 꼭짓점-only 볼록껍질 부피
- 각 상대 차이와 `<=5%` 여부
- instance와 prototype 원 payload 동일 여부
- 닫힌 입체·방향 검사 결과

명시 대조군:

- D347에서 이미 통과한 127개 전부
- 같은 꼭짓점/삼각형 수 계층의 조각들
- 실패 조각과 기존 볼록껍질 부피가 가까운 조각
- 최소 두께가 가까운 조각
- 면 하나를 제거한 음성 대조군

## 5. 사전 등록된 판정

`D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED`는 다음이 모두 참일
때만 가능하다.

1. 256/256 witness와 D347 원자료 hash가 불변이다.
2. instance/prototype raw payload가 128/128 일치한다.
3. 두 채널 모두 256/256 닫힌·방향 일관 면 구조다.
4. 원점/중심이동 부피가 256/256 일치한다.
5. 콜백 면 부피와 PhysX 속성 부피가 256/256에서 기존 `5%` 이내다.
6. `link5/part_045`의 기존 꼭짓점-only 방식은 `5%` 밖에 남고, 면 방식만
   `5%` 안으로 들어온다.
7. 면 제거 음성 대조군은 닫힌 입체 검사에서 실패한다.
8. Rerun 기계 계약과 실제 스크린샷 수동 검사가 모두 통과한다.

수치 조건이 하나라도 실패하면
`D348_PHYSX_PROPERTY_QUERY_VOLUME_SEMANTICS_FAIL_STOP`, 수치는 통과하지만
관찰 계약이 미완료면 `D348_RERUN_OBSERVABILITY_INCOMPLETE_STOP`으로 멈춘다.

성공해도 D347의 당시 FAIL 기록은 지우지 않는다. 성공의 의미는
“D347의 유일한 불일치는 충돌 자산 실패가 아니라 콜백 면을 버린 비교기 오류였다”로
제한한다. `g0a_pass=false`를 유지하고, 목표 거리·settle은 별도 승인 case다.

## 6. Rerun 완료 계약

Rerun은 장식이 아니라 완료 게이트다. 128개 조각마다 다음 네 형상을 기록한다.

- instance 콜백 면 형상
- instance 꼭짓점-only 볼록껍질
- prototype 콜백 면 형상
- prototype 꼭짓점-only 볼록껍질

사전 등록 수량:

- 좌표계: 2
- 메시: `128 × 4 = 512`
- 수치 행: `128 × 10 = 1280`
- 사건 행: `132`
- 비시스템 entity: `2308`
- timeline: `blueprint`, `event_idx`, `log_time`, `part_idx`

기본 화면 첫 줄은 `link5/part_045`만 네 표현으로 확대하고, 둘째 줄은
`gripper_link` 전체 네 표현을 보여 준다. 나머지 모든 조각도 RRD에 보존되며,
기본 화면만 사람이 실패 원인을 읽을 수 있도록 좁힌다.

자동 완료만으로 충분하지 않다. 생성된 `2400×1400` 실제 Rerun 스크린샷과
결정 PNG를 원본 해상도로 직접 열어 다음을 사람이 확인한 JSON/MD를 남긴 뒤에만
finalize한다.

- 두 body가 모두 표시됨
- 네 표현 패널이 구분됨
- `part_045`에서 콜백 면과 새 볼록껍질의 차이를 판독 가능함
- 5% 고정값과 128/128 결과가 수치 패널에 보임
- HOME 계약과 최종 판정 event가 보임

## 7. HOME 시작 자세 확인 계약

프로젝트의 명목 HOME은 `[0, 0, 90, 0, 0, 0] deg`다. 그러나 D347의 실제
callback 검사는 **정확한 HOME이 아니라 HOME 근방**에서 실행됐다.

- 환경 초기값은 명목 HOME이다.
- reset은 각 관절에 동결 seed의 `±0.02 rad` 작은 흔들림을 더한다.
- 그 뒤 q5는 `0 rad`, 즉 닫힘으로 다시 고정한다.
- D347은 PhysX step을 한 번도 실행하지 않아 sim counter가 `0→0`이었다.
- 실패 뒤 기록용 목표 자세는 q5 `1.5413 rad` 열린 자세로 순간 배치하고
  `sim.forward`와 0초 상태 갱신만 사용했다. HOME에서 목표까지 물리적으로
  움직인 것이 아니다.

따라서 정확한 표현은 다음과 같다.

> D347 실측은 HOME 근방의 닫힌 그리퍼 자세에서 충돌 API를 검사했고,
> 목표 열린 자세는 물리 step 없이 관찰용으로만 기록했다. D348은 PhysX를
> 다시 실행하지 않고 그 D347 실측 자료를 재판독한다.

D348 자체는 Isaac/PhysX 환경을 만들지 않으므로 시작 자세나 reset이 없다.
D348에서 별도 runtime을 만들고 reset을 exact HOME으로 바꾸면 행동 변수가 되므로 금지한다.
대신 해당 소스 줄·reset 계약·zero-step 증거를 `d348_home_start_contract.json`에
고정한다.

## 8. 실행 명령

```bash
conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py \
  --stage prepare

conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py \
  --stage analyze

# 실제 PNG 두 장을 원본 해상도로 직접 검사하고 manual JSON/MD 작성

conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py \
  --stage finalize
```

## 9. Session progress rule 정당화

이번 case는 학습 승격용 case가 아니라 D347 실제 실패에 대한 reactive comparator
repair다. 256개 원자료 전수 판독은 결과에 따라 다음 target case의 허가 여부가
달라지는 반증 가능한 실험이다. 면 구조가 닫히지 않거나 부피가 5% 이내로 돌아오지
않으면 즉시 FAIL로 멈추므로, 결정을 바꿀 수 없는 형식 검사가 아니다.

## 10. 최종 과학 결과

### 10.1 무엇이 잘못됐는가

D347은 콜백이 준 **면 연결 목록**을 버리고, 꼭짓점만 SciPy Qhull로 다시 감싸 새
볼록 외피를 만들었다. 이것은 콜백이 보고한 형상을 그대로 재는 계산이 아니라 다른
외피를 만드는 계산이다.

`link5/part_045`에서 세 값을 분리하면 원인이 선명하다.

- PhysX가 충돌체 속성으로 돌려준 부피:
  `4.061547542733024e-7 m^3`
- 콜백의 실제 면 연결 목록으로 계산한 부피:
  `4.061547420257619e-7 m^3`
- 두 값의 상대 차이: `3.015486183560612e-8`
  (`0.000003015486%`)
- 꼭짓점만 다시 감싼 Qhull 외피 부피:
  `5.171636397369118e-7 m^3`
- 그 잘못된 비교의 상대 차이: `0.27331672052498324`
  (`27.3316720525%`)

콜백 다각형의 최대 평면 잔차는 `0.3147465198 mm`였다. 즉 Float32 꼭짓점이 완벽히
한 평면에 있지 않아, 꼭짓점만 다시 Qhull에 넣으면 원래 면을 보존하지 않고 다른
외피로 교체한다. D347의 27.33%는 충돌 자산 자체의 부피 실패가 아니라 이 비교기
의미 오류였다.

### 10.2 128개 전수 결과

- instance/prototype 원자료 동일: `128/128`
- 두 채널의 닫힌·방향 일관 면 구조: `256/256`
- 콜백 면 부피 ↔ PhysX 속성 부피, 동결 `5%` 이내: `256/256`
- 전체 최대 상대 차이: `1.362105296456897e-7`
- 전체 중앙 상대 차이: `2.6262618696070446e-8`
- 원점을 중심으로 옮겨 다시 계산했을 때 최대 부피 변화:
  `2.117582368135751e-21 m^3`
- D347의 기존 통과 조각 127개, 실제 최근접 대조군, 같은 꼭짓점/삼각형 계층
  대조군: 모두 PASS
- 면 하나를 제거한 음성 대조군: 닫힌 입체 검사에서 의도대로 FAIL

결론의 적용 범위는 **PhysX 107.3.26과 보존된 D347 콜백 256개**다. PhysX 모든
버전의 내부 구현을 보편적으로 단정하지 않는다.

## 11. Rerun 최종 완료 결과

attempt5의 기계 계약과 실제 화면 검사는 모두 통과했다.

- 좌표계 `2`, 메시 `512`, 수치 행 `1280`, 사건 행 `133`
- 비시스템 entity `2309`, timeline 4종 exact
- 논리 창 `2400×1400`, 실제 PNG `4800×2800`, DPR `2.0`
- 조각 45 네 표현 + 전체 그리퍼 네 표현이 모두 보임
- `5%`, `256/256`, `128/128`, `HOME-near`, `q5=0 CLOSED`, `0 steps`,
  `D348 OFFLINE`, `G0A=false`를 화면에서 판독
- 완료문 `PASS | D347 HOME-near | D348 offline | G0a=false`가 끝까지 표시

Rerun 0.34.1의 내장 글꼴이 한글을 그리지 못한 사실은 숨기지 않는다. 기계 화면은
ASCII로 고정하고, 한국어 설명은 이 문서와 최종 사용자 보고의 책임으로 분리했다.
Rerun 메시와 수치는 Float32 표시 사본이며, 과학 판정 권한은 원본 JSON/Float64
계산과 해시에 있다.

주요 최종 증거:

- 과학 evidence SHA-256:
  `83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6`
- matched controls SHA-256:
  `35bf839e2e3efe2c64d64819d53db9d4e98dd906bc937961d342eed994a17965`
- 결정 PNG SHA-256:
  `00a77861296b048c02c91f18086cf9ff02a728305d923f8c7b751bbdfee46db9`
- 최종 completion SHA-256:
  `bc93b77fbfbeee074b1241b8f48c0317745b62ff5bca5e2196da00d25eb28697`
- 최종 RRD/RBL SHA-256:
  `a953415e41c352d13c9c054399b353401bac033080b0ddb4d02092353baf6f0b` /
  `761041f80a5f119a5ff4a1ba7388ce44127d795981762a39487b8f78df8c90f3`
- 최종 화면 SHA-256:
  `d034ca569e486edc7613b7601aa6bb4d19530c05d987b2503cfe90d8298d3516`

## 12. HOME 질문에 대한 검증 답변

명목 HOME은 `[0, 0, 90, 0, 0, 0] deg`가 맞다. 그러나 D347의 실제 callback
측정 자세는 정확한 HOME이 아니었다.

1. 환경은 명목 HOME을 기본값으로 작성한다.
2. D347 reset은 seed `33201`로 각 관절에 `±0.02 rad` 흔들림을 더했다.
3. 그 뒤 q5만 `0 rad`, 즉 닫힘으로 고정했다.
4. 그 HOME 근방 닫힌 자세에서 callback/속성 API를 읽었고 물리 step은 `0회`였다.
5. 열린 목표 q5 `1.5413 rad`는 실패 뒤 화면 관찰용으로 순간 배치했을 뿐,
   HOME에서 목표까지 PhysX로 움직이지 않았다.
6. D348은 Isaac/PhysX 자체를 다시 시작하지 않은 오프라인 재판독이므로 D348만의
   시작 자세나 reset은 없다.

따라서 “현재 PhysX 검사가 정확한 HOME에서 출발했다”는 표현은 부정확하다.
정확한 표현은 **D347이 HOME 근방의 닫힌 자세에서 0-step API 측정을 했고,
D348이 그 자료를 오프라인에서 다시 판독했다**이다.

## 13. 파라미터 증가 여부

증가하거나 완화한 물리·분해·판정 파라미터는 없다.

- 부피 허용값: `5%` 그대로
- convex 조각: body당 `64`, 합계 `128` 그대로
- 자산/USD/STL 및 decomposition 설정: 변경 `0`
- 목표 `(7,11)mm`, q5 `1.5413rad`: 변경 `0`
- 물리 step, cook 요청, 자산 쓰기, 목표 거리 질의: 모두 `0`
- `numpy==1.26.0`, `psutil==5.9.8`, `rerun==0.34.1` 유지

화면의 DPR `2.0`은 모니터가 논리 픽셀 하나를 실제 픽셀 2×2로 저장한 표시 배율이며,
물리 파라미터가 아니다. 허용값을 늘린 것이 아니라 논리 창과 실제 PNG 크기를 분리해
정확히 기록한 것이다.

## 14. 최종 판정과 다음 단계

최종 판정은 `D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED`다.
D347의 당시 `127/128 FAIL` 기록은 지우지 않지만, 그 유일한 실패가 자산 결함이
아니라 잘못된 부피 비교였음이 독립적으로 설명됐다. 올바른 면 목록 기준에서는
표현 게이트가 `128/128`이다.

그러나 D348은 동결 열린 목표의 실제 거리, settle, ten-trial을 실행하지 않았다.
따라서 `g0a_pass=false`; G0b/RL/ladder는 계속 막힌다.

권장 다음 후보는 별도 승인 D349 measurement-only
`[frozen_open_jaw_target_live_distance_gate]`다. D344 attempt3 자산, D337의
`(7,11)mm/q5=1.5413rad`, D348의 올바른 128/128 비교 계약을 그대로 사용해
raw mesh와 live collider의 목표 자세 거리를 먼저 질의한다. 물리 step과 settle은
금지한다. 그 거리 게이트까지 통과한 뒤에만 별도 후속 case에서 settle을 검토한다.
