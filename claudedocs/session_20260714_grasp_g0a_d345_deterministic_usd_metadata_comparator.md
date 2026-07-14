# 2026-07-14 Grasp G0a D345 — 주소 없는 결정적 USD 메타데이터 비교기

이번 case의 신규 변수: `[deterministic_usd_metadata_comparator]`

상태: **완료 — `D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS`**

## 1. 무엇을 왜 하는가

D344는 D339 attempt2를 새 attempt3 폴더로 복사하고, 미리 등록한 13개 충돌
조각의 꼭짓점·면 목록만 D340 고정점 후보로 교체했다. 정확히 13개가 바뀌고
115개가 보존됐으며, 최소 두께 자료형·비트 128/128과 질량·관성·비물리 파일
보존도 통과했다. 그러나 등록한 39개 형상 속성을 가린 뒤의 합성 USD 의미 해시가
달라 D344는 `D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP`으로 정지했다.

사후 읽기 전용 감사는 차이가 `metadata.apiSchemas` 194개뿐이고, 비교기가
`Sdf.TokenListOp`의 실제 내용 대신 `<... object at 0x...>` 형태의 프로세스별
RAM 주소를 해시한 것이 원인임을 좁혔다. 주소가 아닌 차이는 0개였다. 이 감사로
D344를 소급 PASS로 바꾸지는 않는다.

D345는 immutable D339 source와 immutable D344 attempt3를 읽기만 하며, USD
메타데이터를 실제 자료형과 내용으로 직렬화하는 비교기를 별도로 증명한다. PASS는
D346 live validation의 자격만 열고, attempt3의 실제 PhysX 충돌 성공이나 G0a
성공을 주장하지 않는다.

## 2. D334와 attempt3를 혼동하지 않는 계보

- D334는 attempt3를 만든 사례가 아니다. 닫힌 `q5=0` 자세에서 접촉 원인이 실제
  그리퍼 원본 형상 관통인지, 부풀어진 PhysX 충돌 형상인지, 링크 소유권 오귀속인지
  판별한 성공한 원인 감사다.
- D337이 `q5=0=닫힘`, `q5≈1.541-1.571rad=열림`을 바로잡았다. 열린 조에서는
  고정 `(7,11)mm` 목표의 원본 link5/gripper가 각각 `+4.2726/+11.1751mm`
  떨어졌지만, 기존 link5 PhysX 볼록 형상이 원통을 `-6.2367mm` 관통해 step 0에
  `38.861N` 충격을 냈다.
- D338 attempt1은 callback 결과를 기록하기 전에 부적절한 전역 통계 게이트에서
  정지했다. D339 attempt2는 callback-first 두 독립 cook `64+64` 동일성을
  증명했으나 실제 장면 재cook에서 13/128 조각이 달라졌다.
- D344 attempt3는 분해 설정을 다시 튜닝한 세 번째 파라미터 시도가 아니다.
  attempt2의 불안정한 13개만 PhysX가 한 번 정리한 고정점 후보로 교체하고 115개를
  그대로 둔 세 번째 forward-only 자산 세대다.

## 3. 두 종류의 "주소"

1. `/colliders/link5/.../part_011` 같은 USD 경로는 장면 항목의 논리적 이름이다.
   D334에서 어느 링크가 어느 충돌체를 소유하는지 확인하려면 필요하다.
2. `0x7f...` 같은 값은 Python이 PXR 포장 객체를 그 실행 중 RAM 어디에 놓았는지
   나타내는 임시 주소다. 로봇 좌표, 원통 위치, 파일 경로, 자산 내용이 아니다.
   같은 파일을 다시 읽어도 달라질 수 있으므로 과학 해시 입력으로 금지한다.

## 4. `apiSchemas`를 어떻게 비교하는가

`apiSchemas`는 USD 장면 항목에 어떤 기능 묶음이 적용됐는지 기록한다. D339/D344
볼록 조각의 직접 저장값은 다음 세 토큰을 `prepend` 연산으로 추가한다.

- `PhysicsCollisionAPI`: 충돌체 기능
- `PhysicsMeshCollisionAPI`: 메시 충돌 설정
- `PhysxConvexHullCollisionAPI`: PhysX 볼록 껍질 전용 설정

목록의 최종 토큰만 비교하지 않는다. USD는 여러 층을 합성하므로 다음을 각각
순서 보존해 기록한다.

- 명시 목록인지(`isExplicit`)
- 명시 항목
- 앞에 추가한 항목
- 뒤에 추가한 항목
- 추가·삭제·순서 지정 항목

직접 Sdf 층에 저장된 목록 연산과 모든 층을 합친 Usd Stage의 최종 적용 목록을
별도 판독한다. core-only PXR에서 등록되지 않은 PhysX schema가
`GetAppliedSchemas()`에서 생략될 수 있으므로, 직접 authored 목록과 합성 결과를
서로 대체하지 않는다.

## 5. 결정적 직렬화 계약

비교기는 일반 `repr(...)` fallback을 사용하지 않는다. 지원하는 USD/PXR 값은
자료형 태그와 실제 필드로 JSON화하고, 지원하지 않는 자료형이 하나라도 나오면
FAIL한다. 주요 규칙은 다음과 같다.

- `Sdf.*ListOp`: 연산 모드와 여섯 항목 목록을 각각 기록
- `Sdf.Path`, `Sdf.AssetPath`, reference/payload/layer offset: 실제 필드 기록
- `Sdf.ValueTypeName`: 이름·별칭·역할 기록
- Gf 벡터·행렬·쿼터니언과 Vt 배열: 자료형 태그와 원소 순서 보존
- float: 주소 없는 IEEE-754 비트 표현
- dict: 키 자료형을 보존해 정렬한 항목 목록
- prim/attribute/relationship: 경로·형식·활성·연결·메타데이터를 정렬하되
  의미 있는 목록 내부 순서는 보존

D344에 등록된 정확한 13개 조각의 `points`, `faceVertexCounts`,
`faceVertexIndices` 값만 고정 표식으로 가린다. 허용 개수는 정확히 39개이며 다른
속성이나 메타데이터를 가리지 않는다.

## 6. 두 독립 프로세스와 실패 가능한 반례

등록 명령은 독립된 standalone-PXR worker 두 개를 새 PID와 nonce로 시작한다.
각 worker는 source와 attempt3의 다음 증거를 판정 전에 반환한다.

- 310개 합성 prim의 주소 없는 canonical row hash
- 39개 mask 목록과 개수
- 직접/합성 `apiSchemas` 목록 연산과 실제 토큰
- 합성 `GetAppliedSchemas()` 목록
- 처리한 실제 자료형 분포와 지원하지 못한 자료형 수
- 옛 `repr(TokenListOp)` 방식의 주소 포함 진단 해시

PASS 비교기는 실제로 틀린 값을 거부할 수 있어야 한다.

1. 두 프로세스의 옛 주소 포함 해시는 달라야 하고 새 canonical 해시는 같아야 한다.
2. 실제 `prepend` TokenListOp에서 토큰 하나를 메모리 안에서 삭제하면 canonical
   해시가 달라져야 한다.
3. 최종 토큰이 같아도 `prepend`를 `explicit`으로 바꾸면 직접 목록 연산 해시가
   달라져야 한다.

원본 USD 파일은 세 반례에서 모두 쓰지 않는다.

## 7. 불변성·파라미터 계약

- 신규 변수: measurement-only 1개
  `[deterministic_usd_metadata_comparator]`
- 물리 변수 변경: 0
- 기존 파라미터 증가/변경: 0/0
- 분해 설정 변경: 0
- 허용 기준 완화: 0
- 자산 복사·작성·recook: 0/0/0
- Isaac/Kit/GPU/SimulationContext/물리 진행: 없음/없음/없음/없음/0
- D344 attempt3, D339 attempt2, D344 diagnosis 파일: 실행 전후 정확히 동일
- D344 공식 FAIL, `g0a_pass=false`, G0b/RL/ladder block 유지

## 8. Rerun 생략과 세션 진행 규칙

D345는 좌표, 형상 거리, 자세, 접촉, 궤적, 사건 시간을 판정하지 않는 순수
파일/자료형/스키마/해시 감사다. Rerun 화면은 토큰 비트나 목록 연산을 더 강하게
증명하지 못하므로 새 RRD/RBL/PNG를 만들지 않는다. D346에서 실제 충돌 형상이나
목표 거리를 판정하면 D341 전체 Rerun 완료 계약이 다시 의무다.

이번 control hardening은 D344에서 실제 관측된 comparator 실패에 대한 reactive
수리다. 옛 주소 포함 표현의 독립 프로세스 비결정성과 토큰 삭제·연산 모드 변경
반례가 verdict를 실제로 FAIL시킬 수 있으므로 실패 가능한 perturbation 평가다.
물리나 학습은 이 결함을 검증하지 못하고 범위 밖 변수를 추가하므로 실행하지 않는다.

## 9. 등록 실행 순서

1. D344 attempt3 전체 inventory와 D344의 두 semantic diagnosis 및 root-cause
   audit를 SHA-256으로 봉인한다.
2. D339 source attempt2와 D344 source/session/manifest를 봉인한다.
3. D345 script/session/AGENTS/START_HERE/BACKLOG/parameter audit/Rerun 생략
   문서를 봉인한다.
4. `py_compile`, JSON parse, 금지 import/asset-write 정적 검사를 통과한다.
5. 아래 등록 명령을 정확히 한 번 실행한다.
6. worker A 결과를 먼저 보존하고 worker B 결과를 보존한 뒤에만 교차 판정한다.
7. 어느 조건이든 실패하면 같은 출력 경로에서 재실행하지 않고 FAIL_STOP한다.

등록 명령:

```bash
env \
  PYTHONPATH=/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
  LD_LIBRARY_PATH=/home/cgxr/miniconda3/envs/isaaclab/lib:/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/cyl34_top_view_d345_grasp_g0a_deterministic_usd_metadata_comparator.py \
  --stage run
```

## 10. PASS/FAIL과 중단선

PASS에는 다음이 모두 필요하다.

- 사전 등록·환경·39개 allowlist·입력 봉인 통과
- worker 두 개의 PID/nonce 독립
- 각 worker에서 source==attempt3 canonical rows, 직접 목록 연산, 합성 목록 통과
- 두 worker 사이 canonical row/hash exact
- 지원하지 않은 PXR 자료형 0
- 옛 주소 포함 표현은 교차 프로세스에서 달라져 비결정적임을 검출
- 토큰 삭제와 `prepend→explicit` 반례 모두 거부
- 실행 전후 D339/D344 불변
- 자산/Isaac/Rerun/물리 출력 0

통과 판정은 `D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS`, 하나라도 실패하면
`D345_DETERMINISTIC_USD_METADATA_COMPARATOR_FAIL_STOP`이다. 어느 판정이든 D345는
즉시 정지한다. PASS 후에도 D344는 FAIL로 남고, 별도 사용자 승인 D346만 immutable
attempt3의 callback 256개·실제 128조각·고정 목표 거리·Rerun을 검증할 수 있다.

## 11. 실행 전 비교기 검증

scientific run과 출력 폴더 생성 전, 기록을 남기지 않는 비교기 단위검사를 수행했다.
이 검사는 D339와 D344의 최종 동등 판정을 내리지 않고, 등록 명령이 시작되기 전에
비교기 결함을 발견하면 실행을 중단할 수 있는 실패 가능한 준비 검사다.

- 독립 코드 검토: 확정적인 Python/PXR 중대 런타임 오류 0건
- `py_compile`: PASS
- 출력 경로 강제: JSON/문자 쓰기 함수는 `g0a_d345/` 밖의 경로를 거부
- 금지 쓰기 정적 검사 5/5 PASS: 자산 저장·복사·이름 변경 호출 0, 직접 쓰기
  모드 `open` 0, shell subprocess 0, 허용되지 않은 `write_text` 0, 두 허용 쓰기
  함수의 출력 경로 guard 2/2
- D339 한쪽 자료형 dry-run: 합성 prim 310, time sample 0, 미지원 자료형 0,
  canonical 주소 패턴 0
- 실제 `part_011` 목록 연산 반례 10/10 PASS: 원본은 non-explicit prepend 3개,
  토큰 하나 삭제를 검출했고 적용 토큰 수가 정확히 하나 줄었으며,
  `prepend→explicit` 변경도 최종 토큰이 같아도 검출했다.

독립 검토가 지적한 세 항목—출력 폴더 강제와 금지 쓰기 검사, 변경 후 토큰 내용
보존, 전체 속성 time sample=0 확인—을 모두 사전등록 전에 반영했다. worker의 성공
시 표준 오류도 길이·행 수·SHA-256으로 기록하고 빈 값이어야 통과하도록 보강했다.

아직 D345 scientific run은 실행하지 않았고 `g0a_d345/` 출력 폴더도 만들지 않았다.
다음은 최종 소스와 현재 작업 트리를 봉인하는 `--stage prepare`이며, 그 뒤에는 봉인된
문서를 고치지 않고 위 등록 명령을 정확히 한 번 실행한다.

## 12. 실제 실행 순서와 결과

위 11절은 실행 직전 봉인된 기록이다. 그 뒤 다음 순서로 진행했다.

1. `--stage prepare`를 한 번 실행해 사전등록·파라미터 동결·Rerun 생략 사유 세
   파일을 생성했다.
2. 사전등록을 다시 읽어 D339 attempt2 `18`개와 D344 attempt3 `9`개, 정확한
   `39`개 허용 속성, worker `2`개, `numpy==1.26.0`, `psutil==5.9.8`,
   OpenUSD `0.24.5`를 확인했다.
3. 등록한 `--stage run` 명령을 정확히 한 번 시작했다.
4. worker A가 끝나고 `d345_worker_a.json`이 먼저 생긴 뒤 worker B가 시작됐다.
5. B가 끝난 후에만 두 파일을 교차 비교하고 evidence·summary·한글 report를 썼다.
6. 도구의 최초 대기 호출이 종료 코드를 전달하지 않은 채 끝났지만 원래 부모와
   worker 프로세스는 계속 살아 있었다. 명령을 재실행하지 않고 같은 프로세스를
   감시했고, A→B→최종 판정 파일 생성 후 원래 프로세스가 종료된 것을 확인했다.

두 독립 worker의 내부 PID는 `13`과 `46`, nonce는 각각
`15f8bc138e52ce78562b381d9c29e8fd`와
`d124e55ea4f83d04a6a8e470f3e0d26d`로 달랐다. 두 worker 모두 등록한
Python 3.11과 OpenUSD 0.24.5를 사용했고 표준 오류는 `0`바이트였다.

## 13. 주소 없는 전체 의미 비교

각 worker가 D339 원본과 D344 attempt3를 따로 직렬화했으므로 비교 방향은 총
네 개다: `A-원본`, `A-attempt3`, `B-원본`, `B-attempt3`.

- 합성 장면 항목: 네 방향 모두 `310`
- 정규 JSON 크기: 네 방향 모두 `164,675,173`바이트
- 정규 SHA-256: 네 방향 모두
  `3f85d121439060ef5c6deb49cab7860dbc72eb94e23e54617c4ac2b1f7cdcd09`
- 주소 패턴: 네 방향 모두 `0`
- 지원하지 못한 자료형: 네 방향 모두 `0`
- 시간에 따라 변하는 속성값: 네 방향 모두 `0`
- 가린 값: 등록한 13개 조각 × 3개 형상 속성 = 정확히 `39`
- 직접 물리층 `apiSchemas`: 네 방향 모두 `149`, 행·목록 연산 exact
- 합성 장면 `apiSchemas`: 네 방향 모두 `194`, 행·최종 목록 exact

여기서 `149`와 `194`는 서로 충돌하는 수가 아니다. `149`는 현재 물리층 파일에
직접 목록 연산이 쓰인 항목 수이고, `194`는 참조·하위 층까지 합친 최종 장면에서
해당 메타데이터가 있는 항목 수다. 대표 `part_011`은 파일에 세 토큰을
`prepend`로 썼고, 합성 결과는 같은 세 토큰의 `explicit` 목록이다. core-only
PXR의 `GetAppliedSchemas()`는 등록되지 않은 PhysX 전용 플러그인 토큰을 해석하지
못해 앞의 두 core 토큰만 반환하지만, 직접 작성값과 합성 메타데이터에는 세 토큰이
모두 보존됐다.

## 14. 비교기가 틀린 내용을 실제로 거부하는지

### 14.1 옛 메모리 주소 방식

옛 `repr(TokenListOp)`에는 194개 행 모두 RAM 주소가 있었다. worker A의 주소 포함
해시는 `85a69a023565111a3b99f75aa701b0ef34b436a74ddf67f38b288ce8c17942e1`,
worker B는 `67a2480bab2ae9af769eb8f0d48bf0f0ccf8b7719043fa4f2870afbb419eddff`로
달랐다. 파일 내용이 아니라 실행할 때의 메모리 배치가 해시를 바꾼다는 대조군이다.
두 해시는 진단용으로만 저장했고 과학 판정 입력에서는 거부했다.

### 14.2 토큰 하나 삭제

대표 직접 작성값은 다음 세 토큰을 앞에 추가한다.

1. `PhysicsCollisionAPI` — 충돌체 기능
2. `PhysicsMeshCollisionAPI` — 메시 충돌 설정
3. `PhysxConvexHullCollisionAPI` — PhysX 볼록 껍질 설정

원본 목록 연산 해시는
`0cf931d1353486685dead88b6a4026eacf3c4bdeaf1e85b902c9c912e77ece57`이다.
메모리 안에서 세 번째 토큰만 삭제하자 적용 토큰은 정확히 `3→2`로 줄고 해시는
`a277e0ac19ed2acd4c05c5cd6f43c5e4280e7c67fdf0107cb4b00b9a0cba70d8`로
바뀌었다. 두 worker가 모두 이 잘못을 검출했다.

### 14.3 최종 토큰은 같지만 작성 연산이 다름

세 최종 토큰을 유지한 채 작성 방식을 `prepend`에서 `explicit`으로 바꾸자 해시는
`4a2c5d3b9e6dd46c2d67db99b46ba37a6280a145ee4d0cdc36bc900c69652e0d`로
바뀌었다. 즉 새 비교기는 “무슨 토큰이 남았는가”뿐 아니라 “그 목록을 어떤 USD
연산으로 작성했는가”도 구별한다.

## 15. 불변성·변수·Rerun 결과

- D339 attempt2: `18→18`, 묶음 해시
  `0dae41fd3937a0a8aea18488019c74f097d32f7b8de916943ff31334e30464a1`
  실행 전후 exact
- D344 attempt3: `9→9`, 묶음 해시
  `ea6965199ff1f195a6d19d9c55febfe44cc9838f12651570c80d5bb97fa6caf1`
  실행 전후 exact
- D344 전체 출력: `19→19`, 묶음 해시
  `bd07acc09a8c89f4f8da78da7e9492546a8f2974864e3e42c9069a211195a3c8`
  실행 전후 exact
- 사전등록한 코드·규칙·상태·입력 파일 해시: 전부 exact
- D345와 무관한 미커밋 저장 보관 작업 묶음 해시: 실행 전후 exact
- 신규 변수: 비교기 1개
- 기존 파라미터 증가/변경, 기준 완화, 분해 설정 변경: `0/0/0/0`
- 자산 작성/복사/재조리, Isaac runtime, 물리 진행: `0/0/0/0/0`
- RRD/RBL/PNG: `0/0/0`; USD/STL 등 자산 출력도 `0`

Rerun은 누락된 것이 아니라 등록된 예외로 생략했다. D345 판정에는 공간 위치,
거리, 접촉, 자세, 궤적, 시간축이 없어서 화면을 추가해도 토큰·자료형·해시 증명이
강해지지 않는다. 다음 실제 형상 사례 D346에서는 이 예외가 끝나고 D341의 RRD,
RBL, 검증, 화면 캡처, 실제 육안 확인이 모두 다시 필수다.

별도 읽기 전용 감사자는 summary를 그대로 인용하지 않고 prereg, worker A/B,
evidence와 현재 입력 inventory를 교차 확인했으며 불일치 0건으로 보고했다. 증거 한계도
기록한다. 164,675,173바이트 정규 JSON 전체는 파일로 보존하지 않았고, 각 worker가
남긴 310개 행 해시와 전체 해시를 보존했다. 이는 사전등록 계약과 일치하지만, 독립
감사는 세 번째 worker로 164MB 정규 표현을 다시 생성한 재실행 감사는 아니다.

## 16. 최종 판정의 일상어 번역

최종 판정은 `D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS`다. 쉽게 말하면,
D344에서 실패한 한 항목은 충돌 자산의 숨은 의미가 달라서가 아니라 비교기가
임시 RAM 주소를 파일 내용처럼 취급했기 때문에 생긴 거짓 경보다. 주소를 버리고
실제 토큰·목록 연산·자료형을 기록하자, 원본과 attempt3는 허용한 39개 형상값을
제외한 모든 비교 대상에서 두 독립 실행 모두 정확히 같았다. 또한 토큰 삭제와
작성 방식 변경은 제대로 잡았으므로 단순히 모든 차이를 무시한 비교기도 아니다.

그러나 D345는 과거 D344를 소급 PASS로 바꾸지 않으며, attempt3가 Isaac/PhysX에서
원통을 피하고 안정적으로 물체를 잡는다는 뜻도 아니다. `g0a_pass=false`, G0b/RL/
ladder 차단은 유지한다. D345가 연 것은 별도 승인 D346의 자격뿐이다. D346은
immutable attempt3를 새 프로세스에서 읽어 callback 256개, 실제 충돌 조각 128개,
고정 `(7,11)mm` 열린 조 목표의 거리, 그리고 D341 Rerun 완료 계약을 검증해야 한다.

교수님 방향의 원통 재질·실물 제작 계약은 `claudedocs/BACKLOG.md`에 후속 후보로
분리했다. 현재 `0.72kg`, 마찰 `1.5/1.2`는 자동 물리값이 아니라 임시 코드값이며,
D345나 비교기 PASS에 섞어 보정하지 않았다.

## 17. 결과 파일

- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_parameter_freeze_audit.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_rerun_omission_justification.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_worker_a.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_worker_b.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_deterministic_usd_metadata_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_deterministic_usd_metadata_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_deterministic_usd_metadata_report.md`

## 18. 세션 종료 상태

실험 판정 시점에는 사전등록한 18개 코드·규칙·상태·입력 해시가 모두 exact였다.
판정이 끝난 뒤 Current-State Protocol에 따라 이 세션 문서와 `START_HERE.md`를 결과로
갱신하고, `DECISIONS.md` D345와 실험 원장 행을 추가했다. 따라서 현재의 세션 문서와
`START_HERE.md` 해시는 사전등록 값과 의도적으로 다르며, 이는 과학 실행 도중 변경이
아니다. D345 script, AGENTS 규칙, BACKLOG 항목과 D339/D344 입력은 실행 후 수정하지
않았다.

현재 작업 트리는 미커밋 상태다. D345 소유 변경은 `AGENTS.md`, `START_HERE.md`,
`claudedocs/BACKLOG.md`, `DECISIONS.md`, `EXPERIMENT_LEDGER.md`, 이 세션 문서,
D345 script와 `g0a_d345/` 출력이다. 기존에 있던 저장 보관 sidecar 변경
`claudedocs/storage_plan_cube10cm_visual_dataset_d232.md`,
`claudedocs/dataset_archives/`,
`claudedocs/session_20260714_cube10cm_0_999_usb_archive_preflight.md`는 수정하거나
실행하지 않았다. 최종 재검사에서 이 세 경로의 inventory 해시는 사전등록과 각각
exact였고, 전체 dirty 묶음에 새로 나타난 경로는 의무 갱신한 `DECISIONS.md`와
`EXPERIMENT_LEDGER.md` 두 개뿐이었다. 사용자 요청이 없으므로 commit/push도 하지
않았다.
