# 참조 해제 검사 정정 — 새 revision 사전 계약

작성: 2026-09-13 20:48 KST, root. 적용 대상은 앞으로 동결할 rev28 이후의 명시 후보뿐이다.
이 문서 자체는 GPU 또는 production GO가 아니다. 새 물리 변수: [].

## 왜 정정하는가

root가 rev26/27 준비 진단에 추가한 '5개 weakref가 즉시 모두 사망해야 한다'는 조건은 renderer가
소유한 참조를 놓았다는 요구보다 강하다. actual16에서는 camera2/scene/robot은 사망했지만 sim은
생존했고, 사전 조건에 따라 child/runner96 및 readiness FAIL로 끝났다. 이 결과는 그대로 보존한다.

IsaacLab2.3 공식 및 설치 `clear_instance()`는 STOP callback 정리 후 부모 cleanup을 호출한다.
부모는 callback과 singleton을 정리한다. 객체의 즉시 weakref 사망을 보장한다는 계약은 아니다.
공식 build_simulation_context의 finally 자체도 sim local을 유지한 채 clear를 호출한다. 이는
API cleanup 성공과 즉시 객체 소멸이 다름을 보여 주지만 actual16의 정확한 잔류 소유자는 밝히지 못한다.

이는 연구 결과 임계값 수정이 아니라 root가 추가한 **종료 구현 진단의 과도한 조건 정정**이다.
원래 scientific criteria.json, 물리/경로/입력, 정상 종료 및 시간 한도는 불변이다. 검사 난도를
낮추기 위해 old FAIL을 재평가하지 않으며, 앞으로의 실제 실행에서 아래 요구를 직접 관측한다.

## 새 사전 계약

1. 기존 clear_instance 호출이 반환하고 실패 없이 기록돼야 한다. 순서는 clear → 소유 참조 해제
   → 기존 probe/update → 기존 close이며 추가 lifecycle 조작은 없다.
2. 실제 release callback이 cams/scene/robot/sim 각 셀을 None으로 만들고, 이 **실제 셀 값**에서
   계산한 네 `is None` 결과와 공개 `SimulationContext.instance() is None` 결과를 반환한다.
   helper의 CloseProbe.step 분류가 exact keys와 exact bool 타입/True를 검사해 JSON으로 영속 저장한다.
   repr 문자열, 상수 True 또는 truthy 값으로 관측을 대신하지 않는다. 누락/잘못된 타입/False/분류오류는 실패다.
3. Camera side/top, scene, robot 4개는 관측 가능하고 실제 weakref 사망이 필수다. 대상 누락이나
   weakref 미지원/None 대체로 통과시키지 않는다. cams dict은 weakref 미지원 사실을 명시한다.
4. sim weakref도 관측 대상에서 제거하지 않는다. 생존/사망을 그대로 기록하되 **정보성 관측**으로
   분리한다. required_survivors와 informational_survivors를 혼동하지 않는다. sim 생존만으로
   외부 엔진 소유/누수/안전/정상 해제를 단정하지 않는다. 공용 instance가 None이 아니면 여전히 실패다.
5. 이 관측은 기존 정리예산 안에서 실행하며 모든 호출 전후 기록/실제 시간/제어예외 전파를 유지한다.
   실패는 기존 rc96이다. 정상 close 반환, 필수6시간, 실제 child/runner rc0, 신호/timeout 없음,
   정리 포함 총180초 및 owned group 부재는 별도의 필수 조건으로 그대로 둔다.
6. 런처의 rc96 설명은 probe 미측정으로 단정하지 않고 실제 참조/clear/probe 영수증을 안내해야 한다.
   설명 수정 외 launcher 성공 boolean, runner와 timeout 처리는 바꾸지 않는다.

## 검증과 범위

- 실제 callback AST의 셀/instance 관측을 가짜 소유자 환경에서 실행하는 작은 CPU 대조를 포함한다.
  helper 양성, sim만 자기순환으로 생존하는 정보성 양성, camera/robot 잔류 음성, 각 소유셀 미해제,
  singleton 잔류, 분류 keys/타입/False/오류 및 hash-byte 손상을 검출한다. 테스트가 관측 대상에
  강한 임시 참조를 남겨 결과를 바꾸지 않아야 한다. native 효과 검증으로 과장하지 않는다.
- 새 rev28, 새17planned 및 미존재17actual 경로만 사용한다. source/mini/external/criteria 해시를
  다시 고정한다. 기존 rev26/27 및 actual16을 편집·삭제·재실행하지 않는다.
- 실제 준비 실행은 CPU 독립 검수 후 별도 정확 GO1회가 있어야 한다. production GO는 그 후다.
- gc.collect, 직접 __del__, 전역 reset/destroy, 강제 rc0/skip_cleanup, 엔진 설정 변경, 패키지 설치 금지.
- 새 물리/학습/A/B/C/하드웨어/commit/push 금지. 기존9시간 비용 승인 유지, root 원장 배타 소유.

## 근거

- 공식 `isaaclab.sim.simulation_context — Isaac Lab Documentation`, v2.3.0:
  https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/simulation_context.html
  clear_instance 및 build_simulation_context finally를 root가 직접 읽었다.
- 설치 `.../isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py:639` 및 `:1070`;
  SHA340450726276d321c48b57de35f846c5a231c30a358b4922b5b7dbb8d42ec80e.
- 설치 IsaacSim5.1 `.../isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py:195`
  공개 instance와 :211 clear; SHAebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028.
- 독립 `REV27_SIM_SURVIVOR_DIAGNOSTIC_01.md`, actual 감사22/23 SHA
  9f73e5303a542003d7240e72b43ab6a9731afc9b80de0c21b8f0c55a9b57e803.
- CPU 의미검토 수락 msg_afb6eac535f2. actual16 원본/실제8PNG 검수는
  `ROOT_READINESS_INSPECTION_07.json` SHA45c52abd49f6a59650f574840c31a59aefd750ca8f8f040d11e5c969eab8b90d.
