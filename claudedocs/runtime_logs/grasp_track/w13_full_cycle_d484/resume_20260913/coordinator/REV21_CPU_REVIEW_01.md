# rev21 변경부 root 검사 — GPU 전 결함 검출

2026-09-13 18:08 KST. close_diagnostics.py 전체와 rev20→21 renderer/launcher diff를 읽었다.
생산 자기검사14/14를 독립 인수로 대체하지 않는다. rev21 GPU 미실행 보존.

1. 진단 step이 BaseException을 기록한 뒤 삼키고 returned=true를 적는다.
   SystemExit/KeyboardInterrupt 전파와 정상반환/예외 구분이 필요하다.
2. stop 조건에 사용하는 두 번째 get_status에는 before/after가 없으므로 그 호출에서
   멈추면 앞 호출까지의 영수증만 남는다.
3. bounded_update의 elapsed 식은 실제 소요가 budget을 넘으면 budget값으로 잘린다.
   실제 t_after-t_before와 budget 초과 여부를 분리해야 한다. native app_update 단일 호출을
   Python 루프 조건만으로 제한했다고 주장할 수 없고 기존 외부 실행기가 최종 경계다.
4. probe는 bounded_close 밖에서 실행되며 probe 실패가 renderer.ok/rc에 반영되지 않는다.
   진단·실제close·전체cleanup 계측을 나누고 누락/오류를 성공으로 삼지 않아야 한다.
5. 기존 장면 위에 놓는 검정 패널은 팔 상단을 가릴 수 있고 긴 라벨은1600px폭을 넘길 수 있다.
   기존 카메라 영상은 유지하고 caption전용 여백/실제 폰트 폭에 맞춘 줄바꿈이 필요하다.
6.180초=child135+close27+grace18이므로 외부TERM은162초다. 생산본문135초 주장은 정정 대상.

msg_bcb387d139fc로 위 묶음을 새rev22에서만 수정하도록 했다. auditor에는
msg_5bc8c06c62e2로 느린/예외/status 가짜 호출 음성대조와 연결 검수를 배정했다.
새 과학 임계/물리/경로/조건 변경이 아니며 production GO0, GPU HOLD다.

## NVIDIA 원문과 관측/개입의 구분

설치 omni.replicator.core1.12.27+107.3.3/IsaacSim5.1.0에서 root가 확인한 원문:

- [Omni Replicator Python API,1.12.27](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.orchestrator.stop)
  stop은 정지 상태가 될 때까지 기다리는 API가 아니라 정지 명령을 제출하는 API다.
  문서의 설명은 이번 native 호출의 신속한 반환이 실제로 보장됐다는 증거가 아니다.
- 같은 버전 문서의 set_capture_on_play는 timeline 조작에 Replicator가 반응할지를 바꾼다.
  따라서 capture-off를 stop보다 먼저 실행하는 진단은 상태/호출순서 개입이다.
  기존 close의 순수 읽기 관찰이라고 주장하지 않는다. 물리 연구조건 변경과는 별개다.
- 로컬 설치 `isaacsim/extscache/omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/orchestrator.py:826`
  는 STOPPING설정→event dispatch→비동기 완료 coroutine예약을 한다. 공개stop은 :1229.
  정확 native blocking frame은 아직 미확정이다.
- 처음 시도한 py/replicator/1.12.16 문서URL은 latest로redirect됐으므로 버전일치근거로 쓰지 않았다.
  위 kit/docs/omni_replicator/1.12.27의 실제본문을 별도로 확인했다.

root가 추가한 표시는 바닥 높이/벽 범위를 이해할 수 있어야 한다는 관찰 목적의 보완이다.
그림 속 표시선을 실제 충돌면이나 새 물리구조로 세지 않는다. 기존 실패그림 판정은 그대로 보존한다.

## 18:12~18:18 독립 반례와 연결 보완

독립 `REV21_CANDIDATE10_CHANGED_PARTS_AUDIT_02.json`을 root가 전체 읽고 SHA256
01c58d5d0792e8e501a55d342a9f5396dd62861b0bb8315ead7f3545a9f9bb29를 재확인했다.
4/12인수조건통과·8미달·CPU NO-GO. 실제 가짜호출0.052089초가0.01초로 잘려 기록됨,
두 라벨1911/1922px>1600px. 패널214px/side900px는 가림 위험 정량이지 새 과학 임계가 아니다.

root는 작업중rev22의CD/renderer diff도 읽고 동결 전에 같은 요구의 재발을 알렸다:
루프내bucket호출이flush를생략,enum실값대신repr substring,app_update예외뒤STOPPED문자열로
빠져나가면falsePASS가능,renderer외부BaseException재흡수,프레임별caption높이변동.
msg_2020e857ed59/msg_dc9b5fbc2940으로실제값비교·전후기록·전체실패전파·공통caption높이·
기존27초정리할당안의진단+close잔여시간을요청했다. 새physics/criteria/runner수정은아니다.

이전W12소스 `w12-isaac-replay/.../replay/sim_isaac_replay_w12.py:530`은20초뒤
os._exit(0)를호출하는daemon을시작한뒤:531에서close를호출한다. 따라서옛rc0는정상close의
양성대조가아니다. 이번확인은소스만이며그fallback의실제발생여부나옛과학판정을새로재검증한것이아니다.
