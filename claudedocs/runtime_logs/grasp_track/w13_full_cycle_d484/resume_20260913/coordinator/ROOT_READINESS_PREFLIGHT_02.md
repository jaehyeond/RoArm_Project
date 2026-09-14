# rev13 준비 실행 검수 — NO-GO

2026-09-13 16:06 KST. 이 검수는 읽기 전용이며 새 Isaac/DEME 실행0. root는 준비 런처169줄, 카메라 계산135줄, 렌더러1064줄을 읽었다. 실행기/수치의 독립 대조는 감사 워커에 배정했다.

## 관측한 실행 전 연결 결함

1. `src/readiness_launcher.py:108` 이하에서 임시 `MANIFEST_prospective.json`에 planned_outputs만 넣는다. 실행기는 `external_frozen_inputs_sha256`를 직접 읽는다. 또한 임시 `frozen_copies_sha256={}`이며 호출한 실행기 파일은 임시 revision 바깥이다. 따라서 필수 입력 키와 자기 해시 계약이 충족되지 않는다. 해시 검사 자체를 약화하지 말고 정상 동결 구조를 사용해야 한다.
2. 런처가 `--out` 안에 `_readiness_revision`, `_readiness_attempt`, `readiness_plan.json`을 만든 뒤 같은 경로를 렌더러에게 준다. 렌더러139~147줄은 비어 있지 않은 출력 폴더를 거절한다. 계획/실행 영수증과 렌더 출력은 서로 다른 새 하위 경로여야 한다. `--plan-only`로 채운 폴더를 같은 명령으로 재사용하는 것도 현재 런처에서 거절된다.
3. 런처는 `--synthetic-phases`를 전달하지 않는다. 기존6PF 벽 시험만 읽으면8상한이어도6raw만 재생되고 운반·배출 위치·HOME 표시는 확인되지 않는다. 기존 승인 범위인 최대8장 안에서 명시 합성 자세를 포함하는 정확한 계획이 필요하다. 합성은 실제 물리 결과로 세지 않는다.
4. 렌더러1034~1048줄의 close timeout은 `os._exit(rc_)`로 나가므로 기존 렌더ok이면rc0가 된다. 런처는 `close_timed_out`을 기록하지만 `honest_gaps`/`ok` 판정에 사용하지 않아 close timeout을 준비완료로 받아들일 수 있다. 시간초과는 전체 워크플로 비성공으로 남겨야 하며, 화면 품질 판정과 분리한다. SIGALRM의 합성 대조 성공을 실제 Isaac C++ 종료의 무조건적 시간 보장으로 쓰지 않는다. 외부 실행기 제한은 여전히 필수다.

## 구도 계산 주장 범위

카메라 계산 자체는 제공된 점 집합을 대상으로 한다. 현재 렌더러는 S1 노드 AABB를 두 대각 점만 넣고, 어깨·툴 원점은 넣지만 모든 팔 링크/합성 자세 외곽은 넣지 않는다. 따라서 "모든 팔/합성 자세가 반드시 담긴다"는 주장은 아직 성립하지 않는다. AABB에는8모서리를 쓰고 선택된 합성 자세의 툴 및 표시 팔 외곽을 포함하거나, 계산 범위를 정확히 한정한 뒤 실제8프레임 검수에서 누락을 검출해야 한다. 물리/경로 변경은 요구하지 않는다.

## 검수 시점 해시 불일치

07:06 UTC에 `rev13/REVISION_PIN.json`의24개 항목을 실제 재대조해22일치·2불일치를 확인했다. 실행 후 원시 변조가 아니라 **미실행 후보가 검수 중 바뀐 사건**이다. 이 상태를 immutable로 인수하지 않는다.

| 파일 | pin | 실제 |
|---|---|---|
| src/bridge_numerics.py | f1d1ec4facbb2648575f6ece4f1a0d9b6a0f4b38b3315a9440d4dae7248119a4 | 01e55cbee1954179e15bb318be5c2a69e92686f0b93ac18d9233bb87fa01f4af |
| src/run_production.py | 88c2341a4ac6aac2aa3c00ecabfb355e2672d77a0ba984293720ce76e113d2f7 | 5ad5992656d8fbd62f7d54baa18466ece1d69501c949608885f755a98e2eaac3 |

root가 읽은 세 파일 pin: camera `600aef751a0d5abd2ce6c665e8ec42a2b3f7f28a53b5b626aecd9a5981f8a33c`; renderer `e289181e85c7bdf85fb0326c46fd2271f92d57e64b9c1ebcf438c6de4e08910f`; launcher `afce1a3863826c7dedcd30ab13b5e48acc9c7be6e7bebf32c708d047b63d4564`.

## 다음 판정

기존 후보를 되돌리거나 덮지 않는다. 보존 이력을 기록한 새 revision에서 위 연결과 기존 실행기 마감 결함을 닫고, GPU 없이 **실제 런처→실제 실행기→가짜 렌더 작업** 연결 대조를 수행한다. 단순 AST 존재/plan-only13PASS로 이 연결을 대체하지 않는다. 그 뒤 정확 argv/입력해시/새 출력 경로의 제한적 Isaac 준비 GO를 검토한다. 생산 물리 GO는 계속0이다.

## 16:10 공식/설치 카메라 규격 대조 — 계산 비율 확인

설치 IsaacLab2.3.0의 `.../isaaclab/sim/spawners/sensors/sensors_cfg.py:63`은 horizontal_aperture 기본20.955, `:72`는 vertical_aperture 기본None이다. `.../isaaclab/sensors/camera/camera.py:130`~131은 None일 때 horizontal×height/width를 실제 대입한다. 따라서 이번 시야 계산에 쓴 가로 구경 값과 화면 종횡비 처리는 설치 코드와 일치한다. 이 값은 카메라 기본값/프로젝트 설정이지 엔진 hard limit가 아니다. focal20은 프로젝트가 정한 표시 설정이며 공식기본24와 구별한다.

공식 문서/소스: **isaaclab.sim.spawners.sensors.sensors_cfg — Isaac Lab2.3.0**, https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/spawners/sensors/sensors_cfg.html ; **isaaclab.sensors.camera.camera — Isaac Lab2.3.0**, https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sensors/camera/camera.html . 로컬 공통 prefix는 `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/`다.

단위 주의: 해당 설정 소스의 focal/horizontal 설명은 cm, vertical 설명은 mm로 서로 다르게 적혀 있다. 프로젝트 helper의 `_mm` 필드명을 실측 렌즈 mm로 인용하지 않는다. FOV 계산은 동일 authoring 단위의 aperture/focal **비율**이므로 이 이름 문제로 임의10배 변환/렌더값 변경을 하지 않는다. 이번 대조는 점 집합 누락이나 실제 구도 검수 문제를 대신 해결하지 않는다.

## 16:38 연결 검토 후속 — 이전 판정 덮어쓰기 금지

root가 생산 `verify_resume.py`의 preflight 부분을 끝까지 읽었다. 현재 검사는 `readiness/INSPECTION.json`과 한 단계 glob `*/render_manifest.json`을 읽어 새 `*/render/render_manifest.json`을 선택하지 못한다. 또한 일반 `MANIFEST_prospective.json`을 읽고, 선택된 revision의 COMMANDS가 지정하는 manifest는 읽지 않는다. 이 상태의63/64 보고를 새 후보의 실제 준비완료 증거로 인수하지 않는다.

생산에 요청한 범위(msg_75d2624a767b)는 새 버전 검사기/명시적 CLI 경로 선택뿐이다. 옛 실패 INSPECTION과 원자료는 그대로 두고, 새 root 육안 검수·render manifest·launcher/runner 영수증 및 선택된 revision manifest를 읽는다. 초기화 단계의 Isaac 물리스텝과 표시 프레임 중 시간 비진행을 구별하고, 종료 포함 runner 전체 소요를 렌더 구간 시간으로 대체하지 않는다. 새 준비 실행 전에는 해당 게이트가 계속 실패해야 한다. 새로운 과학 임계나 재실행 허가가 아니다.

같은 시각 기존508파일/HEAD의 승인된 보존 검사 root:G1 재실행PASS. 검사 프로세스exit0/EXPECT일치/output SHA b4898deb232ce0dc4ba60f530be29db6c5df556b484a175807442690bc291ccb. 전체 게이트 도구 exit1은 나머지3개 수동 완료조건 미달이며 보존 실패가 아니다.
