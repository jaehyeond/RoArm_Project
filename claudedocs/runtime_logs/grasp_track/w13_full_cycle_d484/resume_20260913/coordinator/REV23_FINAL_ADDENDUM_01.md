# rev23 actual13 — 실제 준비 실행 최종 인수와 남은 실패

2026-09-13 19:12 KST. 새 DEME 물리0. 이번 case 신규 변수 []이며 표시/종료 절차만 검증했다.

1. GO_isaac_readiness_rev23_01.md의 정확한 한정 명령을 생산 worktree에서 1회 실행했다. 외부 시작09:57:15Z, 실행기 단계09:57:16Z~09:59:58Z. launcher2387684/runner2387720/renderer2387721, 두 PGID2387684/2387721. root 호스트 정확 ID 필터에서 전부 부재 확인(19:04 KST). 수동 신호0/자동 재시도0.
2. 8 PNG를 root가 전부 실제 열었다. 시각 식별만 인수한 ROOT_READINESS_INSPECTION_04.json SHA b29b3b169370ab57446d8f4d69f388c1a97ea8a7b8c9fc8b9eff0e56bb598ca4가 정본이다. 그림에 있는 면의 전체 가시성이 아니라 바닥 높이/네 벽의 범위를 경계선과 설명으로 식별한 것이다. 색 분리 한계·부분 가림·상면 좌측 어두운 직사각형의 미확정 원인은 남으며 재실행용 미관 기준으로 추가하지 않는다.
3. root PNG 헤더8개/ffprobe: 모두1600x2102, MP4 8프레임·10fps·.8초·329555bytes. 원자료2행은 모두t0/sync0, 합성6은 입자 원행1복사와 선언된 자세이며 실제 운반/배출 결과가 아니다. 표시8회 시계 .08333333767950535 불변, 초기 warmup8은 별도다.
4. 진단 파일의 6 API 호출은 모두 반환했고 1 app update 뒤 실제 STOPPED를 관측했다. 전체0.0413초/stop .0008초. 설치 Kit 로그3983/3985행은09:57:33Z에 close 진입/Replicator shutdown finished,4026~4031행은09:59:58Z 외부TERM 뒤 재진입/Stage could not be closed/프레임워크 종료를 기록했다. 이번 지연은 그 사이 stage close 구간으로 좁혀지지만 정확한 native 함수 원인은 미확정이다. 옛 rev20나 모든 Replicator 연관 경로를 배제하는 일반 결론은 철회한다.
5. 원본 stderr의 `ValueError: Invalid object in Py_Graph in getWrappedGraphFromNode`를 root rg -c로 직접 재계수해267806회를 확인했다. 크기109800376bytes, SHA ef0a49135b1de4c2f219bf4a919cfecfa537998d95dba4eb8e4bb54889676c70. Kit 로그SHA31d1ab0517db3e079bfd5389d43ab5fd72eaa9b44ba961f0e6d5f8543cca7fa4. 둘 다 보존한다.
6. 최종 stage162.652548초≤180, TERM162초, KILL 불필요. childrc0이지만 timeout=true여서 runner/launcher124, 전체 그룹 부재/active[]/READINESS.ok=false. close_s/cleanup_total_s는 미측정이다. 별도 파일에 존재하는 진단 시간조차 manifest에 복사하지 않은 renderer1390~1435의 결함은 생산 보고 뒤 root도 소스에서 확인했다. 미완료 cleanup_total과 이미 측정된 진단의 저장 누락을 구분한다.
7. root 명시 경로 v2 preflight는64/66·rc1/W13R_PREFLIGHT_V2_FAIL. 시각 부분은 통과하나 close 시간 및 launcher 성공이 실패했다. 04의 READINESS_PASSED 키는 시각 입력 전용이고 전체 준비 성공/production GO가 아니다.
8. 독립 REV23_READINESS_ACTUAL_AUDIT_01.json SHA8ab7623769a9c56996f0f805bfd6279723cf558b81a2c25600a0c525816afcb8은13/13 실패 상태·매핑 검증 통과, 실제 준비 판정은 READINESS_FAILED_CLOSE_TIMEOUT이다. root는 첫210줄과 전체 의미 필드를 읽고 해시를 재계산했다. 긴 중복 pin 행은 표로 요약해 읽었고 핵심 원자료/그림/로그는 별도로 직접 검증했다. root의 첫 화면 축약 함수가 숫자 raw행[0,1]을 hash행처럼 요약한 표시 오류는 원 JSON 그대로 재출력해 바로 정정했으며 감사 원본은 변경하지 않았다.

출력 기준 경로: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_13_actual/`.
최종 영수증/그림6+8 SHA는 ROOT_READINESS_INSPECTION_04.json에 있다.
Kit 로그: `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20260913_185716.log`.

다음 경계: 기존 두 워커의 카메라 출력 자원 소유/공식 정리 순서 CPU 진단 제안만 진행. 코드 변경/추가 GPU 실행은 새 검수 전 금지. 본 실행 GO0, 사용자9시간 비용 승인 유지. 실물/학습/A/B/C/설치/commit/push0.

## 19:28 KST 정정 및 다음 CPU 후보

위 8번의 13/13 및 SHA8ab762…는 감사자가 완료 보고하기 전 root가 읽은 작성 중 사본이다. 최종 보고의 14/14 및 SHA `a8ef9cd522991e024dcdbbca005068969fa43f3dff6c28238b03d85b675fe3a4`를 root가 추가 검사 항목까지 읽고 19:27 재해시했다. 추가 항목은 완료된 진단 시간의 manifest 복사 누락이다. 원본 실행 자료 변경이나 새로운 준비 PASS가 아니다. 최종 verdict는 그대로 READINESS_FAILED_CLOSE_TIMEOUT이다.

설치 `isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py:235`, `:638`, `:960`과 [Isaac Lab 2.3 공식 소스](https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/simulation_context.html)를 대조했다. STOP 때 다시 playing일 때까지 render를 반복하는 콜백이 있고, `clear_instance()`는 이를 구독 해제한다. 설치 base `isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py:211`도 root가 읽었다. 다른 callback/singleton 및 backend/device를 종료 시 정리하므로 실행 중 호출이 아니라 마지막 표시 뒤에 한정한다. Kit 로그 4019~4025의 onStop 뒤 두 렌더 엔진 재생성은 이 가설과 맞으나 콜백 스택 증명은 아니다.

독립 진단 최종 읽은 사본 `REV23_SYNTHETICDATA_CLOSE_DIAGNOSTIC_01.md` SHA `a94841bc4188c240557c945b427772c33c6fb41abcf5f9b6e089e3b571ff94fa`는 카메라 정리까지 포함한 후보를 제안했다. root는 인과를 좁히기 위해 **rev24에서 clear_instance 선행만** CPU 구현하도록 선택했다(msg_d1e4f0b148ed). 카메라 참조/GC/render-product 재조회·destroy, 전역 texture/graph reset, 명시적 timeline stop은 넣지 않는다. 완료된 PHASE를 pre-close dump에 복사하는 저장 결함도 고친다. 6개 필수 시간 필드, 기존 cleanup/외부 제한, 모든 물리·구도·입력은 유지한다. 별도 새 준비 GO 전 GPU0, production GO0이다.

감사 배정 msg_6b697bd5faaa의 발신 핸들에는 root가 8fcf를 빠뜨리는 오타를 냈다. 같은 내용은 올바른 root 핸들의 msg_c4e5d1816ba6으로 정정·확인했다. 다른 조정자나 새 lifecycle 권한으로 취급하지 않는다.
