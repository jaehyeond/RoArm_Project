# rev19 변경 소스 root 검토 — 2026-09-13 17:26 KST

판정: **준비 실행 NO-GO**. GPU/DEME 실행 없이 실제 mini 사본의 새 모듈 누락을 검출했다.

읽은 파일: `rev19/src/arm_link_bounds.py` 전체, rev17→rev19 `isaac_replay_w13.py` 및 `readiness_launcher.py` diff 전체, 런처 `REVISION_SRC`와 실제 복사 루프, `READINESS_CANDIDATE_08.json` 전체.

직접 재계산 SHA256:

- arm_link_bounds.py: `0ba2949f0e6e0055993e5be6810d8a80c5a36a63bd48fe7181f9b2d1947863f7`
- isaac_replay_w13.py: `c510628118a75274cebb58e78d61c20903cda02a93a79ca47398b1c13e0f5090`
- readiness_launcher.py: `290f3c2e3cb1649e0238d099bec3b5a3bbe784d3d46c81299b8a93fe5e0c51d8`

표시 변경은 canonical tool-relative 문/셸 변환, 기존 URDF/STL local bounds와 exact XML joint transform, source floor/벽의 색·라벨, 공식 close 대기 옵션, close 필드 누락의 null 표기다. 물리/FK/criteria 변경은 이 diff에 없다. 실제 화면/close 해결 여부는 아직 미검증이며 코드 변경만으로 준비 PASS를 선언하지 않는다.

실행 연결 결함: 런처59~60행의 `REVISION_SRC` 6모듈 목록에는 신규 `arm_link_bounds.py`가 없다. 복사된 렌더러40행은 해당 모듈을 import한다. outer rev19의 26파일 핀과 실제 mini의 8파일 자기일치는 누락을 검증하지 않는다.

실제 CPU 재현(CWD `/tmp`, 원 후보 폴더/PYTHONPATH fallback 없이 `-I -B`):

```text
/home/cgxr/miniconda3/envs/roarm/bin/python -I -B -c 'import sys; sys.path.insert(0, "/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_09_planned/plan/revision/src"); import arm_link_bounds'
```

rc1, `ModuleNotFoundError: No module named 'arm_link_bounds'`. renderer/Isaac를 실행하지 않았고 새 파일도 쓰지 않았다. 독립 감사도 msg_5fd3777472a6에서 목록·실제 mini 핀·파일 부재를 확인했다.

최소 후속 승인: msg_fade6d8ef8c1로 새rev20의 복사 목록1항목 및 자기경로/메타데이터만 수정하도록 배정. 원rev19/09_planned 보존, 핵심 renderer/helper 동일 SHA는 검수 재사용. 새 plan-only 실제 mini에서 격리 import 및 asset 읽기 양성 대조, 전체 local-import 포함을 확인한다. 원래 runner/guard 전체시험 재실행이나 GPU 실행 허가는 아니다.
