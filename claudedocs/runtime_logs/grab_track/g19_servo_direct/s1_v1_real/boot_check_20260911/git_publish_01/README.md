# S1 / W10 게시 검증

대상은 `https://github.com/jaehyeond/RoArm_Project`, branch `master`다. 이 폴더는 게시 시 원본과 Git 내용의 일치 여부를 기록한다. 새790/배출 기울임/반복5회 성공을 주장하는 결과가 아니다. 현재 실행 상태는 `START_HERE.md`를 따른다.

- `index_audit_01.json`: 일반 Git 크기 제한, LFS pointer, 공백 검사 결과.
- `byte_integrity_01.json`: 최초 준비1008개 파일의 원본 SHA256·크기 및 index 일치 여부.
- `manifest_audit_01.json`: 동결 실기/사진/배출 검토 manifest88항목 검증.
- 게시 후 검증은 같은 폴더에 새 파일로 기록한다.

큰 RRD와 W10 canonical NPZ는 Git LFS를 사용한다. 새 checkout에서 원본 재생/분석 전에 `git lfs pull`이 필요하다. Git 웹 화면의 pointer만 다운로드하면 원본 RRD가 아니다. Rerun 시각 증거는 기록된 SDK/CLI0.34.1로 재생한다. 기존 manifest를 LFS pointer의 해시와 비교하지 말고 내려받은 원본 SHA256과 비교한다.

기존 원시 CSV의 CRLF, 동결 소스/OBJ/diff의 공백은 원본 증거이므로 변경하지 않았다. 전체 `git diff --check`에는 이 기존 공백 경고가 남는다. Python 구문 검사 및 장치 없는 문 목표 유지 회귀 검사는 별도로 통과했다.

범위: S1 설계/자산/코드, W10 결과·재생·검증, 실기 PID/900 원시 데이터·사진 계량, 배출 회전 검토, 프로젝트 상태 문서. 무관한 g16/발표자료/print 도구·로컬 백업은 로컬에 보존한다. 기존 W9 MP4와 외부 worktree의 초기 펠릿 더미까지 모두 포함한 배포 패키지는 아니다.

기존 W10 설정에는 `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz` 입력 의존성이 남아 있다. 결과 배열/RRD의 확인과 시뮬레이션 재실행 환경 복원은 구별해야 한다. 이 게시 과정에서 외부 worktree나 원래 입력 경로는 변경하지 않았다.
