# 2026-09-14 설명 정리·새 세션 인계·Git 게시

이번 case의 신규 변수: []. 사용자 요청은 Downloads Markdown 작성, 다음 세션 프롬프트, main/Orca 작업의 검토·commit/push다. 이번 요청이 앞선 commit/push 금지를 이 게시 작업 범위에서 명시적으로 해제했다. 하드웨어·새 연구 실행의 승인은 아니다.

## 진행과 범위

`unlazy` Solo 완료 게이트를 먼저 작성했다. 읽기 순서는 현 상태/relay, W13 실행 영수증과 독립/root 감사, 기존 Git 게시 규약, worktree 변경과 원격이다. 별도 작업자를 새로 실행하지 않고 같은 세션에서 원자료/문서/인덱스/원격이라는 독립 층을 교차 검사한다.

새 Markdown은 최근 질문인 W13 결과·준비와 물리 시간·병목 후보·반복 비용·Isaac Lab 학습 구조를 정리한다. 새 continuation은 원자료 규약 수정과 CPU 검수부터 시작하도록 하며, 이후 재생 수정·운반 원인·성능 측정을 분리한다. 프롬프트를 새 세션에 전달하는 것이 그 새 작업의 요청이며 지금 연구 case를 실행한 것은 아니다.

문서/근거 파일은 `claudedocs/research/closeout_20260914/`, 게이트는 `.unlazy/research_closeout_20260914/GATES.md`. Downloads에는 같은 이름의 출력용 Markdown을 새 파일로 복사하며 기존 세 문서와 PPT를 덮어쓰지 않는다.

## Git 사전 교차 검사

원격은 기존 `git@github.com:jaehyeond/RoArm_Project.git`, 시작 master/원격 master 모두 `3267dcb38369f01bb77b923066a89ca92060691d`다. worktree10개 중 main과6개 worker에 미커밋 결과가 있다. 나머지 decision-oracle/deme-force/kinect-hm은 clean이고 해당 HEAD가 기존 master에 포함돼 있다. 작업 종료 accounting과 두 차례 파일 스냅샷을 대조한다. 열린 AI 터미널을 강제 종료하지 않는다.

worker6개의 각 branch를 그대로 게시한다. 특히 pellet-model의 옛 BACKLOG/코드 변경은 그 branch의 이력으로 보존할 뿐 main 상태에 merge하지 않는다. 기존 main의 BACKLOG/원장/최근원장/세션 변경도 유실 없이 게시한다. 세 `.bak` 사본과 캐시는 제외·로컬 보존하며 삭제0. 각 scoped 실험의 NPZ/PNG/MP4/RRD/관련 로그는 가려진 무시 규칙을 확인한 뒤 명시 경로로 stage하고 필요한 파일만 LFS 속성을 붙인다.

게시 전 스캐너는 양성/음성 대조를 통과했다. 강화된 패턴 후보5개는 새 스캐너의 example.invalid 테스트 URL1개와 기존 감사 결과의 대문자 성공표식4개로 확인했다. 이외 알려진 형식의 비밀키/토큰 후보는 없었다. 이는 패턴 검사와 문맥 검토이지 모든 가능한 비밀정보가 없다는 수학적 증명은 아니다. 새 credential 파일·환경/SSH 설정은 읽거나 게시하지 않는다.

## 연구·시각화 규칙 적용

이번 요청은 문서 전달과 Git 게시다. 실패 가능 실험/학습을 새로 하지 않는 사유는 사용자가 연구 작업을 새 세션에서 하겠다고 명시했고 이번에는 그 준비·게시만 요청했기 때문이다. 기존 파일의 시간·해시·스키마 검사를 수행하지만 새 기하/물리 판정을 만들지 않는다. 새 RRD는 순수 문서/파일/해시 감사라 생략한다. 과거 RRD·이미지·실패 판정은 보존한다.

설치 IsaacLab2.3.0의 DirectRLEnv 코드와 같은 버전 NVIDIA 공식 소스를 대조해 행동→물리→보상/관측의 의미를 설명했다. `num_envs`가 외부 DEME를 자동 복제한다는 주장은 하지 않는다. 온라인 DEME main과 설치2.4.0의 차이도 출력용 문서에 표시했다. 새 코드/물성/dt/보호선/학습/하드웨어/설치 변경0.

## 결과 기록

세부 게시 파일·브랜치·제외 목록과 검수 결과는 `research/closeout_20260914/GIT_PUBLICATION.md` 및 `PUBLICATION_MANIFEST.json`에서 관리한다. 이 절은 실제 전달/게시 후 append한다.

### 22:24 KST 종료 검수와 남은 게시

Downloads 새문서는181줄·21,977bytes·SHA256 `c5bb9b81159784ff5a3cd0a2f984ef59414f381af3807df971f4c399ef0b296c`, repo사본과바이트일치다. 설명로컬링크13/continuation10실재, W13원시NPZ/JSON·부분영상·원본PPT4해시유지. 숫자검사에서16,304sync/283PF·24.486802938176766초·파생24,486,803스텝/10회3.61328일·40회14.45313일을원자료로재계산했다. 새로운성능실험이아니다.

main자료커밋672fab2 및worker6개커밋완료. 3,394파일의stage바이트대조불일치0, Python961구문오류0, LFS포인터1,016/고유982·약3.32GB. 7branch의로컬LFS fsck PASS. 원본과모든동결실패자료변경0이며새LFS속성만붙였다.

원격push는미완료다. 최초worker6branch atomic push는LFS전송중SSH연결이종료돼최종rc141. 메인push명령은목적지/전송범위추가확인을요구하는자동보안검토거부로실행되지않았다. 사용자가원래commit/push를요청했다는사실은보존하되검토거부를우회하지않았고, 정확한공개origin과원자료/영상범위를질문했다. 현재답전재시도하지않는다. 사후원격master3267dcb·대상workerref없음을확인했으므로백업완료라고보고하지않는다.

게이트초기실행은CWD상대경로해석오류로모듈찾기FAIL, 절대repo경로로정정후delivery/continuation/numbers PASS. publication은실제Remote mismatch master를검출했다. 원격게시미완료는사용자승인후이어갈명시인계로남긴다. 종료상태/이보고는후속로컬문서커밋으로보존하며승인전새push없음. 원장/DECISIONS의새실험추가0, 상태소유권은종료후다음세션에인계한다.
