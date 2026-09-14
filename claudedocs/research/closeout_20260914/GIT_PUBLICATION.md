# 2026-09-14 Git 게시 검토와 복원 안내

사용자 명시 요청으로 이번에만 commit/push를 실행한다. 기존 원격 `git@github.com:jaehyeond/RoArm_Project.git`를 사용하며 force push·merge·rebase·reset·파일 삭제는 하지 않는다. 저장소는 기존 공개 프로젝트 목적지다. 이 보고는 연구 성공 판정이 아니라 파일 게시·보존 검수다.

## 브랜치별 소유권

| worktree / branch | 이번 보존 대상 | 커밋 |
|---|---|---|
| main / master | W11 원자료, W12/W13 감독 기록, 실물 종료 관찰, 발표·Downloads 문서, 최신 상태·relay | 672fab2 (자료 커밋; 종료 기록은 후속 커밋) |
| pellet-model / jaehyeond/pellet-model | 측정 펠릿/더미 구현·원자료, 물리 발표 조사 | 26966ea |
| research-survey / jaehyeond/research-survey | 연구 조사와 PPT 구성 브리핑 | 9888de4 |
| w12-input-audit / jaehyeond/w12-input-audit | W12 입력 독립 감사 | aecb468 |
| w12-isaac-replay / jaehyeond/w12-isaac-replay | W12 재생 코드·원자료 연결·영상·검수 화면, 렌더 발표 조사 | d90916e |
| w13-cycle-audit / jaehyeond/w13-cycle-audit | 독립 감사 코드, 음성 대조와 원자료/재생 실패 기록 | fc3d80a |
| w13-full-cycle / jaehyeond/w13-full-cycle | 전체 경로 시도 코드·수정 이력·준비 실패·본 부분 원자료·재생 | 0213da9 |

decision-oracle/deme-force/kinect-hm은 이번 미커밋 변경이 없고 각 HEAD가 시작 master의 조상임을 확인했다. 이미 master 이력에 포함된 내용을 새로운 작업으로 중복 커밋하지 않았다. worktree를 삭제하거나 브랜치를 정리하지 않았다.

worker 브랜치는 master에 병합하지 않았다. 특히 오래된 worker의 START/원장/relay를 최신 상태로 사용하면 안 된다. pellet-model의 BACKLOG 수정은 그 브랜치의 과거 연구 이력으로만 게시하며 메인 상태의 대체가 아니다. 원장 소유권은 메인만 유지한다.

## 실제 검수

1. 모든 worktree의 status/branch/HEAD, 원격 HEAD를 읽었다. 시작 master와 origin/master는 `3267dcb38369f01bb77b923066a89ca92060691d`였다.
2. 일반 untracked뿐 아니라 대상 실험 아래 ignored 파일도 확인했다. 원시 NPZ·영상·검수 PNG가 `.gitignore` 때문에 빠지는 문제를 확인했다.
3. 게시 파일의 크기·SHA256 스냅샷을 만들었다. 각 선택 파일에만 LFS 속성을 추가했고 기존 원본 바이트는 바꾸지 않았다. 후속 재생성 파일을 기존 원자료 대신 넣지 않았다.
4. stage 직전 스냅샷 해시를 다시 비교해 중간 파일 변경을 차단했다. stage 후에는 일반 blob을 원본과 비교하고, LFS 포인터의 OID/size를 원본 SHA256/크기와 대조했다.
5. 새 Python 파일들은 실행하지 않고 `ast.parse`로 구문만 검사했다. 연구 코드의 기능·물리 성공을 이 구문 검사로 인정하지 않는다. worker6개의 stage 3,207파일에서 원본 불일치0·구문 오류0이며 일반 blob 최대598,320bytes다.
6. 공개 게시 전 비밀정보 패턴을 검사했다. 스캐너 양성/음성 대조 PASS. 검출 후보5개는 스캐너 테스트 URL1개와 감사 성공 표식4개임을 코드 문맥까지 확인했다. `.env`, 키, SSH·계정 설정은 게시 범위가 아니다. 패턴 검사만으로 모든 비밀정보 부재를 보장하지 않는다.
7. 각 브랜치를 독립 커밋하고 기존 origin으로 push한다. 완료 여부는 아래 사후 기록과 최종 `git ls-remote --heads origin` 대조로 판단한다. 단순 로컬 커밋을 업로드 완료로 부르지 않는다.

초기 전체 스냅샷 기준 LFS 고유 객체982개, 원본 합계3,316,053,800bytes(약3.32GB)다. 같은 SHA256 원자료의 여러 경로는 LFS에서 같은 객체를 가리킨다. 실제 업로드량은 원격에 이미 있는 객체와 전송 과정에 따라 달라진다. 대상은 W11/W12/W13과 연결된 펠릿·발표 근거이며, 전체 PC 백업은 아니다.

## 포함하지 않은 것과 복원 주의

- pellet-model의 `sim_deme_pile.py.bak_20260910_pre_lens`, `sim_pellet_model.py.bak_20260909_pre_planar`, `sim_pellet_model.py.bak_20260910_pre_lens` 세 사본은 로컬 보존. 삭제/이동하지 않았다.
- Python 캐시·환경·기존 무관한 학습/수집 데이터와 다른 case의 ignored 자산은 포함하지 않았다. 모델/데이터/PC 전체가 백업됐다는 뜻이 아니다.
- Downloads의 원본 PPT는 사용자 PC 파일이며 이번 Git 게시에 새로 넣지 않았다. 출력 문서의 repo 사본은 master에 포함된다. 미디어 복사/PPT 제작은 기존 인계 안내에 따라 다른 PC에서 한다.
- Git clone만으로 이 PC의 `/home/cgxr/orca/...` 절대경로가 만들어지지 않는다. 해당 branch를 별도 worktree로 복원하고 Git LFS 파일을 내려받은 뒤 상대경로·SHA256을 검증해야 한다. LFS 포인터 텍스트를 NPZ로 열면 안 된다.
- 과거 실행 계약/manifest는 당시의 절대경로와 이전 HEAD를 고정한다. 그것을 새 Git 커밋에 맞게 고치거나 오래된 GO를 재실행하지 않는다. 앞으로의 실행은 새 경로·새 pin으로 만든다.

원본별 SHA256/크기/LFS 여부와 branch 매핑은 `PUBLICATION_MANIFEST.json`에 남긴다. 동적인 START/relay/이번 종료 기록은 게시 확인 이후 추가 문서 커밋으로 갱신할 수 있으며, 동결 연구 증거와 구분한다. manifest 자체는 자기 자신의 해시를 담지 않는다.

## 사후 게시 기록

**원격 게시 미완료 — 로컬 커밋 보존, 사용자 확인 대기.**

- 자료 커밋: main `672fab2738cf42f053d0c5e12d5b37fe77953142` 및 위6worker커밋. 전체 stage3,394파일·Python961개구문오류0·원본바이트불일치0·LFS포인터1,016개/고유982개. 일반Git최대blob2,364,241bytes. 7개branch의 `git lfs fsck --objects` 모두PASS.
- 첫 작업branch6개 atomic push는 LFS전송 중 `Connection to github.com closed by remote host.`를 출력했고 최종rc141로종료했다. 전송프로세스종료확인. 일부객체가원격에전송됐을수있지만 객체별최종업로드완료를확인하지못했으며 branch게시성공으로세지않는다.
- 메인 push 요청은 실행 전에 자동보안검토가거부했다. 사유는 기존공개원격과 원자료/영상LFS를포함한payload의구체적인승인확인요구다. 이명령은실행되지않았다. 거부된명령을스크립트나우회경로로실행하지않았다.
- 사용자에게 정확히 `git@github.com:jaehyeond/RoArm_Project.git`의 master 및 해당6workerbranch, 연구코드·상태문서·W11/W12/W13원자료·영상·검수증거(고유LFS약3.32GB) 게시를확인요청했다. 답전새push하지않는다. 알려진민감정보후보5개는검사fixture/성공표식으로판명됐지만자동보안거부를임의로무시하지않는다.
- 22:24KST의사후 `git ls-remote --heads origin`: master는여전히 `3267dcb38369f01bb77b923066a89ca92060691d`, 대상worker6개원격ref없음. 기존 `docs/readme`는변경하지않았다. 따라서 이번작업을원격백업완료로보고하면안된다.
- 명시확인후재시도할때는SSH keepalive를명령범위로적용하는방법을검토하고(전역설정/.ssh편집금지), 이미전송된LFS객체는재사용한다. forcepush없이ref대조와 `verify_closeout.mjs publication`을통과해야게시완료다. 이번실패로시뮬레이션을재실행할이유는없다.

문서검수는 delivery/continuation/numbers PASS다. publication 검사는실제로 `Remote mismatch master`를검출했다. 처음 gate 실행은 CWD를 ledger상대경로로잘못해석해 모듈찾기FAIL이났고, repo절대경로로정정한뒤앞의3검사를통과했다. 미완료게시게이트는PASS로바꾸지않았다.
