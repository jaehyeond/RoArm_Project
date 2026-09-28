# 2026-09-16 교수 질의 소스 감사 / LFS 로컬 보류 / 재개 안내

이번 case의 신규 변수: []. 사용자는 dt1/2µs 비교의 의미와10µs/밀리초 제안, 원래다음계획, 실제물성·계산식·7구모델·Isaac렌더를 상세히 설명하고, LFS를ignore하여나중명단을남기며새세션프롬프트를요청했다. 실제새연구작업은새세션에서한다.

## 범위와 부팅

- AGENTS/START/ACTIVE/RECENT/relay와 원래9/14 continuation, 연결된 W13부분결과·이전 physics/render 보고를 읽었다. 옛9/11 reboot차단/중간로봇자세/GO를 현재로 쓰지 않았다.
- unlazy Solo: 완료조건5개를 `.unlazy/professor_review_20260916/GATES.md`에 먼저 작성했다. 새worker를 실행하지 않았고 기존독립감사·이번별도산술·공식/설치소스대조로교차검토했다.
- 새실험0: 사용자질의·문서/Git보관설정만현재요청, 실제실험은새세션이라는명시범위가 Session progress 예외사유다. 학습/perturbation을 임의실행하지않았다.
- 새RRD/기하렌더0: 원본파일/스키마/파라미터산술감사이며 새공간·궤적판정은내리지않았다. 과거영상전체시청/새시각검수완료라고하지않는다.
- 실물조회·구동·PID/토크·카메라수집·DEME/Isaac실행·패키지설치·commit/push·과거커밋재작성0.

## 관찰과 검증 순서

1. 원래첫case는 W13원자료판정2결함(phase-only11/기록25,source바닥최하단식)의 새revision수정+CPU회귀임을9/14continuation§4에서확인했다. 그후재생3결함→운반원인→성능계측을분리한다.
2. 과거10µs cell_DE_c 설정/타임라인/실행로그/엔진stderr를대조했다. 저장t2.56118s/max13.0303m/s,엔진오류22,291.97m/s>10,000,rc134,완결resultJSON없음. 10µs가처음시도되는조건이아니었다.
3. canonicalNPZ SHA659d6b0…8812를다시읽었다. 20,000clumpID각7행=140,000구,실제폭3.601263379mm대목표3.8(-5.229911%),질량20.257382129mg과MOI3성분을독립산술로재계산해W10/W11/W13JSONmirror대조했다. 기존MonteCarlo적분을다시실행한것은아니다.
4. W10/W11 effective params diff는 timestep_s/render_timeline_path뿐. 기존541/517알질량도템플릿질량으로재계산했다. 수렴/실물정합으로승격하지않았다.
5. DEME2.4.0설치API/Models/force/helper/integration8개를공식고정commit12f13cb15805d891eddc7b5b545d1f6823f523d7과SHA대조해모두일치했다. Hertz·접선이력·Coulomb·Crr·τ/I·기본EXTENDED_TAYLOR를설명하고실물보정/일반Euler자이로항의검증한계를분리했다.
6. 독립DEME_SOURCE_BOUND의누적시간규칙을CPU재계산했다. 0.1ms요청은dt1/2/10/100/1000µs에서약101/102/110/200/1000µs가된다. 새GPU실측이아닌소스기반산술이다.1ms비교가실제제어간격까지바꾸는점을후속설계에명시했다.
7. W12/W13실제renderer/manifest를읽었다. 한알7구294정점/560삼각면의표시prototype,개별20,000instance,색분류·카메라·저장주기·W12scene step/W13display render-only를대조했다. 카메라focal_mm명칭과USDauthoring단위혼동을추가정정:숫자40/20·aperture20.955,센서fx1954.665039px를기록하고실제40mm렌즈라고하지않았다.
8. Git의기존LFS명단만exactpath로ignore/index제외,전수원본해시보존·index범위·HEAD대조를수행했다. 아래는파일보관설정이며원격백업완료아님.

## Git 변경 정확한 범위

| worktree | branch | index에서만 제외한 경로 |
|---|---|---:|
| main | master | 24 |
| pellet-model | jaehyeond/pellet-model | 109 |
| w12-isaac-replay | jaehyeond/w12-isaac-replay | 481 |
| w13-cycle-audit | jaehyeond/w13-cycle-audit | 17 |
| w13-full-cycle | jaehyeond/w13-full-cycle | 385 |

합계1016경로/982고유객체/3,316,053,800bytes(3.0883158GiB). 실제작업복사본에는중복파일도존재하므로고유객체합과디스크파일합은다르다. research-survey/w12-input-audit는선택LFS0이며수정안함. pellet모델의기존3개.bak보존.

- 각5worktree `.gitignore` 끝에정확한선택경로를추가. `.gitattributes` 보존. `git rm --cached -- <exact paths>`만사용,실제파일삭제없음.
- 변경전mainHEAD `fc557db0f1c2f3e27877e119bb3bac6c3f4ca172` 및workerHEAD는 LFS_BEFORE/AFTER에보존·사후불변확인. 과거커밋에LFS포인터가남으므로새ignore/추적제외만으로원격push의업로드필요가사라지지않는다. push보류.
- 기존게시script force-add/stage/push재사용금지. 계정잔여용량과과거일부전송객체수는미확인. 별도공개범위·용량·게시이력전략승인없이는이력수정/forcepush하지않는다.
- Node의git subprocess진단이샌드박스EPERM으로한번막혔다. 동일제한작업을명시한승격검토후 prepare/apply/verify를실행했다. 원격전송거부를우회하지않았다.

## 근거 파일

- [상세 보고](research/professor_review_20260916/REPORT.md): 물리식/매개변수/누락/단위/표시/승인경계.
- [전수 파라미터](research/professor_review_20260916/PARAMETERS_ALL.md), [원자료 대조](research/professor_review_20260916/PARAMETER_AUDIT.json), [공식8소스 대조](research/professor_review_20260916/OFFICIAL_SOURCE_CROSSCHECK.json).
- `audit_physics.py`는CPU·NumPy/trimesh/AST만사용하며DEME/Isaac모듈import없음. `verify_review.mjs`의physics/git/docs모드로다시대조한다. `defer_lfs.mjs apply`는이미완료한일회작업이라재실행하지않는다.
- [전체 LFS 명단](LFS_DEFERRED_20260916.md), [사전 스냅샷](research/professor_review_20260916/LFS_BEFORE.json), [사후 검사](research/professor_review_20260916/LFS_AFTER.json).
- [새 continuation](CONTINUE_20260916_PHYSICS_AUDIT_DT.md), [이전 원래 계획](CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md).

## 종료 인계 / 기존 실패 보존

W13 TIMEOUT/원자료FAIL2/재생FAIL3와실물종료정본은그대로다. 원자료수정case도교수dt실험도아직시작안함. 새GATES는문서/감사의완료기준이지W13실험합격기준이아니다. START/이새session/relay갱신, BACKLOG에후속후보만append. 기존DECISIONS(D484)/ACTIVE/실험원장/RECENT에는새실험이나지속규칙을꾸며추가하지않았다.

최종검증상태는이문서말미의종료검증기록과 `research/professor_review_20260916/FINAL_VERIFICATION.json`을따른다. 다음세션은9/16continuation요청문으로시작하고메인이상태원장소유권을넘겨받는다.

## 종료 검증 기록

- 완료기준 G1/G2는 원래순서·공식/설치소스·보고내용을 수동검토했고, G3/G4/G5는실행검사로PASS했다. 최초 gate-check 결과 ALL MET(5/5), 추가최종reverify결과는FINAL_VERIFICATION에기록한다.
- G3: CPU NPZ/템플릿·질량/MOI·입자ID·render prototype산술,실제JSON대조,공식8파일hash일치,10µs실패/시간누적/카메라pixel값대조.
- G4: 1016파일원본SHA/size보존,전부index제외/ignore,예상항목외index삭제0,5branchHEAD불변,MD전체1016행대조.
- G5: 문서5개/로컬링크36개실재와현재START/relay/재개승인경계대조. `git diff --check`도오류없음.
- 이PASS는현재질의·보관감사의완료이며W13실험PASS로승격한것이아니다. 새로운물리합격선/속도상한변경0. 물리/렌더원본삭제0·commit/push0.
