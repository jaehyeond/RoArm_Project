# 2026-09-12 W11 계산 간격 비교

이번 case의 신규 변수: [dt].

사용자 승인: W10의 기존 dt=2e-6s와 새 dt=1e-6s 한 셀 비교, 나머지 조건 유지, 실제 시뮬레이션과 원자료/Rerun 검수, 종료 상태/relay 갱신. 실물 조회/구동/카메라·commit/push·A/B/C·학습 금지.

출력: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/` (사용자 지정 날짜 이름 유지; 실제 실행일은 09-12). 사전 기준 `PREREG.md`, 완료 기준 `GATES.md`.

## 부트 복원

AGENTS/START/DECISIONS_ACTIVE/LEDGER_RECENT/CONTINUE 전체, 양쪽 relay, 최신 실물 종료/전체 scoop/영상 W9 W10 검토 세션, 저장 영상 워커 브리핑, closeout BRIEFING·관찰/준비감사·검증 JSON, W10 REPORT·verification·실행기·params와 물리 코드를 확인했다. 최신 상태는 실물 종료·마지막 기록 HOME이며 중간 열린 자세/과거 재부팅 차단을 현재로 쓰지 않는다. 이번 세션 하드웨어 접근 없음.

W9=W8 재생, W10=별도 dt2μs 결과. 연구 진행은 고정 S1·고정 배출 위치에서 취점 선택이며, 현재는 입력 전이의 dt 민감도 확인만 수행. 실물은 새 scoop1회가 중간정지4회 후 끝났고 24g의 컵 포함 여부는 미확정. 순질량을 확정하거나 물성을 맞추지 않는다.

샌드박스 nvidia-smi는 장치 접근 실패, 호스트 읽기에서는 드라이버580.178.04 정상 확인. 이 구분은 재부팅 필요 판정이 아니다. 기존 roarm DEME2.4.0·torch2.7.1·numpy2.2.6·psutil7.2.2, isaaclab numpy1.26.0·psutil5.9.8·Rerun0.34.1 확인. 설치 없음. HEAD는 현재 `3267dcb38369f01bb77b923066a89ca92060691d`로 직전 문서보다 진행했으며 사용자 기존 변경을 보존한다. 상세 해시는 preflight에 저장한다.

실패 가능한 연구 실행은 본 dt 섭동 한 셀이다. unlazy 스킬의 solo 완료 기준을 기록했고 새 subagent/워커는 만들지 않는다.

## 실행 착수와 분석 기준 점검

10:12:55 KST 신규 셀 시작(`launch.json`). 6입력 SHA16 일치, 기존 입력·W10 산출 등158파일 SHA256 보존 목록을 `preflight.json`에 기록. 설정 차이는 timestep_s 및 render_timeline_path 정확히2개. shell/경로/seed/종료가드를 launch로 기록했다. 자동 재시도 없음.

대조군 재계산에서 포획541개·10.959243732020918g·저장 sync 최대5.3291m/s 확인. 원본 초기 더미와 엔진 초기 readback의 좌표는 최대1.4901161193847656e-8m 차이가 있어 완전 일치라는 분석기 가정이 틀렸다. 입력 해시 불일치나 새 물리 실패가 아니다. 초기 readback 오차를 보고값으로 남기고 두 조건의 실제 초기 위치/자세 배열을 직접 완전 일치 대조하도록 분석기를 수정했다. 과학 dt 설정·보호 기준은 그대로다. W11 결과를 보기 전 대조군으로 분석기를 점검한 것이며 사후 수렴 허용값을 추가하지 않았다.

Rerun 변환은 W10의 검증된 로거를 새 W11 파일로 복제하여 명칭/출력만 변경했다. `roarm_rl.viz_debug`를 사용한다. RRD 대조기는 시점·배치 길이·스칼라뿐 아니라 모든 툴 노드/접촉/입자 표시좌표까지 원자료의 Float32 사본과 대조하도록 작성했다. 원자료를 Rerun 값으로 보정하지 않는다.

공식 출처: NVIDIA **nvidia-smi — NVIDIA System Management Interface program**, https://docs.nvidia.com/deploy/nvidia-smi/index.html (현행 공통 CLI 문서, 고정580.178.04판 아님). 적용 버전은 preflight.json의 로컬 드라이버580.178.04·torch CUDA build12.6와 호스트 연산 결과다. `nvidia-smi` 표시 CUDA13.0을 torch 빌드 버전으로 부르지 않는다. DEME 공식 https://github.com/projectchrono/DEM-Engine README는 main만 참고(2.4.0/v2.4.0 URL은 조회 실패), 설치2.4.0 API 보증으로 사용하지 않았다. `/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/DEME.py:3`은 deme 호환 import이며, 실제 동작은 보존된 프로젝트 코드 `sim_deme_scoop_s1.py:454`의 시간 간격 설정과 실행 자료로 판단한다. Rerun timelines 공식 https://rerun.io/docs/concepts/logging-and-ingestion/timelines 는 웹0.37.2라 설치0.34.1보다 최신; 로컬0.34.1 로거/검증 API를 사용한다. 신규 NVIDIA 물리 한계 주장은 하지 않는다.

앵커 정정: 시간 간격 설정 호출의 실제 줄은 `sim_deme_scoop_s1.py:459`(`SetInitTimeStep`)이다. 위 :454는 속도 사전의 위치였으며 원 코드 변경은 없다.

RRD 대조기 첫 실행에서 NPZ 필드를 각 프레임마다 다시 읽어 전체 배열을 반복 해제하는 후처리 비용을 발견했다. 해당 읽기 전용 검증만 중단(exit130), 배열을 한 번 읽는 방식으로 고친 뒤 정상 W10 전체14엔티티·2774sync·입자64프레임의 원시 좌표/스칼라 대조PASS. 다음에는 기대 고정부 좌표를 메모리에서1mm 이동한 음성 대조를 넣어 고정부 엔티티 하나만 FAIL함을 확인했다. `rrd_baseline_positive.json`, `rrd_baseline_negative.json`, `rrd_controls.json`. 물리 실행은 중단/수정하지 않았고 원자료 파일 변조는 없다. 실행 중 CPU 분석이 병행됐으므로 W10/W11 벽시계 비는 통제된 성능 벤치마크가 아니다.

## 실제 실행 종료와 원자료 대조

10:58:25.240669 KST rc0 종료. `terminal.json`: 총2730.058862522초, guard_stop=null. 자동 재시도0·추가 물리 셀0. 원래 물리 루프2720.74초, 저장2771sync·실제3.141871초·입자64프레임. settle25→descend351→close2072→lift137→reclose186을 거쳤다(원시 rows는 마지막 lift 후처리3행을 포함). 두 조건의 초기 clump/tool/door 위치와 quaternion 6배열 완전 일치. 실제 dt 필드 외에는 출력 경로만 다르다.

포획을 동일 공동 기하식으로 독립 재계산했다. W10 541개/10.959243732020918g, W11 **517개/10.47306656091463g(반올림10.4731g)**. 차이−24개·−0.486177171106g·−4.436229%. capture mask/입자ID/템플릿질량/JSON 반올림 일치. 이는 공구 내부 포획이지 컵 배출량이 아니다.

첫 닫힘 정지: W10 명목3.307/실제3.235°, M1.783022N·m; W11 명목3.494/실제3.390°, M1.765517N·m. 모두 servo_stall(기존1.764N·m 선). 재닫기: W10 명목3.069/실제2.993°, M1.765356N·m·단일2.5997N로servo_stall; W11 명목3.076/실제2.954°, M1.405012N·m·단일3.0089N로**pinch_guard**(기존3N). 실제 메시 맞춤 각도는 각각3.234891/3.390305 및2.992974/2.954386°로 quaternion 반올림과 일치했다. W11 최종 서보 환산은 명목5.576/실제5.454°, 립 등가6.165mm·물림진단5개(W10 6.152mm·0개). 보호코드/수치는 변경하지 않았으며 실제 파손 관측으로 해석하지 않는다.

전체 저장 sync 최대속도5.3291→2.7255m/s, 5m/s초과1→0. 완주+최대<5+초과0이라는 사전 경고 개선 조건PASS. 다만 W11 재닫기2m/s 속도/2N 접촉 이벤트는 남는다. 최대속도 시점은 W10 close2.529456s, W11 reclose3.119970s. 모든 내부 dt를 저장한 속도 상한은 아니다. 두 실행의 stderr에는 기존 matplotlib 한글 폰트 경고가 남는다.

높이맵은 원래5mm격자44×62에서 재계산. 초기 지형차이0, 재안착pre 전체MAE0.000258839mm/최대0.109184533mm. 종료post 전체MAE0.303284404/RMSE2.445510459mm, 반경80mm(812셀) MAE0.933979490/RMSE4.472167191mm, 최대49.162920564mm·5mm초과9셀. 반경80mm의pre−post 변화량 차이MAE0.934041953mm. 양의 높이감소체적53.961871425→53.428138117cm³. 이 post는 높이로carried를 제외한 동적 마지막 스냅샷이며 재정착/안식각 정본이 아니다. 평균만 보고 지형 동등성 판정 금지. 원시 대조 전체 `comparison.json.evidence_pass=true`.

## Rerun 완결과 실제 육안 검수

`cell_dt1e6_seed460/scoop_s1_seed460_w11.rrd`(145,550,081B), `.rbl`, `_rerun_validation.json`, `_coverage.json`. SDK/CLI0.34.1·save-only·첫로그전sink·종료/flush후footer PASS, 정확 엔티티/타임라인/필수컴포넌트 PASS. 전체2771sync 툴노드/접촉점/힘벡터/9수치와64프레임의구전개입자를 RRD 읽기로 저장정밀도에서 정확히 대조했다. 원자료→표시사본 방향만 사용.

실제로 `view_image`로 연 그림과 관찰(공통 출력 폴더 기준):

- `cell_dt1e6_seed460/scoop_s1_seed460_w11_inspection.png`: 중앙 최종 공구 내부 주황 입자와 아래 더미의 분리, 오른쪽 첫닫힘 접촉점/고정부·문 노드, 아래 높이맵 확인. 왼쪽은초기0s커서, 중앙최종/오른쪽첫정지는정적사본이라동일시간아님. 입자색은최종분류고정. 초기이벤트창빈상태를원인판정에쓰지않음. 재닫기별도정적화면을봤다고주장하지않음.
- `cell_dt1e6_seed460/scoop_s1_seed460_w11_decision_frames.png`: frame2447 명목/실제축이거의겹치고라벨중첩. 0.104°차이는PNG축척으로확정못함,원시수치로대조.
- `heightmap_comparison.png`: pre거의겹침, post취점왼쪽고립된높은셀위치차이. 차이가국소집중되어평균만해석하지않음. pre/post차이색범위다름.
- `timeline_comparison.png`: 문각큰추세는겹치지만정지이후토크다름. W10 close속도5m/s선초과, W11재닫기끝피크잔존. 선은저장시점연결.

4이미지 경로/SHA256/관찰·한계 정본 `inspection.json`. PNG생성만으로검수처리하지않았다. RRD/RBL/모든검증물은신규run폴더안,대형영상없음.

## 최종 해석 / 종료 상태 인계

판정 `W11_DT1E6_COMPLETE__SAMPLED_SPEED_WARNING_REDUCED__RECLOSE_PINCH_GUARD__CONVERGENCE_UNPROVEN`: **계산간격을절반으로한실행은완료됐고저장속도경고는줄었지만,포획과재닫기정지원인이달라졌다.** 두번토크정지유지false. 한쌍만으로실행간변동과dt효과를분리할수없고,동등성/수렴/실물정합을선언하지않는다. 시간비1.454241은관측값이며통제된벤치마크아님. raw non_claims/상속_note의기본설명은실제params/별도Rerun완료를대체하지않는다.

START를W11완료로갱신,6열원장행을끝에append,LEDGER_RECENT를실제최근20행(:536~544,:582~592)으로압축해낡은중간자세표현은당시상태라고구분,from_codex §2덮어쓰기. 기존연구/부트문서를삭제하지않음. **DECISIONS/DECISIONS_ACTIVE append0**: 이번은사전기준에따른dt섭동관찰이며새영구규칙/보호선변경이없어서D484유지. 원장과DECISIONS의기존prefix바이트보존은최종G5에서검사한다. 기존BACKLOG/CONTINUE/실물종료세션은이번턴무수정.

다음행동은사용자결과검토. 현재승인한한셀을넘겨추가dt/반복/물성/A잔류허용/B추가제거/C통합배출/학습을실행하지않는다. 하드웨어조회/구동/카메라0,commit/push0. 최신START→본세션/REPORT→raw로재개하며이전CONTINUE프롬프트를반복실행하지않는다.

## 최종 완료 기준 재검증

unlazy 완료검사 첫 샌드박스 실행은 자식 CHECK가 rc0인데 출력 없음으로 G1/G2/G3/G5의 EXPECT가 맞지 않아 **4개 미충족**으로 처리했다. 같은 closeout를 직접 Python으로 호출하면 통과했고, 무해한 중첩 detached Node 문자열 출력도 샌드박스에서 사라지는 현상을 확인했다. 기존 스킬/물리 코드는 수정하지 않고 호스트에서 동일 선언 명령을 다시 실행했다. 환경 차이를 확인한 것이며 OS 내부 원인을 완전히 규명한 것은 아니다.

호스트 재검사: **G1/G2/G3/G5 모두 rc0+EXPECT일치**, G4는 위 PNG4장 실제 관찰 근거. **충족5 / 미충족0 / 포기0**. `GATES.md`에 cwd/shell/출력SHA256 기록. 원시 수치 재계산 일치,158개 기존파일 크기/SHA256 불변, DECISIONS/LEDGER 기존prefix 바이트 불변, HEAD불변, isaaclab 핀 유지, `git diff --check` PASS. 이 실행은 저장 원자료 검사뿐이고 새 물리/하드웨어 재시도0이다. 수치 정본 REPORT/comparison, 최종 파일 목록과 지문은 `manifest.json`으로 마감한다.
