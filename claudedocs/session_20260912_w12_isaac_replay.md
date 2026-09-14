# W12 — W10/W11 Isaac replay comparison (in progress)

이번 case의 신규 변수: [] — 기존 dt2μs/dt1μs 저장 결과의 관찰 계층만 추가.

사용자가 W9 방식의 Isaac 장면 재생/비교를 별도 Orca worktree에 맡기고 메인에서 결과를 수신·브리핑하도록 승인했다. 이어 Claude Opus5, Codex gpt5.6 sol high를 명시했다. 새 물리 실험을 실행하지 않는 이유: 이번 범위는 이미 실행된 W10/W11의 재생·검수이며 새 물리/학습은 승인되지 않았다. 실패 가능한 검사는 좌표/시간/입력 동일성 및 재생 검수 게이트이다.

실물 종료 유지. 로봇 조회/구동·카메라 수집·PID/토크·DEME 재실행·학습·A/B/C·commit/push 금지.

## 진행

- 전용 조정 mailbox 생성: term_2805f4ba-1593-4ba2-bf09-be097b6b32ee. 기존 사용자 터미널은 재사용/중단하지 않았다.
- Run run_8e14e889054d. 최초 Run 생성은 sender 미지정으로 실패했으며 작업/워커 생성 전이었다. 정식 전용 터미널 handle로 재시도하여 Run 생성 완료.
- 사전 계약: .unlazy/w12_isaac_replay_20260912/PLAN.md 및 각 GATES.md.
- 조정 출력: claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/coordinator/.
- 요청 모델 식별자: claude-opus-5, gpt-5.6-sol/high. Orca worker-start 영수증 및 실제 워커 반환으로 적용 확인 예정.
- 초기 HEAD 3267dcb38369f01bb77b923066a89ca92060691d. 기존 미커밋 W11/실물종료/상태 변경 보존.

완료 수치/그림 검수/정리 결과는 실행 후 아래에 append한다.

## 워커 기동 영수증

- 두 worker-start rc0, ready/input_accepted. 요청값과 effective 일치, setup skip, residualResources []. 전체 선택필드는 coordinator/launch_receipts.json.
- Claude claude-opus-5: `/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay`; Task task_23902725f61e / Dispatch ctx_dc822d889e06 / terminal term_23c1d985-dceb-48e2-afb2-89081b3993bd.
- Codex gpt-5.6-sol/high: `/home/cgxr/orca/workspaces/RoArm_Project/w12-input-audit`; Task task_e73407b3b659 / Dispatch ctx_80f9c974177e / terminal term_10f08cc7-587b-4a3c-bd09-3a508c20ac8c.
- 독립 두 leaf를 launch wave ready-1에 시작 기록한 뒤 seal. 결과 읽기/대기는 seal 이후. 정식 gate 파일은 `.unlazy/w12_isaac_replay_20260912/gates/leaf-{render,audit}.md`; 앞의 leaf별 GATES 초안은 이동/삭제하지 않고 보존.

## 중간 관측 (최종 검수 전)

- 메인 verify_inputs.py snapshot: W11 기존 보호 입력 대조 PASS 후 210개 파일 기준선 저장. 새 입력/소스만이 아니라 W10/W11 결과·USD도 포함한다.
- Claude 실제 응답 transcript의 message.model=claude-opus-5 확인. Codex worker projection/UI= gpt-5.6-sol high, acceptance heartbeat 일치. Claude acceptance도 수신했다. 기존 실행 중인 사용자 워커의 모델/설정은 바꾸지 않았다.
- Codex 11:49:40 KST 중간 감사: 두 particle timeline에 재닫기 phase4 프레임 없음, appended final만 포함. close/reclose 사건 시각과 nearest particle 시각 차이를 표시해야 한다. final captured_ids 고정 색상은 순간 포획량이 아님. 해당 제한을 render Dispatch에 전달했다; 최종 수치는 반환 원자료로 별도 재검수 예정.
- 현재 단일 Isaac 렌더 담당/CPU 감사 담당 두 워커 active. 새 DEME/학습/실물 작업은 없음.

## 메인 1차 재검수 / 한정 보완

- `preservation_checkpoint_01.json`: 210파일/HEAD 불변 PASS.
- W10 stdout 64/64, 최대 립 오차1.873mm, 최종541/541 공동 내. 아직 전체 시각계약 완료 판정 전.
- 메인 실제 열람: `/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/w10/frames/f_00063.png`. 로봇·상자·더미·그랩과 초록/자홍 원자료 메시 진단이 보이고 빈 프레임 아님. 상단 sync dt=0와 EVENT reclose/final은 사건차12ms를 따로 안 읽으면 오독 가능, 아래 captured 색은 최종ID라는 설명이 부족. msg_84bf365200c4로 주석 파생본 보완 요청; 원자료/기존렌더 물리 재실행 불필요.
- Codex audit 원래 완료 msg_98bee219a341 수신, wave leaf-audit return 기록. REPORT/GATES 및 검증기643줄 전체 검토 후 --check 실행 rc0. 정상 mismatch0/direct acceptance true; W11 재닫기 max_single_contact_N +0.25N 임시사본은 의미값/raw force/hash 불일치로 거부.
- 감사 검증기에서 mutable root gate 파일을 frozen source로 해시한 재실행 의존성 발견. 상태 기록 변경으로 과학 데이터가 안 바뀌어도 future check FAIL할 수 있다. 동일 내용의 불변 사전 계약 초안을 가리키는 새 repair_01 증거를 만들도록 동일 워커에 한정 보완 요청. 원래 감사 산출물 보존, 수치·과학 해시 변경 금지.
- 후속 Task task_76306739043d / Dispatch ctx_3503a5d8fd4f, 같은 Codex terminal/processIncarnation b2fd10c7-3500-49e0-aa84-2e8ab343c855 재사용(rc0 input_accepted). 기존 model/high 유지, 새 agent/model 대체 아님. 원래 완료 Delivery는 이 소유권 이관 후 ack.

## 감사 보완 수락 / 두 렌더 숫자 확인

- msg_769bd1a4daa7 보완 완료 수신. `audit/repair_01/REPORT_w12_audit_repair_01.md`, 원래 코드 대비 diff 전체 읽고 repaired --check 재실행 rc0. 원래8파일·과학 입력29해시·계산 결과 전체 동일. 정상 mismatch0, 새 +0.25N 임시사본은 의미/raw force/hash 모두 거부. 수락 표기할 활성 gate 대신 byte-identical frozen contract를 사용한다.
- 감사 후속 Dispatch ctx_3503a5d8fd4f released / closed_agent_terminal / transcript captured. 원래worktree·증거파일은 보존. Delivery ack 완료. 원래 audit/verify_w12_audit.py는 mutable gate 의존을 가진 역사 시도이므로 향후에는 repair_01 검증기를 사용한다.
- 메인 `verify_replay_numeric.py` 독립 실행 PASS: W10/W11 총128프레임 시각·source quaternion 기반 actual door·명목각·카메라/환경 매핑 동일성·최종 수량 대조. W10 maxlip1.873, W11 1.867mm. 저장 XYZ가 소수5자리여서 norm 재계산 차이 최대0.009165/0.009000mm이며 반올림 오차한계0.018mm 내. 비트 단위 동일성으로 주장하지 않는다.
- 같은 메인 검사에 rendered door +1°를 넣은 메모리 사본은 두 셀 모두 거부. 원본 수정 없음. 최종 포획541/517, 두 입력 same-index nearest pair max3.076999992ms. 두 실제 Isaac 렌더는각64프레임 완료, 메인 시각계약 전체 수락은 아직 후처리/Rerun 대기.

## 검증 실행 환경 / Rerun 중간 검수

- unlazy 중첩 CHECK는 샌드박스에서 rc0인데 출력이 비어 EXPECT 불일치. 과학 검사와 무관한 `/bin/echo W12_CAPTURE_CONTROL`도 동일하게 실패했다. 같은 제어 검사와 같은 감사 검증기를 호스트에서 실행하니 출력/EXPECT PASS. 확인한 것은 실행 환경에 따른 출력 수집 차이이며 커널 원인이나 GPU 고장으로 단정하지 않는다. 도구/EXPECT/과학 기준은 완화하지 않았다.
- 호스트 `gate-check --reverify gates/leaf-audit.md`: 3 met, 재실행1, rc0·EXPECT matched. 출력 SHA256 `851c585be6730b7d1472147a0e424ac30c04c8f84d3228abdc3362c2782e4455`, 5021 bytes. 승인 저장소는 저장소 밖 전용 `/tmp/w12-unlazy-approvals.gR8Haa`.
- 메인 실제 열람 추가: replay `paired/montage_initial.png`, `montage_first_close.png`, `montage_reclose.png`. 초기 배치·로봇/상자·더미가 보이고, 첫 닫힘 몽타주는 W10 frame52 +16.008ms / W11 frame51 -24.947ms를 표기한다. 이는 사건별 nearest frame 비교이지 동시점 짝 영상이 아니다. 재닫기는 두 run 모두 마지막frame63이며 사건후 약12ms다.
- `rerun/trial2/w12_decision_inspection.png`를 메인에서 실제 열었다. 입자와 원자료/렌더 문 노드, 최종 포획 ID 표시, 사건 표, 개수 시계열이 채워져 있다. 전체 로봇 장면은 별도 Isaac 영상의 증거이며 이 RRD 캡처를 전체 로봇 영상으로 부르지 않는다. mm 오차와 개수를 같은 y축에 둔 가독성 한계는 별도 오차 PNG로 보완한다.
- 224.7MB 전체 재생 RRD의 헤드리스 캡처에는 일부 패널 로딩 경쟁이 있다고 워커가 보고했다. 완전히 표시된 7.0MB 결정 전용 RRD 캡처와 전체 재생 RRD의 읽기 검사를 분리한다. 전체 캡처가 완전히 표시되었다고 주장하지 않는다.
- 첫 RRD 색상 결함 시도는 증거 보존 대상. `trial2/`가 수정본이다. 메인은 첫 시도 파일의 원래 경로에 동일 사본 복원, 영상 파생본의 FINAL ID 색상/사건차/잘린 범례 수정, 정확한 정지 순간이라는 표현 수정 요청을 보냈다. 이 문단 시점에 최종 반환/수락은 아직 대기 중이다.

## 메인 RRD 재읽기 / 보고서 반증 검토

- `coordinator/verify_rrd_root.py --report rrd_root_reverification_01.json` rc0. SDK/CLI0.34.1, 두 RRD/두 RBL footer를 다시 검사했고 각 SHA256이 정확 entity/timeline/component 검증 당시 파일과 일치했다. 전체 재생 RRD의30개 entity에서 각64프레임의 시간·배치·값을 원자료로 재계산해 대조. source/rendered 좌표로 오차 스칼라 재계산, 두 RRD의12개 정적 nearest-frame geometry/particle 배열 대조 PASS. W11 최종 노드 한 점 z+5mm 기대값은 같은 검사로 거부. 원본/뷰어/Isaac 재실행0.
- root 검수 기록 `coordinator/inspection_root.json` 추가. `paired/montage_lift.png`, `mapping_error_per_frame.png`도 직접 확인했다. 최대오차는 프로젝트 표시 기준5mm보다 작으며, 이것을 PhysX 한계 또는 dt간 물리 오차라고 부르지 않는다.
- 워커 보고서 초안346줄 전체 읽고 msg_2a9c1de6affc로 한정 수정 요청: 실물 팔 재현 가능성 과장 제거, 포획ID 부분집합의 공동내 잔류 확인과 전체 공동동일성 구분, 정지사건 JSON 반올림/nearest-frame 차이 구분, entity 개수 재계산, lift 최초저장프레임 표시, 과거 명령을 기존경로에 재실행 금지. 보고서의 PASS 선언만으로 수락하지 않았다.
- Rerun CLI의 analytics 초기화가 샌드박스 읽기전용 경로에서 경고를 냈지만 파일 검사들은 rc0로 완료했다. 설정 설치/수정 없이 결과를 분리했다.

## 주석 파생본 검수 / 보조 진단 정정 요청

- `annotated/montage_first_close_annotated.png`, `montage_reclose_annotated.png`, `frames_paired/p_00063.png`를 메인에서 직접 열었다. frame↔sync와 frame↔event, 전후 시점, FINAL ID 고정 색상 및 포획ID 부분집합의 현재 공동내 개수가 명확하다. 글씨는 촘촘해서 원해상도로 읽어야 하지만 범례 잘림은 해소됐다. 원본 영상은 보존하고 사용자 열람은 annotated 파생본을 우선한다.
- `coordinator/verify_media_root.py --report media_root_reverification_01.json` rc0. 원본3+주석3 MP4 모두 디코드64프레임·10fps·6.4초. 주석128프레임의 기존 글자 영역 외 장면 픽셀은 원본과 완전 동일, 짝64장도 pairing.csv의 지정 프레임을 그대로 합성했다. trial1의 RRD/RBL/PNG/validation/coverage 총5파일이 원래경로와 archive 양쪽에 byte-identical하게 존재한다.
- 메인 보고서 검토 중 별도 보조진단 오류 발견: `sim_isaac_replay_w12.py:209`의 `DOOR_R_MAX_M` 계산은 DEME nodes와 표시원점이 더해진 TOOL_W를 혼용한다. 축 회전거리에는 축에 수직인 반경도 써야 한다. 이 값은 실제 렌더/매핑 계산에 쓰이지 않고 보고서의7.7822mm/deg와 파생0.18mm 설명에만 영향을 준다. msg_1369f5b3ef7c로 동결 renderer/gates 보존·새 정정 JSON·보고서 정정만 요청. 실제 매핑 노드 변환은 원점을 올바르게 더하며 전프레임 읽기 검사도 PASS했다.

## 원 재생 반환 수신 / 한정 진단 후속 소유권

- 원 worker_done `msg_6c779fc70138`, Task task_23902725f61e / Dispatch ctx_dc822d889e06 수신. 시간/색상 주석·원경로복원·주요 보고서 과대주장 수정은 수락했다. wave ready-1 COMPLETE2/2. 후속 보조진단 메시지가 원 완료 보고서에 반영되지 않아 해당 부분만 미수락 유지.
- 같은 Claude Opus5 processIncarnation `e617af4b-21ca-405f-afcb-ca25248e7bac` / terminal term_23c1d985-dceb-48e2-afb2-89081b3993bd를 재사용했다. 새 Task task_1770949b6819 / Dispatch ctx_8ba52dad5016, ready/input_accepted, residualResources[]; 모델·worktree 대체 없음. 소유권 이관 뒤 원 완료 Delivery를 ack했다.
- 신규 후속 범위는 `coordinator/REPLAY_DIAGNOSTIC_FIX_BRIEF.md`: 원자료 좌표계로 보조 회전거리 정정 JSON/간단 검사 + worker REPORT만. 추가 Isaac/RRD/입자 계산은 요청하지 않았다. 메인은 이미 검토한 leaf G1-G3를 호스트 unlazy oracle로 재실행하고, G4는 정정 수락까지 남겨 둔다.
- `preservation_final.json`: 기존210개 파일과 시작 HEAD 모두 불변. 메인 상태 문서/산출은 별도 신규 경로에만 기록했고 worker checkout의 git status는 각각 새 W12 출력 폴더뿐이었다.

## 최종 수신·검수·종료

- 진단 후속 수락 heartbeat msg_aa4274ee95ec에서 claude-opus-5/동일worktree 확인. 완료 msg_763c91913620(Task task_1770949b6819 / Dispatch ctx_8ba52dad5016) 수신. 정정 JSON과 읽기전용 검사기 전체를 메인이 읽고 실행해 rc0. 올바른 축수직 반경0.117898378773m,2.057714892354mm/°(두 run 동일), 소스/표시 프레임 계산값 일치. 기존 잘못된7.7822 값은 역사적 메타데이터로 보존, 실제 매핑/과학판정 불변.
- root `diagnostic_root_reverification_01.json`: 원식 재현→원기록 round6/round4 일치, 올바른 계산과는 불일치, 독립 노드변위 역산과 반올림 허용폭 내 일치, renderer/per-run gates 해시불변PASS. 역산 실제값 W10 0.117881709993m/W11 0.117906946314m.
- worker REPORT와 완료 메시지에는 마지막 두 역산값의 전사 오타0.117984/0.117895가 남았다. 늦은 send는 dispatch_inactive로 거부되어 중복 전송/새 워커를 만들지 않았다. 워커가 종료한 뒤 메인이 보고서 한 줄을0.117882/0.117907로 바로잡고 변경 사실을 그 문단에 명시했다. 보고서 수정전 SHA256 `c06ac979a7de44409c45bde9e681ef052112f51f1e424949457a605c4d8f0a65`, 수정후 `dc873d3ba11c2b1deb562c3d1fad670d9d72ab4a5e75a185d59d901d6687947f`. 정정 JSON/검사기/원자료/영상 변경0. 완료 메시지의 prose 오타보다 JSON/검사 결과가 우선한다.
- 최종 render resource ctx_8ba52dad5016 released / closed_agent_terminal / transcript captured 후 완료 Delivery ack. audit resource ctx_3503a5d8fd4f도 앞서 동일 종료. `orca_resources_final.json`: active0, reclaimable0, 실제 released 자원2. 원 Task의 historical retained 행은 후속 소유권 이관 이력이지 남은 worker 프로세스가 아니다. 전용 coordinator shell도 신원을 확인한 뒤 종료(ptyKilled=true), 기존 사용자 터미널 미접촉. worktree/산출 삭제0.
- 메인 수신 보고서 `coordinator/REPORT_w12_received.md`에 절차→정량→근거 경로→판정/승인 경계를 정리했다. START 완료 상태, EXPERIMENT_LEDGER :593 끝 append, LEDGER_RECENT 실제최근20건/82줄, relay/from_codex 갱신. DECISIONS/ACTIVE D484 유지, 새 규칙 추가 없음. BACKLOG/CONTINUE·기존 사용자 변경 보존.
- 과학 결과: 원시541/517개·10.959243732/10.473066561g, 재닫기servo_stall/pinch_guard 차이 그대로. 표시 립최대1.873/1.867mm, 원자료시간64쌍 최대차3.076999992ms. 실제화면검수/원자료오류대조와 전체64프레임 RRD 읽기 검사PASS. 화면으로 재닫기 연속 입자 운동/실물 서보 가능성/동등성/dt수렴을 주장하지 않는다. 추가 물리·실물·학습·A/B/C·commit/push0.
- 재생 leaf의 호스트 재검수: G1/G2/G3 rc0·EXPECT matched, 출력 SHA 각각 `ec5034f3b347f30e7aa1b50e4f1c41c306dd4d66320efef245f06061d07be5de` / `4350976aca5671aa14214e08bc9c0beb74fcbb087c6299d3e2230f96c5bd6425` / `54dc9ac0c49f42b60ebb3f6b26a559af7e49ed1db2d39096d9033b9375a64edd`. G4 작은 정정 검사도 rc0(3792bytes, SHA `29130c3ae2346729aca7210adf1f43b6ba47dbea4ebb96c3fbfe8dfc6a8978e6`), leaf4/4 PASS. 루트 최종 게이트 재실행 결과는 아래에 append한다.

### 종료 검사 결과

- 루트 G2 입력/HEAD 보존과 G3 매체/픽셀/원경로복원, 감사 G3 원시/오류대조를 호스트에서 다시 실행: 전부 rc0·EXPECT matched. 출력 SHA 각각 `caedd2cc9d649905a6bec0dc13a1303d2caae95a2ced025b8bdc062e6d01a645`(25bytes), `aeeab23e318c30ef024a68d4f91732b9e4d7408ad6cac859709a12988627ba57`(81bytes), `851c585be6730b7d1472147a0e424ac30c04c8f84d3228abdc3362c2782e4455`(5021bytes).
- 측정된 전체 gate 합계: **11 met / 0 unmet / 0 abandoned**(root4+audit3+render4). dispatch wave는 별도 **2/2 returned, COMPLETE**이며 gate 수에 더하지 않는다. scope lease 해제. Orca 실제 worker 자원2개 모두 release, active/reclaimable0; coordinator 전용셸 종료.
- `numpy1.26.0`, `psutil5.9.8`, `rerun-sdk0.34.1` 설치 메타데이터 재확인PASS. 설치0. `git diff --check` PASS, 원장593줄/최근색인20건·82줄. 시작/종료 HEAD `3267dcb38369f01bb77b923066a89ca92060691d`; commit/push0.
- unlazy lint 오류0. 사람 검수가 필요한 모델/종료상태·보고서 의미·과학 원자료 검토 게이트의 manual 경고는 실제 기록/재계산으로 보완했으며 자동검사만으로 사진을 검수했다고 주장하지 않는다.
- 최종 판정 **W12_ISAAC_REPLAY_VERIFIED__SOURCE_TIME_PAIRED__PHYSICS_VERDICTS_UNCHANGED**. 작업 완료, 다음 별도case/실물/학습은 새 승인 경계. 두 worktree 산출은 메인 push에 자동 포함되지 않으므로 보존할 것.
