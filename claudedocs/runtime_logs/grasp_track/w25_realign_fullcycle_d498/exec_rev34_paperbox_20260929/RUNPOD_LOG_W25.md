# RUNPOD_LOG_W25.md — 2026-09-29 병행(hedge) 실행 로그 (append-only, 메인 Claude Fable 5.1)

규칙: 우리 pod 이름 `roarm_w25_*` 만 생성·조작, 이 파일에 적힌 pod ID 만 stop/terminate 대상. 타인 pod 8개(계정 05:4x 조회: RUNNING 2 = PRO 6000 `g4ra0ol3n2nxno`·`t41m6mb79u45h6`) 불가침.
동결본: `EXEC_PIN.json`(GPU 결과 0 시점), 꾸러미 `bundle/w25_bundle_20260929_0557.tar.gz`(606 파일, 46,047,485 B, sha `c0137a2a759f28decee3f77b08faa5c365b42feeef17ac193e277376ca220b26`).
사전등록 판정(podB GO): `COMMANDS_w25_podB_pro6000x2.json.pre_registered_go_rule` — smoke 3종 rc0 ∧ G0 3/3 ∧ R = settle(wall/물리초)_4090 / settle_PRO6000×2 ≥ 1.3. 결과 본 뒤 변경 금지.

| 시각(KST) | 사건 |
|---|---|
| 05:53 | exec 폴더 생성: rev34 핀 54/54 sha 일치(COPY_RECEIPT), 2차 NPZ sha `31dd2897…` 일치, 새 criteria(cap 115,200/114,000 s) |
| 05:56 | pod 스크립트(bootstrap v5 경로판·run_w25.py·smokes)·COMMANDS podA/podB 작성 |
| 05:57 | 꾸러미 빌드 + EXEC_PIN(77 파일) |
| 05:58 | **podB 생성** `th14bds3fsjg8n` PRO 6000 ×2 SECURE US-CO-1 CUDA 13.2 4.18 $/h (receipts/POD_B_CREATE.json). podA 4090 1차 생성 실패("no instances available", LOW→0) → DC 지정 재시도 |
| 05:59 | **podA 생성** `5pzwyyqhi9f1gd` 4090 ×1 SECURE EUR-IS-1 CUDA 13.0 0.74 $/h (receipts/POD_A_CREATE.json, DC 지정 2차) |
| 06:01 | podB 꾸러미 업로드 sha 3/3 일치(tar·manifest·bootstrap) → 부트스트랩 기동(로그 pod:/workspace/w25_bootstrap/). 드라이버 595.91.07, GPU 2개 인식 |
| 06:02 | podA 꾸러미 업로드 sha 3/3 일치 → 부트스트랩 기동(로그 pod:/workspace/w25_bootstrap/) |
| 06:04 | 세션 문서 §13 append · local_tools/{retrieve_smokes.sh,go_full_cycle.sh} 작성(동결본 밖 보조, 꾸러미 미포함) |
| 06:1x | 독립 감사 A1~A9 PASS/FAIL 0(receipts/AUDIT_exec_frozen_copy_20260929.md), 정오표 5건 기록(핀 파일 무수정) |
| 06:05 | podB BOOTSTRAP_OK(21:01:11Z, 606/606 대조 0 불일치) → smoke 3종 기동 |
| 06:05~06:08 | **smoke 1·2 rc0 양쪽**: podA import300 77 s(sim 74.2 s)·G0 회귀 103 s / podB import300 32 s(sim 28.9 s)·G0 60 s, GPU 2개 92~93 % 동시 사용(DEME nGPUs=2 자동) → Blackwell sm_120 JIT·정적 커널 동작 확인. smoke 3(67.7k settle 0.1 s) 21:08Z 양쪽 기동 — 벤치마크 셀 |
| 06:10 | **podB smoke 3종 rc0**(31/60/102 s). G0 회귀 n_in_cavity **301**(268~362)·10.7788 g·servo_stall·diverged false → **G0 PASS**. settle 67,737알 0.1 s: 25 sync **86.98 s = 869.6 s/물리초**(GPU 2개), 수치 증거 l=4.912e-12 적용·도메인 z 상한 0.563 일치·Initialize OK·abort None. 회수 runs/podB_pro6000x2/(G0_CHECK.json·SETTLE_SPEED.json) |
| 06:05 | podA BOOTSTRAP_OK(06:02:37Z 기준 21:02:37Z, 꾸러미 606/606 대조 0 불일치, DEME 정적 lib·NVRTC sha 로컬 일치) → smoke 3종 기동 |
| 06:15 | **사전등록 판정**: podA settle 2222.1 s/물리초(259 s smoke) · podB 869.6 → **R = 2.556 ≥ 1.3**, smoke rc 0/0/0 양쪽, G0 3/3 양쪽(306·301알) → podA GO + **podB GO(장비 혼합 반복 n=2 라벨)**. receipts/PODB_GO_DECISION.json. 본 실행 기동(러너 cap 115,200 s·grace 1,200·sim --max-wall-s 114,000). 로컬 10분 폴러 local_tools/poll_pods.sh → runs/POLL_LOG.txt |
| 07:52 | 진단(정지 의심 → 아님): podA stdout 마지막 줄 07:14(approach 400/530, w=3627), timeline JSON 07:32 갱신 = approach 530 종료(8.4 s/sync). 이후 표면 문 열기(fine sync 0.1 ms) 단계 — podB 는 이 단계 첫 출력까지 ≈1,950 s(w 1870→3819), approach_end w=4876(t=5.455 s), plunge 240 w=5709(F/D 296/55, Fz 6.15 N). podA 첫 door_open 출력 예상 ≈08:55. 양쪽 프로세스 132/133 스레드·GPU 95~97 | 07:52 | 진단(정지 의심 → 아님): podA stdout 마지막 줄 07:14(approach 400/530, w=3627), timeline JSON 07:32 갱신 = approach 530 종료(8.4 s/sync). 이후 표면 문 열기(fine sync 0.1 ms) 단계 — podB 는 이 단계 첫 출력까지 ≈1,950 s(w 1870→3819), approach_end w=4876(t=5.455 s), plunge 240 w=5709(F/D 296/55, Fz 6.15 N). podA 첫 door_open 출력 예상 ≈08:55. 양쪽 프로세스 132/133 스레드·GPU 95~97 %·stderr 0 B. 재진단 기준: 08:55 까지 podA stdout 불변이면 D493 정지 의심 |
| 08:20 | podB: descend_end t=6.612 s(tool_residual 71) → close_stop t=7.685 s **관절 3.405°(서보 ≈5.9° > 3.6°) → chatter_close_1 진입**(rev34 실물 절차 첫 물리 실행). podA stdout 여전히 07:14 줄(문 열기 fine-sync 단계 예상), GPU 63~96 %. Monitor 도구 3회 연속 무이벤트(폴러는 기록 중) → 알림을 백그라운드 Bash 대기 루프로 교체 |
| 08:48 | podA 정지 의심 해소: `approach/surface_plus_50mm 200/269 w=7686 s`(08:21) 출력 — p1_tool_vertical 530 → surface_plus_50mm 269 순(rev34 접근 2단계), 07:14~08:21 침묵은 출력 없는 구간. podB: close_stop(tool_residual 428) → chatter → **lift 60 z_lip 54.2 mm, w=8501 s**(F/D 156/150). 양쪽 GPU 96~97 %, 오류 0 |
| 08:50 | podB **bridge_clearance_decision t=8.971 s**(source 63,357·tool_residual 411·in_flight 7·ambiguous 3,962) 통과 → 운반 단계. podA `door_open_at_approach 1000 q=2.252° w=9400 s`(표면 문 열기 진행, GPU 34 % = fine-sync 구간) |
| 09:10 | podA door_open_at_approach 2800 q=9.016° w=10409 s(GPU 97 %) · podB transport/post_lift_travel 200/219 w=10311 s(bridge 후 운반 중). 오류 0 |
| 09:41 | podA **approach_end t=5.4554 s w=12334 s**(podB 와 물리 시각 동일 = 결정적 제어 흐름) → plunge 시작. podB stdout 은 09:07 `post_lift_travel 200/219 w=10311` 이후 30분 무출력 → 신선도 점검(아래) |
| 10:01 | podB `transport/place_retract_base90 200/383`(v_max 2.25 m/s 관측, 상한 5 미만) — 무출력 의심 해소 · podA `descend/plunge 120 z_lip 30.59 mm w=13400 s`(Fz 3.31 N) |
| 10:21 | podA **descend_end t=6.5236 s**(podB 6.6117 s — 힘 조건 종료라 장비 간 물리 시각 상이, tool_residual 81 vs 71) → close 100 q=25.2° w=14850 s · podB place_retract_base90 200/383 유지 |
| 10:41 | podA close 600 q=13.98° w=16009 s(GPU 82 %) · podB stdout 10:01 `place_retract_base90 200/383` 이후 무출력 46분 → 신선도 재점검 |
| 11:12 | podB **release_before t=13.640 s**: receiving_bin 16·tool_residual 351·**spill 25**(운반 중, W19 A 는 124)·ambiguous 3,942, 재닫기 뒤 관절 2.390° → 배출 door_open 800 q=4.19°. podA **close_stop t=7.599 s**(tool_residual 441·in_flight 193·관절 3.358°) → chatter_close_1 300 w=17888 s. 물리 판정 아님(원자료 회계 전) |
| 11:32 | podA chatter_close_2 900 q=3.473°(2회차 재닫기, w=19080 s) · podB 배출 door_open 2000 q=14.9°(w=18835 s). GPU 96~98 %, 오류 0 |
| 11:52 | podB **release_after t=14.759 s**(원시 stdout 값, 회계 전): receiving_bin **584**·tool_residual 11·spill 25·ambiguous 3,713 → discharge_wait. podA 채터링 종료 → lift 0 z_lip 18.23 mm w=20034 s. (판정 승격 아님 — rev31 회계·정착 cadence 감사 뒤) |
| 12:12 | podA **lift_end t=8.909 s**(tool_residual 388·in_flight 4, 관절 3.306°) → reclose 시작 w=21500 s · podB discharge_wait 200(t=15.56 s) w=21224 s |
| 12:22 | podA **bridge_clearance_decision t=8.922 s**(tool_residual 417·in_flight 3; podB 는 t=8.971 s·411) → 운반 · podB discharge_wait 300(t=15.96 s) w=21843 s |
| 12:32 | podB **wait_end t=16.260 s**(receiving_bin 584 유지·spill 25) → close_after_discharge 100 w=22623 s · podA 운반 중(bridge t=8.922 s 이후) |
| 12:5x | 후처리 준비(replay-renderer, GPU 0): **post06** = post05(sha `1a1c48c0…` 재계산 일치) 사본 + `metadata_json.w25_frame` 의 R·t 를 입자·owner·CAD·셸·용기·마커·카메라 점집합·표시 상자 20개·IK 입력에 적용(`deme_to_disp = p@R.T + origin_disp`, 없으면 post05 동일식). 자체검사 9/9: rev34 스텁 첫 프레임 로봇 좌표 중심 (0.2500, −0.0000) m·217.5×307.5 mm·긴 변 y · W19 A 회귀 8항목 최대 차 0.0 · CAD 항등식 1.1e−16 m · 재투영 8.10 mm(미패치 507.7 mm). 폴더 `<case>/replay_post06_convA_20260929/`(sha `926d67ed…`). D341 항목(렌더·verify·육안)은 전부 미충족으로 남김 — 본 실행 원자료로 재검사 후 렌더 |
| 12:53 | podB close_after_discharge 500 q=16.2° w=23717 s · podA post_lift_travel 200/219 w=23252 s(v_max 1.93). 오류 0 |
| 13:23 | podB close_after_discharge 2000 q=3.64° w=25634 s(귀환 직전) · podA 운반 중(마지막 출력 12:53 post_lift_travel 200/219) |
| 13:43 | podB **close_after_discharge_end t=17.485 s**(receiving_bin 585·tool_residual 11·spill 25) → return_home(스텁 기준 물리 ≈4.9 s 남음) · podA place_retract_base90 200/383 |
| 14:34 | podB return_home/place_up_travel 200/268(귀환 중) · podA place_retract_base90 200/383 |
| 14:54 | podB return_home/return_retract_base0 200/383(베이스 복귀 중) · podA transport/place_target 200/268 w=30928 s |
| 15:04 | podA **release_before t=13.591 s**(원시): receiving_bin 13·tool_residual **119**·**spill 123**·ambiguous 3,844, 운반 문 관절 **3.036°** ↔ podB 같은 지점 tool_residual 351·spill 25·관절 2.390°. 같은 입력·다른 장비에서 재닫기 각도 차(3.04° vs 2.39°)가 운반 흘림 차(123 vs 25)로 이어진 관측 — 회계 전 원시값, 원인 단정 금지. podB return_retract_base0 200/383 |
| 15:55 | podA **release_after t=14.681 s**(원시): receiving_bin **251**·tool_residual 10·spill 123 ↔ podB 584/11/25. podB return_home/return_home 200/530(마지막 구간) w=33484 s |
| 16:22~16:32 | **podB 완료·회수·terminate**: 러너 `completed_rc0`(rc 0, 36,372.8 s = 10.10 h, 타임아웃/kill 없음), 시뮬 abort None·diverged False. 회수 14/14 sha 일치·0 불일치·`_obj` 4 포함(NPZ 1,770,948,514 B sha `af313f05…`) → RETRIEVAL_COMPLETE → `delete-pod th14bds3fsjg8n` 204. 과금(조회 시점) **40.21 $**(GPU 40.12·디스크 0.09; 09-28 12.64 + 09-29 27.58). 원시 회계(승격 아님): definite 585 개 11.8506 g·possible 665 개 13.4712 g·layer1 integrity true·spill 25. receipts/POD_B_TERMINATE.json |
| 16:45 | podA **wait_end t=16.181 s**(receiving_bin 250·spill 123) → close_after_discharge 100 w=37774 s. podB 동일 구간(wait_end→종료) 13,964 s × 2.5 → podA 완료 외삽 ≈09-30 02:2x KST |
| 17:26 | **podB 회계 1단계(raw-accountant, CPU 38분)**: 보존 14/14 · rev31 규약 v2 재분류 275프레임×67,737 = 18,627,675 칸 **불일치 0**(생산식·독립 NumPy식 둘 다) · 스키마 27항목 22 PASS/5 FAIL(ERRATUM_04 선언 5개 null·visual_mapping·time_mapping·매니페스트·정착층) · criteria 45항목 35 PASS/2 FAIL/8 판정불가(FAIL = settlement_window_UNCALIBRATED, policy.no_threshold_change_after_outcomes[hard_fail — criteria sha 가 EXEC_PIN 에는 있으나 EXECUTION_RECEIPT 에 없음, 문구 그대로 읽은 결과]) · 정착 cadence 5프레임<6·간격 0.100025 s>0.05 s **FAIL** → 배출은 585~665알(11.85~13.47 g) 구간으로만 · 채터링 3회 후 retries_exhausted(서보 6.09→5.99→5.81°) · 흘림 25알 전부 transport/place_retract_base90 t 11.50~13.20 s 이탈 · bridge CLEARANCE_CERTIFIED 여유 62.8 mm. 메인 원자료 대조: delivery 585/665·inventory_final·결정 관절각(3.4047/3.3108/2.3905/2.3924°)·return_home_end t=22.903 s 일치. `runs/podB_pro6000x2/postprocess_20260929/REPORT_postprocess_w25_podB.md` |
| 17:46 | podA **close_after_discharge_end t=17.407 s**(receiving_bin 251·spill 123) → return_home. podB 동일 구간 ≈9,500 s × 2.5 → 완료 외삽 ≈09-30 00:3x KST |
| 20:17 | podA return_home/return_home 200/530 w=49,562 s. podB 같은 구간(200→종료) 2,828 s × 2.5 → 완료 외삽 ≈22:00 KST |
| 20:57~21:1x | **podA 완료·회수·terminate**: `completed_rc0`(rc 0, 52,707.9 s = 14.64 h), 회수 14/14 sha·0 불일치·`_obj` 4(NPZ sha `8b341be1…`) → `delete-pod 5pzwyyqhi9f1gd` 204. 과금 **10.62 $**. 원시: 확정 251 개 5.0846 g·가능 297·흘림 123. **R_full = 52,707.9 / 36,372.8 = 1.449**(스모크 R 2.556 은 접촉 많은 단계 한정: approach·close A/B 2.3~2.5, 운반·배출·복귀 0.8~1.3). **기하 라벨(lift_end 더미 위 들린 알) podA 836 vs podB 832(0.5 %)**, tool_residual 분류 388 vs 397. 되떨어짐 podA 281 vs podB 44. lift_end 구덩이 A↔B RMS 1.93 mm(퍼내기 효과 11.8 mm). 우리 pod 전부 종료, 총 과금 50.83 $ |
