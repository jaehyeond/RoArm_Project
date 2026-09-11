# 78th (2026-09-03 오후 ~ 09-04) — 그랩 기구 S1 전환 · 벤더 STEP 분해 · 실물 대조 · Isaac 파지 · 출력 슬라이스

정본 결정 = `DECISIONS.md` **D480 `:30127`**. 원장 행 `EXPERIMENT_LEDGER.md:543`. 정본 폴더 `claudedocs/runtime_logs/grab_track/g19_servo_direct/`.
이번 case 의 신규 변수: ① 구동 인출 = 순정 가동 조 제거 후 서보 디스크 직결 ② 폐합 = 한쪽 가동(S1). g18 은 동결(참조 전용).

## 0. 흐름 (사용자 질문 순)
1. "순정 조 안 열고 셸만 열리나" → 불가(셸 무동력, D478 영상은 근사). 명칭: 셸/그랩/순정 가동 조=서보 레버.
2. 방식 A(순정 조 제거·디스크 직결) 채택 → 실측 항목 10 → 벤더 STEP 발견으로 3 으로 축소.
3. "M2 여분 조 절단은 공업소 필요 → 그냥 인쇄" → "인쇄면 셸을 직접 붙이자" → **S1** (사용자 승인).
4. 설계 스크립트 → URDF/USD → Isaac 렌더 → Isaac Lab 파지 → 조립 순서도 → 출력 승인 → 분할·슬라이스 → 전송 직전 대기.

## 1. 벤더 STEP (`vendor_step_parts/`)
- 출처 `files.waveshare.com/wiki/RoArm-M3/RoArm-M3_STEP_260310.zip` (sha256 `1e2111145276aac1…`), 2D `RoArm-M3_2Dsize.zip`. cadquery/OCP 는 새 conda env `cadstep`(isaaclab 무접촉).
- 추출: `step_xde.py`(bbox/COM 716 leaf) · `step_gripper_detail.py`(원형 에지) · `step_export.py`(STL 13 + `manifest.json`) · `step_tree.py`/`step_place.py`.
- 조립: 손목롤 디스크 → 손목 b(M3×4 4+1) → 그리퍼 베이스(M3×6 2) → 서보(PA2×5 8, 베이스 벽 통과) → 고정 조(+Y 쪽 서보 나사와 동체 추정) → 구동/종동 디스크 → 가동 조(뺨당 M3×4 4).
- `stock_jaw_interface.json` = 설계 입력 전부(프레임 변환 STEP↔link5: X=link5 Z−236.967 / Y=−(link5 X)−0.88 / Z=346.07−link5 Y).
- 🔴 CAD 오차: 구동 디스크 플랜지가 뺨 안쪽면과 0.88 겹침(종동측 정합). 포크는 뺨 안쪽면 복사라 무관.

## 2. 실물 대조 (`real_arm_confirmation_20260903.json`, 사진 20장 sha 기록)
- 라벨: 그리퍼 **ST3215-HS**(20 kg·cm@12 V=1.96 N·m) · 손목롤 ST3215. 위키 FAQ: Pro 는 손목2·끝관절 외 금속 하우징.
- 자 사진: 뺨 1.45~1.55 · 포크 폭 39.3~40.2 · 고정 조 35.8. 저울: M2 여분 세트 9.48/9.49 g. 고정 조 3구멍 M3 볼트·너트 OK.
- 그림: p42 A/B(어디를 재나), p43(방식 A 계획), p44(g18 vs S1), p45(조립 순서 0~9).

## 3. S1 설계 (`s1_v0/`, `scoop_grab_s1_design.py`)
- 게이트 10: 볼록·자중·공동≥B1·입58@≤30°·스윕 간섭(KD-트리 표본)·고정부 침입 0·고정부 간섭·립 정합·외팔보·플랜지. 전부 PASS. 함정: 암 사각형이 고정 캡 영역 침범(→ L자) · 자중 63.9→판 3 mm+창 · 근접 거리 삼각형 질의 OOM(→ 표본 KD-트리) · 퇴화 삼각형 196(→ simplify+제거) · 핀 구멍-창 0.1 mm² 부스러기(자동 제거).
- 렌더 검정 문제: 원인 미해결(D480 §5). 다음 세션: `sim_render_grab_closeup.py`(g18) 와 비교, `Camera` 센서 경로 사용 권장.

## 4. Isaac Lab (`s1_v0/isaaclab_grasp_sphere/`)
- `sim_isaaclab_grasp_sphere_s1.py`: 문 상한 1.96, 팔 8.0(데모). 1차 실패(구가 고정 립 아래) → 2차 ok. 영상 `grasp_sphere_s1.mp4`, 띠 `strip_s1.png`.

## 5. 출력 (`s1_v0_print/`)
- `split_for_print.py`: 문 Y=0 두 쪽(측판 바닥, 오버행 0), 고정부 한 덩어리(판 바닥; 진짜 오버행 937 mm² = 스파인 밖 벽 두 띠 25~45°, 캡 아래 1.5 mm). 설계 수정: 다리 X 8~18 + 각구멍 3.5(M3×45) + 측판 ⌀3.4, 스파인 45° 플레어.
- 프로필 신설 `process_support_slow_roarm.json`(slow 기반 + support_on_build_plate_only=1). 3mf `output/roarm_s1_v0.3mf` sha `c72f36ce44bb51bc`. 13/13 PASS, 8611 s / 36.7 g / 24.1 mm. dry-run 4/5(idle, FINISH, hms 0).
- **전송 안 함**: 베드 청소·필라멘트 확인 대기. 전송 = `~/miniconda3/bin/python send_print_job.py claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0_print/print_job.json --yes` 후 subtask_name 대조.

## 6. 검증 명령
```
grep -n '^## D480' claudedocs/DECISIONS.md                       # 30127
sed -n '543p' claudedocs/EXPERIMENT_LEDGER.md | cut -c1-80
~/miniconda3/envs/3dgrut/bin/python scoop_grab_s1_design.py     # all_gates_pass = True, 53.78 g
~/miniconda3/envs/3dgrut/bin/python sim_isaaclab_grasp_sphere_s1.py --solve-only
~/miniconda3/bin/python send_print_job.py claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0_print/print_job.json   # dry-run 4/5
```
