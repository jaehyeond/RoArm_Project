# TASK_SPEC — W26 측정 1번 "따라 하기" 영상 (Claude Opus 5.5, 격리 worktree)

## Target
- 이 worktree 안 새 폴더 `claudedocs/research/w26_measure1_guide_20260930/` (새로 만든다) + 완성본 사본 `~/Downloads/measure1_guide_20260930/`.

## Change (만들 것)
사용자가 **영상을 보면서 바로 따라 할 수 있는** 한국어 안내물 한 벌:
1. `measure1_guide.mp4` — 1920×1080, 30 fps, H.264, 5~8분. 챕터 표지 + 단계 카드(카드당 10~25 s, 큰 한국어 글씨, 도식·실제 사진), 화면 상단에 "단계 n / N" 진행 표시, 마지막에 요약 체크리스트 카드. 음성 없이 자막형이면 된다. 챕터별로 나눈 짧은 mp4 도 함께(`chapters/`).
2. `checklist_A4.png`(+ 가능하면 `.pdf`) — 한 장짜리 체크리스트(준비물·순서·기록 칸).
3. `record_sheet.csv` 와 `record_sheet_template.json` — 측정값을 적는 기록표(재료 이름·봉지·온습도·100알 질량 ×3·캘리퍼 L/W/T 표·모양 개수·밀도 무게들).
4. `compute_measure1.py` — 기록표를 읽어 알 1개 질량(mg)·알/g·3회 변동계수, 치수 통계(평균·표준편차·분위), 모양 비율, 알코올 밀도·고체 밀도, "질량 ÷ 고체 밀도 = 알 1개 부피" 와 캘리퍼 치수(타원체 π/6·L·W·T) 비교를 계산해 `result_measure1.json` 과 짧은 요약을 출력. 판정 문턱: 3회 질량 각각이 평균의 ±2 % 안, 고체 밀도 3회 표준편차 ≤ 0.01 g/cm³ → 벗어나면 "재측정 권고" 표시.
5. `README.md` — 사용 순서, 파일 목록, 영상 챕터 시각표.

## 영상에 들어갈 절차 (이 순서·내용을 지킬 것)
근거 문서(읽기 전용): 메인 repo `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/research/w25_learning_plan_20260929/MEASUREMENT_AND_DATA_STRATEGY.md` §1, `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/MEASUREMENT_PROTOCOL_20260909.md` M1~M3, `/home/cgxr/Downloads/CALIBRATION_AOR_BULK_20260930.md` §3.
0. **준비물**: 0.01 g 저울(Riwonas, 최대 500 g), 작은 그릇, 디지털 캘리퍼, 모눈종이, 핀셋, 폰 카메라, 좁은 목 병(약 100 mL, 목에 표시선), 99 % 이소프로필 알코올, 스포이트, 젓개, 휴지, 세제 한 방울 탄 물 한 컵, 기록표. 알코올 안전(환기, 불 금지).
1. **어떤 봉지를 재나**: 지금 종이 상자에 깔린 것부터 재고 봉지 이름·사진을 기록. 다른 봉지는 따로 같은 절차.
2. **무작위 채취**: 봉지를 흔들어 섞고 위·중간·아래에서 한 줌씩.
3. **질량 — 100알 × 2번 세기 × 3회**: 모눈종이에 10×10 으로 늘어놓아 센다 → 다시 센다 → 그릇 영점(TARE) → 100알을 담아 무게 기록. 새 한 줌으로 3회. 알 1개 = 무게 ÷ 100. **왜 한 알씩 달지 않나**: 0.01 g 저울에서 수십 mg 은 눈금 몇 칸이라 반올림·저울 0점 오차가 크다(9/10 기록 사진 `펠릿무게.jpg` 의 한 알 0.06 g 읽음을 예로 보여 주되 "믿을 수 없는 읽음"으로 설명).
4. **치수 — 캘리퍼 30~50알**: 길이 L = 가장 긴 쪽, 폭 W = 납작한 면에서 L 에 수직, 두께 T = 가장 얇은 쪽(알을 눕혀 두고 턱을 살짝만 닫음, 누르지 않기). 한 알씩 L·W·T 를 기록. 두께가 가장 불확실한 축이라 특히 정확히.
5. **모양 비율**: 같은 알들을 모눈종이에 펼쳐 사진 한 장 → 렌즈형(가운데 오목한 납작 타원) / 짧은 원기둥 / 둥근 구슬 / 불규칙 으로 개수를 센다. 9/10 사진(`KakaoTalk_20260910_104636645*.jpg`, `KakaoTalk_20260910_110856317*.jpg`, `KakaoTalk_20260910_105619511.jpg`)을 모양 예시로 활용.
6. **물 뜨기 확인**: 세제 한 방울 탄 물에 10알 → 뜨면 밀도 < 1.0(순수 PP 쪽), 가라앉으면 충전제 가능성 → 기록.
7. **고체 밀도 — 알코올 치환(병 방식)**. PP 는 물에 떠서 물로는 부피를 못 잰다는 점을 먼저 설명.
   a. 빈 병 무게 m0 → 물을 표시선까지 → 무게 → 병 부피 V = (물 무게) ÷ 0.9977 g/mL(약 22 °C).
   b. 병을 말리고 알코올을 표시선까지 → 무게 m1 → 알코올 밀도 = (m1 − m0) ÷ V.
   c. 알 약 30 g 을 따로 달아 m_p → 병의 알코올을 일부 덜고 알을 넣은 뒤 알코올을 표시선까지 채우고 젓개로 저어 기포 제거 → 무게 m2.
   d. 알 부피 = (m1 + m_p − m2) ÷ 알코올 밀도, 고체 밀도 = m_p ÷ 알 부피. 3회. 전체 무게가 500 g 을 넘지 않게.
8. **기록·계산**: 기록표 채우는 법과 `compute_measure1.py` 실행법(예시 입력은 "예시"라고 크게 표시).

## Constraints
- 로봇·카메라·GPU·RunPod 접근 금지. 설치 금지(있는 ffmpeg·python·matplotlib·PIL 사용; 한글 폰트는 `fc-list :lang=ko` 로 확인한 Noto CJK KR 사용).
- 상태 원장(`START_HERE.md`, `claudedocs/DECISIONS*.md`, `claudedocs/EXPERIMENT_LEDGER.md`, `claudedocs/LEDGER_RECENT.md`, `claudedocs/relay/`) 쓰기 금지. git commit/push 금지. 기존 파일 수정 금지 — 새 폴더와 Downloads 사본만.
- **측정값을 지어내지 말 것.** 예시 숫자는 화면·기록표에 "예시"라고 표시. 시뮬 현재 알(렌즈 4.5 × 3.8 × 2.5 mm, 20.26 mg)은 "비교용 참고, 정답 아님"으로만.
- 그림은 도식(matplotlib/PIL)과 위 실제 사진으로. 사진은 원본을 수정하지 말고 사본을 잘라 쓴다.

## Ownership
- 편집 가능: 이 worktree 의 `claudedocs/research/w26_measure1_guide_20260930/` 와 `~/Downloads/measure1_guide_20260930/` 만.

## Observable acceptance
1. `ffprobe` 결과(해상도·길이·코덱·프레임률)를 `inspection.md` 에 기록.
2. **카드마다 프레임 1장을 추출해 직접 열어 보고**(Read 로 이미지 확인) 한글 깨짐(□)·글자 잘림·겹침·오탈자가 없음을 카드별로 기록. 문제 있으면 고치고 다시 렌더.
3. `compute_measure1.py` 를 예시 입력으로 실행한 출력과 판정 표시가 동작함.
4. Downloads 사본과 원본의 sha256 일치 목록.
5. 최종 `worker_done` 에 3문장 요약 + 산출 경로 + 영상 길이.
