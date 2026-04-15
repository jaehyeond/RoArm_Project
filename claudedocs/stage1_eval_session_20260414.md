# Stage 1 정량 평가 결과 — 2026-04-14

## 설정
- Checkpoint: `outputs/smolvla_v6/checkpoints/015000/pretrained_model`
- 명령: `python deploy_smolvla.py --checkpoint ... --port /dev/ttyUSB1 --open-loop --n-chunks 2 --start-pos init --log-csv`
- 포트: USB0=Leader, USB1=Follower (4/1 매핑 유지 — 4/14 토크-off 물리 검증)
- Plan 3 gripper unlock 적용 (deploy_smolvla.py:668, 824)

## 최종 결과 — Stage 1 baseline

| Zone | 성공 | 비율 | Base final 범위 | Gripper lock |
|------|------|------|----------------|--------------|
| CENTER | 5/5 | **100%** | -1 ~ +7° | 18-19° |
| LEFT | 5/5 | **100%** | +40 ~ +47° | 16-18° |
| RIGHT | 0/5 | **0%** | -11 ~ -26° (peak) | 0.7-1.8° (허공 close) |
| **Total** | **10/15** | **67%** | | |

## 핵심 발견 — LEFT/RIGHT 비대칭성
- CENTER/LEFT 양쪽 100% 성공 → **카메라 위치 정상** (틀어졌다면 전부 실패).
  - Kinect 스냅샷 vs 학습 프레임 비교: 배경/테이블/로봇/케이블 위치 동일.
- RIGHT 0/5: 모델이 base -20~-30° 방향으로 reach 시도하나 파지 실패 (gripper lock 0.7-1.8° = 허공 close).
- 원인 가설: 학습 데이터 approach 30% base 분포 비대칭
  - `<-15°` (유저 RIGHT): 9ep (18%)
  - `> +15°` (유저 LEFT): 26ep (52%)
  - CENTER: 15ep (30%)
- 방향 컨벤션 확정:
  - **유저 LEFT = 로봇 base +방향** (ep 많음, 성공)
  - **유저 RIGHT = 로봇 base -방향** (ep 부족, 실패)

## 스펀지 orientation 교란 변수 배제
- 유저 확인: RIGHT 시도 시 스펀지는 **세운 상태**였음. 실패 중 팔이 쳐서 눕혀진 것.
- 즉 **순수 RIGHT 방향 실패** (orientation은 항상 세움, 학습 분포 일치).

## 무효/예외 기록
- CENTER #2 (171130): 외부 간섭 (사람이 스펀지 침) — 재시도 171300 성공
- LEFT #4 (130414): 스펀지 회수 못 함 — 재시도 130414 성공 (그대로 사용)
- LEFT #3 (130202): "아슬아슬" 성공, grip_max=64.8, lock=4.0 (비정상 signature지만 유저 물리 확인)

## 로그 파일 목록
- CENTER: deploy_20260413_170838, 171130(무효), 171300, 174428, 174533, 174650
- LEFT: deploy_20260414_125646, 130045, 130202, 130414, 130526
- RIGHT: deploy_20260414_131728, 131837, 131946, 132117, 132223

## Stage 1 베이스라인 확정
- **66.7% 성공률 (10/15)**. 공간 방향성에 강한 비대칭 존재.
- Plan 3 gripper unlock 패치 = grasp 필수 조건 확인됨 (CENTER/LEFT lock signature 일관).
- open-loop n-chunks=2 (100 step) = 현재 baseline 모드.

## Next Steps (다음 세션)
1. v7 수집 계획: base **-방향** (유저 RIGHT) 샘플 증량. 현 9ep → 25-30ep, 대칭성 확보.
2. closed-loop DIST breach (step 32, 420mm) 별도 이슈 해결.
3. Stage 1 baseline 67% 결과 교수님 보고 → Stage 2(다물체/sequential) vs Isaac Sim 방향 결정.
4. (선택) 데이터 증강으로 v6에서 RIGHT 개선 가능한지 테스트 (mirror flip 등) — v7 재수집 전 cheap 실험.
