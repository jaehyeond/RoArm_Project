# T3 trust-region 크기 스윕 — TRANSIT_POLISH_DEV_DEG 도입 근거 (2026-08-06, 22nd 세션)

## 질문

재검증 `wf_3cea04db-7c2` 생존 MAJOR ②(transit 명령 잔차 1.2~2.55mm)가
관절 trust region(`--waypoint_max_joint_dev_deg`) 크기를 키우면 해소되는가?

## 방법

p9 개정판(밴드 0.5mm + trust region 적용, **폴리시 도입 전** 리비전)의
`_solve_q_vertical`을 직접 import, 4개 T2-PASS 포즈의 HOME→approach 체인을
dev ∈ {12, 14, 16, 18, 20, 24}°로 재실행. 스크립트 = 세션 스크래치
`dev_sweep.py`(사전비행 v2와 동일 waypoint/슬루 산식). isaaclab env python.

## 결과 (worst 명령 pe [mm] / 비관 슬루 [mm] / 도착 (pe[mm], tilt[deg]))

```
dev= 12.0 | seed0_S1: pe=1.222 slew=9.59 arr=(0.016,0.200) | seed0_S2: pe=1.520 slew=8.92 arr=(0.016,0.197) | R1_center: pe=1.308 slew=9.05 arr=(0.016,0.199) | R2_center: pe=1.351 slew=8.69 arr=(0.016,0.198)
dev= 14.0 | seed0_S1: pe=1.222 slew=9.51 arr=(0.016,0.200) | seed0_S2: pe=1.520 slew=9.19 arr=(0.016,0.197) | R1_center: pe=1.308 slew=9.31 arr=(0.016,0.199) | R2_center: pe=1.351 slew=8.96 arr=(0.016,0.198)
dev= 16.0 | seed0_S1: pe=1.222 slew=9.90 arr=(0.016,0.200) | seed0_S2: pe=1.520 slew=9.14 arr=(0.016,0.197) | R1_center: pe=1.308 slew=9.71 arr=(0.016,0.199) | R2_center: pe=1.351 slew=9.35 arr=(0.016,0.198)
dev= 18.0 | seed0_S1: pe=1.346 slew=9.90 arr=(0.016,0.200) | seed0_S2: pe=1.520 slew=9.60 arr=(0.016,0.197) | R1_center: pe=1.308 slew=9.71 arr=(0.016,0.199) | R2_center: pe=1.351 slew=9.40 arr=(0.016,0.198)
dev= 20.0 | seed0_S1: pe=1.263 slew=9.90 arr=(0.016,0.200) | seed0_S2: pe=1.520 slew=9.60 arr=(0.016,0.197) | R1_center: pe=1.308 slew=9.71 arr=(0.016,0.199) | R2_center: pe=1.351 slew=9.40 arr=(0.016,0.198)
dev= 24.0 | seed0_S1: pe=1.264 slew=9.90 arr=(0.016,0.200) | seed0_S2: pe=1.520 slew=9.60 arr=(0.016,0.197) | R1_center: pe=1.308 slew=9.71 arr=(0.016,0.199) | R2_center: pe=1.351 slew=9.40 arr=(0.016,0.198)
```

(ok = 전 waypoint True — 3mm 수용 게이트 기준으로는 전부 통과; 문제는 마진.)

## 판독

1. **worst 명령 pe(1.2~1.5mm)는 dev와 무관** (12→24° 동일) — 잔차의 원인은 trust
   region 클립이 아니라 **바이어스(w_axis=0.03)-위치 DLS 평형 자체**다. 클립을
   키워도 잔차는 안 줄고 슬루 상한만 커진다(9.59→9.90mm).
2. 순수 위치-only 전환(w_axis=0)은 도착점 재배향을 한 waypoint에 몰아 D423-R1이
   경고한 "도착 재배향 부담" 폭증 경로 — 기각.
3. → **채택 수리 = 2단 솔브**: 바이어스 해(수직화 진행 보존)를 시드로 bias-free
   폴리시(`TRANSIT_POLISH_DEV_DEG=2.0`, 비악화 가드 `pe_p < pe`)를 얹는다.
   사전비행 v2 실측: worst pe 1.222~1.520 → **0.468~0.492mm**, dev 12.00→12.05°
   (폴리시 실사용 0.05°), 도착 tilt 불변.

## 층위

수치·설계 근거 문서 — 물리 verdict 아님. `g0a_pass=false` 불변.
