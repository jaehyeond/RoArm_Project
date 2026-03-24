---
name: projector_idea_analysis
description: 2026-03-24 Unity+Projector+SAM3 시스템 포지셔닝 분석 — CoRL 2026에 부적합 판정
type: project
---

Unity + Beam Projector + RoArm-M3 + SAM3 시스템 아이디어 분석 (2026-03-24).

**판정: CoRL 2026에 부적합. 기존 data-centric 방향 유지 권고.**

**Why:**
1. 문제가 없는 기술 — 유저가 어떤 문제를 푸는지 모른다고 인정함
2. SAR은 25년 된 성숙한 분야 (Raskar 2001~). GreenAug (CoRL 2024)도 projector 기반
3. 로봇 카메라에 투사된 이미지 = SigLIP OOD 문제 재등장
4. Unity는 이 파이프라인에서 Python/OpenCV로 대체 가능 — 리뷰어 지적 불가피
5. 타임라인 불가: projector 구매 + 설치 + Unity 씬 + SAM3 + task 설계 + 데이터 수집 = 8-12주 필요, 남은 시간 9주
6. 기존 계획은 74ep 결과 + v1(0%) vs v3(100%) 증거 이미 보유

**Unity 기술을 논문에 활용하려면 (향후):**
- Isaac Lab + Unity assets → synthetic training data (sim-to-real) 방향이 가장 defensible
- 현재 CoRL 2026 scope 밖

**How to apply:** 유저가 projector/Unity 아이디어를 다시 꺼내면 이 분석을 참조. 현재 data-centric 방향에서의 이탈을 막아야 함.
