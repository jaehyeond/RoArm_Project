# Archive — AGENTS.md `## Current Status (2026-04-28)` 스냅샷

> 출처: 분리 전 `AGENTS.md` **622-655행**. 원본 전체는 `AGENTS_full_20260825_pre_split.md`.
> 아래 본문은 원본에서 **바이트 동일**하게 이동했다 (2026-08-25).
>
> 🔴 **이것은 죽은 상태 문서다. 현재 상태로 절대 사용하지 말 것.**
> 2026-04-28 시점 스냅샷이며, 여기의 `Active Blockers` #5("B200 SERVER cleanup")와
> `Next Steps` ST-B("B200 finetune 1.5h")는 **HARD RULE #27(B200 전면 금지, 2026-05-22 lease 만료)과 정면 충돌**한다.
> 규칙 파일에 상태가 섞여 있어 매 세션 이 모순이 주입되던 것이 (가) 분리의 사유다.
> 현재 상태의 진실은 `START_HERE.md` 하나뿐이다 (상태 ≠ 규칙).

---

## Current Status (2026-04-28)

> Historical snapshot — 현재 상태의 진실은 `START_HERE.md` (상태 ≠ 규칙 원칙).

### Completed
- **v6 데이터 수집 완료 (4/01)**: 50 ep, 6942 frames, L-F 텔레옵, single Azure Kinect (`lerobot_dataset_v6/`)
- **v6 학습 완료 (4/05)**: 50K steps, smolvla_base pretrained, batch=8 (4090 5.2h)
- **v6 배포 SUCCESS (4/9)**: Plan 3 = JOINT_SPEED_CAPS gripper-only unlock (`speed=1000`). 유저 물리 검증: **다양한 위치/방향 sponge 전부 파지 성공**. git commit `2e840e4`
- **Kinect↔RoArm calibration 완료 (4/15)**: 빨간 스티커 마커, RMSE 10.13mm. git commit `a217cd3`
- **현실 측정 완료 (4/24)**: Hand-eye solve, table plane (z=-12.12mm RMSE 1.24mm), sponge poses 50ep. git commit `1f0d52e`
- **Sim env 구축 완료 (4/24)**: Isaac Sim (`isaaclab` env) + URDF + Kinect calib pose + table USD. SigLIP 0.7222 (48/50 GO ≥0.70). Joint replay RMSE 0.43°. `sim_v1/` (87MB lerobot v3)
- **Stacking scene 시각 검증 (4/28)**: [sim_renders_v2/stacking_initial.png](sim_renders_v2/stacking_initial.png) — A/B/Temp Layout 정확
- **B200 학습 reproducibility 검증 (4/28)**: 4090 동등 (loss bit-exact, weight diff frozen 378/500 bit-exact, max\|diff\|=0.0319 saturate). 1.4h vs 5.2h (3.7×). git commit `18abcef`
- **Stacking task pivot (4/24)**: 교수님 target = N=2 sponge stacking (3-step pick-place). Layout A(+280,0)/B(+280,+130)/Temp(+280,-110)
- **Sim2real gap 정량화**: SigLIP 0.7222, sim 70% brighter than real (dome light), LEFT zone weakest

### v6 Stacking Feasibility Analysis (4/28)
- Pick z 분포 ✅ in-distribution (elbow > 90: 22.3%, elbow < 50: 33.5%)
- Place 동작 ❌ v6에 없음 (sim demos 필수)
- Tower context image ❌ OOD (single sponge만 학습)
- v6 trajectory ~50% 재사용 가능, place + tower visual은 sim에서 새로 학습

### Active Blockers / Pending Decisions
1. Stacking 3-step vs 4-step 순서
2. Curriculum 도입 (Phase A 단독 pick → B 1-stack → C 2-stack)?
3. Safety limit hard-code (`z_world > +148+3mm`)?
4. Sim demo 생성 방식 (Procedural IK vs Leader teleop in sim vs Isaac Lab Mimic)
5. B200 SERVER 5K/10K/15K cleanup (1개월 대여)
6. 단톡방 발송 (Vulkan ICD 정책 미답)

### Next Steps (Phase ST-A → ST-B → ST-C, 2-3주)
1. **ST-A (1-2일)**: stacking_scene.py 2-sponge 패치 (현재 1 sponge spawn) + procedural pick-place script 설계
2. **ST-B (1-1.5주)**: Sim에서 50-100 stacking demos 생성 → `sim_to_lerobot.py` 변환 → Co-training (v6 real + sim) → **B200 finetune 1.5h**
3. **ST-C (3-7일)**: Real deploy A→Temp → A→B → Temp→B 3-step, dataset_mean 시작
