# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약

- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo 를 여는 **다른 도구**(Codex/Cursor).
  Claude 가 연속으로 두 번 열려도 이 파일이 아니라 `START_HERE.md` 로 재개한다.
- **덮어쓰기.** 최신 인계 1건만 §2. 상태 정본은 `START_HERE.md`(여기 안 베낌). 규칙은 `AGENTS.md`.
- `HANDOFF.md` 아님(HARD RULE #7). 중복 금지: 상태→`START_HERE`, 활성 결정→`DECISIONS_ACTIVE`,
  최근 실험→`LEDGER_RECENT`, 규칙→`AGENTS.md`. 여기엔 만진 것·만지지 말 것·함정·승인 대기만.

## §2 2026-09-03 Claude → Codex (77th 후반 3, 시각·시뮬 검증 D477 + Isaac Lab 구 파지 D478)

세션 성격 = 검증 → 설계 수정(D476) → 시각·시뮬 검증(D477) → **Isaac Lab 물리 시행: 구 파지 영상(D478)**. 로봇 0·출력 0·입자 물리 0. 원장 D475~D478 · LEDGER `:538~:541`. **커밋됨**: `9f4241d`(설계·프로브·자산) · `84ffa46`(시뮬 스크립트·검증 JSON) · `a2201c2`(원장·상태 문서) (PNG·mp4·조각 STL·USD 는 gitignore, 로컬만). 상태 정본 = `START_HERE.md`.
- 신규 `sim_isaaclab_grasp_sphere.py`(numpy FK/IK + 1 env + `Camera` 센서 → mp4). 산출 `g18_nut_trap/isaaclab_grasp_sphere/` (mp4·strip·JSON, 프레임 PNG 삭제).
- 🔴 함정 8: Isaac Lab `ImplicitActuatorCfg` 에 `effort_limit_sim` 없으면 USD maxForce(URDF 1.9) 상한 → 팔 처짐. 데모 8.0 은 **비물리**(인용 금지).
- 🔴 함정 9: 셸 폐합 도중 배가 립보다 3.7 mm 깊다(−39.80 @22°) — 바닥·DEME 경계 여유는 립이 아니라 스윕 최저점 기준. `grab_v1_meta.json` 에 아직 없음.

**만진 것 (미커밋, D476 분에 더해)**
- `sim_urdf_to_usd.py`: `convert_mimic_joints_to_normal_joints` **True→False**(반전 플래그, 근거 주석). USD 재생성됨(gitignore).
- 신규 `sim_viz_grab_assembly.py`(실메쉬 3D) · `sim_render_grab_closeup.py`(자동 조준 근접 8장 + `closeup_poses.json`) · `sim_isaaclab_parallel_smoke.py`(N env 스텝·구동·계측, JSON 선기록 + `os._exit` 워치독).
- 산출 `g18_nut_trap/{viz/, isaaclab_smoke/}` · `local_assets/roarm_m3/usd/{g18_robot_full.png, g18_closeup_v3/}`.

**만지지 말 것**
- 🔴 `g17_yoke_alu/`·`g9`·`g16*` 낡음(g17 브래킷은 볼트 못 들어감). `usd/config.yaml` 은 변환 후 git checkout. 원장(배타).
- 🔴 `sim_render_grab_usd.py` 카메라(0.72 m) 로 찍은 `bowl_*.png`·`g18_bowl_*.png` 인용 금지 — 근접 정본은 `g18_closeup_v3/`.

**🔴 함정 (이번 세션 실측)**
1. 🔴 **`UrdfConverterCfg.convert_mimic_joints_to_normal_joints=True` = PhysX mimic 생성**(`urdf_converter.py:130` → `set_parse_mimic`). 독립 관절은 **False**. D474 ④ 는 오독.
2. 🔴 **순간이동 렌더(`set_joint_positions`)는 드라이브 결함을 못 잡는다** — 512 env 스텝에서 셸 R 이 0.931 rad 상한 고착으로 드러났다.
3. 🔴 **Isaac Lab 앱에서 Replicator `BasicWriter` 금지** — 매 프레임 기록 폭주(32 GB/23,646 파일). annotator `get_data()` 1장.
4. 🔴 **`SimulationApp.close()` 무한 대기**(1,108 s·8.7 h) + `timeout` SIGTERM 무시 → 결과 JSON **선기록** + `os._exit` 워치독 + `timeout -k 30`. 증거는 stdout 이 아니라 JSON(`python -u`).
5. ⚠️ Replicator 캡처 뒤 물리 뷰가 사라진다(`get_joint_positions()` None) → `world.play()`+`art.initialize()` 재초기화, `orchestrator.step(pause_timeline=False)`.
6. ⚠️ pxr USD 검증은 `Usd.TraverseInstanceProxies()` 필수. 스쿱 옆면 검정 삼각형 = 벤더 link5 backface(무해).
7. ⚠️ 셸 R 독립 구동 = 기어 커플링 근사. mimic 으로 모델하려면 gearing 부호·연성(25 Hz)·한계 확장 검증 필요(미실행).

**승인 대기 / 다음**
- 실물(팔): 순정 구멍 나사산·⌀·재질, 플랜지 대조, 어댑터 전압(D476). 🔴 출력 착수(Phase 5) 는 사용자 승인.
- Isaac Lab 환경 정의(스쿱 작업공간·heightmap 관측)·실 서보 게인 반영·64 env 격자 스냅샷(orchestrator 행 회피). DEME 연결 병행.

**검증**
```bash
grep -n '^## D47[567]' claudedocs/DECISIONS.md            # 29879 · 29936 · 29992
head -n 29990 claudedocs/DECISIONS.md | md5sum            # == md5sum claudedocs/DECISIONS.md.bak_20260903_pre_d477
grep -n '^## Schema errata' claudedocs/EXPERIMENT_LEDGER.md   # 542 → 표 끝 :540
OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 600 ~/miniconda3/envs/isaaclab/bin/python sim_isaaclab_parallel_smoke.py --headless --num_envs 512   # smoke_512.json "ok": true
```
