# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약

- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo 를 여는 **다른 도구**(Codex/Cursor).
  Claude 가 연속으로 두 번 열려도 이 파일이 아니라 `START_HERE.md` 로 재개한다.
- **덮어쓰기.** 최신 인계 1건만 §2. 상태 정본은 `START_HERE.md`(여기 안 베낌). 규칙은 `AGENTS.md`.
- `HANDOFF.md` 아님(HARD RULE #7). 중복 금지: 상태→`START_HERE`, 활성 결정→`DECISIONS_ACTIVE`,
  최근 실험→`LEDGER_RECENT`, 규칙→`AGENTS.md`. 여기엔 만진 것·만지지 말 것·함정·승인 대기만.

## §2 2026-09-03 새벽 Claude → Codex (76th 연장, Phase 3 자산화)

세션 성격 = **Phase 3 실체화** — 구동 1축 URDF → 로봇 합성 → Isaac 5.1 USD 임포트 → RTX 렌더. 물리 시뮬 0.
원장 D474 append 완료. 커밋 다수(`c6a64d3`~`3d63b36` + full-robot 렌더). 상태 정본 = `START_HERE.md`.

**만지지 말 것**
- 🔴 벤더 `local_assets/roarm_m3/urdf/roarm_m3.urdf` **무수정**. 합성본은 `roarm_m3_with_grab.urdf`(별도).
- `grab_track/g17_yoke_alu/`(정본 형상) · `g9_sidefix`·`g16*`(낡음). 원장(코디네이터 배타 소유).
- `local_assets/roarm_m3/usd/config.yaml` — 변환 도구가 덮어쓰는 공유 파일. **매 변환 후 추적본 복구**(git checkout).

**🔴 함정 (전부 이번 세션 실측)**
1. 🔴 **커스텀 그랩은 순정 그리퍼 "교체" 아님, "추가"** — 브래킷=순정 고정 조, 서보 크랭크=순정 가동 조에
   볼트로 물려 **순정 서보로 구동**(D462). 순정 조 2개 남음. 렌더에선 클러터라 숨겼을 뿐.
2. 🔴 **Isaac 기본 collider `convex_hull` 은 오목 스쿱 보울을 채운다 = D446 함정.** 조각별 볼록 STL(350)로
   임포트해야 공동 산다. `sim_urdf_to_usd.py --collider convex_hull`(조각이 이미 볼록).
3. 🔴 **mimic 조인트는 Isaac articulation 이 구동 못 한다** — `convert_mimic_joints_to_normal_joints=True` 로 독립화.
4. 🔴 **Isaac 카메라 ~0.5 m 면 프레임 놓친다**(빈 렌더). 전체 로봇은 **AABB→거리 1.14 m**(`sim_render_robot_full.py`).
5. 🔴 스쿱 = 그랩 입(로컬 -Y=link5 +Z)이 **아래(-Z)**. 스쿱 각도 셸은 옆에서 짜부라져 보울로 안 읽힘(격리 3D 로 확인).
6. ⚠️ D447: `SimulationApp.close()` 예외 삼켜 exit 0 → 산출 USD 를 pxr 로 별도 검증. USD·PNG·조각 collision gitignore.

**승인 대기 / 다음**
- 🔴 **구동 인출 실물 검증**(Phase 2 잔여, 팔 필요) — 순정 가동 조→서보 크랭크→4절→셸.
- USD↔DEME 연결(브리핑만). 관절형 그랩 DEME 표현(D464 크래시)이 관문.

**검증**
```bash
git log --oneline -8
head -n 29846 claudedocs/DECISIONS.md | md5sum   # 9a44add0ceb2 (D474 append 전 불변)
~/miniconda3/envs/3dgrut/bin/python -c "from pxr import Usd; s=Usd.Stage.Open('local_assets/roarm_m3/usd/roarm_m3_with_grab.usd'); print([p.GetName() for p in s.Traverse() if 'grab' in p.GetName()])"
```
