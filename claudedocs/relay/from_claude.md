# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약

- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo를 여는 **다른 도구**(Codex CLI / Cursor).
  Claude가 연속으로 두 번 열려도 이 파일이 아니라 `START_HERE.md`로 재개한다.
- **덮어쓰기.** append-only 아님. 최신 인계 1건만 §2에 둔다. 과거 인계는 보존 대상이 아니다
  (실제 기록은 `claudedocs/session_*.md`가 소유).
- **상태 정본이 아니다.** 현재 상태·active case·다음 행동·수치는 전부 `START_HERE.md`가 소유하며
  **여기에 한 줄도 베끼지 않는다.** 어긋나면 `START_HERE.md`가 이긴다.
- **`HANDOFF.md`가 아니다.** HARD RULE #7은 `HANDOFF.md`라는 **파일명**을 금지한다
  (PreToolUse 후크로 기계 강제 중). 이 파일은 그 이름을 쓰지 않으며, 상태 대시보드를 대체하지도 않는다.
- **중복 금지 지도**: 현재 상태 → `START_HERE.md` / 활성 결정 → `claudedocs/DECISIONS_ACTIVE.md` →
  정본 `claudedocs/DECISIONS.md` / 최근 실험 → `claudedocs/LEDGER_RECENT.md` → 정본
  `claudedocs/EXPERIMENT_LEDGER.md` / 규칙 → `AGENTS.md`.
  **여기 적을 것은 그 어디에도 안 들어가는 것뿐** — 직전 도구가 무엇을 만졌나, 무엇을 만지지 말아야 하나,
  다음 도구가 밟을 함정, 사용자 승인 대기 항목.

## §1 템플릿 (§2를 덮어쓸 때 이 골격을 쓴다)

```
## §2 2026-08-31 Claude → Codex (74th 세션)

세션 성격 = **실물 그랩 설계 + 시뮬 검증**. 73rd 산출물(D457~D461)을 이어받아
D462(기구 전환)·D463(구동 전환)을 냈다. 물리 0, 로봇 0, 펠릿 0 — 전부 기하·문서.

**한 일 (repo에 남은 변경)** — HEAD `9664f91`, 미커밋 1건(내 설정 백업)뿐.

| 커밋 | 내용 |
|---|---|
| `d1f4e32` | 계약 웨이브 — 형상 게이트·heightmap 계약·DEME 입자 시뮬·예측 모델 골격 (워커 4개 산출) |
| `7b4ae74` | 그랩 v1 설계(`scoop_grab_v1_design.py`) + 부착 프로브(`sim_scripts/p37_*`) + 조각 STL gitignore |
| `9664f91` | D463 기어→링크 전환 + 감사 6건 |

**만지지 말 것**

- `scoop_shell_design.py`(v0) 와 `claudedocs/runtime_logs/scoop_shell_v0/` — **동결**. forward-only.
- `.claude/worktrees/grab-v1` (브랜치 `track/grab-v1`) — **트랙 A 워커 작업 중**
- `.claude/worktrees/pellet-sim` (브랜치 `track/pellet-sim`) — **트랙 B 워커 작업 중**
- `.claude/settings.local.json.bak_20260831_pre_orca_allow` — 되돌리기 경로. 커밋 대상 아님.
- 원장(`START_HERE`·`DECISIONS*`·`session_*`·이 파일) — 코디네이터 배타 소유.

**함정** (전부 이번 세션 실측)

1. 🔴 **`orca-ide worktree create` 를 쓸 것. `git worktree add` 를 쓰면 Orca 사이드바에 안 뜬다.**
   74th 는 `git worktree add` 로 만들었고 그 결과 `git worktree list` 3개 / Orca 등록부 1개가 됐다.
   **격리·워커 배치는 정상 작동**하지만(`worker-start --worktree <절대경로>` 로 해결) 사용자가
   UI 에서 탭을 못 찾아 승인 프롬프트 처리가 번거로워진다.
   **사용자 결정(08-31): 현행 2개는 그대로 두고, 다음 웨이브부터 Orca 방식.**
2. 🔴 **`orca orchestration check` 는 `--from` 을 받지 않는다.** 유효 플래그는 `--terminal` / `--run`.
   그리고 호출 전에 `run-use --id <run> --from <handle>` 로 재바인딩하지 않으면 `consumer_fenced` 가 난다.
   동작 순서: `run-use --from <handle>` → `check --terminal <handle>`.
3. 🔴 **워커가 권한 프롬프트에서 멈춘다. 코디네이터가 대신 못 누른다** (`agent_prompt_blocked` — 의도된 설계).
   지금까지 걸린 것: `orca orchestration send` → `orca-ide orchestration inbox` →
   `python sim_deme_pile.py` → `cd` (경로 우회 방지). **사람이 눌러야 진행된다.**
   근본 해결 = `.claude/settings.local.json` 의 `permissions.allow` 에
   `Bash(orca orchestration:*)` · `Bash(orca-ide orchestration:*)` 추가 (사용자 승인 사안, 미실행).
4. 🔴 **충돌 검사에 `mesh.convex_hull` 을 쓰지 마라.** `link5` 실부피/볼록껍질 = **0.386**.
   속 빈 프레임을 통짜로 만들어 −52.9 mm 를 과대보고했다. `fcl` 부재 + non-watertight 이므로
   **1 mm 복셀 점유**를 쓴다 (`p37` 의 `link5_occupancy()`). D453 과 같은 계열의 오류다.
5. ⚠️ **여유가 조절 파라미터에 거의 반응하지 않으면 그 파라미터는 원인이 아니다.**
   스탠드오프를 32→38 mm(+6) 키웠는데 여유가 −0.834→−0.744(+0.09) 였다. 진짜 원인은
   배치 변환의 깊이 **부호**였다. 무작정 그 축을 키웠으면 그랩만 멀어졌다.
6. ⚠️ **numpy 스칼라는 JSON 직렬화가 안 된다.** `np.bool_` 에서 두 번 죽었다. `default=` 핸들러 필수.
7. ⚠️ **생성기가 출력 폴더를 비우지 않는다.** `scoop_grab_v1/` 에 구버전 조각
   (`bracket_base_z{m,p}_*` 14개)이 잔류한다. 디스크에서 슬라이스하면 구 조각을 집을 수 있다.
8. ⚠️ **조각 STL 은 `.gitignore` 로 제외했다** (`claudedocs/runtime_logs/scoop_grab_*/*.stl`,
   `_ALL.stl` 만 예외). 미커밋 179 → 28 로 줄었다. 생성기로 결정론적 재생성 가능하다.

**승인 대기** (사용자 결정 없이는 착수 금지)

- 🔴 **`.npz` 저장 정책** (D463 §5-3). 전역 `*.npz` 무시로 DEME 더미 데이터가 git 추적 **0개**다.
  `GATES.md` 의 G2/G3/G4 CHECK 명령이 그 파일을 인자로 받으므로 **클론하면 게이트가 안 돈다**.
  DEME 는 bit-재현이 안 되므로(D463 §3) 잃으면 sha256 검증이 영구 불가.
  코디네이터 권장 = 게이트가 쓰는 **3개만 화이트리스트(1.83 MB)**, `g0b_d420` 선례와 같은 폴더 단위 패턴.
  학습 데이터는 npz 대신 heightmap 으로(2.73 GB → 69 MB). **D232 대상이라 승인 필요.**
- `permissions.allow` 에 orchestration 명령 추가 (함정 ③).
- 미커밋 백업 파일 처리.
- 펠릿 조달(진행 중) · 배출 용기 규격.

**검증 방법**

```
git worktree list                                    # 3개, Orca 등록부는 1개 (함정 ①)
orca-ide worktree list --json                        # 1개만 나온다
git log --oneline -3                                 # 9664f91 / 7b4ae74 / d1f4e32
python sim_scripts/p37_g2_grab_v1_attach_probe.py    # G1~G8 재현
git ls-files claudedocs/runtime_logs/sim_deme/ | grep -c npz   # 0 (승인 대기 §npz)
python -c "import trimesh;m=trimesh.load('local_assets/roarm_m3/urdf/meshes/link5.stl');\
m.apply_scale(1000);print(m.volume/m.convex_hull.volume)"      # 0.386 (함정 ④)
```
