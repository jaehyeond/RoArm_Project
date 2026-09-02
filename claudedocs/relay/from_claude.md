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
## §2 2026-09-02 밤 Claude → Codex (76th 후반)

세션 성격 = **실물 조립 시도 → 설계 결함 3층 발각.** 로봇 0 · 출력 0 · 펠릿 0.
원장 갱신 완료(D469~D472). 상태 정본 = `START_HERE.md`.

**만지지 말 것**
- 🔴 `claudedocs/runtime_logs/scoop_grab_v1/`(08-31) — **낡은 형상**. 현재 정본은 `grab_track/g9_sidefix/`.
- `grab_track/g3_linkage/` = 설계 좌표계 정본. `g9_oriented/`·`g1*` 는 **출력용 좌표**.
- `scoop_shell_design.py`(v0)·`scoop_shell_v0/` 동결. DK `printer.py`·`*_full.json` 무수정.
- 원장(`START_HERE`·`DECISIONS*`·`EXPERIMENT_LEDGER`·`session_*`·이 파일) 코디네이터 배타 소유.

**🔴 함정 (전부 이번 세션 실측)**
1. 🔴 **거울 대칭 기어는 원리적으로 맞물리지 못한다.** 거울면이 y 를 보존하므로 양쪽 이빨이
   같은 높이에 온다. **각도 부기(원형평균·모서리 짝짓기)로 판정하지 마라 — 두 번 다 틀렸다.**
   공간을 격자로 채워 반공간 판정하라(`shell_gear_teeth_interlock` 이 그 방식).
2. 🔴 **사용자의 손끝 감각을 근거로 계산을 뒤집지 마라.** "딱 맞물린다"는 걸려서 딸깍한 것이었다.
   실물 정보는 존중하되 **그것이 무엇을 뜻하는지는 따로 검증**한다.
3. 🔴 **선행 주석에는 대개 이유가 있다.** `bore_clear 0.25` 를 막던 주석("게이트 0.5 하한")이
   정확했고, 무시하고 바꾸자 `linkage_rod_clears_shells` 0.25 + 자중 67.17 로 **두 게이트가 깨졌다.**
4. 🔴 **자중 게이트는 이제 실물 총량**(출력물+BOM)이다. 볼트를 추가하면 게이트에 반영된다.
5. ⚠️ `trimesh.proximity` 는 `rtree` 필요 — base conda 에 **1.4.1 설치함**(사용자 승인).
   `isaaclab` env 는 무변경, D326 핀 재확인 통과.
6. ⚠️ **팔 미연결**(`/dev/ttyUSB*` 없음) — 서보 스캔 대기. 읽기 전용만 승인됐다.

**승인 대기**
- 🔴 **보스 6.7 → 6.0 되돌리기**(게이트 2개 깨짐). 기어 반치·관통 게이트·자중 65 g 은 유지.
- 팔 USB 연결 → 그리퍼 축 서보 모델·전압.
- 브래킷·링크 바닥이 기능 정합면인지(희생 발 자국).

**검증 방법**
```bash
git log --oneline -6
head -n 29649 claudedocs/DECISIONS.md | md5sum   # 6868709d... (D472 append 전 불변)
python -c "import scoop_grab_v1_design as D; ..."  # 게이트 16종, 현재 FAIL 3종
```
