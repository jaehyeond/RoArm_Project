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
## §2 2026-09-02 Claude → Codex (76th 세션)

세션 성격 = **실물 출력 2건 완주 + 워커 P4 트랙 판정.** 로봇 0 · 펠릿 0 · 신규 물리 시뮬 0.
원장은 코디네이터가 갱신 완료(D469~D471). 상태 정본은 `START_HERE.md`.

**한 일** (master `d9fc96a` 이후 원장 커밋 1건 더 붙는다)

| 커밋 | 내용 |
|---|---|
| `6b835df` | 전송 게이트 6번 — 발행 후 프린터가 실제로 잡았는지 확인 |
| `b26aa89`/`7630aa5`/`ae8a38a` | 워커 3트랙 머지 (decision-oracle · P4/P4b/P4c · N1/B1) |
| `a69901d`/`71487a1` | g10 접착 수정 + 무게 게이트 · 희생 발 + 링크 축 0° |
| `5e26b90`/`4bb7674`/`d9fc96a` | 원장 D467·D468 · D469~D471 |

**만지지 말 것**

- 🔴 **`claudedocs/runtime_logs/scoop_grab_v1/`(08-31)은 낡은 형상이다.** 읽지도 말 것.
  현재 형상 = `grab_track/g9_sidefix/`. 이걸 혼동한 것이 D470 사고다.
- `grab_track/g3_linkage/` — 설계 좌표계 STL 정본. `g9_oriented`·`g1*` 는 **출력용 좌표**라
  조립 변환을 걸면 틀린다.
- `scoop_shell_design.py`(v0)·`scoop_shell_v0/` — 동결. `scoop_grab_v1_design.py` 는 **워커 보호 파일**.
- `~/Documents/DK/DTR/bamboo-3dprinter/profiles/*_full.json`·`printer.py` — **DK 원본, 무수정.**
  사본만 고친다(`process_adh_roarm.json` 이 현행).
- 원장(`START_HERE`·`DECISIONS*`·`EXPERIMENT_LEDGER`·`session_*`·이 파일) — 코디네이터 배타 소유.

**🔴 함정 (전부 이번 세션 실측)**

1. 🔴 **하드코딩 경로가 낡은 설계를 조용히 읽는다.** `forward-only` 규칙이 옛 경로를 절대
   안 깨뜨리므로 **깨져서 발각될 기회가 없다.** 산출 JSON 에 `source_sha256_16` 을 남길 것.
2. 🔴 **`design.json` 의 `supersedes` 는 뒤만 가리킨다.** 그 폴더 안에서는 자기가 낡았는지 알 수 없다.
3. 🔴 **모니터 실행 중 별도 MQTT 조회 금지.** P1S 가 동시 클라이언트를 제한해 한쪽이 리셋된다 —
   오늘 "MQTT 먹통" 의 상당수가 이것이었다. 카메라(6000)는 영향 없다.
4. 🔴 **카메라는 접속 직후 버퍼된 옛 프레임을 준다.** 첫 장 버리고 둘째를 쓸 것.
   판을 치운 뒤인데 치우기 전 프레임이 나와 오보할 뻔했다.
5. 🔴 **복구 가능한 정지를 자동 중단하지 마라.** `print_error != 0` 이 뜬 단 한 번이 필라멘트
   소진(0x0300_8004)이었고 중단했다면 27층을 날렸다. 살아 있는 PAUSE 에는 `mqtt_resume()` 이 작동한다.
6. ⚠️ **`overhang` 지표는 베드에 닿는 아래 면까지 센다.** 희생 발을 붙이면 그만큼 늘어나는데
   실제 늘어질 면이 생긴 게 아니다.
7. ⚠️ **`trimesh` 단면(`section().to_2D()`)은 `shapely` 없으면 죽는다.** 바닥면 삼각형 직접 선별로 우회.
8. ⚠️ BambuStudio glfw 에러는 **무해**하다. CLI 슬라이싱은 정상 완료된다.

**승인 대기 / 사용자 결정 필요**

- 🔴 **브래킷·링크 바닥이 기능 정합면인가** — 희생 발 자국이 남는다. 출력 전 확인 필요.
- 🔴 **그랩 v2 착수 여부** — B1 까지 2.4 cm³ 부족, 보울 체적 확대 필요(바닥판으로는 안 됨).
- `orient_for_print.py` 에 "축 정확히 정렬" 후보 추가 여부(기존 성공 배향에 영향 가능).
- D470 처방 3건 구현 · 안전 후크 2건 배선 · 프린터 IP DHCP 예약 · 펠릿 조달.

**검증 방법**

```bash
git log --oneline -8
head -n 29347 claudedocs/DECISIONS.md | md5sum        # 956fcfe6... (D469 append 전 불변)
grep -n '^## Schema errata' claudedocs/EXPERIMENT_LEDGER.md   # 537 = 표 블록 끝 +2
python make_print_job.py <3mf> <dir> <name> <STL>     # 게이트 13종
```
