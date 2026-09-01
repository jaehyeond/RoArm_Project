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
## §2 <날짜> Claude → Codex (<n>th 세션)
세션 성격 / 한 일(커밋 표) / 만지지 말 것 / 함정 / 승인 대기 / 검증 방법
```

## §2 2026-09-01 Claude → Codex (75th 세션)

세션 성격 = **실물 3D 출력 3연속 실패 종결 + 워커 2트랙 회수 + 게이트 정비.**
실물 출력 4회(3실패 1성공) · DEME 시뮬 5회. **로봇 0 · 펠릿 0.**

**한 일** — HEAD 는 아래 커밋 뒤 원장 커밋 1건이 더 붙는다. 미커밋은 설정 백업 1건뿐.

| 커밋 | 내용 |
|---|---|
| `478a100`/`e63b940` | P1 deme-force 회수 + merge |
| `c7c052f`/`a575129` | P2b kinect-hm 회수 + merge |
| `c415594` | P2b D341 검수 png 화이트리스트 |
| `38d5880` | P1 n=3 + 설정 동일성 게이트 + npz 화이트리스트 |
| `1445e69` | 출력 파이프라인 정정 2회 + 게이트 3종 + P1 n=5 + 조립 뷰어 |

**만지지 말 것**

- `claudedocs/runtime_logs/grab_track/g3_linkage/` — **설계 좌표계 STL 정본.**
  `g4_flat`·`g5_oriented`·`g7_oriented` 는 **출력용으로 눕힌 좌표**라 조립 변환을 걸면 틀린다.
- `scoop_shell_design.py`(v0)·`scoop_shell_v0/` — 동결. forward-only.
- `~/Documents/DK/DTR/bamboo-3dprinter/profiles/*_full.json`·`print_cli.py` — **DK 원본, 무수정.**
  사본만 고친다(`process_nosupport_roarm.json`·`process_support_roarm.json`).
- 원장(`START_HERE`·`DECISIONS*`·`session_*`·이 파일) — 코디네이터 배타 소유.
- `.claude/settings.local.json.bak_20260831_pre_orca_allow` — 되돌리기 경로. 커밋 대상 아님.

**🔴 함정 (전부 이번 세션 실측)**

1. 🔴 **BambuStudio 프로필 JSON 은 값이 전부 문자열이어야 한다.** `brim_width` 만 정수 `8` 로
   적혀 있어 슬라이서가 **오류 없이 무시하고 0** 으로 떨어뜨렸다 → 브림 0 인 gcode 로 출력해
   부품 4개가 전량 탈락했다. 프로필을 고칠 때는 **전 키 타입 검사**를 돌릴 것.
2. 🔴 **게이트는 "슬라이서가 실제로 쓴 값"을 봐야 한다.** 프로필을 읽으면 내가 넣으려던 값을
   보게 된다. 3mf 안 `Metadata/project_settings.config` 가 실제 값이고, 그것도 부족해서
   **gcode 툴패스**(`; FEATURE: Brim` / `Support`)까지 세야 한다.
3. 🔴 **배향을 접지 하나로 최적화하면 안 된다.** 기어축이 90° 눕고(이빨이 층으로 쌓임)
   오버행이 5.8배가 됐다. 힌지축(설계 Z)에 기어·피벗보스·핀·로드아이·셸크랭크허브가
   **전부 동축**이므로 그 축을 수직으로 세우는 제약을 먼저 건다.
   ⚠️ **원통의 축은 bbox 최소축이 아니다** — 원판은 최소축, 긴 축은 최대축이다.
   **나머지 둘(=지름)과 다른 축**이 맞다. 첫 판정이 이걸로 틀렸다.
4. 🔴 **`support_on_build_plate_only=1` 은 모델 위 오버행을 못 받친다.** 볼트 구멍 오염은
   막지만 서포트가 z=7.40 mm 에서 끊겼다(오버행 최고 57.5 mm, 무방비 243 mm²).
5. 🔴 **스크립트 하드코딩 기본값이 아티팩트의 `params` 와 다를 수 있다.**
   `sim_deme_scoop.py` 기본값(`close_end_deg=0`·`insert_depth_mm=18`)이 rep1/rep2 를 만든
   값(6·8)과 달랐고, `close_end_deg=0` 은 **이미 발산한다고 문서화된 값**이었다.
   재실행 전 `params` 를 26개 전수 대조할 것. `force_stats()` 가 이제 강제한다.
6. ⚠️ **`--orient 1`(슬라이서 자동 배향)을 믿지 마라.** 09-01 실측에서 높이를 59.5→60.7 로 키웠다.
7. ⚠️ **`make_print_job.py` 지역 변수는 모듈 상단 이름과 겹치면 안 된다.** `_a`(argv)를
   면적 배열로 덮어써서 죽었다.
8. ⚠️ **Bambu MQTT 는 바뀐 필드만 보낸다.** 매번 새 dict 로 덮어쓰면 `UNKNOWN`·`베드 0.0` 이
   나온다. **누적 병합**해야 한다.
9. ⚠️ **파이프 뒤 종료코드는 마지막 명령의 것이다.** `python ... | tail -1` 로 만든
   `until` 루프가 조기 탈출했다.
10. ⚠️ 워커 산출물의 **`.npz`/`.png` 는 머지로 따라오지 않는다**(gitignore). worktree 에서
    복사해야 하고, 게이트·판정의 **입력**이면 화이트리스트해야 한다.

**승인 대기 / 사용자 결정 필요**

- 🔴 **브래킷 출력 방법 미정.** 기능축을 세우면 접지 **41 mm²**(높이 59 mm)로 1차 실패 수준이다.
  배향으로 못 푼다 — 브림 확대·러프트·형상 분할 중 결정 필요.
- 🔴 **임계값 2개가 근거 없다**: 1층 접지 10 mm²/mm · 무방비 오버행 300 mm².
  **실패 사례에서만** 잡았다. g7 실물(무방비 243 mm²)이 첫 교정 데이터 — 사용자 실물 검수 대기 중.
- P1 **D341 미이행 3건**(`.rbl`·헤드리스 스크린샷·육안검수). 루트 `GATES.md` **G7~G13 전부 미완**.
- 안전 후크 2건 미배선(D458 §6) · 프린터 IP DHCP 예약 · 펠릿 조달 · 배출 용기 규격.

**검증 방법**

```bash
git log --oneline -6
python make_print_job.py <3mf> <dir> <name> <STL>     # 게이트 11종
python sim_deme_scoop_report.py                       # n=5 통계 + 설정 동일성 게이트
head -n 28958 claudedocs/DECISIONS.md | md5sum        # f48770d7... (D465 append 전 불변)
grep -n '^## Schema errata' claudedocs/EXPERIMENT_LEDGER.md   # 536 = 표 블록 끝 +2
```
