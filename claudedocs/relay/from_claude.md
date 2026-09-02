# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약

- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo를 여는 **다른 도구**(Codex CLI / Cursor).
  Claude가 연속으로 두 번 열려도 이 파일이 아니라 `START_HERE.md`로 재개한다.
- **덮어쓰기.** append-only 아님. 최신 인계 1건만 §2에 둔다.
- **상태 정본이 아니다.** 현재 상태·active case·다음 행동·수치는 전부 `START_HERE.md`가 소유하며
  **여기에 한 줄도 베끼지 않는다.** 어긋나면 `START_HERE.md`가 이긴다.
- **`HANDOFF.md`가 아니다.** HARD RULE #7은 그 파일명을 금지한다. 이 파일은 그 이름을 쓰지 않는다.
- **중복 금지 지도**: 현재 상태 → `START_HERE.md` / 활성 결정 → `DECISIONS_ACTIVE.md` → `DECISIONS.md` /
  최근 실험 → `LEDGER_RECENT.md` → `EXPERIMENT_LEDGER.md` / 규칙 → `AGENTS.md`.
  **여기 적을 것은 그 어디에도 안 들어가는 것뿐** — 직전 도구가 만진 것, 만지지 말 것, 함정, 승인 대기.

## §2 2026-09-02 심야 Claude → Codex (76th 연장)

세션 성격 = **D472 순서 실행 → Phase 1 조립 정의 완료 + 실물 로봇 장착 검증.**
로봇 = 읽기 전용 T:105 1회만. 출력 0 · 펠릿 0. 원장 전량 갱신 완료(D473). 상태 정본 = `START_HERE.md`.
커밋 3건 = `bcde1df`(보스 되돌리기+벽1.6) · `541cda3`(요크+장착검증+알루볼트+게이트3종) · `d58f6b6`(G8b+롤제약).

**만지지 말 것**
- 🔴 `grab_track/g9_sidefix/`(09-01)·`g16~g16e/`(09-02 반복본) — **낡음**. 정본은 `g17_yoke_alu/`.
- `grab_track/g3_linkage/` = 설계 좌표계 정본. `scoop_shell_design.py`(v0)·`scoop_shell_v0/` 동결.
- 원장(`START_HERE`·`DECISIONS*`·`EXPERIMENT_LEDGER`·`LEDGER_RECENT`·`session_*`·이 파일) 코디네이터 배타 소유.
- DK `printer.py`·`*_full.json` 무수정.

**🔴 함정 (전부 이번 세션 실측)**
1. 🔴 **커스텀 EOAT 는 자기 내부 간섭만 보면 안 된다** — 붙을 로봇 본체(link5·link4·순정 조) 간섭을
   같이 봐야 한다. `p37_g2_grab_v1_attach_probe.py`(`build_bracket` import → 브래킷 변경 자동 검증)로 돌린다.
   verdict `G2_ATTACH_OK` 여야 한다.
2. 🔴 **단일 자세 게이트가 "전 구간"을 주장하면 안 된다** — p37 G8 이 롤 0 한 자세만 보며 "롤 전 구간"이라
   적었다가 개방+대롤 servocrank↔link4 충돌을 놓쳤다. 신설 **G8b** 가 롤 스윕을 본다.
3. 🔴 **벽두께 함정** — 정본 셸은 `GRAB_WALL_MM=1.6` override 산물이었는데 파일 기본값이 2.0 였다.
   기본값을 1.6 으로 정합해 뒀다. 이제 override 없이 돌려도 정본과 일치. env override 남발 주의.
4. 🔴 **손목 롤 제약** = 개구>44 mm → |롤| 제한(완전개방 |롤|≤14°). **구성 의존 한계, 문서화됨**(재설계 안 함).
   충돌부는 servocrank(링크 선재)지 요크 아님(요크는 link4 서 44 mm).
5. ⚠️ **서보 모델·전압은 시리얼로 못 읽는다**(펌웨어 확정, T:1051=위치+부하뿐). 육안만.
6. ⚠️ **자중 게이트 65 는 사용자값** — 넘겨서 통과시키지 말 것. 알루미늄 피벗볼트로 61.41 로 맞췄다.

**승인 대기 (팔 필요)**
- 어댑터 라벨(12V?)·그리퍼 서보 하우징 재질/각인 육안 → Phase 0b 종결.
- 서보 출력 인출 실측(순정 가동 조 볼트 스팬 25 mm) → Phase 2 잔여.
- 펠릿(3–5 mm 불투명 백색 2종)·배출 용기 조달.

**검증 방법**
```bash
git log --oneline -4                         # d58f6b6 · 541cda3 · bcde1df · 341de18
head -n 29762 claudedocs/DECISIONS.md | md5sum   # f47d107d0c97... (D473 append 전 불변)
python scoop_grab_v1_design.py /tmp/x         # 게이트: self_load_ratio 만 FAIL(펠릿 대기), 나머지 PASS
python sim_scripts/p37_g2_grab_v1_attach_probe.py /tmp/y   # verdict G2_ATTACH_OK, G8b PASS
```
