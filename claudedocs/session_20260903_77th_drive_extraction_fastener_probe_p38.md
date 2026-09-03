# 77th 세션 — 구동 인출부(순정 가동 조 볼트 → 서보 크랭크판) 체결구까지 3D 검증 → `G2_DRIVE_EXTRACTION_BLOCKED`

2026-09-03 · Claude · master `4e563d1` 기준 (이 세션의 코드·문서는 **미커밋** — 사용자 요청 없음)
**이번 case 의 신규 변수: 전반 0 / 후반 2** (전반 = 검증만. 후반 §7~ = 사용자 "권고대로 진행해" 이후 ① 브래킷 체결 패턴 ② 크랭크 체결 종단)

이 문서는 append-only 상세 로그다. 요약·판정은 `DECISIONS.md` **D475 `:29879`**(전반, BLOCKED) · **D476 `:29936`**(후반, 해소) 가 정본이고, 여기 수치와 어긋나면 **원문이 이긴다.**

---

## 0. 이 세션이 한 일 (한 줄)

D474 ③이 "팔 필요"라고 넘긴 Phase 2 잔여 — **순정 가동 조 볼트(스팬 25) → 서보 크랭크 → 4절 → 셸 의 구동 인출이 실물에서 성립하는가** — 를
팔 없이 3D 로 먼저 검증했다(사용자 권고 "바로 할 것 1)"). 구멍·좌면은 맞았고(G10·G11 PASS), **체결구 자리가 없었다**(G12·G13 FAIL).

로봇 0 · 출력 0 · 물리 시뮬 0 · 펠릿 0. 순수 기하(메쉬 + 볼록 조각 + ISO 공칭 체결구).

---

## 1. 무엇을·왜

p37(D463~D473)은 그랩 **몸체**(셸·브래킷·링크)와 로봇 본체(link5·link4·순정 가동 조)의 간섭을 스윕했고 9종 전부 PASS 였다.
그러나 **체결구(볼트 머리·너트·볼트 꼬리)는 한 번도 모델되지 않았다.** 설계 게이트 `fastener_seat_present` 는 스스로
"머리/너트가 이웃 부품과 간섭하는지, 렌치 접근이 되는지는 여기서 안 본다"고 적어 두었다.

구동 인출부는 그 사각지대 한가운데다: 크랭크판은 **두 순정 조 사이 간극**(4.05 mm)에 들어가고, 바로 그 자리에
브래킷 볼트가 관통하는 고정 조 구멍쌍(Z 83.46)과 크랭크 볼트가 관통하는 가동 조 구멍쌍(Z 82.98)이 **0.48 mm 차이로 겹쳐** 있다.

## 2. 절차 (관측 가능한 순서)

1. 부팅 절차 5단계 read → START_HERE(09-03 새벽판)·DECISIONS_ACTIVE §5·LEDGER_RECENT :537·relay·git.
2. D474·D462·D463·D464·D473 원문 + `PHASE1_ASSEMBLY_DEFINITION.md` + p37 소스 + `scoop_grab_v1_design.py` `build_linkage` 를 읽어
   크랭크판이 어디에·어떻게 물리는지 확인: `jaw_bolt_yz_mm = [[-12.66, 82.98],[12.45, 82.98]]`, 판 두께 3.0, 판 = 가동 조 **안쪽면**(x −6.0).
3. 순정 가동 조 메쉬(`gripper_link.stl`)를 p37 `jaw_in_link5()` 로 서보 0° link5 프레임에 놓고 단면·래스터·점유 스캔:
   - 블레이드 x[−5.98, −4.47](1.51) · 구멍 (−12.66, 82.96) ⌀3.18 · (12.45, 82.97) ⌀3.19 (래스터 0.25) — 설계값과 0.02 mm.
   - 바깥면 밖(x > −4.47) 구멍 둘레 반경 3 mm 원통에 12 mm 까지 재료 0. 측벽 |Y| 18.3–19.8 (x −5.98–0.7, Z 41–87), 림 x 8–15.3.
4. **의심점 식별**: 고정 조 안쪽면 −10.03 ↔ 가동 조 안쪽면 −5.98 = 4.046. 크랭크판 x[−9.0, −6.0] → 잔여 **1.03**.
   그 자리에 브래킷 "반대편 너트"(BOM)와 크랭크 볼트 머리가 동시에 와야 한다.
5. 전용 프로브 `sim_scripts/p38_g2_drive_extraction_probe.py` 작성(p37 헬퍼 import, 서보 0° 정적, 체결구 = ISO 공칭치 표 내장):
   G10 구멍 일치 / G11 좌면 접촉·무관통 / G12 크랭크 볼트 자리 / G13 브래킷 4볼트 자리(체결 방식 A·B·C).
6. 실행 → `g17_yoke_alu/p38_drive/{p38_results.json, p38_fastener_sections.png}`. 단면 그림 육안 검수(한글 폰트 Noto Sans CJK 지정 후 재생성).
7. 대안 설계용 추가 실측: 측벽 창 Z 범위, 팁 구멍(Y −0.75, Z 115.91) 안쪽 여유, 고정 블레이드 Y 폭, 가동 조 바깥 영역 점유.

## 3. 결과 (수치 + 출처)

출처 = `claudedocs/runtime_logs/grab_track/g17_yoke_alu/p38_drive/p38_results.json` (입력 design.json sha256_16 `67144ad0248103b5`).

```
G10 crank_holes_match_stock_jaw      PASS  max_err 0.02 mm   고정↔가동 구멍쌍 ΔY −0.68/−0.60 · ΔZ 0.48
G11 crank_plate_seats_on_jaw         PASS  seat_gap 0.016    웨브 0.087 · 암/아이/핀 7.949 (무관통)   발자국 재료비 0.51(프레임 조)
G12 crank_fastener_envelope          FAIL  바깥 자유 12.0(≥) / 안쪽 잔여 1.05(스텝 0.05, 정확 1.03) < 1.35+0.3
G13 bracket_fastener_envelope        FAIL  A(BOM ×10+안쪽 너트) 4구멍 ❌ · B(안쪽 버튼머리) Z83 ❌ Z103 ✅ · C(×8+안쪽 너트) Z83 ❌ Z103 ✅
verdict = G2_DRIVE_EXTRACTION_BLOCKED
```

| 항목 | 값 (link5 mm) | 근거 |
|---|---|---|
| 고정 조 블레이드 | x[−11.54, −10.03] (1.51) | D462 §2 (p37 상수) |
| 가동 조 블레이드 @구멍 | x[−5.98, −4.47] (1.51) | p38 evidence `movable_blade_x` |
| 블레이드 간극 | **4.046** | evidence `gap_between_blades_mm` |
| 크랭크판 | x[−9.0, −6.0] · Y[−19.24, 15.76] · Z[77.48, 88.48] | evidence `crank_plate_bounds` |
| 잔여 간극 (고정 조 안쪽면 ↔ 판 −x 면) | **1.03** | evidence `gap_residual…` |
| 브래킷 볼트판 | x[−15.54, −11.54] (4.0) · Y[−18.34, 16.85] · Z[78.46, 107.9] | evidence `bracket_bolt_plate_bounds` |
| BOM 볼트 M2.5×10 안쪽 돌출 | 10 − 4.0 − 1.51 = **4.49** | G13 scheme A |
| 팁 구멍 안쪽 여유 / 고정 조 팁 폭 | 4.05 / Y[−7.94, 6.91] | 이 세션 스크래치 실측 (§2-7) |
| 측벽 창 (Y −19) | Z 66–86 에서 x 0.7–9.7 (Z 84: 0.7–7.0 · Z 86: 0.7–5.0), Z ≥ 88 닫힘 | 동상 |
| 가동 조 바깥 영역 x(−4.47, 8)×|Y|<17.5×Z78–88 | 재료 0 (0 히트) | 동상 |

체결구 공칭(ISO, 실물 대조 전): 소켓캡 k 2.5 · 버튼 k 1.35 · 접시 k 1.5(카운터싱크 필요) · 너트 m 2.0(5 AF) · 얇은너트 1.6 · 히트셋 인서트 L 3.0/OD 3.5(벤더 의존).

### 3.1 단면 그림 육안 검수 (`p38_fastener_sections.png`)

- XY@Z 83.2: 검정 = link5(고정 블레이드 두 선 x −11.54/−10.03, 구멍 자리에 노치), 파랑 = 가동 조(블레이드 + 구멍 + 측벽 L자), 주황 = 크랭크판, 초록 = 브래킷 볼트판.
  빨강(브래킷 너트 2.0, 안쪽면)이 주황과 약 1 mm 겹치고, 보라(크랭크 버튼머리 1.35, 판 안쪽면)가 고정 블레이드 선을 넘는다. "잔여 간극 1.03" 화살표.
- XZ@Y −13: Z 83.46 너트가 크랭크판 조각과 겹침, Z 102.9 너트는 파랑 블레이드까지 2.05 여유. 의도한 대비가 그대로 보인다.
- ⚠️ Rerun RRD 미기록(D341 계약 미충족) — 정적 단면 2장 + JSON 이 판정 근거. p37 계열(D463·D473)과 같은 상태이며 이 문단이 그 명시다.

## 4. 판정 (일상어) + 다음 승인 경계

**크랭크판과 순정 가동 조는 구멍이 맞고 딱 붙는다. 그러나 볼트를 조일 자리가 없다.** 두 순정 조 사이 4 mm 틈에 크랭크판 3 mm 를 넣으면 1 mm 만 남는데,
그 1 mm 안에 브래킷 볼트의 너트(2 mm)와 크랭크 볼트의 머리(최소 1.35 mm)가 같은 위치에 들어가야 한다. 어떤 시판 볼트로도 안 된다.
게다가 BOM 의 M2.5×10 은 브래킷 판과 블레이드를 뚫고 4.5 mm 가 남아 **네 구멍 모두** 반대편 가동 조에 닿고, 크랭크 볼트는 BOM 에 아예 없다.

→ **Phase 2 는 "팔만 있으면 되는 잔여"가 아니라 설계 수정이 먼저인 BLOCKED 다.** 팔이 답할 것은 순정 구멍의 **나사산 유무**(탭이면 너트가 필요 없어 표가 바뀐다)·실측 ⌀·전압뿐이다.

**다음 승인 경계 (사용자 결정)** — D475 §3 의 실측 사실 위에서 택1:
- (가) 크랭크판을 가동 조 **바깥면**으로 (간극 비움, 4구멍 성립) — 웨브가 측벽 창(Z 66–86)을 지나야 하고 링크 스윕 전부 재실행.
- (나) 브래킷 Z 83.46 쌍 포기 → Z 102.9 쌍 + 팁 구멍 3점 — 링크 무수정, 대가 = 레버 19.44 → 12.99 mm(−33%), 팁 얇음, 볼트판 연장.
- (다) 크랭크 체결 종단(히트셋 인서트 L3.0 / 포켓 너트) + BOM 항목 추가 — (가)(나) 공통 필수.
- (라) 순정 블레이드 카운터싱크 — 비가역 개조라 규칙상 ✗ (사용자만 뒤집을 수 있음).
어느 쪽이든 **p37 + p38 재실행 PASS** 가 Phase 2 종결 조건이다. 실물 확인 항목은 START_HERE "우선순위 1" 에 적었다.

## 5. 이 세션이 주장하지 않는 것

- 실물 미확인: 나사산 유무·⌀·재질·서보 0° 실물 닫힘각·그리퍼 서보 전압. 체결구 치수는 ISO 공칭.
- 서보 토크 대 폐합 부하(펠릿 물성 대기). 인서트 열삽입 강도·렌치 접근성. 동적·처짐.
- 대안 (가)(나)(다)는 열거만 — 미구현·미검증. `G2_ATTACH_OK`(D473 ④)는 몸체 판정으로 계속 유효.
- 이 세션은 실험(학습/섭동) 0 — Session progress rule 정당화: 실물 조립 성립 여부를 가르는 기하 판정이며 결과가 다음 결정을 바꿨다(BLOCKED).

## 6. 산출물

- 코드: `sim_scripts/p38_g2_drive_extraction_probe.py` (신규, p37 import)
- 결과: `claudedocs/runtime_logs/grab_track/g17_yoke_alu/p38_drive/p38_results.json` · `p38_fastener_sections.png`
- 원장: `DECISIONS.md` D475 `:29879` (append 전 md5 `bcccb20a…` = `head -n 29877` 불변) · `EXPERIMENT_LEDGER.md :538` (표 블록 끝, errata `:540`)
- 상태: `START_HERE.md` overwrite · `DECISIONS_ACTIVE.md` §5·§9 · `LEDGER_RECENT.md` · `relay/from_claude.md` · auto-memory `MEMORY.md`
- 백업: `DECISIONS.md.bak_20260903_pre_d475` · `EXPERIMENT_LEDGER.md.bak_20260903_pre_77th`
- 스크래치(비산출): `/tmp/claude-1000/…/scratchpad/scan_iface.py` (§2-3·§2-7 실측 재현)

---

# 2부 (후반) — 사용자 "권고대로 진행해" → (나)+(다) 이행 → `g18_nut_trap` → p37·p38 PASS → URDF·USD 재생성

## 7. 무엇을·왜

D475 §3 의 권고 (나)(브래킷 Z 83.46 쌍 포기 → Z 102.9 쌍 + 팁 3점) + (다)(크랭크 체결을 판 안에서 종단) 를 사용자가 채택했다.
종단 방식은 (다)의 두 선택지 중 **너트 트랩/포켓**(기존 M2.5 너트, 구매·납땜 0) 을 골랐다 — 인서트는 벤더 치수가 미확정이라 src UNCONFIRMED 리스크가 하나 더 붙는다.

## 8. 절차 (관측 가능한 순서) — 세 번 고쳤다

1. 코드 읽기: `build_bracket()` 의 볼트판이 `plate_with_holes(hole_axis="x")` 로 생성됨을 확인 → **구멍이 밴드 중앙(z=0)에만** 생기는 함수였다.
   g17 조각 좌표로 확정: `bolt_plate_1/2` Y[−18.34,−2.44]·[0.95,16.85] → 구멍 = Y −2.44~0.95(중앙선). 블레이드 구멍(Y −13.34/11.85)과 무관.
   p37 G3 는 `dy/dz` 파라미터끼리 비교 → 0.00 PASS(거짓). p38 1차 그림의 초록 판 가운데 빈 틈이 그 구멍이었다(당시엔 link5 보스로 오독).
2. **1차 설계**(터널안): 3점 구멍을 `to_local` 로 옮겨 **신설 `plate_holes_x()`** 로 판 생성 · 쌍 구멍 바깥면에 U-터널(측벽 2 + 캡 1) · 팁 구멍은 스파인 레일 뿌리 슬롯(레일 3조각) · 크랭크판 2층(바닥 1.0 + 포켓 2.0) · 백포스트 y 11→12(슬롯과 0.87 겹침 회피) · BOM 7종 · 게이트 `fastener_stack_terminates` 신설. 설계 19/20.
3. p37 실행 → **G3 FAIL(구멍 1/3 검출)·G6 FAIL(터널 측벽이 link5 에 −0.895)**, G7 0.007. 조사:
   - G3: 구멍은 존재·폐쇄(래스터 열/행 전이로 확인). 격자가 조각 경계 Z 101.2 와 정확히 겹쳐 경계선이 비고 fill_holes 가 구멍을 바깥과 이었다 → 반 피치 오프셋 + closing.
   - G6: link5 단면 Z 96~106 에서 x<−11.6 재료가 **|Y|≥15.22/16.25 양 옆**에만 있고 x −22.5(Z 96)→−12.6(Z 106) 로 45° 경사 = **플랜지**. 터널 상자 안 link5 표면점 205/157. **옛 g17 판 상자(x<−11.7) 안 link5 표면점 12,854** — 옛 판도 관통. G6 는 체결면을 x 관통만 검사해 가렸다.
   - G7: p37 `clearance_to` 경로 재현 → 0.007 은 **AABB 분리거리**(조각 경계상자가 link5 경계상자 밖으로 0.007) — 실제 거리 11.27 mm. g17 의 0.964 도 동일.
4. **2차 설계**(현 g18): 쌍 구멍 = 바깥 볼트(머리 판 위) + 안쪽 너트(간극 2.49 < 4.05), 터널 제거 · 팁 = 슬롯 유지 · 판 폭 = 플랜지 안쪽 −0.3 = link5 Y[−15.95, 14.92] (실측 파라미터 `link5_flange_y_edges`·`link5_flange_z_max` 신설). 바깥 머리·렌치 경로(x −40~−15.54)·안쪽 너트 자리 무점유 실측.
5. p37 정정: G3 = 판 조각 래스터로 실제 구멍 검출·대조 / G6 = 체결면 조각 vs 블레이드 면 점을 뺀 link5 표면점구름 ≥ −0.02 / `clearance_to` = AABB 하한 <1 mm 면 정확 재측정.
   p38 v2: G13 을 구멍별 방식(쌍 outside_in · 팁 inside_out)으로, 바깥 요소 vs link5 · 렌치 경로 · 간극 안 요소(너트+꼬리/머리) 스윕 추가.
6. 실행: 설계 19/20 → **p37 `G2_ATTACH_OK`** → **p38 `G2_DRIVE_EXTRACTION_OK`** → 3면도·단면 그림 육안 검수.
7. 자산: `export_grab_urdf.py g18_nut_trap`(소스 인자화) → `compose_roarm_grab_urdf.py g18_nut_trap/urdf` → `sim_urdf_to_usd.py --collider convex_hull`(isaaclab, 핀 확인 numpy 1.26.0·psutil 5.9.8) → `config.yaml` 복구 → pxr 검증.

## 9. 결과 (수치 + 출처)

출처 = `g18_nut_trap/design.json`(sha `1e1cf09073719b3c`) · `p37_attach/g2_results.json` · `p38_drive/p38_results.json` · `urdf/grab_v1_meta.json`.

| 항목 | 값 |
|---|---|
| 설계 게이트 | 19/20 PASS (FAIL = self_load_ratio 펠릿 밀도 대기). 신설 `fastener_stack_terminates` PASS |
| 자중 | **58.53 g** = 출력물 51.26 + 하드웨어 7.27 (M3×75 알루 ×2·M3 너트·와셔 / M2.5×8 버튼 ×3·너트 ×3 / M2.5×4 버튼 ×2·너트 ×2) |
| p37 | G3 0.022(3/3 실측) · G4 1.001 · G6 0.5, 플랜지 0.296 · G7 1.026 · G8 14.326 · G8b 4.411/44.0 · G9 1.023 → **G2_ATTACH_OK** |
| p38 | G10 0.02 · G11 0.016 · G12 물림 1.49/돌출 0 · G13 쌍 4.05≥2.79, 머리 vs link5 0.59/1.06, 렌치 경로 OK · 팁 4.05≥1.65, 슬롯 OK · 간극 요소 스윕 1.56 → **G2_DRIVE_EXTRACTION_OK** |
| URDF | grab_base 12.99 g · 셸 15.17 ×2 · 합성 12링크/11조인트/collision 360(벤더 7 + 조각 353) · 미해결 메시 0 |
| USD | 프록시 포함 2,275 prim · 그랩 collision 353 convexHull · `bracket_spine_rail_01`·`bolt_plate_7` 존재, `nut_tunnel` 0 · 조인트 11(셸 R 독립 revolute) · ArticulationRoot `/roarm_m3/world` |
| 판 폭 | link5 Y [−15.95, 14.92] (플랜지 −16.25/15.22 − 0.3). 쌍 구멍 랜드 0.86/1.35 |

### 9.1 육안 검수

- `p38_drive/p38_fastener_sections.png`: XZ@Y −13.34 — 보라 머리가 판 바깥, 빨강 너트+꼬리가 간극 안(파랑 가동 조까지 1.56), 주황 크랭크판 안 빨강 포켓 너트, "Z83.46 쌍 포기" 주석. XY@Z 102.9 — 판이 검정 플랜지 곡선 안쪽에 머물고 머리가 플랜지와 0.59/1.06. XY@Z 115.91 — 올리브 레일 뿌리에 슬롯, 빨강 너트, 간극 안 보라 머리.
- `bracket_3view.png`(AABB 근사): 판·레일·요크·크랭크 배치가 link5·순정 조 점군과 겹치지 않는다. Rerun RRD 미기록(정적 단면·3면도가 근거, 1부와 동일 상태).

## 10. 판정 (일상어) + 다음 승인 경계

**볼트를 조일 자리가 생겼다.** 브래킷은 고정 조 구멍 3개에, 크랭크판은 가동 조 구멍 2개에 각각 표준 M2.5 볼트·너트로 물리고, 머리·너트·볼트 꼬리가 어디에도 닿지 않는다(간극·플랜지·순정 조·크랭크 스윕 전부 실측). 그 과정에서 **g17 까지의 브래킷은 애초에 볼트가 들어갈 수 없는 판**(구멍 중앙선)이었고 **link5 플랜지에 박혀 있었다**는 사실이 드러났다 — 몸체 게이트 9종이 전부 PASS 인 채로.

→ **Phase 2 = 설계상 종결.** 남은 것은 실물: 순정 구멍 나사산 유무·⌀·재질, 어댑터 전압. 그 다음 Phase 4(갭 항목 등록) → Phase 5(출력: `docs/reference/printing.md` 필독, g18 STL).
**다음 승인 경계** = 출력 착수(g18 브래킷·링크·셸) 또는 DEME 연결 — 사용자 결정.

## 11. 이 세션이 주장하지 않는 것 (2부)

- 실물 미확인 항목은 1부와 동일. 너트 물림 1.49 mm(3.3 산)·PLA 포켓 바닥 1.0·쌍 구멍 랜드 0.86/1.35·팁 블레이드 강도·간극 안 너트 조립 손놀림은 **실물 시험 대상**.
- 출력 0 · 물리 0 · 토크 미계산 · RRD 없음. USD 는 자산·운동학·시각화용(D474 ⑥).
- 실험(학습/섭동) 0 — 정당화: 조립 성립 여부를 가르는 기하 판정이며 결과가 결정을 바꿨다(BLOCKED → 설계상 종결).

## 12. 산출물 (2부)

- 코드: `scoop_grab_v1_design.py`(백업 `.bak_20260903_pre_nuttrap`; `plate_holes_x`·3점·포켓·플랜지 파라미터·게이트 ⑱) · `sim_scripts/p37_*.py`(G3·G6·clearance_to 정정) · `sim_scripts/p38_*.py`(v2, 백업 `.bak_20260903_pre_nuttrap`) · `export_grab_urdf.py`·`compose_roarm_grab_urdf.py`(인자화, 기본 g18)
- 형상·검증: `claudedocs/runtime_logs/grab_track/g18_nut_trap/`(STL 500 · design.json · p37_attach · p38_drive · urdf · bracket_3view.png)
- 자산: `local_assets/roarm_m3/urdf/roarm_m3_with_grab.urdf` + `meshes/{grab_*.stl, collision/*}`(추적) · `usd/roarm_m3_with_grab.usd`(gitignore, 재생성) · `usd/config.yaml` 복구
- 원장: D476 `:29936` · LEDGER `:539` · 상태 문서 일체. 백업 `DECISIONS.md.bak_20260903_pre_d476`. **전부 미커밋.**

---

# 3부 (후반 2) — 사용자 "시각적으로도 확인하면서 해 … sim 에서 렌더 … Isaac Lab 병렬 수천·수만" → 3층 검증 → D477

## 13. 절차 (관측 가능한 순서)

1. **① matplotlib 실메쉬**(`sim_viz_grab_assembly.py`): link5 + 순정 조(서보각 회전) + 브래킷 + 4절(`linkage_pose`) + 셸(±44.5°) 를 삼각형 그대로.
   닫힘/개방 × 등각·앞·옆·위 + 체결부 근접 → `g18_nut_trap/viz/assembly_4view.png`, `assembly_fastening_closeup.png`. 육안: 순정 조 89° 눕고 링크가 따라감, 판 아래 가장자리가 플랜지 위.
2. **② Isaac RTX**: `sim_render_grab_usd.py`(D474) → `g18_bowl_{closed,open}.png` **동일 그림, 로봇 프레임 아래 잘림** — 옛 `bowl_*.png` 도 같음 → 카메라 0.72 m 결함(D474 자기 경고 0.85 m 위반).
   `sim_render_robot_full.py` → `g18_robot_full.png` 정상(팔 끝 클램셸 개방).
3. **③ Isaac Lab 병렬**(`sim_isaaclab_parallel_smoke.py`, `InteractiveScene` 복제, `ImplicitActuator` 300/30, dt 1/120, 셸 0→0.777→0 사인 240 스텝):
   64 env+카메라 1차 = BasicWriter 폭주(32 GB/23,646 PNG, 14 분) → 강제 종료·삭제. 512 env = **FAIL**: 셸 R 0.931 rad 상한 고착(L 0.036), 전 환경 동일.
4. USD 검사(pxr): `grab_shell_R_joint` 에 드라이브 0, `physxMimicJoint gearing −1·25 Hz·ζ 0.005`, 한계 −8.9°~53.4° → 소스 추적: `isaaclab/.../urdf_converter.py:130` `set_parse_mimic(cfg.convert_mimic_joints_to_normal_joints)` + `isaacsim.asset.importer.urdf/tests/test_urdf.py:475~491`(parse_mimic=True ⇒ PhysxMimicJointAPI) → **플래그 반전**.
5. `sim_urdf_to_usd.py` 플래그 **False** → USD 재생성(exit 0, `config.yaml` 복구) → pxr: 셸 R 드라이브 = L 동일, 한계 0~44.5°, mimic 0.
6. 재실행: **512 env ok**(`smoke_512.json`) · **64 env ok**(`smoke_64.json`, 격자 스냅샷은 orchestrator 행으로 미생성 → 8.7 h 멈춤, `timeout 900` 무시 → `kill -9`).
7. 근접 렌더 신설 `sim_render_grab_closeup.py`(grab_base 자동 조준·1.0 m·70 mm·바닥 없음·관절 읽기값 JSON): 1차 = 물리 뷰 소실로 2장만(Replicator 캡처가 타임라인 정지) → annotator + `pause_timeline=False` + 뷰 재초기화 → **8/8**, `closeup_poses.json`: 닫힘 L/R 0.0 · 개방 0.7767 · 스쿱 팔 [0, 0.39, 1.39, 1.34, 0]. 대조표 `viz/isaac_closeup_sheet_v3.png`(사용자 전송).

## 14. 결과 (수치 + 출처)

| 항목 | 값 | 출처 |
|---|---|---|
| 512 env 스텝 | mean 5.55 ms · max 7.86 · finite · 셸 오차 max 0.112 / last 0.036 · L/R 최종 0.0365/0.0365 · 편차 2.6e-8 · 팔 처짐 0.0195 | `isaaclab_smoke/smoke_512.json` |
| 64 env+카메라 | mean 8.76 ms · max 882(초기 튐) · ok | `isaaclab_smoke/smoke_64.json` |
| 외삽 | ≈11 µs/env/스텝 → 4,096 env ≈ 45 ms/스텝(선형 가정, **미측정**) | 계산 |
| GPU | RTX 4090 Laptop 16 GB, 64 env 카메라 스모크 중 10.9 GB | nvidia-smi |
| 셸 R (전→후) | mimic gearing −1·한계 −8.9~53.4° → 드라이브 1.745/0.0175/2.94·한계 0~44.5° | pxr 검사 |
| 근접 8장 | 관절 읽기값 = 목표 (4 자세 전부 ok) | `g18_closeup_v3/closeup_poses.json` |

## 15. 판정 + 다음 승인 경계

**자산은 Isaac Lab 병렬에서 돈다** — 512 환경이 NaN 없이 5.6 ms/스텝으로 스텝되고 두 셸이 같은 명령을 같은 각으로 따른다. 다만 처음 USD 는 셸 R 이 "따라 도는 척"만 하는 mimic 이라 병렬에서 바로 고착됐고, 그것을 잡은 것은 렌더가 아니라 **스텝**이었다.
다음 경계 = Isaac Lab **환경 정의**(스쿱 작업공간·heightmap 관측·보상 없음 — 로봇 동작은 정해진 경로) 와 실 서보 게인/토크 반영. 입자는 DEME 별개(불변).

## 16. 이 세션이 주장하지 않는 것 (3부)

- 입자 물리 0. 드라이브 게인 300/30 임의. 셸 R 독립 구동 = 기어 커플링 근사(mimic 모델은 gearing 부호·연성·한계 확장 검증 필요, 미실행). 4,096 env 외삽. 64 env 격자 스냅샷 미생성. 스쿱 옆면 검정 삼각형 = 벤더 link5 얇은 판 backface(무해).

## 17. 산출물 (3부)

- 코드: `sim_viz_grab_assembly.py` · `sim_render_grab_closeup.py` · `sim_isaaclab_parallel_smoke.py`(신규 3종) · `sim_urdf_to_usd.py`(플래그 False + 근거 주석)
- 산출: `g18_nut_trap/viz/{assembly_4view, assembly_fastening_closeup, isaac_closeup_sheet_v3}.png` · `g18_nut_trap/isaaclab_smoke/smoke_{512,64}.json` · `local_assets/roarm_m3/usd/{g18_robot_full.png, g18_closeup_v3/}` (usd 폴더는 gitignore)
- ⚠️ 인용 금지: `usd/g18_bowl_*.png`·`bowl_*.png`(프레임 놓침), `usd/g18_closeup/`(옛 USD 혼재 → 삭제)
- 원장: D477 `:29992` · LEDGER `:540`. 백업 `DECISIONS.md.bak_20260903_pre_d477`. **전부 미커밋.**

---

# 4부 (후반 3) — 사용자 "isaac lab 에서 간단한 구 하나 집어봐, 렌더링해서 영상으로" → 3회 시행 → D478

## 18. 절차

1. `sim_isaaclab_grasp_sphere.py`: 벤더 URDF 체인 + 그랩 부착을 numpy FK 로 — 스쿱 자세 FK (0.2487, 0.0135, 0.1206) = 근접 렌더 실측과 일치.
   키프레임 IK(q1+q2+q3 = π 로 입 아래 구속, 격자 → 좌표 하강, 오차 ≤ 0.21 mm): high 0.22 / pre 0.10 / down / lift 0.18 (피벗선 중심 z, x 0.25).
   Isaac Lab 1 env: 고정 베이스 로봇 + 구(r 15 mm, 10 g, μ 1.0) + 바닥 + `Camera` 센서(960×600, 20 fps, writer 아님) → 프레임 PNG → ffmpeg mp4. 구 위치·셸 각·grab_base 를 매 프레임 JSON.
2. **1차 FAIL**(`ok:false`): 접근 끝 grab_base z 0.042(계획 0.100), 어깨 0.681 rad(계획 0.436) — 액추에이터 상한 = USD maxForce 1.9. t 1.75 s 에 처진 그랩이 구를 쓸어 78 mm 밀어냄, 셸은 빈 채 0/0.
   조치: 팔 `effort_limit_sim` 8.0(비물리 데모), 게인 400/40, HOME→high(0.22)→pre 경유.
3. **2차 ok(운)**: 폐합 구간 내내 셸 0.776(못 닫힘) → lift 중 닫힘 → 구 0.1515. 원인 = 하강 피벗 0.0362(계획 0.0381, 처짐 1.9)에서 폐합 도중 배가 바닥에 박힘.
   설계 조각으로 셸 최저점 계산: 0° −37.97 / **22.25° −39.80** / 44.5° −35.99 (립 −36.06).
   조치: down = 39.8 + 2.0 + 2.0(처짐) = 43.8 mm, 폐합 2.5 s·유지 1.0 s.
4. **3차 ok**: 폐합 끝 셸 L 0.012 / R 0.325(구에 걸림, 독립 구동 비대칭) → lift 끝 grab_base 0.1725 · 구 **0.1512** · xy 0.0008 · 셸 0.000/0.005 · NaN 0. 270 프레임 13.5 s → `grasp_sphere_run3.mp4`, 6장 스트립 `strip_run3.png`(사용자 전송).

## 19. 결과 (출처: `g18_nut_trap/isaaclab_grasp_sphere/`)

| 런 | 하강 피벗 계획/실측 (m) | 폐합 끝 셸 L/R | 구 z 끝 (m) | xy 거리 (m) | 판정 |
|---|---|---|---|---|---|
| 1 | 0.0381 / 0.036(접근부터 처짐) | 0.000/0.000 | 0.015 | 0.078 | FAIL (토크 상한) |
| 2 | 0.0381 / 0.0362 | 0.776/0.776 (lift 중 닫힘) | 0.1515 | 0.0045 | ok (운) |
| 3 | 0.0438 / 0.039 | 0.012/0.325 → 0.000/0.005 | **0.1512** | **0.0008** | **ok** |

육안(스트립): HOME 의 빨간 구 → 개방 클램셸 접근 → 하강(구가 두 셸 사이) → 닫힘(립에 구 일부) → 폐합(구 안 보임) → 들어올림(그랩 위로, 구 내부).

## 20. 판정 + 다음 경계

**시뮬에서 그랩이 물체를 집어 올린다.** 다만 두 실패가 남긴 사실이 더 중요하다: (a) 토크 상한을 USD 가 정한다 — 실물 전압·토크 확정 후 같은 시퀀스를 실 상한으로 재현해야 한다. (b) 폐합 도중 배가 립보다 3.7 mm 깊다 — 바닥·DEME 경계 여유의 기준을 바꿔야 한다.
다음 경계 = 실 토크·처짐 반영 재현 → 환경 정의 → DEME 연결. 입자 물리는 여전히 0.

## 21. 산출물 (4부)

- 코드: `sim_isaaclab_grasp_sphere.py`(신규). 산출: `isaaclab_grasp_sphere/{grasp_result.json, grasp_log.json, keyframes.json, grasp_sphere_run3.mp4, strip_run3.png, grasp_sphere_run2.mp4, strip_run2.png}` (프레임 PNG 는 삭제, mp4 로 대체).
- 원장: D478 `:30046` · LEDGER `:541`. 백업 `DECISIONS.md.bak_20260903_pre_d478`. **전부 미커밋.**

> **커밋 (사용자 요청, 세션 말미)**: `9f4241d`(설계·프로브·자산) · `84ffa46`(시뮬 스크립트·검증 JSON) · `a2201c2`(원장·상태 문서). 렌더 PNG·영상 mp4·조각 STL·USD 는 .gitignore 정책대로 미추적(로컬 보관).

---

# 5부 (후반 4) — 사용자 "순정 서보 조인트로 묶어서 다시 돌리고 영상, 실물에서도 어떻게 할지 제대로" → D479

## 22. 절차

1. `sim_isaaclab_grasp_sphere.py` 결합판: 명령은 `link5_to_gripper_link`(순정 서보, effort 2.5) 하나. 매 스텝 실측 서보각 → `grab_v1_meta.json` `servo_shell_mouth_nonlinear` 표 보간 → 두 셸 목표. 셸 드라이브는 링크 대체(운동학 결합).
   결과 **ok**: 서보 0→1.553 rad, 셸 0→0.777(입 58.0), 순정 조 끝 최저 z 0.112(바닥 무접촉), 폐합 끝 서보 0.019·셸 L 0.022/R 0.318(구에 걸림), lift 끝 구 0.1518·xy 6.7 mm. 영상 `isaaclab_grasp_sphere_servo/grasp_sphere_servo_coupled.mp4` + 스트립(사용자 전송; 6.5 s 프레임에 순정 조 89° 벌어짐).
2. 펌웨어 원문(D473 ① 이 받아둔 `EffectsMachine/roarm-m3`) 감사: `roarm-m3.ino:113` 부팅 `moveInit()`, `RoArm-M3_module.h:238-239` 그리퍼 → 중앙 2047 = π, `:117` 토크 1000 복귀, `:59-61` EOAT rad = 스텝·2π/4096(오프셋 0), `json_cmd.h:60-64` 예시 "grab = 3.14", `:67` T:107 토크, `module.h:346-347` 클램프 [700, 2596], `uart_ctrl.h:66-71` T:106 핸들러(cmd 없으면 0).
   SDK `roarm_sdk/common.py:147,180`: `hand = π − h` / `h = 180 − h` → SDK 그리퍼 각도 = 조 개방각 = 설계 servo_deg.
3. `docs/reference/hardware.md` 의 "해결 방법 1: T:106 ESP32 리셋" 이 **오기**(맨 `{"T":106}` = cmd 0 → 700 스텝 = 조 118.5° 개방) — 취소선 + 근거 + 규약 절(표 + 5조) 추가. `AGENTS.md` 안전 제약에 한 줄 주석.

## 23. 판정

**얹는 구조에서 "순정 그리퍼가 먼저 열리는" 국면은 없다** — 조와 크랭크판이 한 몸이라 서보 명령 = 그랩 명령. 실물의 안전 조건은 설계 게이트가 아니라 **펌웨어의 자동 동작**(부팅 닫힘·토크 복귀·클램프)이며, 이제 원문 근거로 규약화됐다. 부팅 = 닫힘 명령이라 그랩 장착 상태에서 무해(팔 HOME 이동 공간만 확보).
남은 실물 항목: 부팅 속도(600)로 링크 내구 · 서보 0 오프셋과 립 접촉 캘리브 · T:107 200 충분성 · 벌어진 순정 조(89°)와 작업 공간 간섭.

## 24. 산출물 (5부)

- `sim_isaaclab_grasp_sphere.py`(서보 결합) · `isaaclab_grasp_sphere_servo/{grasp_result.json, grasp_log.json, keyframes.json, grasp_sphere_servo_coupled.mp4, strip_servo_coupled.png}`(프레임 삭제)
- `docs/reference/hardware.md`(T:106 정정 + 규약) · `AGENTS.md`(안전 주석) · D479 `:30084` · LEDGER `:542`. 백업 `DECISIONS.md.bak_20260903_pre_d479`.
