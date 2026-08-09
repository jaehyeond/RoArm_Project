# START_HERE.md

Last updated: 2026-08-10 KST (**39th 세션**). **G0b case `g0b_d420` 계속.**
31st = Gate-0 verdict `GATE0_SOURCE_ABSENT`(D427) → 정지·사용자 결정 대기(D426 ①).
32nd 죽은 자산 단서 → 33rd 실물 사진 → 34th-b C1 CONFIRMED/C2 REFUTED →
35th-b C5 크기 축 SETTLED → 36th 캘리퍼 프로토콜 → 37th/37th-b 가설 C·D → 38th/38th-b 적대 패널.
**39th = 마지막 미결 리드(38th §11) 판별 → 반증 종결 + 38th §12 미적용 6건 집행 완료.**

> ✅ **86.4 트랙 종결 — `RESOLVED_NO_NEW_MECHANISM`** (사용자 승인, 38th §10-3).
> 신규 기구 없이 설명 가능하고 자산 오류도 아니다. **잔여 각주 3건 = 38th §10-5**
> (Q1·Q2·Q4 미기입 ⇒ 정지규칙 F1/F2 기계적 발동 불가·F3 판정 불가 / P-bare 불필요 판단 /
> 【E】·【G】 준수 기록). **이 각주들을 "해결"로 인용하지 말 것.**
> ⛔ 86.4 신규 가설 생성·신규 패널 발사 금지. ⛔ G0a(Ø2.2)·G0b(웹 창) 실물 테스트 금지(38th-b 기각).

> ✅ **38th §11 리드 = 반증 종결 (39th, D429)** — `SEC11_LEAD_REFUTED_NO_MISSING_GEOMETRY`.
> `gripper_left_link.stl`의 **9,461 정점 전부가 `link5.stl` 표면 34.1 µm 이내** ⇒ 출하 파일은
> URDF 소스가 이미 가진 기하의 **재테셀레이션**이고 **결손은 0**이다.
> 38th의 "10.84%"는 산술 오류가 아니라 **대상 선택 오류 2겹**이었다: 기준을 `link5.stl` 전체가
> 아닌 6,222-tri 면-연결 조각으로 잡았고, 거리를 정점-대-**정점**으로 쟀다(19,126 vs 6,222 tri).
> 기준을 고칠수록 **1026 → 998 → 425 → 0**으로 붕괴한다.
> ⇒ **D427 `GATE0_SOURCE_ABSENT` 불변**, D426 ① 분기(수제 저작 승인 or 정지·재상의) 불변.
> ⛔ 이 리드 재개 금지 · 이 판별 재실행 금지(결정을 바꿀 수 없음).

## ⚡ 현재 진실

- **Gate-0 수치 불변·재실행 금지 (31st, D427)**: fixed link5 `l_vis` **4.4576mm**(peak
  r=10.12mm·az 172°, link5 좌표 (−10.0258, 1.4090, **119.8856**)) / moving 피크 **3.9559mm**
  @ q5=5.10°. L_MIN **5.5** ⇒ 둘 다 FAIL. 권위 = `g0b_d420/t3r_gate0_vismesh_results.json`
  (sha `d7d2ce6a…b310`). **39th가 이 통계를 델타 0.000e+00 · n_pts 2,266,503 정확 일치로 재현.**
- **좌표 규약 (39th에서 소스 재고정 — 기억 금지)**: `roarm_m3.urdf:129-135` link5 visual
  origin **항등** + scale 0.001 ⇒ **`link5.stl`의 mm 좌표 = link5 프레임 좌표**.
  TCP = link5 **z 115.428mm**(`:234-239`), 공구축 = link5 **+z**.
  depth = z−115.428(+ = 원위), r = hypot(x,y). finger window r≤30 / footprint r≤14.5 /
  wall annulus r∈[12.5,20] / rim band depth∈[5,15]. 샘플 간격 0.5mm, **미터 단위로 샘플링**.
- **`gripper_left_link.stl` = 고정 조 독립 파일 확정, 열람 승인 완료 (금지 해제)**
  sha `1dfb7722…f71bc8` / 19,126 tri / 9,461 verts / bodies 3 / watertight False / euler 37 /
  AABB 34.4676 × 23.9384 × 103.8000 / 볼록껍질 **105.6842**.
  ★ 이 sha는 Gate-0 실행 당시 이미 `t3r_gate0_vismesh_results.json` `/sources/dead_asset/sha256`에
  핀돼 있었다. **"죽은 자산" 규정(29th~37th)은 오분류였다.**
  ★ 39th: `link5.stl` 표면 **34.1 µm 이내 완전 포함**(정점 단위, 38th의 4-스칼라 대조에서 승격).
  단 **비트 동일 아님** — 테셀레이션이 다르다(19,126 vs 7,250 tri).
- **`link5.stl`**: sha `1d63f374…9c7eb` / 14,092 tri / 6,926 verts / watertight False / euler 19 /
  AABB 46.4960 × 35.5200 × 120.6351, z ∈ [−0.7495, **119.8856**].
  정점-연결 body **6개** = [**7250**, 5276, 1054, 288, 192, 32].
  최대 body(고정 조) 7,250 tri / 3,519 verts / **열린 모서리 0** / 껍질 105.68419 /
  z ∈ [16.0856, 119.8856] / 면-연결 조각 6222+564+348+116.
- **`gripper_link.stl`(OpenJaw, 이동측)**: sha `7946a374…65a56` / 13,698 tri / watertight /
  성분 1 / AABB 77.8500 × 25.2400 × 39.3676 / **볼록껍질 81.4065005834**
  (독립 코드경로 5 + 적대 공격 2회 생존 — **재도출 금지, 결정 불변**).
- **joint5**: `roarm_m3.urdf:225-231` origin (0, 18.821, 52.035)mm, rpy(−1.5708,−1.5708,0),
  limit 0~1.571 rad. R=[[0,1,0],[0,0,1],[1,0,0]] ⇒ **회전축 = link5 +y**, 조 길이축 +x → link5 +z.
- **사용자 실물 실측 (자 눈금, ±1mm, n=1, 서보 완전 분해)**: OpenJaw **80** / 고정 조 **105**.
  둘 다 [축방향 폭, 껍질] 구간 내. **판독 정밀도가 고정 조 허용 창(1.88mm)과 동등**하므로
  **총체적 축척 오류(수 % 규모) 배제까지만 지지**하고 하부 정밀도 판별력은 없다.
  ⇒ **s ∈ 약 [0.98, 1.03] — C5의 0.985와 순수 1.000이 모두 생존.**
  σ 값(껍질 −0.19/+0.71 · 축 +3.55/+2.21)은 **둘 다 측정 노이즈 안, 어느 쪽도 근거 아님.**
- **D421 = T2_PASS**: 완전 수직은 **베이스 근측 annulus에서만** 도달 가능, p7 후보 4/8이 그 안.
- 실기 최대 개방 **88.3°** 출처 = `claudedocs/direction_20260708_grasp_pivot.md:26`.
- 물리 진실 불변(D424/D425): attempt3 **현 상태**에서 top-down 상면 중심 파지 기하 불가.
  **D419 타깃 변경은 교수님 사안 — HARD RULE #18, 단독 변경 금지.**

## Active Case — `g0b_d420` (범위 불변)

- 물체 = 원통 D29×H50 / 24.83g 기립(#18). 파지 = 수직 상부 상면 중심(D419) 유지.
- 이번 case의 신규 변수: **[P, F]** (36th~39th 신규 변수 0)
- 출력: `g0b_d420/`. 신규 자산은 `g0b_d420/repair_assets/arm*/`, 태그 `t3r_*`.
- 39th 신규 산출물: `t3r_n6_subsetloc_{results.json,diagnostic.png,timeline.rrd,timeline.rbl,
  rerun_validation.json,inspection.png,script.py.txt,run_stdout.log,run_stderr.log}`

## 다음 세션 — 순서 고정

**⛔ 1. T2 재실행은 하지 않는다** (39th §11 근거). 사용자 【5】가 원한 "IK 실패와 파지 실패의
   원인 분리"는 **이미 증거로 존재한다**: ① T3 스폰이 **이미 T2 PASS 후보에 핀돼 있다**
   (`t3_prereg.md:47` `seed0_S1` (+0.213696, −0.195719), supersession S-1 — 설계값 (0.300,0)은
   T2 annulus 경계라 폐기됨; D421 `seed0_S1` descend pos_err 0.0147mm·tilt **0.1989°**·
   `pass_both=True`) ② attempt1이 **approach 44wp 완주·도착 vertical PASS** 후 descend
   wp006에서 접촉으로 정지했고, prereg 자신이 **"controller/IK 실패가 아니라 descend 목표
   자체가 기하 위반"**으로 결론(`t3_prereg.md:175-183`), D424가 verdict로 확정.
   ⇒ 재실행은 **결정을 바꿀 수 없는 검증**(AGENTS.md 금지) + `t2_*`는 Frozen.
   ⚠️ 새 격자가 필요해지는 유일한 조건 = **스폰이 `seed0_S1`에서 바뀔 때**.

**🔴 2. 사용자 결정 대기 — D426 ① 분기가 이 트랙의 유일한 열린 결정이다**
   D424(조 목구멍 폐색) → D425(기하 원인) → **D427(원인은 cook이 아니라 저작 소스 자체)**
   ⇒ 분기는 **둘뿐**: **(A) 원위 조 기하 수제 저작 승인**(Arm-F "양조 원위 손가락 증분";
   치수 근거 = T1 실물 물림 0~12mm + 권장 L 밴드 [9.5, 13.5]mm(24th §4-1) + D426 ④ 3조건
   prereg) **또는 (B) 정지·재상의**. **D427 ④ = 사용자 결정 전 착수 금지.**
   ★ 39th는 이 분기를 **넓히지 못했다** — 오히려 "출하 자산을 쓰면 된다"는 마지막 우회로를
   닫았다(D429).

**3. (A) 승인 시에만 T3 재개** — grasp → hold → move → place.

**4. 사용자 【4】(link5 64 헐 중 타 헐 내부에 완전 포함되는 개수) = 강등 유지** —
   T3에서 **접촉 이상이 실제로 관측될 때만** 실행하는 1순위 용의자 조사.

**5. ⛔ 재개 금지 목록**: 38th §11 리드(D429 종결) · 이 판별 재실행 · 이동측 껍질 상한
   81.4065 재도출 · 86.4 신규 가설/패널 · G0a·G0b 실물 테스트 · Gate-0 재실행 · T2 재실행.

## T 사다리 현황

| 단계 | 상태 |
|---|---|
| T0/T1 | 완료 (D419/D420) |
| T2/T2b | **완전 종결 (D421~D423). 재실행 금지** — T3 스폰이 이미 T2 PASS 후보 `seed0_S1`에 핀돼 있고(`t3_prereg.md:47`), attempt1이 approach 도착 vertical PASS를 실측했다(39th §11) |
| T3 | Gate-0 = SOURCE_ABSENT(D427) → 34th C1·C2 → 35th-b C5 → 36th 프로토콜 → 37th/37th-b 가설 C·D → 38th/38th-b 적대 패널(반증 13·정정 17) → **39th §11 리드 반증 종결(D429). 미결 리드 0.** → 🔴 **D426 ① 사용자 결정 대기(수제 저작 승인 or 정지·재상의)가 유일한 차단 요인** |
| T4 실물 재현 / T5~T7 | 대기 (`g0a_pass=false` 불변, 프로포절 일정에 실물 파지 미포함) |

## Open Risks / Claim Limits

- **D427 자체는 살아 있다** — 시각 소스에 조 원위부 기하 부재. 분기 = **수제 저작 승인 or
  정지·재상의 둘뿐**(D426 ①, C 소멸). 39th는 이 분기를 **바꾸지 않았다**.
- 39th 잔여 425점을 "마운팅 보스"로 부른 것은 **기하 기술 기반 분류**(완전 회전체·이산 z
  레벨 9·z=80.5518 카운터보어와 동축 0.05 µm)이지 **CAD 부품표 확인이 아니다** [추론 표기].
- `watertight`는 **병합 허용오차 ≤1e-6mm에서만** 성립 — 인용 시 허용오차 병기 필수.
- **ρ_real은 여전히 측정이 아니라 모델**(장착 변환 미측정, 규약 차 0.7645mm).
  s 확정 바닥 ±0.6%, 표본 n=1, 재질 미기록.
- 34th의 ρ 67.3±2.0은 flower 좌표 규약(0.80mm) 때문에 ~3% 편향 가능.
- 광학 앵커 5개(78/82.2/83.54/84.0/84.15) 상호 모순 미해소. **78은 캘리퍼 값 아님**(사진 측광).
- **계측기 신원 미확정** — 33rd 사진 10장의 유일 계측기 = DIGITAL ANGLE RULER, LCD 전부 꺼짐,
  캘리퍼 0장. "86.4 = 86.4°"는 **가설**이며 사용자 확인 전 근거 사용 금지(#18).
- 로브형 ±z 개구가 joint5 피벗 자리인지 미해결(축에서 0.7674mm 어긋남).
- 36th "포크 간격 36.688"은 확인도 반증도 안 됨(37th 36.3676은 AABB 면 추론 의존).
- 실물 실루엣 좌우 비대칭 0.46~1.36mm(p04) 미해소. 렌즈 왜곡 미보정.
- 서보 폐지력·자율 재현성 = null. `g0a_pass=false` 불변. **"T1이 파지력 증명" 금지.**
- 가드 G-c/G-d/G-e = 예방적. attempt3 world/link1~4 legacy collider 잔존(D425 ③).
- 마찰 0.40/0.30 미실측 / marker 비증거 / exit code 판정 채널 금지.
- AnyGrasp 인용 시 26th doc §6 MISMATCH 3건 주의.
- 코드 충돌 미해소: `deploy_smolvla.py:685-689` vs `safety_p0_guards.py:145-146` (T4 전).
- 별건 1: 25th scratchpad 118MB(`6e109ebc-*/scratchpad`) 처분 지시 대기.
- 별건 2: MEMORY.md 용량 — 남은 압축 대상은 #8이 "불변"으로 못박은 섹션 → **명시 승인 필요**.

## Frozen — Do Not Retry or Overwrite

- 격리 트랙 전체 — 사용자 호출 시에만. p7 원본 재실행 금지.
- attempt3 **원본** 불변(파생 사본만 — 재분해는 D426 하 **Arm-A 한정**).
- T2·T2b·t3_grasp{,2,3,4}_*·t3_jaw_audit{,2,3}_*·t3r_* 산출물 덮어쓰기 금지
  (**t3r_gate0_vismesh_* 6종 = 완결 증거, 재실행 금지** — 확대 열람은 허용).
  **`t3r_n6_subsetloc_*` 7종 추가 (39th 완결 증거, 재실행 금지).**
- ✅ **`gripper_left_link.stl` = 금지 해제** (Q3 승인 + 38th 열람 + 39th 정점 대조 완료).
  "죽은 자산" 규정은 오분류였다. 단 **URDF 배선 변경은 별건이며 승인 없이 금지.**
- **`58.419`·`48.3706`·`33.2843` C5 기준선 재사용 금지.**
- **36th 금지 유지**: 손끝 끝면 폭(3.8431)·"팁에서 5mm 뒤 폭"·"노출 날 길이"를 측정 항목으로
  쓰지 말 것 / **판 두께(1.5000)·구멍/보어 지름을 s 추정에 쓰지 말 것** /
  **전장·최대폭에 대칭 ±허용대역 금지**(단측 상향 편향 ⇒ 최소값 탐색).
- **37th 금지**: STL을 `process=False`로 읽었으면 위상 사용 전 **반드시 `merge_vertices()`**.
- **39th 금지(D428 #29/#30)**: 정점-대-정점 거리로 기하 결손 주장 금지(정점-대-표면으로 잴 것) /
  비교 기준은 **파이프라인이 실제 소비하는 객체**여야 함(body 아닌 파일 전체).
- "bbox 77.85 ≈ 실물 78mm" 인용 금지 / marker=접촉 증거 금지 / exit code 판정 채널 금지.
- HANDOFF.md·TASKS.md 불신 / `/half-clone`·`/handoff` 금지(#11, **거부 40회**) /
  commit·push는 사용자 요청 시에만. `isaaclab` env pin(rerun 0.34.1 / numpy 1.26.0 /
  psutil 5.9.8), D326 절차.

## Must Read First

1. `AGENTS.md` → 2. **`claudedocs/session_20260810_39th_g0b_t3r_n6_subset_locate.md`**
   (**§5(판별)·§7(집행)·§10(다음 순서)** 먼저)
3. this file → 4. `claudedocs/DECISIONS.md` **D429 → D428 → D427** → D426 → D425(⚠️ "중앙 플러그"
   반증됨) → D424 → D421
5. `claudedocs/session_20260809_38th_*.md` **§9(적대 패널)·§10(사용자 실측)**
   — ⚠️ **§11은 D429로 반증됨, 인용 금지**
6. `g0b_d420/t3r_n6_subsetloc_results.json`(sha `819e624ea5f10f76`) ·
   `t3r_gate0_vismesh_results.json`(sha `d7d2ce6a…b310`)
7. 필요 시: 37th doc §12 → 36th §2·§4-1 → 35th §11 → 34th §11 → 31st §3~§7
8. `local_assets/roarm_m3/urdf/roarm_m3.urdf:129-135,225-231,234-239` · 24th doc §4-1 · 27th doc §3

## Git

- HEAD == `79df2b3` "8월 6일자 변경". **29th~39th분 미커밋** (수정 3 + 신규 34, 39th 시작 시점).
  39th에서 신규 9건(`t3r_n6_subsetloc_*` 7 + 39th session doc + 로그 2) 추가.
- commit/push는 **사용자 요청 시에만**. DECISIONS는 39th에서 **D428·D429 2건 append**.
