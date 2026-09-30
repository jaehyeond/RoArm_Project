# notes_B — 플라스틱 펠릿/폴리머 입자 DEM 보정(calibration) 사전값 조사

작성 2026-09-30. 읽기·검색만 함. 숫자는 **직접 연(opened) 원문**에서만 옮겼음. 검색엔진 AI 요약 문장은 출처로 쓰지 않음.
"opened" = 원문 전문 또는 파라미터 표가 보이는 페이지를 실제로 열어 읽음. 열지 못한 논문은 표 (1)/(2)에 넣지 않고 §4 gaps 에만 적음.
대상 조건 참고: PP 쌀알형 펠릿(긴 축 ≈3.8 mm, 반투명), 4 cm 층, 골판지 상자, PLA 그랩, DEME 2.4.0 Hertz–Mindlin.

**핵심 경고 (먼저 읽을 것)**
- 열린 문헌 중 **"PP 쌀알형(장타원/렌즈형) 펠릿 + 골판지/종이 벽 + PLA 벽"** 조합을 보정한 논문은 **0편**. 벽 재질은 전부 강철·유리·PMMA·폴리카보네이트·아크릴.
- 대부분 논문이 **강성(E/G)을 1e-3 배 등으로 낮춰** 씀 → 그 논문의 마찰·구름저항 값은 그 강성·그 입자모양(구 근사)·그 스케일링과 한 묶음. 값만 떼어 DEME 에 넣으면 안 됨.
- 구(sphere) 근사 논문은 **모양 효과를 구름마찰(μr)로 흡수** → μr 값은 모양 표현에 강하게 종속 (P5·P5b 가 명시).
- 같은 PP 인데 PP–PP 정지마찰이 0.432(직접 측정 인용, P3) ~ 0.52(P1 보정) ~ 0.6(P6 보정) ~ 0.8(P5 분말 보정)로 퍼져 있음. **평균 내지 말 것** — 모양·스케일·μr 과의 조합이 다름.

---

## (1) 보정(calibration)/파라미터화 논문 표 — opened 7편

| # | 인용 (저자, 연도, 제목, 저널, DOI) | opened | 재료·형상·표현 | 보정 실험 | 접촉모델·코드 | 최종 파라미터 (표/쪽) | 측정 bulk density · AoR |
|---|---|---|---|---|---|---|---|
| P1 | Sudsawat S., Chongchitpaisan P., Arunyanart P. (2023) "Calibrating polypropylene particle model parameters with upscaling and repose surface method", *EUREKA: Physics and Engineering* 6:34–46, doi:10.21303/2461-4262.2023.002968 | yes (PDF 전문, journal.eu-jr.eu download/2968/2330) | **PP**, "spherical" 입자 L 3.00 / W 3.01 mm (Table 6); prolate spheroid 근사, 2배 업스케일(6 mm) | 고정 깔때기(fixed funnel) AoR + 영상처리, Plackett–Burman → CCD/RSM (Table 3·5) | EDEM Hertz–Mindlin, Rayleigh 시간간격(비율 미기재), 벽 = **강철** | 최적: μs PP–PP **0.52**, e PP–PP **0.55** (본문 p.44). 고정: ν 0.36, E **1.3 GPa**(감소 안 함), μs PP–강철 **0.26**, "dynamic friction" PP–PP 0.05, PP–강철 0.246; 강철 E 198 GPa ν 0.3 ρ 7800. Table 1 문헌 범위: e PP–PP 0.3/0.49/0.9/0.97, e PP–강철 0.55/0.71, μs PP–PP 0.1–0.6, PP–강철 0.26/0.3 | 입자밀도 910±0.889 kg/m³, AoR **30.18±0.51°** (Table 6). DEM AoR 29.14°(3 mm)/29.67°(6 mm). bulk density 값은 본문 추출본에 없음 |
| P2 | García-Montagut J., Paz R., Monzón M. (2025) "Development of a Non-Spherical Polymeric Particles Calibration Procedure for Numerical Simulations Based on the DEM", *Polymers* 17(20):2748, doi:10.3390/polym17202748 | yes (PMC12566918 전문) | **LDPE** 3 mm 펠릿을 **분쇄한 분말** D[4,3] 579 µm (펠릿 아님); 2-구 중첩(14개 크기군, Table 8) | 부피밀도, hollow cylinder(들어올림) AoR, ledge box, draw-down; 유전알고리즘+Kriging 85회 | EDEM Hertz–Mindlin + 히스테리시스(HSCM) + 선형 응집(LCCM); 벽 = **아크릴** (ν 0.4, ρ 1385, G 1.6e10 Pa, Table 4) | Table 11 (Op85): ρ **664**(가상, 실제 918), ν 0.20, e pp 0.662, μs pp 0.526, μr pp 0.149, e pw 0.463, μs pw **0.950(탐색 상한에 붙음)**, μr pw 0.107, bN 0.081, γT 1.0, ξ 5750 J/m³. G **1.0e7 Pa 고정**(감소, Table 4). 시간간격 수치 미기재 | 부피밀도 366±10.8 kg/m³, tap 428 (Table 6); AoR(HC) 35.6°, ledge-box 60.7°, DD 55.4°/38.9° (Table 7) |
| P3 | Thieleke P., Bonten C. (2021) "Enhanced Processing of Regrind as Recycling Material in Single-Screw Extruders", *Polymers* 13(10):1540, doi:10.3390/polym13101540 | yes (EuropePMC PMC8151302 XML 전문) | **PP** 버진 펠릿(Ducor DuPure G 72 TF) + PE-HD + 분쇄품; 구 & superquadric (Table 1) | 부피밀도(DIN EN ISO 60, 컵 51 mm) 시뮬-실험 비교, 압축시험으로 신규 접촉모델 조정 | LIGGGHTS 3.8.0, 2단 히스테리시스(Walton–Braun 기반) 신규 모델; 외부(벽) 재질 본문에 명시 없음(압출기 배럴/스크루 → 강철로 추정, **미확인**) | Table 2 (PP): ρ 0.91 g/cm³(데이터시트), E **1850 MPa**(데이터시트), ν 0.399, e 내부(p-p) **0.81**, e 외부(p-w) **0.85**, 마찰 내부 **0.432**, 외부 **0.304** (각주: "Thieleke [52]에서 결정" = 박사논문 추정, 참고번호 불일치). PE-HD: 0.945/850/0.443/0.87/0.83/0.498/0.303. μr·시간간격 미기재 | 버진 펠릿은 구 모델이 실험 부피밀도와 <1% 차 (Fig 20, 수치는 그림만) |
| P4 | Hlosta J., Jezerská L., Rozbroj J., Žurovec D., Nečas J., Zegzulka J. (2020) "DEM Investigation of the Influence of Particulate Properties and Operating Conditions on the Mixing Process in Rotary Drums: Part 1", *Processes* 8(2):222, doi:10.3390/pr8020222 | yes (PDF 전문, mdpi-res) | **ABS** 구슬 6 mm (D6_ABS, 구) + 나무·강철 | 직접측정: 기울임 팔(정지·구름마찰), 낙하시험(e p-w), 이중 진자(e p-p); 보정: packing, piling AoR(90/150 mm 받침), 회전드럼 140 mm 3 rpm, 호퍼 배출 | EDEM Hertz–Mindlin; 모듈러스 ×0.001 (GPa→MPa, p.16) | ABS ρ 1050, E 2.25 GPa(문헌, Table 2), ν 0.3. DEM 입자밀도 1768 kg/m³(질량 맞춤, Table 3). **측정값**: μs ABS–강철 0.49±0.08, –PMMA 0.60±0.05, –Al 0.22±0.02, –유리 0.43±0.05 (Table 8); μs ABS–ABS 0.35±0.06 (Table 9); μr ABS–강철 0.03±0.01 (Table 10); e ABS–강철 0.72±0.02, –PMMA 0.62±0.01, –Al 0.67±0.06, –유리 0.71±0.01 (Table 11); e ABS–ABS 0.87±0.01 (Table 12). 시간간격: Table 13 추출 깨짐 — G 1e6 Pa→Δt 2.45e-4 s, 1e7→7.77e-5, 1e8→2.45e-5 로 읽힘(**불확실**) | packing 층높이 DEM 115±0.5 vs 실험 112±1.3 mm (Table 7). AoR ABS 실험/DEM: piling 90 mm 27.4±1.1 / 26.5±1.6°, 150 mm 37.4±1.4 / 34.0±1.7°, 드럼 30.3±1.5 / 32.8±5.5° (Table 15, 기하법) |
| P5 | Pourandi S., van der Sande P.C., Ostanin I., Weinhart T. (2025) "Calibration of a DEM contact model for wet industrial granular materials", arXiv:2512.08685 (Consensus 표기: *Powder Technology* 2026) | yes (arXiv PDF) | **PP 반응기 분말** PP1 D50 736.7 µm, PP2 D50 904.6 µm (Table 1); 구, 업스케일 l = 3, 5 | 회전드럼 동적 AoR (D 14 cm × 18 cm, 폴리카보네이트 측벽, 5 rpm, Fr 0.04) | MercuryDPM Hertz–Mindlin + 구름저항(구름강성 = 접선강성); 강성 ×1e-3; Δt = 20% Rayleigh | Table 2: E 1325 MPa, G 400 MPa, e 0.5 (DB/문헌). Table 4 (건조, PP2 는 Pourandi 2024 PT 446:120176 에서 인용): μs **0.8**, μr PP1(l=5) 0.3, PP2(l=3) 0.4, PP2(l=5) 0.3. 벽 마찰 미기재 | 폭기 부피밀도 393 / 368 kg/m³ (Table 2). AoR 값은 그림만 |
| P5b | Pourandi S., Ostanin I., Weinhart T. (2026) "Granular mixing and flow dynamics in horizontal stirred bed reactors", arXiv:2604.07082 | yes (arXiv PDF) | 같은 PP 분말(368 kg/m³), 구, l = 3 | AoR 로 재보정 (p.6) | MercuryDPM 동일; **벽 접촉 = 입자-입자와 동일 값** (p.7); Δt 20% Rayleigh; 강성 ×1e-3 | E 1.325e9 Pa, G 4e8 Pa, e 0.5 (Table 2); μs **0.8**, μr **0.4** (p.6) | — |
| P6 | Mesnier A., Peczalski R., Mollon G., Vessot-Crastes S. (2020) "Mixing of Bi-Dispersed Milli-Beads in a Rotary Drum…", *Processes* 8(9):1166, doi:10.3390/pr8091166 | yes (PDF 전문, mdpi-res) | **PP 구슬**(본문 3 mm; Table 1 행은 "PP 2 mm" — 논문 내부 불일치) + 셀룰로오스 아세테이트 구슬 | p-p μs: 평판 위 자연 경사(정적 AoR) 맞춤; p-w μs·μr·e: 드럼 분리지수 맞춤 | EDEM Hertz–Mindlin, Δt ≤ 40% Rayleigh; 드럼 = **강철**, 유리 | Table 2 (p.6): PP ρ 910, ν 0.42, E **2.84 MPa**, G **1 MPa**(속도용 감소), μr **0.01**, e **0.3**; 강철 E 182 GPa G 70 GPa ν 0.3, μr(강철) 0, e 0.3; 유리 94.3/39 GPa, μr 0.01. Table 3 (p.7): μs PP–강철 **0.3**, PP–유리 **0.2**, PP–PP **0.6**, PP–CA 0.6 | PP AoR 수치는 그림(Fig 2)만. 동적 AoR 29.8/24.7°(실험) vs 26.8/26.4°(시뮬) 는 CA 혼합층 값 (Table 4) |

참고(비보정, 표에서 제외했으나 열람함): Brüning & Schöppner 2022 *Polymers* 14(2):256 (PMC8777651) Table A4 — PP(Borealis RD204CF) 스크루 마찰 0.112, 배럴 마찰 0.28, 내부 마찰 0.5, 표준 부피밀도 540, 평균 펠릿 지름 4.55 mm (단위 행 깨짐, 값 출처 미기재; 해석모델 입력). PMC12252033(*Polymers* 2025 원심건조기)·PMC10237442(*PLoS One* 2023 깔때기)는 일반 가정값이라 제외.

---

## (2) 직접 측정(보정 아님) 표

| 값 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| AoR **30.18 ± 0.51°** | PP 구형 3.00 mm, 고정 깔때기→컵, 영상 회귀 | P1 Sudsawat 2023, Table 6 | yes | 쌀알형 아님(구형) |
| 입자밀도 **910 ± 0.889 kg/m³** | PP, ASTM D854 비중병 | P1 Table 6 | yes | |
| μs PP–강철 **0.26** / 동마찰 0.246 / e PP–강철 **0.55** | Table 1 출처 "j = experiment results" | P1 Table 1 | yes | 출처 기호 j,d,e 가 값 2개에 3개라 어느 값이 자체 측정인지 **모호** |
| 마찰 PP–PP **0.432**, PP–외부 **0.304**; e PP–PP **0.81**, PP–외부 **0.85**; ν 0.399 | PP 버진 펠릿(DuPure G 72 TF), 측정법·벽재질 본문 미기재 | P3 Thieleke 2021 Table 2 (각주: Thieleke 박사논문에서 결정) | yes (2차 인용) | 원 측정 조건은 박사논문(열지 못함) |
| 벌크밀도 ρ0 **0.531 g/cm³**(렌즈형), **0.491**(원통형); 낙하높이 5.5 mm 에서 0.419 / 0.389 | PP Moplen HP400H, 컵 Ø50 mm, 낙하높이 2–20 mm + 50 mm(DIN EN ISO 60), 3회 평균 | D2 Johann 2022 *Polymers* 14(5):898, Table 3 | yes | 낙하높이 의존 → 4 cm 층 채우는 방식에 따라 부피밀도 달라짐 |
| 펠릿 치수 h 2.28±0.01, d1 3.70±0.07, d2 4.35±0.04 mm, ESD 3.8 mm, 29.2 mg/알 | PP 렌즈형 버진, 100알 | D2 Johann 2022 Table 1 | yes | 우리 펠릿(긴 축 3.8 mm)과 크기 비슷, 모양 다름 |
| μs ABS–강철 0.49±0.08, –PMMA 0.60±0.05, –Al 0.22±0.02, –유리 0.43±0.05; ABS–ABS 0.35±0.06; μr ABS–강철 0.03±0.01 | ABS 6 mm 구슬, 기울임 팔 10회 | P4 Hlosta 2020 Table 8–10 | yes | PP 아님. 플라스틱–플라스틱(ABS–PMMA 0.60 > ABS–강철 0.49) 경향 참고용 |
| e ABS–강철 0.72±0.02, –PMMA 0.62±0.01, –Al 0.67±0.06, –유리 0.71±0.01; ABS–ABS 0.87±0.01 | 낙하시험 / 이중 진자 | P4 Table 11–12 | yes | |
| CoR 크기 LLDPE–강철 **0.624 / 0.598** (평균 0.611); LLDPE–폴리카보네이트 **0.704 / 0.638** (평균 0.671); 법선 CoR 강철 0.350, PC 0.294; 접선 CoR 강철 0.783, PC 0.870 | LLDPE 원판형 펠릿 0.0453 g, 45° 경사판(강철 3 mm, PC 2 mm), 충돌속도 ≈70–93 cm/s | D1 Borsa et al. 2019 *Lat. Am. Appl. Res.* 49(2):143–147 (CONICET hdl 11336/115934), Table 2–3 | yes (exa 전문 레코드) | 표의 "launch height 2 / 3 cm" 는 속도와 맞지 않음 — **해석 미해결** |
| 벽마찰각 (그림만, 축 10–35°) | 연질 PE 펠릿, SS304·Al6061 연마/비연마(Ra 0.11–1.71 µm), 500 / 4600 / 10000 Pa, 22·37 °C | D3 Han 2011 *KONA* 29:118–124 | yes | 거칠기↑·수직압↓ → 벽마찰각↑. 수치 표 없음 (>10년) |
| PLA–강철 정지마찰력 9 N @ 30 N 하중 (→ μs ≈ 0.30, **내가 계산**), 70 N 에서 +55% (→ ≈ 0.20, 계산) | 3D 프린트 PLA 원통 Ø10 mm, 연마 Ra 0.123 µm, 건식, 120 s 정지 | D4 Stoimenov et al. 2024 *Tribology in Industry* 46(1):97–106 | yes | PP–PLA 아님. COF 자체는 그림만 |
| PP–골판지/종이 마찰 | — | 없음 | — | **못 찾음** (§4) |
| PP–PLA 마찰 | — | 없음 | — | **못 찾음** (§4) |
| PP 펠릿/구 반발계수(판 충돌) | — | Yurata 2021 APT 32(4):1004 (PP 구슬 3–6 mm, 아크릴·강철판) | **no** (유료) | §4 |

---

## (3) 검색 쿼리 기록 (엔진 · 쿼리)

**보정 논문 (PP 우선)**
1. WebSearch · "DEM calibration polypropylene pellets angle of repose"
2. WebSearch · "HDPE pellets discrete element calibration parameters restitution friction"
3. WebSearch · "plastic pellets DEM hopper discharge calibration polymer granules"
4. Consensus · "discrete element method calibration polypropylene pellets"
5. Exa · "paper calibrating DEM parameters for polypropylene pellets with restitution coefficient and friction measured, angle of repose test"
6. WebSearch · "García-Montagut … Polymers 2025" / "Landgraeber … Advanced Powder Technology"
7. Exa · "Landgraeber Bruening discrete element method polymer processing material models …"
8. WebSearch · "DEM simulation plastic pellets rotating drum calibration polypropylene PET multi-sphere"
9. WebSearch · "polyethylene pellets DEM angle of repose calibration hopper experiment Powder Technology"
10. WebSearch · "plastic granules screw feeder DEM calibration parameters polypropylene granules"
11. WebSearch · "\"PET pellets\" OR \"PET granules\" discrete element calibration angle of repose"
12. WebSearch · "\"polypropylene\" pellets \"rolling friction\" DEM EDEM calibration \"angle of repose\" cylinder lifting"
13. Consensus · "plastic pellets hopper discharge discrete element simulation experiment validation" (다른 2건은 rate limit)
14. Exa · "open access paper: DEM parameter calibration of plastic pellets (PE or PP granules) … hopper discharge …"
15. Exa · "MDPI article DEM simulation of polymer pellets with experimentally determined friction coefficient and coefficient of restitution …"
16. WebSearch · "DEM pellet feeding extruder additive manufacturing fused granular fabrication calibration …"
17. WebSearch · "microplastic pellets DEM simulation parameters polyethylene polypropylene restitution friction measured"
18. WebSearch · "塑料颗粒 离散元 参数标定 休止角 聚丙烯"
19. WebSearch · "rotating drum DEM polypropylene beads measured coefficient of restitution sliding friction rolling friction …"
20. WebSearch · "polyoxymethylene POM spheres DEM parameters measured friction restitution rotating drum hopper"
21. WebSearch · "polyester chips discrete element parameters calibration angle of repose PET chips silo"
22. Europe PMC REST · (plastic/PP/PE/PET/HDPE pellets) AND (DEM) AND restitution → 4건
23. Europe PMC REST · (PP/PE/plastic/polymer) AND (pellets/granules) AND "discrete element method" AND "angle of repose" AND calibration → 9건
24. Europe PMC REST · ("Hertz-Mindlin" OR DEM) AND (PP/PE/POM/ABS) AND restitution AND "rolling friction" AND "angle of repose" → 2건
25. arXiv MCP · abs:polypropylene AND abs:granular → 4건 (P5, P5b 발견)
26. arXiv MCP · plastic pellets/polymer pellets/plastic beads friction restitution → 무관
27. HAL API · "Mesnier drum" (P6 발견) ; Semantic Scholar / OpenAlex API · OA 위치 확인

**직접 측정**
28. WebSearch · "coefficient of restitution polypropylene pellets drop test measurement DEM"
29. WebSearch · "Yurata 2021 parameter-dependent coefficient of restitution polypropylene beads acrylic steel drop test"
30. Exa · "Borsa Paulo Petit Piña coefficient of restitution … polyethylene pellets steel polycarbonate"
31. WebSearch · "friction coefficient polypropylene against paperboard cardboard measured inclined plane"
32. WebSearch · "polypropylene pellets wall friction angle steel measured Jenike shear tester plastic granules"
33. WebSearch · "PLA 3D printed surface friction coefficient against polypropylene PP measured tribology"
34. WebSearch / Exa · "Frictional properties of pellets and silo wall materials … silo honking"
35. WebSearch · "\"angle of repose\" polypropylene pellets measured degrees granules flowability study"
36. WebSearch · "polypropylene granules sliding friction coefficient steel plate inclined plane measurement pellets"
37. WebSearch · "Mechanisms of static and kinetic friction of PP, PET and HDPE pairs during sliding"
38. Europe PMC REST · (polypropylene AND pellets) AND friction coefficient AND steel AND "bulk density"

접근 실패: ScienceDirect(403/캡차), MDPI htm(봇 차단 → mdpi-res PDF 로 우회), ResearchGate(403), Twente 저장소(Cloudflare), AIP PDF(403/timeout). Playwright 가 repo 에 `.playwright-mcp/` 를 만들었던 것은 scratchpad 로 옮기고 빈 폴더 제거 — `git status` 깨끗함 확인.

---

## (4) Gaps — 못 연 것 / 없는 것

**존재 확인했지만 못 연 논문 (값 인용 금지, 필요하면 사용자가 기관 접속으로 확인)**
- Landgraeber J., Brüning F. (2025) *Adv. Powder Technol.* 36(8):104968, doi:10.1016/j.apt.2025.104968 — 여러 폴리머 펠릿+분쇄품, 구/다중구, AoR 로 μr 보정 + 전단셀 + 부피밀도. **CC-BY-NC-ND OA 인데 ScienceDirect 캡차로 못 엶. 가장 우선순위 높은 미열람 문헌.**
- Pourandi S. et al. (2024) *Powder Technol.* 446:120176, doi:10.1016/j.powtec.2024.120176 — PP 분말 건조 보정 원 논문 (CC-BY, Twente 저장소 Cloudflare 차단).
- Martignoni A., Iorio L., Strano M. (2024) *Granular Matter* 26:104, doi:10.1007/s10035-024-01474-8 — 분쇄 폴리머 폐기물 6종, lifting cylinder AoR (유료, 표 못 봄).
- Fan J. et al. (2021) *J. Nat. Gas Sci. Eng.* 88:103854 — PP 입자 관내 CFD-DEM (유료; P1 이 e PP–PP 0.49, ν 0.41 등 출처로 인용).
- Yurata T. et al. (2021) *Adv. Powder Technol.* 32(4):1004–1012 — **PP 구슬 3–6 mm 의 판(아크릴·강철) 반발계수, 속도·온도 의존** (유료). PP 반발계수 직접측정으로 가장 적합.
- Moysey P.A., Thompson M.R. (2007) *Chem. Eng. Sci.* 62(14):3699–3709 — HDPE/PS/PC 구의 강철 충돌 반발·마찰 (유료, >10년).
- Chavez-Sagarnaga J. et al. (2004) "Frictional properties of pellets and silo wall materials for the investigation of silo honking" — PET·PP 펠릿 vs 알루미늄·스테인리스 Jenike 벽마찰 (전문 못 찾음, >10년).
- Cho D.H., Bhushan B., Dyess J. (2016) *Tribol. Int.* — PP–PP, PET–PET, HDPE–HDPE 정지/운동 마찰 (유료). **PP–PP 직접 마찰로 가장 적합.**
- Hastie D.B. (2013) *Chem. Eng. Sci.* — 불규칙 PE 펠릿 반발계수 (유료).
- Tangri H., Guo Y., Curtis J. (2019) *Chem. Eng. Sci. X* 4:100040 — OA 이나 ScienceDirect 차단; exa 초록은 "steel cylindrical particles" 로만 나와 플라스틱 여부 불확실.
- Brüning F., Sommer L., Schöppner V. (2023) AIP Conf. Proc. 2884:110002 — 링전단기로 플라스틱–플라스틱·플라스틱–강철 마찰 역보정 (PDF 403).
- Elskamp F. et al. (2017) *Granular Matter* 19:46 — POM 구 입자단위 마찰·반발 측정 (유료).

**문헌에서 못 찾은 값 (실측이 필요)**
- **PP 펠릿–골판지/종이 마찰**: 열린 문헌 0건. 경사판(기울임) 실측 권장.
- **PP 펠릿–PLA(3D 프린트) 마찰**: 0건. PLA–강철(D4)·ABS–PMMA(P4)만 있음. 실측 권장.
- **PP 쌀알형/장타원 펠릿의 AoR**: 구형 PP(30.18°, P1)만 있음. 쌀알형은 실측 필요.
- **PP 펠릿–골판지/PLA 반발계수**: 0건.
- **DEME(DEM-Engine) 에서 보정된 플라스틱 펠릿 세트**: 0건 (전부 EDEM·LIGGGHTS·MercuryDPM). 코드 간 구름저항 모델 정의가 달라 μr 이식 불가.
- 시간간격: P1(Rayleigh, 비율 없음), P5/P5b(20% Rayleigh), P6(40% Rayleigh), P4(추출 깨짐)뿐 — 절대값 대부분 미기재.

---

## 부록: 원문별 raw 메모 (수집 순서)

## RAW P1 Sudsawat 2023 EUREKA (opened full text via exa fetch of PDF download/2968/2330)
- PP spherical particles 3.00 mm (L 3.00, W 3.01; Table 6), density 910±0.889 kg/m3 (Table 6), AOR 30.18±0.51 deg fixed funnel, image analysis (Table 6).
- Spheroid, upscaled x2 (6 mm). EDEM Hertz-Mindlin, Rayleigh time step (fraction not given), steel wall.
- Table 1 literature ranges: PP nu 0.36,0.41; rho 889,910,1107; E 1.3,1.9,2 GPa; mu_s PP-PP 0.1,0.153,0.22,0.3,0.6; PP-steel 0.26,0.3 (sources j=own expt, d, e); "dynamic friction" PP-PP 0.05,0.44; PP-steel 0.246,0.28; COR PP-PP 0.3,0.49,0.9,0.97; PP-steel 0.55,0.71 (j,e).
- Final: mu_s PP-PP 0.52, COR PP-PP 0.55 (RSM optimum); fixed nu 0.36, E 1.3 GPa, mu_s PP-steel 0.26, dyn PP-PP 0.05, dyn PP-steel 0.246; steel E198GPa nu0.3. DEM AOR 29.14 (3mm) / 29.67 (6mm).
- Bulk density: method described, value NOT shown in extracted text.

## RAW P2 Garcia-Montagut, Paz, Monzon 2025 Polymers 17(20):2748 doi 10.3390/polym17202748 (opened PMC12566918 full text)
- LDPE (Total 1200 MN 18 C, 0.918 g/cm3), 3 mm pellets MICRONIZED to powder D[4,3]=579 um (NOT pellets). 2 overlapped spheres, 14 size groups (Table 8).
- Tests: bulk density, hollow cylinder (lifting), ledge box, draw-down; walls acrylic (nu 0.4, rho 1385, G 1.6e10 Pa, Table 4). Hertz-Mindlin + HSCM (hysteretic) + linear cohesion (LCCM). EDEM.
- G LDPE fixed 1.0e7 Pa (Table 4; reduced, "EDEM software").
- Final Table 11 (Op85): rho 664 kg/m3 (calibrated virtual), nu 0.20, e_pp 0.662, mu_s pp 0.526, mu_r pp 0.149, e_pw 0.463, mu_s pw 0.950 (upper bound!), mu_r pw 0.107, bN 0.081, gammaT 1.0, xi 5750 J/m3.
- Expt Table 7: AOR HC 35.6, ledge-box shear 60.7, DD shear 55.4, DD AOR 38.9, mass DD 3.3 g, bulk mass 59.4 g; Table 6 bulk 366 kg/m3, tap 428, Hausner 1.17.
- Time step: not stated numerically in text found.

## RAW P3 Thieleke & Bonten 2021 Polymers 13(10):1540 doi 10.3390/polym13101540 (opened EuropePMC fullTextXML PMC8151302)
- PP virgin = Ducor DuPure G 72 TF homopolymer pellets; PE-HD Lupolen 4261AG; + regrinds. Shapes: sphere and superquadric (Table 1). LIGGGHTS 3.8.0, new 2-stage hysteresis (Walton-Braun based) contact model.
- Table 2 (params used): PP density 0.91 g/cm3 (datasheet), E 1850 MPa (datasheet), nu 0.399, internal COR 0.81, external COR 0.85, internal friction 0.432, external friction 0.304 (footnote 2: "determined in Thieleke [52]" = PhD thesis Stuttgart 2020 presumably; ref numbering mismatch in XML). PE-HD: 0.945, 850 MPa, 0.443, 0.87, 0.83, 0.498, 0.303.
- External = particle-wall; wall material not explicitly stated in text (extruder barrel/screw, steel implied - NOT confirmed).
- Validation: bulk density per DIN EN ISO 60 (cup 51 mm) sim vs exp; spheres <1% deviation for virgin (Fig 20, numbers only in figure). Compression 20,000 N.
- Time step not given in extracted text; rolling friction not listed.

## RAW (not a calibration) Bruening & Schoeppner 2022 Polymers 14(2):256 (PMC8777651, opened)
- Table A4 (analytical-model data; unit row garbled): PP (Borealis RD204CF) screw friction 0.112, barrel friction 0.28, internal friction 0.5, standard bulk density 540, avg pellet diameter 4.55 mm. PA6 0.068/0.17/0.37/718.6/2.68; LLDPE 0.08/0.2/0.5/550/1.41; PS 0.156/0.39/0.38/580. Source of friction values not stated in extracted text. Not a DEM calibration.

## RAW P4 Hlosta, Jezerska, Rozbroj, Zurovec, Necas, Zegzulka 2020 Processes 8(2):222 doi 10.3390/pr8020222 (opened PDF via mdpi-res)
- Materials incl. ABS plastic beads 6 mm (D6_ABS; sphere), plus wood, steel. Walls: steel, PMMA, aluminum, glass. EDEM Hertz-Mindlin.
- Table 2: ABS rho 1050, E 2.25 GPa (lit); PMMA 1250/3 GPa; steel 7850/210. nu=0.3 default. Moduli reduced x0.001 (GPa->MPa) (p.16 text).
- Table 3: D6_ABS single mass 0.200 g, DEM volume 113.1 mm3, DEM density 1768 kg/m3 (as given; mass-matched).
- Measured (inclined arm tilt): Table 8 mu_s ABS-wall: steel 0.49±0.08, PMMA 0.60±0.05, Al 0.22±0.02, glass 0.43±0.05. Table 9 mu_s ABS-ABS 0.35±0.06. Table 10 mu_r ABS on steel 0.03±0.01 (tilt 1.5°). Table 11 e ABS-wall: steel 0.72±0.02, PMMA 0.62±0.01, Al 0.67±0.06, glass 0.71±0.01 (drop test). Table 12 e ABS-ABS 0.87±0.01 (double pendulum).
- Packing test Table 7: bed height D6ABS DEM 115±0.5 vs exp 112±1.3 mm.
- AoR ABS Table 14 (graphical) piling 90mm 27.4±3.1 exp vs 35.0±0.7 DEM; 150mm 33.1±4.1/32.4±9.1; drum 140mm 3rpm 34.1±3.1/33.9±0.9. Table 15 (geometric) 27.4±1.1/26.5±1.6; 37.4±1.4/34.0±1.7; 30.3±1.5/32.8±5.5.
- Time step: Table 13 garbled in extraction; appears G 1e6 Pa -> dt 2.45e-4 s, 1e7 -> 7.77e-5, 1e8 -> 2.45e-5 (uncertain parse).

## RAW P5 Pourandi, van der Sande, Ostanin, Weinhart, arXiv 2512.08685 (2025) "Calibration of a DEM contact model for wet industrial granular materials" (Powder Technology 2026 per Consensus) (opened arXiv PDF)
- PP reactor POWDERS (Innovene), PP1 D50 736.7 um, PP2 D50 904.6 um (Table 1); spheres, upscaled l=3,5. MercuryDPM Hertz-Mindlin + rolling resistance (rolling stiffness = tangential).
- Table 2: E 1325 MPa, G 400 MPa, e 0.5 (databases/literature), aerated bulk density 393 / 368 kg/m3. Stiffness reduced by 1e3 in sims.
- Table 4 (dry calibration, from ref [27] Pourandi et al. Powder Technol 446 (2024) 120176 for PP2): mu_s 0.8 all; mu_r PP1(l=5) 0.3, PP2(l=3) 0.4, PP2(l=5) 0.3.
- Rotating drum D 14 cm x 18 cm, polycarbonate sidewalls, 5 rpm, Fr 0.04. dt = 20% Rayleigh. Wall friction not stated in extracted text. AoR values only in figures.

## RAW P6 Mesnier, Peczalski, Mollon, Vessot-Crastes 2020 Processes 8(9):1166 doi 10.3390/pr8091166 (opened PDF via mdpi-res)
- PP spherical beads (text: 3 mm, Marteau & Lemarie; Table 1 row says "Beads PP 2 mm" - inconsistency in paper) + cellulose acetate beads. EDEM Hertz-Mindlin, dt max 40% Rayleigh. Drum: steel, glass (front?).
- Table 2 (p.6): PP rho 910 (mfr), nu 0.42 (mfr), E 2.84 MPa & G 1 MPa (user-defined reduced for speed), mu_r 0.01 (fitted), e 0.3 (fitted); steel rho 7800 nu 0.3 E 182 GPa G 70 GPa, mu_r(steel) 0, e 0.3; glass 2500/0.21/94.3/39, mu_r 0.01, e 0.3.
- Table 3 (p.7) fitted mu_s: PP-steel 0.3, PP-glass 0.2, PP-PP 0.6, PP-CA 0.6. p-p via static AoR (pour on flat plate) match; p-w, mu_r, e via drum segregation index fit.
- Table 4: DAR (bi-size CA bed, not PP) exp 29.8/24.7 vs sim 26.8/26.4. PP AoR numeric not found in text (Fig 2).

## RAW D1 Borsa, Paulo, Petit, Pina 2019 Lat. Am. Appl. Res. 49(2):143-147 (OA, CONICET hdl 11336/115934; LAAR OJS article/view/35). Opened via exa library full-text record (tables visible).
- LLDPE commercial pellets, disc shape, mean mass 0.0453 g. Dropped onto 45° inclined plate: steel 3 mm, polycarbonate 2 mm. Table 2 CoR modulus (CoRm): PE-PC 0.704 & 0.638 (mean 0.671); PE-steel 0.624 & 0.598 (mean 0.611); Vbi PE-PC 70.659/91.390 cm/s, PE-steel 79.323/92.763 cm/s. Table 3: normal CoR PE-PC 0.365/0.222 (0.294), PE-steel 0.364/0.335 (0.350); tangential PE-PC 0.891/0.850 (0.870), PE-steel 0.802/0.764 (0.783). "Launch height [cm]" column reads 2 / 3 (possibly level index; velocities suggest ~25-45 cm drops - UNRESOLVED).

## RAW P5b Pourandi, Ostanin, Weinhart arXiv 2604.07082 (2026) "Granular mixing and flow dynamics in horizontal stirred bed reactors" (opened arXiv PDF)
- Same PP powder (aerated bulk 368 kg/m3). Table 2: E 1.325e9 Pa, G 4e8 Pa, e 0.5 (databases). Scale factor 3; recalibrated via AoR: mu_s 0.8, mu_r 0.4 (p.6). Particle-WALL contact = same as p-p (p.7). dt 20% Rayleigh, stiffness reduced 1e3. MercuryDPM Hertz-Mindlin + rolling (rolling stiffness = tangential). Spheres.

## Not counted (not calibrations / generic params): PMC12252033 (Polymers 2025 centrifugal dryer: generic polymer pellet sphere 3 mm, rho 1380, E 2e9, nu 0.4, mu 0.4, e 0.6, Table 1, source not calibrated); PMC10237442 (PLoS One 2023 slit funnel: 6 mm, 1.19 g/cm3, nu 0.22, e 0.93, mu 0.40, Table 2; material not identified in extract).

## RAW D2 Johann, Mehlich, Laichinger, Bonten 2022 Polymers 14(5):898 doi 10.3390/polym14050898 (opened EuropePMC PMC8912641)
- PP homopolymer Moplen HP400H (LyondellBasell). Table 1: lenticular virgin PP h 2.28±0.01, d1 3.70±0.07, d2 4.35±0.04 mm, ESD 3.8 mm, grain mass 29.2 mg; cylindrical (lab-cut) h 2.88±0.40, d1 2.86±0.21, d2 4.43±0.19, ESD 3.79, 27.8 mg. PE-HD lenticular 35.2 mg, cyl 28.6 mg.
- Table 3 bulk density (Grunschloss fit; cup 50 mm dia; DIN EN ISO 60 at 50 mm dumping): PP cyl rho0 0.491 g/cm3, at 5.5 mm dumping height 0.389; PP lenticular rho0 0.531, at 5.5 mm 0.419. PE-HD cyl 0.529/0.459; lent 0.582/0.492. Friction coefficients NOT measured (authors say error-prone).

## RAW D3 Han 2011 KONA 29:118-124 "Comparison of wall friction measurements by Jenike shear tester and ring shear tester" (opened PDF via pdfs.semanticscholar)
- Soft PE plastic pellets "A" (Dow), walls SS304 & Al6061 polished/non-polished (Ra 0.11-1.71 um Table 2). Wall friction angles only in Figs 4-6 (axis 10-35 deg); increase with roughness and with lower normal pressure (500 vs 4600 vs 10000 Pa); 22 vs 37 C no significant change. >10 y old.

## RAW D4 Stoimenov et al. 2024 Tribology in Industry 46(1):97-106 doi 10.24874/ti.1546.08.23.10 (opened PDF)
- 3D-printed PLA cylinder (10 mm, sanded Ra 0.123 um) vs steel counter-body, dry: static friction FORCE 9 N at 30 N normal load (=> mu_s ~0.30, derived by me), +55% at 70 N (=> ~0.20, derived). COF values themselves only in figures. PLA vs steel, not vs PP.
