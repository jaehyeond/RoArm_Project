# notes_A — PP 펠릿 DEM 사전값(prior) 문헌·제품자료 조사

작성: 2026-09-30, 읽기/검색 전용. 대상 = 인형용 PP 펠릿(반투명 백색, 쌀알형, 긴축 3.8±0.3 mm), DEME 2.4.0 Hertz-Mindlin.

규칙: 직접 연 페이지/PDF만 인용 ("opened: yes"). 검색엔진 요약·스니펫은 출처로 쓰지 않음.
"opened: yes (PDF)" = PDF 를 내려받아 `pdftotext` 로 원문 확인. "opened: yes (page)" = HTML 원문(또는 exa 원문 크롤)을 확인.
"환산" = 내가 계산한 값(kgf/cm² × 0.0980665 = MPa). 원문 값이 아님.
로컬 사본: `scratchpad/pdf/*.pdf` (이 세션에서 받은 것).

⚠️ 공통 주의: 제조사 데이터시트 값은 전부 "typical value, not specification" 이며, 시편은 **사출(또는 압축) 성형 시편**이지 펠릿 자체가 아님. 펠릿(3.8 mm 쌀알)의 입자 밀도·탄성률을 직접 잰 자료는 이번 조사에서 찾지 못함.

---

## Topic 1. PP 고체 밀도 (homo / random / impact, 충전 그레이드)

### 1a. 시험 규격 (무엇을 재는가)

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| ISO 1183-1: Method A 침지법(분말 제외 무공극 고체), Method B 액체 비중병법(입자·분말·플레이크·**granules**), Method C 적정법 | 규격 개요(abstract) | https://www.iso.org/standard/74990.html , "ISO 1183-1:2019 Plastics — Methods for determining the density of non-cellular plastics — Part 1", ISO, 2019-03 (2025-06 폐지, ISO 1183-1:2025 로 대체) | yes (page) | abstract 에 "density of plastic materials depend upon the choice of specimen preparation method" 명시. 펠릿 그대로 재려면 Method B 가 해당 |
| ASTM D792: Method A 물 속 변위, Method B 물 이외 액체 | 규격 scope | https://store.astm.org/d0792-20.html , "D792-20 Density and Specific Gravity (Relative Density) of Plastics by Displacement", ASTM, 2020 | yes (page) | Note 1: "not equivalent to ISO 1183-1 Method A"; ISO 는 27±2 °C 추가 허용 |
| ASTM D1505: 밀도 구배관(density-gradient), 정확도 0.05 % 보다 좋게 설계 | 규격 scope, §4.2 | https://store.astm.org/d1505-18.html , "D1505-18 Density of Plastics by the Density-Gradient Technique", ASTM, 2018 | yes (page) | Note 1: "equivalent to ISO 1183-2" |

### 1b. 핸드북·편람·논문

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| Homopolymer 0.904–0.908 | g/cm³, 조건 미기재 ("bulk material at ambient room temperature") | https://www.ineos.com/globalassets/ineos-group/businesses/ineos-olefins-and-polymers-usa/products/technical-information--patents/ineos-engineering-properties-of-pp.pdf , "Typical Engineering Properties of Polypropylene", INEOS Olefins & Polymers USA, April 2014, p.1 "Density" | yes (PDF) | 문서 스스로 "Data gathered from numerous literature sources ... no guarantees" |
| Random copolymer 0.904–0.908 | 같음 | 같음 | yes (PDF) | |
| Impact copolymer 0.898–0.900 | 같음 | 같음 | yes (PDF) | |
| TPOs 0.875–0.880 | 같음 | 같음 | yes (PDF) | |
| 0.895–0.93 | g/cm³, 조건 미기재 | https://en.wikipedia.org/wiki/Polypropylene , "Polypropylene", Wikipedia (Mechanical properties 절, ref: Tripathi D., *Practical guide to polypropylene*, RAPRA 2001) | yes (page, WebFetch) | 2차 출처. Tripathi 원문은 미확인 |
| 결정상 0.946, 비정질 0.855 | g/cm³ | 같은 Wikipedia 인포박스 | yes (page, WebFetch) | 인포박스 값에 인용 번호 없음(2차·근거 약함). 결정화도에 따라 이 사이에서 변함 |
| 0.9 | g/cm³ (iPP 판재, isotactic index 95 %) | https://arxiv.org/pdf/0811.2412 , Jia & Raabe, "Crystallinity and Crystallographic Texture in Isotactic Polypropylene during Deformation and Heating", arXiv 2008, §2 Experimental (p.4) | yes (PDF) | 측정법 미기재 |
| 910 ± 0.889 (min 908, max 911) | kg/m³, 3 mm PP 입자, 방법 원문 확인 못함 | https://journal.eu-jr.eu/engineering/article/download/2968/2330/ , Sudsawat, Chongchitpaisan, Arunyanart, "Calibrating polypropylene particle model parameters with upscaling and repose surface method", EUREKA: Physics and Engineering 2023(6):34–46, DOI 10.21303/2461-4262.2023.002968, Table 6 | yes (PDF 원문, exa 크롤) | 제품 PP 입자(3.00×3.01 mm)의 실측. DOI 는 OpenAlex 로 확인 |
| 889 / 910 / 1107 (범위 889–1107) | kg/m³, 문헌 수집값 | 같은 논문 Table 1, Table 2 | yes | 1107 은 PP 가 아닐 수도 있는 문헌값(출처 기호 a); DEM 입력 스크리닝 범위로만 사용된 값 |

### 1c. 제조사 데이터시트 — 미충전 PP (규격·방법 병기)

| 값 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| 900 | kg/m³, **ISO 1183-1/Method A**; 사출 시편, 23 °C·50 % RH 96 h 이상 컨디셔닝 | https://www.borealisgroup.com/storage/Datasheets/BE52-PDS-REG_WORLD-EN-V7-PDS-WORLD-36576-PDS_BE52_7_21122022.pdf , Borealis "BE52 Polypropylene Homopolymer" PDS Ed.7, 2022-12-21, p.1 | yes (PDF) | homo |
| 0.905 | g/cc, **ASTM D1505** | https://polymers.totalenergies.com/sites/g/files/wompnd5016/files/site_collection_documents/Technical%20Datasheets/3281-US.pdf , TotalEnergies "Polypropylene 3281" (homo, US), Rev Feb 2026, p.1 | yes (PDF) | 시편·컨디션 미기재 ("laboratory conditions") |
| 0.88 | g/cc, ASTM D1505 | .../1251-US.pdf , TotalEnergies "Polypropylene 1251", Rev Feb 2026 | yes (PDF) | **신디오택틱**(저결정) — 문서 머리글은 Homopolymer, 본문은 "syndiotactic form of copolymer". 일반 이소택틱 PP 의 대표값으로 쓰면 안 됨 |
| 0.905 | g/cm³, **ISO 1183** (부분 미기재) | .../PPH_9020.pdf , TotalEnergies "PPH 9020" (clarified homo, 유럽), Rev December 21 | yes (PDF) | |
| 0.902 | g/cm³, ISO 1183 | .../PPC_2660_blown.pdf , TotalEnergies "PPC 2660" (heterophasic copolymer), Rev January 25 | yes (PDF) | impact |
| 900 | kg/m³, ISO 1183 | https://www.albis.com/en/products/download/doc/en/SI/LyondellBasell/MoplenHP456J.pdf , Albis/M-Base 재게시 "Moplen HP456J PP HOMO" (LyondellBasell) | yes (PDF) | 2차 게시(유통사). 제조사 페이지 https://www.lyondellbasell.com/en/polymers/p/Moplen-HP456J/5709f1b6-040e-4f02-af23-6954ee12744f 는 "Density 0.900 g/cm³" (방법 미기재) — yes (page) |
| 0.90 | g/cm³, **ASTM D1505** | KPIC(대한유화) YUHWA POLYPRO 4018 (Topic 10 표 참조) | yes | |
| 0.91 / 0.9 | g/㎤, **ASTM D1505** | 한화토탈에너지스 homo HY301·HJ730·HI828 = 0.91, random RJ970Z = 0.9, block CI571 = 0.91 (Topic 10) | yes (PDF) | 소수 2자리 반올림 차이일 가능성, 원문 그대로 기록 |
| 0.9 | g/cm³, **ASTM D1505 "Density-Gradient"** | LG화학 H1500(homo)·M1500(block) (Topic 10) | yes (PDF) | |
| 0.9 / 0.90 | g/㎤, **ASTM D792** | 효성화학 Topilene J801(homo)·R601(random) (Topic 10) | yes (PDF) | 한국 제조사 중 유일하게 D792 사용 |
| 0.9 (ASTM D792) / 0.9 (ISO 1183) | g/cm³ | 롯데케미칼 RANPELEN J-550N (random) (Topic 10) | yes (PDF, 유통사 게시) | 같은 문서에 ASTM판·ISO판 병기 |
| 0.9 | g/cm³, ASTM D1505 | 폴리미래 Adstif HA5034(homo), EA648P·EA5073~5076(block) (Topic 10) | yes (PDF) | |

### 1d. 충전(filled) 그레이드

| 값 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| 1.04 | g/cm³ (23 °C), 20 % talc PP copolymer | https://www.lyondellbasell.com/en/polymers/p/Hostacom-TRC-411N-R-C11515/02a47a75-bd72-45d7-86c4-8c1ce9c143d5 , LyondellBasell "Hostacom TRC 411N R C11515" 제품 페이지 | yes (page) | 방법 표기 없음(페이지). Flexural modulus 1800 MPa (23 °C, Tech. A) |
| 1.03 | g/cm³, PP 20 % CaCO3, 조건 미기재 | https://www.azom.com/article.aspx?ArticleID=827 , AZoM "Polypropylene - PP 20% Calcium Carbonate Filled", 2001-09-06 ("Abstracted from Plascams", RAPRA) | yes (page) | 일반값(특정 그레이드 아님). Flexural modulus 2 GPa |
| 1120 | kg/m³, **ISO 1183**, 30 % 장섬유 유리(LGF) PP | https://plasticker.de/docs/recybase/34087_1758022473.pdf , SABIC "SABIC® STAMAX 30YM240", Revision 20140908 | yes (PDF) | 제3자 사이트 게시본. Tensile modulus 6650 MPa (23 °C, ISO 527/1B), Flexural 5900 MPa (ISO 178) |

→ 요약(사실 나열): 미충전 PP 데이터시트 밀도는 0.88(신디오택틱 특수)·0.898–0.908·0.90–0.91 g/cm³. 충전 그레이드는 1.03–1.12 g/cm³ 로 뚜렷이 큼. 인형용 펠릿이 충전 재생 PP 인지 여부는 판매처 정보가 없어 **미확인** — 실측(비중병/부력) 필요.

### Topic 1 queries run
1. WebSearch: "polypropylene homopolymer technical data sheet density ISO 1183 flexural modulus ISO 178 pdf"
2. WebSearch: "isotactic polypropylene density crystalline phase 0.946 amorphous 0.855 g/cm3 crystallinity density relation"
3. WebSearch: "talc filled polypropylene 20% talc datasheet density ISO 1183 Borealis OR LyondellBasell Hostacom pdf"
4. WebSearch: "SABIC STAMAX OR "SABIC PP compound" glass fiber 30% polypropylene datasheet density ISO 1183 pdf"
5. WebSearch: "calcium carbonate filled polypropylene 20% CaCO3 compound technical data sheet density g/cm3 ISO 1183 flexural modulus"
6. WebSearch: "LyondellBasell Moplen HP456J technical data sheet pdf density ISO 1183"
7. 직접 열람(exa fetch): iso.org 74990, astm d0792-20, astm d1505-18

---

## Topic 2. 탄성률(굴곡/인장)·푸아송비, 그리고 DEM 의 강성 축소 관행

### 2a. PP 탄성률·푸아송비 (재료값)

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| Young's modulus: homo 1,300 / copolymer 1,100 | MPa, 조건 미기재 | INEOS "Typical Engineering Properties of Polypropylene", April 2014, p.1 "Mechanical Properties" | yes (PDF) | 문헌 수집값 |
| Poisson's ratio 0.42 | —, 조건 미기재 | 같은 문서 p.1 | yes (PDF) | homo/co 구분 없음 |
| 마찰계수 PP–강철 정지 0.30 / 동 0.28; PP–PP 정지 0.76 / 동 0.44 | —, 조건 미기재 | 같은 문서 p.1 | yes (PDF) | 참고(마찰은 이번 과제 범위 밖이지만 같은 표에 있음) |
| Poisson 0.413 (0.003) / 0.411 (0.014) / 0.420 (0.020) | —, 사출온도 210/230/250 °C 사출 시편, ISO 527 인장 10 mm/min, 게이지 22 mm, 소변형 구간 모델 산출 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12349465/ , Takayama, Takahashi, Konno, Sato, "Evaluation of Polypropylene Reusability Using a Simple Mechanical Model Derived from Injection-Molded Products", Polymers 17(15):2107, 2025, DOI 10.3390/polym17152107, **Table 5** | yes (page, 원문 HTML 에서 표 직접 확인) | 재료: Japan Polypropylene Novatec-PP MA1B. 같은 표의 문헌값[9] = 0.390/0.405/0.417 |
| E 1491 (37) / 1433 (68) / 1375 (56) | MPa, 같은 조건 (Table 5) | 같은 논문 Table 5 | yes | 인장 초기 기울기 |
| Poisson 0.368–0.377, E 1361–1658 MPa (Homo-PP); Block-PP 0.360–0.368, E 1182–1455 MPa | 압출 온도 180–240 °C 압출물, 같은 모델 | 같은 논문 **Table 6** | yes | 재료: Novatec MA3(homo), BC03AD(block). ⚠️ WebFetch 요약이 이 표를 "사출·MA3 Poisson" 으로 잘못 요약했었음 → 원문 HTML 로 정정 |
| 굴곡 1300 | MPa, ISO 178; 사출 시편 23 °C/50 % RH ≥96 h | Borealis BE52 PDS Ed.7 2022-12-21 p.1 | yes (PDF) | homo |
| 굴곡 1400 | MPa (N/mm²), ISO 178 (23 °C) | Albis/M-Base "Moplen HP456J" | yes (PDF) | ⚠️ LyondellBasell 제품 페이지는 같은 1400 MPa 를 **"Tensile Modulus"** 로 표기 — 두 출처 불일치, 둘 다 기록 |
| 인장 1,515 / 굴곡 1,380 | MPa, ASTM D638 / D790 | TotalEnergies 3281 (homo, US), Rev Feb 2026 | yes (PDF) | |
| 인장 1700 / 굴곡 1600 | MPa, ISO 527-2 / ISO 178 | TotalEnergies PPH 9020 (clarified homo), Rev Dec 21 | yes (PDF) | |
| 인장 1200 / 굴곡 1100 | MPa, ISO 527-2 / ISO 178 | TotalEnergies PPC 2660 (heterophasic), Rev Jan 25 | yes (PDF) | |
| 굴곡 1850 / 인장 1700 | MPa, ISO 178 / ISO 527-1,2 | https://www.pp-mosten.com/Mosten/media/content/PDF_ENG/TDS-MT-230-ENG.pdf , ORLEN Unipetrol "PP MOSTEN MT 230" (homo), Issued 02/2021 | yes (PDF) | |
| **같은 그레이드 ASTM vs ISO 굴곡**: 1,270 MPa (13,000 kgf/cm², ASTM D790) vs 1,030 MPa (10,500 kgf/cm², ISO 178) | 롯데 RANPELEN J-550N (random), 2015-06 | Topic 10 표 | yes (PDF) | 같은 수지인데 규격에 따라 ≈19 % 차이(원문 두 값) |
| **같은 그레이드 ASTM vs ISO 굴곡**: R200P 9,500 kg/cm² (D790) vs 820 MPa (ISO 178); HB242P 18,000 vs 1,800; HB244P 20,000 vs 2,000 | 효성화학 표 | https://www.hyosungchemical.com/upload/board/20260319/c8f34560-74dc-4b67-9faf-a7717ef8ab6b.pdf , "HYOSUNG Chemical Polypropylene (Korean Plant) Properties", PDF 생성 2026-03-18, p.1 | yes (PDF) | 표 단위란은 "kg/cm3"(오기로 보임) 원문 그대로. R200P: 환산 932 MPa vs ISO 820 MPa |

### 2b. DEM 강성(영률/전단계수) 축소 관행 — 논문 (DOI 전부 OpenAlex/Crossref 로 존재 확인)

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| 결론: 3개 사례 중 2개는 강성 축소가 벌크 거동에 영향 없음, 1개는 영향 있음 | 초록 | Lommen, Schott, Lodewijks, "DEM speedup: Stiffness effects on behavior of bulk material", *Particuology* 12(1):107–112, 2014, DOI 10.1016/j.partic.2013.03.006 | **초록만** (exa library 원문 초록; TU Delft 포털 서지) — 본문 closed(ScienceDirect 403, OA 없음) | 본문의 구체적 값은 **직접 확인 못함** |
| (Yan 등이 요약한 Lommen) "E in the range of 10^7–10^11 Pa has almost no effect on the repose angle"; 압축·관입 시험에서는 차이; bulk stiffness·restitution·shearing·경계 상호작용 모델에선 주의·검증 권고 | Pa | Yan et al. 2015 §4.1.1 (아래) 의 인용문 | yes (Yan 원문) | **2차 인용** — Lommen 원문 대조 안 됨 |
| E = 0.02, 2.0, 200 GPa 에서 퇴적 형상·오리피스 근처 속도·유량 거의 차이 없음; 계산시간 32코어 ≈0.5 h (Δt 5e-6 s) / ≈3 h (5e-7 s) / ≈240 h (5e-8 s) | GPa; 단분산 구 R = 1 mm ≈14,700개, 평바닥 원통 호퍼 D 50 mm, 오리피스 15 mm, LIGGGHTS, Hertz-Mindlin + EPSD 구름저항, Δt = Rayleigh 시간의 10–20 % | Yan, Wilkinson, Stitt, Marigo, "Discrete element modelling (DEM) input parameters: understanding their impact on model predictions using statistical analysis", ***Computational Particle Mechanics*** 2(3):283–299, 2015, DOI 10.1007/s40571-015-0056-5, §3.1, §4.1.1, Fig. 8 | yes (page, Springer 원문 exa 크롤) | ⚠️ 과제문에 "Powder Technol" 로 적혀 있었으나 실제 게재지는 **Computational Particle Mechanics** (OpenAlex 확인). 초록: 반발계수와 영률은 "insignificant impacts ... strongly cross correlated" |
| 강성 축소 = 계산시간 단축 목적의 흔한 관행, Section 7 "Reduction in contact stiffness"; 안정 시간간격 ∝ √밀도, ∝ 1/√접촉강성 | 리뷰 | Coetzee, "Review: Calibration of the discrete element method", *Powder Technology* 310:104–142, 2017, DOI 10.1016/j.powtec.2017.01.015 | **초록·서론·절 미리보기만** (ScienceDirect preview) | Section 7 의 결론·수치는 **미확인** |
| "in pure numerical studies the elasticity is often reduced, neglecting any probable change of numerical response ... awareness ... has increased lately" | 리뷰 초록 | Paulick, Morgeneyer, Kwade, "Review on the influence of elastic particle properties on DEM simulation results", *Powder Technology* 283:66–76, 2015, DOI 10.1016/j.powtec.2015.03.040 | **초록만** (Altair/Siemens community 게시 초록) | 본문 수치 미확인 |
| 문헌 영률을 **두 자릿수(1/100) 축소**, 안식각·벌크밀도 영향 없음(예비 시뮬); 고정값 YM = 5.06×10⁸ Pa; Δt = Rayleigh 의 1/4 | 유리구 d 5 mm, LIGGGHTS Hertz-Mindlin + EPSD | Rackl & Hanley, "A methodical calibration procedure for discrete element models", *Powder Technology* 307:73–83 (OpenAlex 게재연도 2016), DOI 10.1016/j.powtec.2016.11.048, §2.3.4 · §3.4 | yes (PDF, Edinburgh 저장소 accepted manuscript) | "YM significantly influences neither the bulk density nor the angle of repose" |
| (Hlosta 등이 요약한 Chen 2017) 회전드럼 혼합: 영률을 실제값의 0.0007배까지 낮춰도 혼합 결과 불변 | — | Hlosta et al., "DEM Investigation of the Influence of Particulate Properties and Operating Conditions on the Mixing Process in Rotary Drums: Part 1", *Processes* 8(2):222, 2020, DOI 10.3390/pr8020222 (OpenAlex 확인), §2.2.3 | yes (PDF 원문, exa 크롤) | **2차 인용** — Chen 2017 원문 미확인 |
| **PP 입자 DEM**: E_PP 1.3 GPa, ν_PP 0.36 고정(문헌 범위 1.3–2 GPa, ν 0.36–0.41); 실측 안식각 30.18° ± 0.51°; 보정 PP–PP 정지마찰 0.52·반발 0.55; 3 mm 입자 29.14°, 6 mm 업스케일 29.67° | EDEM, Hertz-Mindlin, Rayleigh 시간간격, 고정 깔때기 AOR | Sudsawat et al. 2023, EUREKA Phys. Eng. (6):34–46, DOI 10.21303/2461-4262.2023.002968, Table 1·2·6, §3 | yes (PDF 원문, exa 크롤) | PP 펠릿 계열이지만 **강성 축소는 안 함**(실제 1.3 GPa 사용). 업스케일로 CPU <71 % |

### Topic 2 queries run
1. WebSearch: "Lommen Schott Lodewijks DEM speedup stiffness effects on behavior of bulk material Particuology 2014"
2. exa: "Lommen Schott Lodewijks 2014 DEM speedup stiffness particle shear modulus reduced 1e7 Pa angle of repose abstract"
3. exa: "Coetzee 2017 "Review: Calibration of the discrete element method" particle stiffness reduction shear modulus abstract"
4. exa: "Paulick Morgeneyer Kwade 2015 Review on the influence of elastic particle properties on DEM simulation results abstract"
5. exa: "DEM simulation polypropylene pellets Hertz-Mindlin shear modulus Poisson ratio calibration angle of repose plastic granules paper"
6. WebSearch: "Poisson's ratio isotactic polypropylene measured 0.4 tensile test digital image correlation paper"
7. API: Crossref/OpenAlex/Semantic Scholar DOI 조회 (10.3390/pr8020222, 10.1016/j.partic.2013.03.006, 10.1016/j.powtec.2017.01.015, 10.1016/j.powtec.2015.03.040, 10.1007/s40571-015-0056-5, 10.21303/2461-4262.2023.002968, 10.1016/j.powtec.2016.11.048, 10.3390/polym17202748)
- 참고로 연 것(사용 안 함): García Montagut, Paz, Monzón, *Polymers* 17(20):2748, 2025 (DOI 확인) — **LDPE 분쇄 분말** 보정이라 PP 펠릿 prior 로 부적합, 본문에서 모듈러스 값 추출 못함.

---

## Topic 6. PP 펠릿 벌크밀도와 시험법 (ASTM D1895, ISO 60, ISO 61)

### 6a. 시험법

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| ASTM D1895 적용 범위: 성형 분말 등 플라스틱의 apparent density·bulk factor·pourability. Note 1: **Method A ≡ ISO R 60, Method C = ISO R 61 과 동일** | — | https://store.astm.org/d1895-17.html , "D1895-17 Standard Test Methods for Apparent Density, Bulk Factor, and Pourability of Plastic Materials", ASTM 2017, Scope 1.1 & Note 1 | yes (page) | 최신판 D1895-24 존재(ANSI 목록) — 내용 미확인 |
| Method A: V형 깔때기로 잘 흐르는 미세 과립 → 100 cm³ 원통 컵 | cm³ | https://industrialphysics.com/standards/astm-d1895/ , Industrial Physics (시험기 제조사) "ASTM D1895" | yes (page) | 2차 설명(장비 업체). 이 페이지 일부 서술은 부정확해 보임 → 컵 부피만 인용 |
| **Method B: A 깔때기로 잘 안 흐르는 굵은 과립·dice·펠릿 → 400 cm³ 원통 컵** | cm³ | 같은 페이지 | yes (page) | **펠릿 = Method B** (이 출처 기준). ASTM 원문 확인 안 됨 |
| Method C: 굵은 플레이크·칩·섬유 → 1000 cm³ 눈금 실린더 + 2300 g 플런저로 압축 후 부피 | cm³, g | 같은 페이지 | yes (page) | ISO 61 과 동일 계열 |
| ISO 60:1977: 지정 깔때기로 **100 cm³ 측정 실린더**에 붓고 곧은 자로 윗면 정리 후 질량 측정, g/mL 로 표기 | — | https://www.iso.org/standard/3698.html , "ISO 60:1977 Plastics — Determination of apparent density of material that can be poured from a specified funnel", ISO (폐지, ISO 60:2023 로 대체) | yes (page) | |
| ISO 60:2023: 지정 깔때기로 부을 수 있는 느슨한 재료(분말·과립)의 apparent density | — | https://www.iso.org/standard/85747.html , ISO 60:2023, Ed.3, 2023-09, 4쪽 | yes (page) | 깔때기 치수는 abstract 에 없음 |
| ISO 61:2023: 지정 깔때기로 **부을 수 없는** 느슨한 성형재료(slice·granular·powder)의 apparent density | — | https://www.iso.org/standard/85748.html , ISO 61:2023, Ed.2, 2023-09 | yes (page) | 방법 세부(플런저 등)는 abstract 에 없음 |

### 6b. PP 펠릿 벌크밀도 값

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| Pellets 32–36 lb/ft³ = **513–577 kg/m³**; Flake 29–31 lb/ft³ = 465–497 kg/m³ | 방법·조건 미기재 | INEOS "Typical Engineering Properties of Polypropylene", April 2014, p.1 "Bulk Density" | yes (PDF) | 문헌 수집값 |
| Average bulk density of PP **32–38 lb/ft³**; 보수적 35, 장기 저장 압밀 시 37 lb/ft³ 권장; "Pellet count (number of pellets per gram)" 가 오차 요인; 안식각 37° (계산에 미반영이라고 명시) | 방법 미기재 | https://www.ineos.com/globalassets/ineos-group/businesses/ineos-olefins-and-polymers-usa/products/technical-information--patents/ineos-polypropylene-silo-capacity.pdf , INEOS O&P USA "Polypropylene Silo Capacity", PDF 생성 2014-04-23, p.1 | yes (PDF) | 같은 회사 두 문서가 상한 36 vs 38 로 다름 → 둘 다 기록. 32–38 lb/ft³ 환산 ≈ 513–609 kg/m³ (환산) |
| **0.525** | g/cm³, 방법란 "ISO 1183" (원문 그대로) | TotalEnergies "PPC 2660" (heterophasic copolymer), Rev January 25, p.1 "Other physical properties" | yes (PDF) | ⚠️ ISO 1183 은 고체 밀도 규격이라 벌크밀도 방법 표기로는 부적절해 보임(원문 오기 가능성) |
| **0.525** | g/cm³, 방법란 "ISO 1183" | TotalEnergies "PPH 9020" (clarified homo), Rev December 21, p.1 | yes (PDF) | 같은 회사 유럽 데이터시트 양식 공통 |
| 없음 | — | Borealis BE52, TotalEnergies US 3281/1251, Mosten MT230, LyondellBasell HP456J(Albis), 한국 7사 데이터시트 전부 | yes | 벌크밀도·펠릿 크기·g당 개수 항목 없음 |

### Topic 6 queries run
1. WebSearch: "ASTM D1895 apparent density bulk factor pourability plastic materials method A method B pellets"
2. WebSearch: "polypropylene pellets "bulk density" datasheet "ISO 60" OR "ASTM D1895" kg/m3 PP resin"
3. WebSearch: "ISO 60:1977 Plastics determination of apparent density of material that can be poured from a specified funnel"
4. WebSearch: "ISO 61:1976 plastics apparent density moulding material that cannot be poured from a specified funnel"
5. WebSearch: "polypropylene homopolymer datasheet "bulk density" pellets "ISO 60" kg/m3"
6. WebSearch: ""pellets per gram" OR "pellet count" polypropylene resin specification pellets/g"
7. 직접 열람: store.astm.org d1895-17, industrialphysics.com astm-d1895, iso.org 3698 / 85747 / 85748 (ptli.com D1895 페이지는 크롤 실패 → 미사용)

---

## Topic 10. 한국 PP 수지사 공개 데이터시트 — 펠릿 크기·형상·g당 개수·벌크밀도 기재 여부

단위 kgf/cm² 옆 괄호는 **환산**(내 계산). "펠릿/벌크" 열 = 펠릿 크기·형상·g당 개수·벌크밀도 중 하나라도 있으면 기재.

| 회사 · 그레이드(종류) | 문서 날짜 | 밀도 (방법) | 굴곡탄성률 (방법) | 펠릿/벌크 | 출처 | opened |
|---|---|---|---|---|---|---|
| 대한유화 KPIC · YUHWA POLYPRO 4018 (사출 homo) | 날짜 없음 | 0.90 g/cm³ (ASTM D1505) | 17,000 kgf/cm² (≈1667 MPa 환산) (ASTM D790) | **없음** | https://www.kpic.co.kr/hp/kr/product/polymer/pol_grade.asp?grade=4018&pm_cd=1110 및 영문 spec sheet https://www.kpic.co.kr/hp/en/product/polymer/specsheet.asp?grade=4018&pm_cd=1110 | yes (page) |
| 한화토탈에너지스 · HY301 (압출연신 homo) | PDF 생성 2022-05-05 | 0.91 g/㎤ (ASTM D1505) | 16000 kgf/cm² (≈1569) (ASTM D790) | **없음** | htpchem.com 제품목록 → 리플렛 `.../ZGSDM5040_SRV/LeafletFileSet('86F0969BBFF91EDCB2EF11970848C306')/$value` | yes (PDF) |
| 한화토탈에너지스 · HJ730 (사출 고강성 homo) | PDF 생성 2024-07-18 | 0.91 (ASTM D1505) | 17000 (≈1667) (ASTM D790) | **없음** | LeafletFileSet('4A30CD28D00C1EDF9197A3AF49DFCC1A') | yes (PDF) |
| 한화토탈에너지스 · HI828 (사출 고강성 homo) | PDF 생성 2022-05-03 | 0.91 (ASTM D1505) | 20000 (≈1961) (ASTM D790) | **없음** | LeafletFileSet('86F0969BBFF91EDCB2D0FF8DCC949506') | yes (PDF) |
| 한화토탈에너지스 · RJ970Z (투명 random) | PDF 생성 2022-05-03 | 0.9 (ASTM D1505) | 11500 (≈1128) (ASTM D790) | **없음** | LeafletFileSet('86F0969BBFF91EDCB2D37E656D5FDF0E') | yes (PDF) |
| 한화토탈에너지스 · CI571 (고충격 투명 block) | PDF 생성 2026-05-08 | 0.91 (ASTM D1505) | 11000 (≈1079) (ASTM D790) | **없음** | LeafletFileSet('76BBDE6F5B771FE192CCF0B12C1A5B0A') | yes (PDF) |
| LG화학 · H1500 (사출 homo) | Issued 2022-01-27 | 0.9 g/cm³ (ASTM D1505, Density-Gradient) | 15000 kgf/cm² (≈1471) (ASTM D790, "Press sheet, 1% Secant") | **없음** | https://www.lgchemon.com/sfc/servlet.shepherd/document/download/0692x00000AlBQ4AAN | yes (PDF) |
| LG화학 · M1500 (사출 block) | Issued 2021-12-07 | 0.9 (ASTM D1505) | 12000 (≈1177) (ASTM D790, Press sheet 1% Secant) | **없음** | https://www.lgchemon.com/sfc/servlet.shepherd/document/download/0692x000009o6kwAAA | yes (PDF) |
| 효성화학 · Topilene J801 (의료 주사기 homo) | 업로드 경로 2023-10-24 | 0.9 g/㎤ (**ASTM D792**) | 16,500 kg/㎠ (≈1618) (ASTM D790) | **없음** (보관·건조 조건만: 40 °C 미만, 결로 시 80–100 °C 2–4 h 예비건조) | https://www.hyosungchemical.com/upload/board/20231024/60d69a95-b9cb-44ea-a119-c6f111a203e7.pdf | yes (PDF) |
| 효성화학 · Topilene R601 (고투명 random) | 업로드 경로 2023-10-24 | 0.90 (ASTM D792) | 11,000 (≈1079) (ASTM D790) | **없음** | https://www.hyosungchemical.com/upload/board/20231024/273e9fc7-8208-41a5-9578-af9652568fd9.pdf | yes (PDF) |
| 효성화학 · 한국공장 물성 일람표 | PDF 생성 2026-03-18 | 밀도 열 **없음** | D790(kg/cm²) + ISO 178(MPa) 병기 (위 2a) | **없음** | https://www.hyosungchemical.com/upload/board/20260319/c8f34560-74dc-4b67-9faf-a7717ef8ab6b.pdf | yes (PDF) |
| 폴리미래 · Adstif HA5034 (homo) | TDS 는 요청 시 생성(인쇄일 2026-09-30) | 0.9 g/cm³ (ASTM D1505) | 22000 kg/cm² (≈2157) (ASTM D790) | **없음** | https://www.polymirae.com/product/product_detail.php?idx=156 → downloadPdf.php (POST action_idx=156, type=tds) | yes (PDF) |
| 폴리미래 · Adstif EA648P (block) | 같음 | 0.9 (ASTM D1505) | 18000 (≈1765) (ASTM D790) | **없음** | .../product_detail.php?idx=84 | yes (page+PDF) |
| 폴리미래 · Adstif HA5029 (homo) | 같음 | **밀도 행 없음** | 22000 (ASTM D790) | **없음** | idx=123 | yes (PDF) |
| 롯데케미칼 · RANPELEN J-550N (random) | 2015-06 | 0.9 g/cm³ (ASTM D792) / 0.9 (ISO 1183) | 13,000 kgf/cm² = 1,270 MPa (ASTM D790) / 10,500 = 1,030 MPa (ISO 178) (원문 병기) | **없음** | https://thienphusigroup.com/wp-content/uploads/2024/06/TDS-PP-J550N.pdf (베트남 유통사 게시본; 문서에 "롯데케미칼 보안문서, 무단 복사/반출 금지" 표기) | yes (PDF) |
| 롯데케미칼 공식 사이트 | — | — | — | 확인 불가 | product.lottechem.com — robots.txt 가 `Disallow: /` 라 자동 열람 안 함 | no |
| SK지오센트릭 · YUPLENE | — | — | — | 확인 불가 | skgeocentric.com — robots.txt `Disallow: /` (Googlebot 만 허용). SpecialChem/Prospector 는 로그인 필요 | no |
| HD현대케미칼 | — | — | — | 확인 불가 | 공개 PP 데이터시트 찾지 못함 (hdhyundaichemical.com 은 주차 도메인; hd.com 소개 페이지만 존재) | no |

### 해외 비교 (펠릿/벌크 항목)

| 회사 · 그레이드 | 날짜 | 밀도 (방법) | 굴곡 (방법) | 펠릿/벌크 | opened |
|---|---|---|---|---|---|
| Borealis · BE52 (homo) | 2022-12-21 Ed.7 | 900 kg/m³ (ISO 1183-1/Method A) | 1300 MPa (ISO 178) | 없음 | yes (PDF) |
| TotalEnergies · PPH 9020 (homo, EU) | Rev Dec 21 | 0.905 (ISO 1183) | 1600 MPa (ISO 178) | **Bulk density 0.525 g/cm³** (방법란 "ISO 1183") | yes (PDF) |
| TotalEnergies · PPC 2660 (heterophasic, EU) | Rev Jan 25 | 0.902 (ISO 1183) | 1100 MPa (ISO 178) | **Bulk density 0.525 g/cm³** (방법란 "ISO 1183") | yes (PDF) |
| TotalEnergies · 3281 (homo, US) | Rev Feb 2026 | 0.905 (ASTM D1505) | 1,380 MPa (ASTM D790) | 없음 | yes (PDF) |
| LyondellBasell · Moplen HP456J (homo) | Albis 게시 날짜 미기재 | 900 kg/m³ (ISO 1183) | 1400 MPa (ISO 178) | 없음 | yes (PDF) |
| ORLEN Unipetrol · Mosten MT 230 (homo) | Issued 02/2021 | 밀도 항목 없음 | 1850 MPa (ISO 178) | 없음 | yes (PDF) |
| SABIC · STAMAX 30YM240 (30 % LGF) | Rev 20140908 | 1120 kg/m³ (ISO 1183) | 5900 MPa (ISO 178) | 없음 | yes (PDF) |
| INEOS O&P USA · 일반 PP 물성표/사일로 문서 | 2014-04 | 위 1b | — | **벌크 513–577 kg/m³(펠릿) / 32–38 lb/ft³, "pellet count" 언급(수치 없음)** | yes (PDF) |

→ 관찰(사실): 열어 본 한국 7사 공개 문서 15건 중 **펠릿 크기·형상·g당 개수·벌크밀도를 적은 문서는 0건**. 해외에선 TotalEnergies 유럽 양식만 벌크밀도(0.525 g/cm³)를 적음. 한국사는 밀도를 ASTM D1505(KPIC·한화·LG·폴리미래) 또는 ASTM D792(효성·롯데 ASTM판)로 적고, 굴곡탄성률은 kgf/cm² (ASTM D790) 단위가 표준 양식.

### Topic 10 queries run
1. WebSearch: "롯데케미칼 PP 폴리프로필렌 grade TDS 밀도 굴곡탄성률 pdf"
2. WebSearch: "Lotte Chemical polypropylene homopolymer grade datasheet density flexural modulus"
3. WebSearch: "효성화학 TOPILENE PP grade 물성표 밀도 굴곡탄성률"
4. WebSearch: "한화토탈에너지스 PP 제품 물성 homo polypropylene grade density flexural modulus"
5. WebSearch: "SK geo centric YUPLENE polypropylene grade datasheet density"
6. WebSearch: "LG화학 PP 폴리프로필렌 grade 물성 TDS lgchem.com polypropylene"
7. WebSearch: "HD현대케미칼 PP 제품 grade 물성표 폴리프로필렌"
8. WebSearch: "Hyosung polypropylene technical data sheet pdf J801R density flexural modulus"
9. Brave: "lgchemon.com polypropylene H1500 technical data sheet density"
10. Brave: "HD Hyundai Chemical polypropylene PP grade product site official"
11. Brave: "롯데케미칼 호모 PP H1500 물성 밀도 ASTM D1505 굴곡탄성률 pdf" → 429 rate limit, 결과 없음
12. 사이트 직접 탐색(curl): kpic.co.kr, htpchem.com 제품목록 AJAX(`/product/product_list?schPcCd=AABBAA|AABBBB|AABBCC`), polymirae.com 제품상세·downloadPdf.php, lgchem.com 제품페이지, hyosungchemical.com, 각 사 robots.txt

---

## Gaps / 미확인

1. **펠릿 자체 실측값 부재**: 인형용 PP 펠릿의 입자 밀도, g당 개수, 벌크밀도, 크기 분포를 적은 제조사 자료는 찾지 못함. 판매처(쿠팡 인형용) 원료가 버진/재생/충전 PP 인지 **미확인** → 비중병(ISO 1183-1 Method B) 또는 부력법 실측 필요.
2. **ASTM D1895 원문 미열람**: Method B = 펠릿, 400 cm³ 컵은 장비업체 설명(Industrial Physics) 기준. 깔때기 치수·낙하 높이는 원문(유료)에서 확인 필요. ISO 60/61 의 깔때기 치수도 abstract 에 없음.
3. **Lommen 2014 본문 미열람**: 구체적 강성 범위("10^7–10^11 Pa")는 Yan 2015 의 2차 인용만 있음. Coetzee 2017 Section 7, Paulick 2015 본문 수치도 미확인(유료).
4. **Chen 2017 (0.0007배)** 은 Hlosta 2020 의 2차 인용만; 원문·DOI 미확인.
5. **LyondellBasell HP456J 1400 MPa 가 굴곡(Albis)인지 인장(LYB 페이지)인지 불일치** — LYB 원 TDS PDF 미열람.
6. **TotalEnergies 벌크밀도 0.525 의 방법 표기("ISO 1183")** 는 원문 오기 의심 — 실제 방법 미확인.
7. **INEOS 두 문서의 벌크밀도 상한 불일치** (36 vs 38 lb/ft³) — 원인 미확인.
8. 롯데케미칼(robots 차단)·SK지오센트릭(robots 차단)·HD현대케미칼(공개 TDS 못 찾음) 의 공식 데이터시트는 직접 열람하지 못함. 롯데는 유통사 게시본 1건만.
9. PP 결정/비정질 밀도(0.946/0.855)는 인용 없는 Wikipedia 인포박스만 — 1차 문헌(예: Polymer Handbook) 미확인.
10. 데이터시트 탄성률은 모두 23 °C 부근 준정적 시험값; PP 는 점탄성이라 충돌 시간척도(µs) 의 유효 강성과 다를 수 있음 — 이 차이를 다룬 자료는 이번에 찾지 않음.
