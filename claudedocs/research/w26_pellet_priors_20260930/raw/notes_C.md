# notes_C — PP 펠릿 DEM 사전값: 재료 수준 마찰·반발 + 시험 표준·방법 민감도

작성 2026-09-30 · 읽기/검색 전용 · repo 파일 수정 0
대상: 쌀알형 PP 펠릿(긴 쪽 ≈3.8 mm) / 종이 상자 / PLA 3D 프린트 그랩 / Hertz–Mindlin DEM

## 0. 읽는 법 (범례)

- **opened**
  - `yes` = 페이지나 PDF 본문을 직접 열어 해당 수치를 눈으로 확인함.
  - `partial` = 출판사 페이지가 403/429로 막혀 **exa 색인이 준 본문 발췌**에서만 수치를 확인함. 인용 전에 원문 PDF로 다시 대조해야 함.
  - `abstract` = 초록만 열었고 수치는 없음.
- 검색엔진 AI 요약에서 나온 수치는 **하나도 쓰지 않았다**.
- 출처끼리 값이 다르면 평균 내지 않고 둘 다 적었다.
- `derived` = 원문 값에서 내가 산술로만 바꾼 값(예: tan(각도)). 원문에 그 값이 적혀 있지는 않다.
- **PE(폴리에틸렌) 펠릿 행**은 PP 자료가 없어 "형상이 비슷한 플라스틱 펠릿" 참고로만 넣었다. 플라스틱 펠릿 DEM 보정 논문은 다른 helper 담당이라 요약 수준으로만 적었다.

---

## 1. 주제 3 — 마찰계수 (정지 μs · 운동 μk · 구름 μr)

### 1a. 논문·기술 문서

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| PP 펠릿–강 **외부 마찰계수 0.25~0.6** | 연마·경화한 스크류강 축 Ra 0.06 µm, 압력 8~20 bar(큰 챔버)·50~200 bar(작은 챔버), 속도 0.1~1.2 m/s, 상온. 속도가 오르면 증가, 압력이 오르면 감소. 긴 원통형 펠릿이 더 높음 | Liu, Zitzenbacher, Laengauer, Kneidinger, "Influence of pellet shape on the external coefficient of friction of polypropylene…", AIP Conf. Proc. 1593, 101–105 (2014), https://doi.org/10.1063/1.4873743 | abstract(AIP) + partial(수치) | 압출기 조건이라 **우리 경우보다 수백 배 높은 압력**. 펠릿 모양 의존성의 근거로만 쓸 것 |
| PP–PP 정지마찰 문헌값 **0.1, 0.153, 0.22, 0.3, 0.6** / PP–강 정지 **0.26, 0.3** / PP–PP 운동 **0.05, 0.44** / PP–강 운동 **0.246, 0.28** | 여러 문헌을 모은 표(Table 1). 출처 기호 "j = experiment results"가 달린 값이 저자 실측(PP–강 정지 0.26, 운동 0.246으로 보임. 표기가 모호함) | Sudsawat, Chongchitpaisan, Arunyanart, "Calibrating polypropylene particle model parameters with upscaling and repose surface method", EUREKA: Physics and Engineering (2023), doi:10.21303/2461-4262.2023.002968 | partial | 출판사 페이지가 봇 차단(429). **보정 결과 PP–PP μs = 0.52**(PP–강 μs 0.26 고정) |
| PP–PP 정지·운동 마찰의 기전 | PP, PET, HDPE 쌍 | Cho, Bhushan, Dyess, Tribology Int. 94:165–175 (2016), doi:10.1016/j.triboint.2015.08.027 | 없음(403, S2 API 초록 비공개) | **수치 미확보**. 원문 확보 필요 |
| PP는 "정상" 고분자처럼 운동마찰 ≈ 정지마찰 (정성) | 깨끗한 유리·금속 위 저속 미끄럼 | Pooley & Tabor, Proc. R. Soc. A (1972), https://royalsocietypublishing.org/doi/10.1098/rspa.1972.0112 | abstract | PP 수치 없음. 초록에 적힌 μ≈0.2(정지)와 μ<0.1(운동)은 **PTFE·HDPE 값**이다 |
| PP 펠릿 단층 **미끄럼각: 아연도금 강판 18°, 플렉시글라스 23°** | 한 층의 입자가 기울인 판에서 "미끄러지거나 구르며" 내려가기 시작하는 각. 조건: 원문 Table 3.1, 하중 = 입자 자중 | Tsai, C.K., "Effect of hopper angle on flow of granular materials through rectangular orifices", MS thesis, Texas Tech Univ. (1991), https://ttu-ir.tdl.org/server/api/core/bitstreams/958ab2b6-3372-4556-93f5-e8c16610308f/content | yes | derived tan: 18° → 0.32, 23° → 0.42. **미끄럼과 구름이 섞인 값이라 순수 μs가 아님** |
| LLDPE 펠릿 벽마찰 (Jenike 직접전단기). **운동 벽마찰각 φw**: PE 용융판 16.5°(0.3) · 아크릴 18°(0.32) · 연강 12.3°(0.22) · Polystone 12.3°(0.22) · 컨베이어 벨트 33°(0.65). **정지 벽마찰각**: 15.8(0.28) · 16(0.29) · 15(0.27) · 11.7(0.21) · 35.1(0.7) | 펠릿 평균 4.55 mm(3.8×5.25). 펠릿끼리 마찰은 펠릿을 녹여 만든 판으로 근사 | Grima & Wypych, "Discrete element simulation validation: impact plate transfer station" (2010, UOW) | partial | **PE임**. 표준 방식(D6128)의 벽마찰을 플라스틱 펠릿에 적용한 사례 |
| PE 펠릿 μs: PE/PE 0.287, PE/아크릴 0.277, PE/연강 0.208. μr(보정값): 0.123, 0.158, 0.158 | 벤치 시험. μr은 DEM으로 역보정 | Hastie, ICBMH 2013 "conveyor rock box", https://ro.uow.edu.au/articles/conference_contribution/…/27703683 | partial | **PE임** |
| FDM PLA–C45 강 운동마찰 **0.08~0.16** | CETR UMT-2 트라이보미터, 50 N·150 N, 0.1·1 mm/s, 두 방향. 50 N·0.1 mm/s에서 0.146/0.160, 150 N·1 mm/s에서 0.080/0.112 | Brăileanu et al., "Comparative Examination of Friction Between Additive Manufactured Plastics and Steel Surface" (2023) | partial | 그랩(PLA)–강 참고값. **PLA–PP 쌍이 아님** |
| **PLA–강 0.112~0.167 · PLA–PLA 0.293~0.364 · PLA–HIPS 0.213~0.263** | 썰매식 장치, Fn 1.585~4.655 N, 층 두께 0.06~0.15 mm, 채움률 18~78 %(Taguchi L9). μ = Fa/Fn | Crăciun et al., "Determination of the Friction Coefficient Magnitude in the Case of Polymer Samples Manufactured by 3D Printing" (2024) | partial | 층 두께의 영향이 큼. **PLA끼리 마찰이 PLA–강의 약 2배** |
| PLA–50CrMo4 강: 초기 **0.3~0.35**, 시험 중 증가(PLA는 끝에서 약 2배) | block-on-disc, 40 N, 0.5 m/s, 200 m, 건조 | SERBIATRIB '25 논문, https://scidar.kg.ac.rs/bitstream/123456789/22509/1/184.pdf | partial | 위 두 논문(0.08~0.17)과 **크게 어긋남**. 속도·하중·마찰열 차이. 평균 내지 말 것 |
| 골판지–골판지 **정지 0.38(±0.01), 운동 0.35(±0.01)**. 판지(board)끼리 0.23~0.29 | 클램프 트럭 모사 시험 | earticle A370111 (한국 논문 초록), https://www.earticle.net/Article/A370111 | partial(초록) | **PP–골판지가 아님**. 골판지 표면 규모 참고용 |
| 종이 마찰은 받침 경도·손 접촉·반복 미끄럼에 민감. 저속에서 정지 ≈ 운동. 겉보기 압력 0.4~6.1 kPa에서는 압력 영향 없음 | 수평 썰매법 | Johansson, Fellers, Gunderson, Haugen, "Paper friction – influence of measurement conditions", USDA FPL (1991), https://www.fpl.fs.usda.gov/documnts/pdf1991/johan91a.pdf | partial | 종이 상자 벽 μ를 잴 때 따를 절차 근거 |
| 참고: 사료 펠릿–ABS μs 0.331 · μr 0.069 · e 0.533 | 경사판 미끄럼각 23.54°를 DEM으로 역산. 구름은 45° 경사 굴림 거리로 보정 | Wang et al., J. Phys.: Conf. Ser. 2951 012070 (2025), https://iopscience.iop.org/article/10.1088/1742-6596/2951/1/012070/pdf | partial | 경사판 → DEM 역산 **절차 예시**(재료는 무관) |
| 목재 펠릿: μp-w(스테인리스) 0.2, **μroll 0.02(보정)**. μroll 0 → 0.05에서 AoR 20.1° → 29.9° | 경사판 틸팅 + AoR 역보정, EDEM | Madrid, Fuentes, Ayuga, Gallego, Agronomy 12(2):424 (2022), https://www.mdpi.com/2073-4395/12/2/424 | yes | 원통형 펠릿의 μr 민감도 근거 |

### 1b. 제품 데이터시트·판매 페이지

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| "Plastics to Steel" **정지 0.30 / 동적 0.28**. "Plastics to Plastic" **정지 0.76 / 동적 0.44** | 시험법·하중·속도 **조건 미기재**. 문서 하단: "여러 문헌에서 모은 값, 정확성 보증 없음" | INEOS Olefins & Polymers USA, "Typical Engineering Properties of Polypropylene" (PDF 2014, 2쪽), https://www.ineos.com/globalassets/…/ineos-engineering-properties-of-pp.pdf | yes | 같은 문서: 펠릿 벌크밀도 513~577 kg/m³, E 1,300 MPa(호모), ν 0.42 |
| PP 행 없음 (PE/강 0.2, 폴리스티렌/강 0.3~0.35, PTFE 0.04 등) | 건조 | RoyMech 마찰계수 표 (Machinery's Handbook 등 인용), https://www.roymech.co.uk/Useful_Tables/Tribology/coefficient-of-friction-table/ | yes | PP는 **없음** |
| "대부분 고분자 μ 0.2~0.6". 플라스틱별 대 강 차트는 이미지라 판독 불가 | 건조 강 | Zeus Industrial Products, "Friction and Wear of Polymers" (2005) | yes | PP 수치 확인 **불가** |

### 1c. 표준 시험법 → 5장 표 참조

- 입자–벽 마찰: ASTM D6128(Jenike), ASTM D6773(Schulze 링 전단)
- 필름 마찰: ASTM D1894(2023 폐지), ISO 8295
- 종이 마찰: TAPPI T815, ISO 15359
- 보고 형식: ASTM G115
- **PP–종이 전용 표준과 PP–PLA 전용 데이터: 없음**

---

## 2. 주제 4 — 반발계수 (e, COR)

### 2a. 논문·기술 문서

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| PP 구 **e = 0.83** (Table 2). 본문에는 "PP·아크릴·POM이 약 0.9" | Ø15.01 mm PP 구(밀도 0.89 g/cm³)를 H = 200 mm에서 304 스테인리스 판(150×150×10 mm)에 자유낙하. 충돌음 간격으로 계산(고강도 충돌 구간만 사용) | Chai, Zhong, Yang, Shi, Zhao, "Restitution coefficient of various particles based on acoustic technology", J. Phys.: Conf. Ser. 2557 012057 (2023), https://iopscience.iop.org/article/10.1088/1742-6596/2557/1/012057/pdf | yes | 크기 15 mm라 우리 펠릿(3.8 mm)과 다름. 표값 0.83과 본문 "약 0.9"가 서로 어긋남 |
| PP 비드–아크릴: e가 충돌 속도와 함께 줄다가 **약 0.85로 수렴**. DEM에서 상수 **0.85**가 고속 구간과 일치. 3~6 mm에서 **직경 영향 없음**. 온도가 오르면 감소 | 낙하 시험, 고속카메라 4000 fps, 최대 낙하 1.5 m → 충돌 5.4 m/s. PP–강은 30~150 °C. E*: PP–아크릴 1.20 GPa, PP–강 1.57 GPa | Yurata, T., PhD thesis, Chulalongkorn Univ. (2019), https://digital.car.chula.ac.th/cgi/viewcontent.cgi?article=9469&context=chulaetd | yes | 학술지판: Yurata et al., Adv. Powder Technol. (2021) S0921883121000698(partial). **PP–강 상온 수치는 그림에만 있어 판독 불가** |
| PP–PP e 문헌값 **0.3, 0.49, 0.9, 0.97** / PP–강 **0.55, 0.71** | 문헌 모음 + 저자 실측 섞임 | Sudsawat et al. 2023 (1a와 같음) | partial | 범위가 매우 넓음. **보정 결과 PP–PP e = 0.55** |
| LLDPE 펠릿(원판형, 0.0453 g): e_m 강 **0.624(2 cm)/0.598(3 cm), 평균 0.611** · 폴리카보네이트 **0.704/0.638, 평균 0.671**. 법선 e_N 강 **0.350**, PC **0.294** · 접선 e_T 0.783 / 0.870 | 45° 경사판, 발사 높이 2·3 cm, 케이스당 10회 평균 | Borsa, Paulo, Petit, Piña, Lat. Am. Appl. Res. 49(2) (2019), https://doi.org/10.52292/j.laar.2019.35 | abstract(yes) + partial(표) | **PE임**. 펠릿은 법선 e가 0.3대로 낮게 나올 수 있음(형상 효과) |
| LLDPE 펠릿 e: PE판 0.7 · 아크릴 0.65 · 연강 0.66 · 벨트 0.4 | 충돌 속도 2~4 m/s 평균, 고속카메라 | Grima & Wypych 2010 (1a와 같음) | partial | **PE임** |
| PE 펠릿 e: PE/PE 0.619 · PE/아크릴 0.649 · PE/연강 0.716 | 벤치 시험 | Hastie ICBMH 2013 (1a와 같음) | partial | **PE임**. 원 측정 논문 = Hastie, Chem. Eng. Sci. 101:828–836 (2013), HDPE 3.79 mm, 스테인리스와 벨트 대상(값은 미확보) |
| 폴리아미드 3 mm 구 e: 유리판 **0.85**, 강판 **0.9** (보정값) | 진공 노즐에서 210 mm 낙하 | Frankowski et al., "Material characterisation for DEM calibration" (2013) | partial | 같은 계열 고분자 구 참고 |

**속도 의존성 요약 (근거가 있는 것만)**

- Yurata: 충돌 속도가 오르면 e가 감소하다가 일정값에 수렴한다.
- Borsa: 발사 높이 2 → 3 cm에서 e_m이 모든 케이스에서 감소했다.
- Chai: 속도 의존성 데이터 없음.

### 2b. 제품 데이터시트

- **없음.** INEOS PP 데이터시트, RoyMech 표 모두 COR 항목이 없다.

### 2c. 표준 시험법

- **입자 반발계수 전용 ISO·ASTM 표준: 없음** (찾지 못함).
- 실무 관행은 낙하 시험 + 고속카메라(Yurata, Borsa, Grima), 또는 충돌음 간격법(Chai).

---

## 3. 주제 5 — 안식각 (AoR)

### 3a. 논문·기술 문서 — PP·플라스틱 펠릿 값

| 값·범위 | 단위·조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| PP 입자 **30.18° ± 0.51°** (범위 29.25~30.86) | 고정 깔때기로 컵 안에 더미를 만들고 이미지 분석 + arctan. 입자 길이 3.00 × 폭 3.01 mm, 입자밀도 910 kg/m³ | Sudsawat et al. 2023 (Table 6) | partial | **크기가 우리 펠릿과 가장 가까운 PP 실측**. 깔때기 높이·받침 재질은 발췌에서 확인 불가 |
| PP 펠릿 **채움(filling) 18° / 배출(emptying) 21°** | 채움 = 쏟아부은 더미, 배출 = 빠져나간 뒤 남은 각. 벌크밀도 0.5344 g/cm³ | Tsai 1991 TTU thesis (Table 3.1·3.2) | yes | Sudsawat 값(30°)과 **약 10° 차이**. 방법·펠릿이 다름. 평균 내지 말 것 |
| PP **37°** | 방법·조건 **미기재**. 사일로 용량 계산 문서에서 "고려하지 않음"이라며 언급만 함 | INEOS O&P USA, "Polypropylene Silo Capacity" (PDF 2014), https://www.ineos.com/…/ineos-polypropylene-silo-capacity.pdf | yes | 판매사 문서. 출처 측정 조건 불명 |
| DEM에서 AoR **15.77~39.27°** (파라미터 범위 안에서) | PB 설계 기여도: PP–PP μs 65.7 %, PP–PP e 21.8 %, 밀도 5.1 % | Sudsawat et al. 2023 | partial | **AoR 하나만으로는 μs와 e가 서로 얽힘**(식별성 문제) |

### 3b. 방법 민감도 (정량)

| 효과 | 정량 | 조건 | 출처 | opened |
|---|---|---|---|---|
| **낙하 높이** ↑ → AoR ↓ (비선형) | 60~80 mm 이하에서 기울기 약 **0.13°/mm**. 120 mm 이상에서 약 **23°**로 수렴. 표준 높이 75 mm가 가장 가파른 구간 | 목재 펠릿, 깔때기 출구 60 mm, 스테인리스 받침(반경 102 mm), CV < 3 % | Madrid et al. 2022 | yes |
| **받침 재질**: 펠릿층 받침 vs 강판 | 펠릿층 받침이 **약 10° 큼**(모든 높이). 스테인리스·적층강·나무·**판지** 사이에도 차이가 있었다고 서술(수치 없음) | 가장자리 10 mm 턱이 있는 용기 | Madrid et al. 2022 | yes |
| **깔때기 출구 막힘** | ISO 4324/UNE 55547의 출구 10 mm는 등가직경 2~3 mm 넘는 입자에 너무 작음. 출구 ≥ 6 × 등가구 직경을 권장 | — | Madrid et al. 2022 | yes |
| 깔때기 높이·받침 규정 | 고정 받침 + 가루를 붙잡는 턱("common base"). 깔때기를 더미 꼭대기에서 **2~4 cm**로 유지. tan α = 높이 / (0.5 × 밑지름) | 제약 분말 | USP ⟨1174⟩ (2024-05-01 공식) | yes |
| 정적 vs 동적 | 동적 AoR(회전 드럼)은 보통 정적보다 **최소 3~10° 작음** | 리뷰 인용 | Müller, Fimbinger, Brand, Powder Technol. 383:598–605 (2021) [→ Al-Hashemi 2018 인용], https://pure.unileoben.ac.at/…/1_s2.0_S0032591021000176_main.pdf | yes |
| 배수(internal) vs 부음(external) | 배수 AoR이 **유의하게 큼** (Cho, Dodds, Santamarina) | 리뷰 | Beakawi Al-Hashemi & Baghabra Al-Amoudi, Powder Technol. 330:397–417 (2018) | partial |
| 방법 6종 비교 | 최고 = ASTM·Cornforth법. 삽 퍼붓기 6 % 작음, Santamarina–Cho 건식 12 % 작음, **원통 들기 35 % 작음** | 모래 6종 | Rousé, "Comparison of Methods for the Measurement of the Angle of Repose of Granular Materials" (Geotech. Testing J., 2014) | partial(초록) |
| 보정 시험에 따라 DEM 파라미터가 달라짐 | 정적(원통 들기) 보정: μs(pp) 0.40 · μs(아크릴) 0.42 · μr 0.092. 동적(드럼) 보정: 0.20 · 0.31 · 0.048 → **35~100 % 차이**. 두 시험을 동시에 맞추는 조합은 없었음 | 밀 | Liu & Chen (2017) | partial(초록) |
| 원통 들기 속도 · 받침 거칠기 | 빠른 들기(7~8 cm/s)가 느린 들기(2~3 cm/s)보다 AoR 작음. 받침이 거칠수록 AoR 큼. 양이 많을수록 AoR 작음 | 모래·자갈 | Liu, Z., "Measuring the angle of repose of granular systems using hollow cylinders" (thesis) | partial |
| 받침 거칠기 · 양 (Miura et al. 1997 요약) | 모래에서 거칠기 영향 **< 1°**, 양 증가 효과 약 **2°** 감소. 투입 속도가 느릴수록 AoR 큼 | 모래·유리구 | Anaraki (2008, TU Delft 석사, Miura 1997 인용) | partial |
| 낙하 높이 · 용기 폭 (DEM) | h/d 20 → 40에서 약 **3.0°** 감소. w/d 20 → 40에서 약 **3.5°** 감소 후 안정. 입경 5 → 9 mm에서 약 4.0° 감소 | 철광석 조립물 | Li, Zhao, Honeyands, Moreno-Atanasio (2016) | partial |
| 깔때기 직경 | 0.7 / 1.4 / 2.2 cm → 최종 AoR 29~32°로 **영향 작음**, 성장 궤적은 달라짐 | 강모래, 고정 깔때기 11 cm | Capaldi et al. (2026) | partial |
| AoR 판독 오차 | 수작업 측정 오차 59.93 % → 절차 개선 후 4.82 % (Nested Gage R&R) | 비료 1.5~5 mm | Measurement system analysis, ScienceDirect S026322412031191X | partial(초록) |

### 3b'. 판매·제품 페이지

| 값 | 조건 | 출처 | opened | 메모 |
|---|---|---|---|---|
| PP 37° | 미기재 | INEOS silo capacity PDF (위와 같음) | yes | |
| PP·PE 행 없음 | — | BisonConvey "Bulk Material Properties Reference", https://bisonconvey.com/tools/bulk-material-properties/ | yes | **없음** |

---

## 4. 이 과제에 주는 시사점 (근거가 있는 추론만)

1. **PP–PP μs 사전분포는 넓다.**
   - 데이터시트: 0.76(정지)/0.44(동적), 조건 미기재.
   - DEM 보정: 0.52 (Sudsawat).
   - 문헌 모음: 0.1~0.6.
   - → 단일 값을 고르지 말고 범위 사전분포로 둘 것.
2. **PP–강 μs는 0.26~0.30 근처로 수렴한다** (INEOS, Sudsawat). 다만 **우리 벽은 종이이고 그랩은 PLA**다.
   - PP–종이 실측: 문헌 **없음**.
   - PP–PLA 실측: 문헌 **없음**.
   - 참고: PLA–PLA 0.29~0.36(Crăciun). PLA–강은 0.08~0.17과 0.3~0.35로 서로 **불일치**.
3. **e (반발계수)**
   - PP 구의 벌크 재료값: 0.83~0.85 근처 (Chai; Yurata, 아크릴 상대).
   - 펠릿 형상(원판·원통)에서는 법선 e가 0.3대로 떨어진 사례가 있다 (Borsa, PE).
   - Sudsawat 보정 결과는 0.55.
   - → "재료 e"와 "형상을 반영한 유효 e"를 구분해야 한다.
4. **AoR 실측은 방법을 먼저 고정해야 한다.**
   - 같은 PP인데 18°(Tsai)부터 30°(Sudsawat), 37°(INEOS)까지 분포한다.
   - 낙하 높이, 받침(펠릿층 vs 매끈한 판, 판지), 깔때기 출구 ≥ 6 × 입경을 기록할 것.
   - 가능하면 정적 AoR과 동적/배수 AoR 두 가지 이상을 재서 μs–μr의 얽힘을 풀 것 (Liu & Chen; Sudsawat 기여도 분석).

---

## 5. 표준 표

| 표준 | 무엇을 재나 | 장비·절차 요지 (공개 범위에서만) | 펠릿 적용성 | URL |
|---|---|---|---|---|
| **ISO 4324:1977** (2024 재확인, 현행) | 분말·과립의 안식각 | 공식 초록: "정해진 부피를 특수 깔때기에 통과시켜, 고정 높이에 둔 완전히 평평하고 수평인 판 위에 생긴 원뿔의 밑각을 잰다. 비슷한 성질의 다른 분말·과립에도 적용." 세부 치수는 원문 미열람. Madrid 2022의 2차 서술(UNE 55547 = ISO 4324): 150 mL, 유리 깔때기 출구 10 mm, 높이 75 mm, 금속 받침 Ø100~150 mm | 출구 10 mm라 **3.8 mm 펠릿은 막힐 가능성** (Madrid: 등가직경 > 2~3 mm이면 막힘) | https://www.iso.org/standard/10196.html |
| **ASTM C1444-00** (**2005 폐지**) | 자유유동 몰드 파우더의 안식각 | 공식 범위: "입자 크기·모양·벌크밀도가 흐름성에 영향. 자유유동 몰드 파우더의 AoR 결정." Al-Hashemi 리뷰 2차 서술: 약 454 g, 받침–노즐 높이 약 3.81 cm, 지름은 1 in 단위로 반올림 | 폐지됨. 분말용 | https://store.astm.org/standards/c1444 |
| **USP ⟨1174⟩ Powder Flow** (PDG 조화, 2024-05-01 공식) | AoR, Carr 압축지수·Hausner비, 오리피스 유출, 전단셀 | 고정 받침 + 가루층을 잡는 턱. 깔때기를 더미 꼭대기에서 2~4 cm로 유지. 진동 금지. tan α = h / (0.5 × 밑지름). 변형: 배수각, 동적각(회전 원통). 흐름성 척도: 25~30° 우수 … >66° 매우매우 불량. "AoR은 고유 물성이 아니며 방법에 크게 의존" | 원리는 적용 가능. 척도는 제약 분말 기준 | https://www.usp.org/sites/default/files/usp/document/harmonization/gen-chapter/20230428HSm99885.pdf |
| **ASTM D6393/D6393M-25** | Carr 지수 8측정 + 2계산 (Carr 안식각·낙하각·차각·벌크밀도 등) | 공식 범위: **입자 ≤ 2.0 mm**, 6.0~8.0 mm 깔때기 출구를 통과해야 함 | **3.8 mm 펠릿은 범위 밖** | https://www.astm.org/Standards/D6393.htm |
| **ASTM D6128-16** (Jenike) | 벌크 고체의 응집강도, 내부마찰, 벌크밀도, **벽마찰**(여러 벽면) | 전단셀. 공식 1.2: 셀 이동 한계 안에 정상상태에 도달하지 못하는 재료(예: 매우 탄성인 입자)에는 부적용 | 펠릿–판지·PLA 벽마찰을 잴 **표준 틀**. Grima 2010이 PE 펠릿에 적용 | https://store.astm.org/d6128-16.html |
| **ASTM D6773-22** (Schulze 링 전단) | 벌크 고체 유동성·벽마찰 | 링 전단셀 | 위와 같음 | https://store.astm.org/d6773-22.html |
| **ASTM D1894-14** (**2023 폐지**) | 플라스틱 필름·시트의 정지·운동 마찰 | 썰매–평판. 공식 4.6: 정지 COF는 하중 속도와 정지 시간에 매우 민감 | **필름용**. 펠릿에는 직접 적용 불가 | https://store.astm.org/d1894-14.html |
| **ISO 8295:1995** (현행 표기) | 플라스틱 필름·시트 마찰 | 썰매–평판 | 필름용 | https://www.sis.se/en/produkter/…/iso8295/ |
| **TAPPI T815** (현행 ANSI/TAPPI T 815 om:2024) | 포장재(골판지 포함)의 정지마찰 = 미끄럼각 | 경사를 일정 속도로 올리다가 미끄러지는 각의 tan | **판지 벽에 펠릿 썰매를 얹는 변형으로 PP–판지 μs 측정에 응용 가능** (표준 자체는 판지–판지) | https://www.intertekinform.com/en-us/standards/tappi-t-815-2012-r2018--1062861_saig_tappi_tappi_2472413/ |
| ISO 15359 / ASTM D4521(폐지) | 종이·판지 정지·운동 마찰(수평) / 골판지 정지마찰 | 수평 썰매 | 판지 측정 참고 | 판매 페이지 요약만 확인(partial) |
| **ASTM G115-10(R18)** | 마찰계수 측정·보고 가이드 | 트라이보시스템 기록 항목과 보고 양식 | **보고 양식으로 채택 권장** | https://store.astm.org/g0115-10r18.html |
| 입자 반발계수 표준 | — | **없음** | — | — |

---

## 6. 검색 로그

**WebSearch**
1. coefficient of friction polypropylene steel static kinetic table
2. polypropylene pellets DEM calibration coefficient of restitution static friction rolling friction angle of repose
3. wall friction angle plastic pellets Jenike shear tester polypropylene granules
4. polypropylene pellet coefficient of restitution drop test measurement impact velocity
5. "angle of repose" "polypropylene pellets" degrees measured
6. "Calibrating polypropylene particle model parameters with upscaling and repose surface method"
7. "Influence of pellet shape on the external coefficient of friction of polypropylene" Liu Zitzenbacher
8. ISO 4324 surface active agents powders and granules measurement of the angle of repose scope
9. ASTM C1444 standard test method measuring angle of repose free-flowing mold powders scope
10. ASTM D6393 bulk solids characterization by Carr indices …
11. ISO 8295 plastics film and sheeting …
12. Grima Wypych "Discrete element simulation validation" …
13. Hastie "conveyor rock box" …
14. bulk material density angle of repose table "polypropylene" pellets conveyor manufacturer

**exa web_search_exa**
1. COR PP beads drop test acrylic steel APT
2. Hastie 2013 COR PE pellets
3. Yurata thesis COR (2 회)
4. Cho/Bhushan PP friction
5. PLA 3D-printed friction vs PP/steel
6. plastic granules on cardboard friction
7. PP pellet static friction vs steel DEM table
8. USP ⟨1174⟩ text
9. AoR method comparison lifting cylinder / funnel / tilting
10. drop height & base roughness AoR
11. PP pellets AoR table
12. PP–paperboard friction
13. plastic particles on cardboard wall friction DEM

**Semantic Scholar API** (초록 3건 조회 → 모두 출판사가 초록 비공개)

**직접 열람**
- WebFetch / exa_fetch / curl+pdftotext:
  - RoyMech, INEOS ×2, Zeus, Chai(IOP), Yurata thesis, PMC6960936(초음파 소성, 무관), AIP 초록
  - ISO 4324, ASTM C1444 / D6128 / D1894 / G115 / D6773 / D6393, SIS ISO 8295, Intertek TAPPI T815
  - USP ⟨1174⟩, Müller 2021 PDF, MDPI Madrid 2022, TTU Tsai thesis, LAAR Borsa, BisonConvey
- 실패:
  - 403: ScienceDirect, Wiley, MDPI(WebFetch — exa로 우회 성공)
  - 429: EUREKA
  - ResearchGate timeout
- Playwright 1회(EUREKA 429)로 생긴 로그·스냅샷 2개는 삭제했다. **`.playwright-mcp/`의 나머지 4개 파일은 다른 세션이 만든 것이라 그대로 두었다.**

---

## 7. 공백 (gaps)

- **PP–판지(종이 상자) μs·μk: 없음.** → TAPPI T815식 경사판에 PP 펠릿 썰매(펠릿을 붙인 판 또는 단층)를 올려 **직접 측정 필요**.
- **PP–PLA(FDM) μ: 없음.** PLA–강 문헌끼리도 0.08~0.17과 0.3~0.35로 불일치. → 실측 필요.
- **PP 펠릿 μr 직접 측정값: 없음.** 모두 AoR 역보정값(PE 0.1~0.2, 목재 0.02).
- **쌀알형 3.8 mm PP 펠릿의 반발계수: 없음.** PP 구(15 mm, 3~6 mm)와 PE 펠릿만 있음. 펠릿의 법선 e가 형상 때문에 낮을 수 있다(Borsa 0.29~0.35).
- Cho/Bhushan 2016(PP–PP 정지·운동)과 Hastie 2013 CES(PE 펠릿 COR) 원문 수치: 미확보(유료).
- ISO 4324 원문 세부 치수: 미열람(2차 서술만 있음).
- Sudsawat 2023: 출판사 차단으로 원문 PDF 대조가 필요하다(표기 j/d/e 매핑 모호).
- `partial` 표시 행은 모두 인용 전에 원문 대조가 필요하다.
