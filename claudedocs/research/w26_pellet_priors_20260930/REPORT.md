# PP 알갱이(펠릿) 물성·측정법 문헌/제품 자료 조사 — W26 (2026-09-30)

- 성격: **읽기·검색 전용 조사**. 로봇·카메라·GPU·시뮬 실행 0, 코드 수정 0, 상태 원장 수정 0.
- 작성: Orca 조사 워커 (task_7c44fc3cddfb). 검색·열람은 도우미 4명(A 물성·데이터시트 / B DEM 보정 논문 / C 마찰·반발·안식각·표준 / D 판매 페이지·밀도 측정법·영상)이 나눠 했다. 핵심 수치는 워커가 원문으로 다시 대조했다(§7-3).
- 대상: 쿠팡 "PP 알갱이 인형 완충재 1 kg"(상품번호 109307604). 판매 페이지 표기는 반투명 백색, 쌀알형, **긴쪽 3.8±0.3 mm**.
  종이 상자(안쪽 310×220 mm)에 4 cm 층으로 깔고, PLA 3D 프린트 그랩으로 퍼낸다. 시뮬레이션은 DEME 2.4.0 Hertz–Mindlin 이다.

## 용어 (처음 한 번만 풀어 씀)

| 용어 | 뜻 |
|---|---|
| DEM (discrete element method, 이산요소법) | 알갱이 하나하나의 충돌·마찰을 계산하는 시뮬레이션 |
| Hertz–Mindlin | 두 알이 눌릴 때 힘을 영률·푸아송비로 계산하는 접촉 모델 |
| 영률 E (Young's modulus) | 재료가 얼마나 딱딱한지. 클수록 덜 눌린다 |
| 푸아송비 ν (Poisson's ratio) | 한쪽으로 누를 때 옆으로 퍼지는 비율 |
| 반발계수 e (coefficient of restitution, COR) | 충돌 뒤 속도 ÷ 충돌 전 속도. 1 이면 완전 탄성, 0 이면 붙어 버림 |
| 정지마찰 μs / 운동마찰 μk | 미끄러지기 시작할 때 / 미끄러지는 동안의 마찰계수 |
| 구름마찰 μr (rolling friction) | 알이 굴러가지 못하게 막는 저항. 알 모양(구가 아님)을 흉내 낼 때 흔히 쓴다 |
| 안식각 (angle of repose, AoR) | 알갱이를 부어 쌓은 더미의 옆면이 바닥과 이루는 각 |
| 고체 밀도 | 재료 자체의 밀도(알 사이 빈틈 제외) |
| 부피밀도 (bulk density) | 부어 담았을 때의 겉보기 밀도(빈틈 포함) |
| 보정 (calibration) | 실험 결과(안식각 등)와 같아지도록 시뮬 입력값을 거꾸로 맞추는 일 |
| 데이터시트 (TDS) | 수지 제조사의 물성표. 값은 성형 시편에서 잰 "대표값"이다 |
| 비중병 (pycnometer) | 부피를 정확히 아는 병. 액체나 기체로 시료의 부피를 잰다 |
| 밀도 구배관 (density gradient column) | 위에서 아래로 밀도가 연속으로 커지는 액체 기둥. 시료가 멈추는 높이로 밀도를 읽는다 |

---

## ① 요약 (한 화면)

1. **고체 밀도는 문헌 범위가 좁다.**
   - 미충전 PP 데이터시트 값은 **0.898–0.91 g/cm³** 이다. 예외로 신디오택틱 특수 그레이드 하나가 0.88 이다.
   - 충전 그레이드(탈크·탄산칼슘·유리섬유)는 **1.03–1.12 g/cm³** 로 크다.
   - 우리 제품 페이지의 재질 표기는 "플라스틱" 뿐이고, "PP" 는 상품명에만 있다. 버진인지, 재생인지, 충전재가 들었는지 모른다.
   - 그래서 문헌 범위를 쓰려면 "이 봉지가 미충전 PP" 라는 사실부터 확인해야 한다.
2. **영률은 문헌 값이 1.0–1.9 GPa 인데, DEM 에서는 이 값을 그대로 쓰지 않는 관행이 있다.**
   - 계산 시간을 줄이려고 1/100~1/1000 로 낮추거나 MPa 수준으로 둔다(Mesnier 2020 은 2.84 MPa).
   - 안식각·부피밀도·배출 유량에는 영향이 거의 없다는 근거가 있다(Yan 2015, Rackl & Hanley 2017).
   - 반면 **압축·관입 시험에서는 차이가 난다**고 Lommen 2014 가 보고했다. 이 내용은 Yan 2015 의 인용으로만 확인했다.
   - 우리 그랩은 층을 파고드는 동작이므로 이 경고가 직접 관련된다.
   - 현재 시뮬 입력 E = 5 MPa 는 이 관행의 범위 안에 있다.
3. **PP–PP 마찰은 출처마다 0.43~0.8 로 흩어져 있다.**
   - 출처별 값: 0.432(Thieleke 2021), 0.52(Sudsawat 2023 보정), 0.6(Mesnier 2020 보정), 0.76(INEOS 데이터시트, 조건 미기재), 0.8(PP 분말 보정).
   - PP–강철 마찰은 **0.26–0.30** 로 비교적 모인다.
   - **PP–종이(골판지) 마찰, PP–PLA 마찰을 잰 자료는 이번 조사에서 한 건도 찾지 못했다.**
4. **반발계수는 두 부류로 나뉜다.**
   - 재료값(PP 구, 판 충돌)은 **0.83–0.85** 이다.
   - DEM 보정값은 **0.3–0.85** 로 넓다.
   - 원판형 PE 펠릿은 법선 방향 반발이 0.29–0.35 로 낮게 나왔다. 알 모양이 유효 반발을 끌어내린다는 근거다.
   - 쌀알형 PP 펠릿을 직접 잰 값은 없다.
5. **안식각은 같은 PP 라도 18°·21°·30°·37° 로 다르다.**
   - 측정법(깔때기·높이·받침판 재질·정적/동적)에 따라 수 °~10° 가 달라진다는 정량 근거가 있다.
   - 표준 깔때기(ISO 4324, 출구 10 mm)는 2–3 mm 보다 큰 알에서 막힌다고 보고됐다.
   - ASTM D6393 은 2 mm 이하 입자만 다룬다.
   - 따라서 **방법을 먼저 정해야 값이 의미를 가진다.**
6. **부피밀도는 PP 펠릿 문헌값 0.513–0.609 g/cm³ 범위에 모인다.**
   - 문헌값: INEOS 0.513–0.577, INEOS 사일로 문서 32–38 lb/ft³, TotalEnergies 0.525, Johann 2022 렌즈형 PP 최대 0.531.
   - 같은 알이라도 **붓는 높이가 낮으면 0.419 까지 떨어진다**(Johann 2022, 낙하 5.5 mm).
7. **알 1개 질량, 개/g, 단축(폭·두께) 치수는 어느 판매 페이지에도 없다.**
   - 쿠팡(우리 제품)·스마일러브·청송뜨개실·썬퀼트·프랜즈얀·해외 Fairfield 모두 표기하지 않았다.
   - 한국 PP 수지 7사의 공개 데이터시트 15건에도 펠릿 크기·개/g·부피밀도가 **0건** 이다.
   - 크기가 가장 가까운 문헌값은 렌즈형 버진 PP 한 가지다: 3.70×4.35×2.28 mm, **29.2 mg/알**(Johann 2022).
8. **물에 뜨는 PP 알의 밀도는 네 가지 방법으로 잴 수 있다.**
   - 아르키메데스법(싱커로 가라앉히거나 에탄올에 담금), 액체 비중병, 기체 비중병, 밀도 구배관이다.
   - 규격 기준으로는 ISO 1183-1 Method B(비중병)가 "granules" 를, ISO 1183-2(구배관)가 "pellets" 를 명시한다.
   - ASTM D792 Method A 는 1–50 g 짜리 **한 덩어리** 시료를 전제로 한다.
   - D792 는 밀도 1 미만·10 g 미만 시료에 0.1 mg 저울을 요구한다. 우리 0.01 g 포켓저울은 규격을 충족하지 못한다.
9. **참고 영상은 아르키메데스식 전자 비중계 시연이다.**
   - 영상은 Hongtuo Instrument, 2017, 42초이며 자막이 없다.
   - 모델명은 영상과 설명 어디에도 없다. 같은 회사의 과립용 제품은 DH-300 이다(분해능 0.001 g/cm³, 뜨는 시료용 부속 포함, ASTM D792·ISO 1183 준수 주장).

**채움 현황**
- §2 의 10개 항목은 모두 표에 있다. 찾지 못한 것은 "없음" 으로 적었다.
- 논문은 DEM 보정 논문 7편을 원문 표로 확인했다(목표 5편 이상). 이 밖에 측정·방법 논문을 11편 확인했다.
- 국내 판매 페이지는 5곳이다(목표 2곳 이상). 여기에 복제 페이지 1곳, 목록 페이지 2곳이 더 있다.
- 밀도 측정법은 6종을 비교했다(목표 4종). 필수 4종에 적정법과 전자 비중계를 더했다.

---

## ② 항목별 표

열 설명: **적용성** = 우리 경우(쌀알형 PP 3.8 mm · 종이 상자 · PLA 그랩 · DEME Hertz–Mindlin)에 바로 쓸 수 있는가와 그 이유.
**신뢰도**: 상 = 원문 표를 직접 확인했고 시험 조건이 명시됨. 중 = 원문은 열었지만 조건 미기재, 2차 인용, 또는 크롤 사본. 하 = 초록·발췌만 봄, 또는 출처 안에서 값이 서로 어긋남.
현재 시뮬 입력값은 `claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929/rev34/params_w25_paperbox.json:3-13` 기준이다.
값: E_pa 5.0e6 · nu 0.3 · CoR 0.3 · mu 0.45 · Crr 0.06 · particle_density 905 kg/m³ · E_mesh 3.0e9 · bulk_density 0.55 g/cm³.
출처 번호 `[Sxx]` 는 `sources.json` 의 id 다.

### 항목 1. PP 고체 밀도

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (a) 편람 | 호모·랜덤 0.904–0.908 / 임팩트 코폴리머 0.898–0.900 | g/cm³, "상온 벌크 재료", 시험법 미기재. 문서 스스로 "여러 문헌 수집값, 보증 없음" | INEOS O&P USA, "Typical Engineering Properties of Polypropylene", 2014-04, p.1 [S01] | 미충전 PP 라면 범위로 쓸 수 있다. 우리 알이 미충전 PP 인지는 미확인 | 중 |
| (a) 논문 | 910 ± 0.889 (min 908, max 911) | kg/m³, 3.00×3.01 mm PP 입자, 비중병(ASTM D854 로 표기) | Sudsawat 등 2023, EUREKA Phys. Eng. (6):34–46, Table 6 [S20] | 크기가 비슷한 PP 입자의 실측값. 출판사 페이지는 429 로 막혀 크롤 사본으로 확인 | 중 |
| (a) 2차 | 결정상 0.946 / 비정질 0.855 | g/cm³ | Wikipedia "Polypropylene" 인포박스(인용 번호 없음) [S02] | 결정화도에 따라 이 사이 값이 된다는 정성 근거로만 쓴다 | 하 |
| (b) 데이터시트 | 0.900 / 0.905 / 0.902 / 0.88 | g/cm³ 또는 kg/m³ | Borealis BE52 (ISO 1183-1 A, 사출 시편 23 °C 50 %RH ≥96 h) 900 [S03] · TotalEnergies 3281 (ASTM D1505) 0.905 [S04] · PPH 9020 (ISO 1183) 0.905 [S05] · PPC 2660 0.902 [S06] · TotalEnergies 1251 **신디오택틱** 0.88 [S07] · LyondellBasell Moplen HP456J 0.900 [S08] | 전부 성형 시편 값이다. 펠릿 자체 값이 아니다 | 상(값) / 중(펠릿 적용) |
| (b) 국내 데이터시트 | 0.90–0.91 | g/cm³ | KPIC 4018 · 한화토탈 HY301·HJ730·HI828·RJ970Z·CI571 · LG화학 H1500·M1500 · 폴리미래 HA5034·EA648P = ASTM D1505 / 효성 J801·R601 · 롯데 J-550N = ASTM D792 [S30–S34][S36][S37] | 같은 이유 | 상 / 중 |
| (b) 충전 그레이드 | 탈크 20 % 1.04 / CaCO₃ 20 % 1.03 / 장유리섬유 30 % 1.12 | g/cm³ | LyondellBasell Hostacom TRC 411N [S09] · AZoM(Plascams) [S10] · SABIC STAMAX 30YM240 (ISO 1183) [S11] | 알이 충전·재생 PP 라면 이 쪽에 가까워진다. **측정으로 가를 수 있다** | 중 |
| (c) 표준 | ISO 1183-1 (A 침지 / **B 비중병: particles·powders·flakes·granules** / C 적정), ISO 1183-2 (구배관, pellets 허용), ISO 1183-3 (기체 비중병), ASTM D792 (A 물 / B 다른 액체), ASTM D1505 (구배관, ≡ ISO 1183-2) | — | [S40][S41][S42][S43][S44] | 항목 9 에서 비교 | 상(scope) |

### 항목 2. 영률·푸아송비, 그리고 DEM 에서 실제로 쓰는 축소 강성

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (a) 편람 | E 호모 1,300 / 코폴리머 1,100 · ν 0.42 | MPa, 조건 미기재 | INEOS 2014 p.1 [S01] (워커 원문 대조) | 재료값의 기준 | 중 |
| (a) 논문 | ν 0.413 / 0.411 / 0.420 · E 1491 / 1433 / 1375 | MPa. 사출 시편(사출 온도 210/230/250 °C), ISO 527 인장 10 mm/min | Takayama 등 2025, Polymers 17(15):2107, **Table 5** [S12] | 재료 ν 의 실측 근거. Table 6(압출물)은 ν 0.360–0.377 | 상 |
| (b) 데이터시트 | 굴곡 1,030–2,157 · 인장 1,200–1,700 | MPa (kgf/cm² 표기는 환산). ISO 178 / ASTM D790 / ISO 527 | Borealis BE52 1300 (ISO 178) · Moplen HP456J 1400 · TotalEnergies 3281 굴곡 1380 / 인장 1515 · PPH 9020 1600/1700 · PPC 2660 1100/1200 · Mosten MT230 1850/1700 · 한국 7사 (§④ 부록 표) [S03–S08][S13][S30–S34][S36][S37] | 준정적 23 °C 값이다. µs 단위 충돌에서의 유효 강성과는 다를 수 있으나, 이를 다룬 자료는 찾지 않았다 | 상 |
| (b) 규격 차이 | 같은 그레이드인데 ASTM D790 1,270 vs ISO 178 1,030 MPa (≈19 % 차) | 롯데 J-550N 2015-06 | [S37], 효성 일람표도 같은 경향 [S36] | 값을 비교할 때는 규격을 맞춰야 한다 | 상 |
| (b) 불일치 | Moplen HP456J 1400 MPa 가 유통사 사본에서는 "굴곡", 제조사 페이지에서는 "인장" | — | [S08] | 둘 다 기록한다. 원 TDS 는 미열람 | 하 |
| (a) DEM 축소 근거 | E 0.02 / 2.0 / 200 GPa 에서 더미 모양·유량이 거의 같음. 계산 시간은 ≈0.5 h / ≈3 h / ≈240 h | 단분산 구 R 1 mm ≈14,700개, 호퍼, LIGGGHTS, 32코어 | Yan, Wilkinson, Stitt, Marigo 2015, **Computational Particle Mechanics** 2(3):283–299, §3.1·§4.1.1 [S14] | 안식각·배출에 대한 축소 근거다. 과제문에 "Powder Technol." 로 적힌 것은 오기(게재지 확인) | 상 |
| (a) DEM 축소 근거 | 문헌 영률을 **1/100** 로 낮춰도 안식각·부피밀도 영향 없음. 사용값 YM 5.06×10⁸ Pa | 유리구 5 mm, LIGGGHTS | Rackl & Hanley, Powder Technol. 307:73–83 (DOI 2016, 권 2017), §2.3.4·§3.4 [S15] (워커 원문 대조) | 같음 | 상 |
| (a) DEM 축소 주의 | 사례 3개 중 2개는 영향 없음, 1개는 영향 있음(초록). Yan 2015 의 인용에 따르면 "10⁷–10¹¹ Pa 에서 안식각에 거의 영향 없음, **압축·관입 시험에서는 차이**" | — | Lommen, Schott, Lodewijks 2014, Particuology 12(1):107–112 [S16] (**초록만 열람**, 수치는 Yan 의 2차 인용) | **우리 그랩은 층에 파고든다(관입).** 축소 강성이 파고드는 힘·문 닫힘 저항에 영향을 줄 수 있다는 경고로 읽힌다 | 하 |
| (a) 리뷰 | 강성 축소는 흔한 관행. 안정 시간간격 ∝ 1/√강성 | — | Coetzee 2017 Powder Technol. 310:104–142 §7 [S17] · Paulick 등 2015 Powder Technol. 283:66–76 [S18] (둘 다 **초록·미리보기만**) | 본문 수치는 미확인 | 하 |
| (a) 보정 논문에서 실제로 쓴 강성 | PP: **E 2.84 MPa · G 1 MPa** (Mesnier 2020) / E 1.3 GPa 그대로 (Sudsawat 2023) / E 1.325 GPa 를 ×1e-3 (Pourandi 2025·2026) / ABS: ×0.001 (Hlosta 2020) / LDPE 분말: G 1.0×10⁷ Pa 고정 (García-Montagut 2025) | — | [S20][S21][S23][S24][S25][S26] | 현재 E 5 MPa 는 Mesnier(2.84 MPa)·Pourandi(≈1.3 MPa)·Hlosta(≈2.25 MPa) 와 같은 자릿수다. 단 그 논문들의 마찰·구름 값은 그 강성과 **한 묶음**으로 보정된 것이다 | 상(표 확인) |
| (c) 표준 | ISO 178 / ASTM D790 (굴곡), ISO 527 / ASTM D638 (인장) | — | 데이터시트 표기 | DEM 접촉 강성용 표준은 없다 | — |

### 항목 3. 마찰계수 (알–알, 알–벽, 구름)

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (b) 데이터시트 | PP–강철 정지 **0.30** / 운동 0.28 · 플라스틱–플라스틱 정지 **0.76** / 운동 0.44 | 조건 미기재("문헌 수집값") | INEOS 2014 p.1 [S01] (워커 원문 대조) | PP–PP 의 상한 쪽 값. 조건을 모른다 | 중 |
| (a) 보정 | PP–PP μs **0.52** (보정), PP–강철 μs 0.26 (고정), 운동 PP–PP 0.05 / PP–강철 0.246. 문헌 모음 표: PP–PP 0.1·0.153·0.22·0.3·0.6 | EDEM, 강철 벽, 고정 깔때기 안식각 | Sudsawat 2023 Table 1, p.44 [S20] | 크기(3 mm)는 비슷하나 입자 모양이 다르다(논문 표기 "spherical", 3.00×3.01 mm) | 중 |
| (a) 보정 | PP–PP **0.6** · PP–강철 **0.3** · PP–유리 0.2 · μr **0.01** | PP 구슬 2–3 mm(논문 안에서 불일치), EDEM, E 2.84 MPa | Mesnier 등 2020, Processes 8(9):1166, Table 2–3 [S21] | 구슬이라 모양이 다르고, 강성 축소와 한 묶음이다 | 상 |
| (a) 모델 입력 | PP–PP (내부) **0.432** · PP–외부 **0.304** | 버진 PP 펠릿(DuPure G 72 TF). 측정법·벽 재질 본문 미기재(각주: 저자 박사논문에서 결정) | Thieleke & Bonten 2021, Polymers 13(10):1540, Table 2 [S22] | 실제 펠릿 값이나 조건을 모른다 | 중 |
| (a) 보정 | μs **0.8** · μr 0.3–0.4 (벽 = 알–알과 같은 값) | PP **분말** D50 0.74–0.90 mm, 구, 업스케일 3–5배, MercuryDPM | Pourandi 등 arXiv:2512.08685 Table 4 [S23] · arXiv:2604.07082 p.6–7 [S24] | 분말이라 펠릿과 다르다. 상한 참고용 | 중 |
| (a) 측정(압출기) | PP 펠릿–스크루강 **0.25–0.6** | 압력 8–200 bar, 0.1–1.2 m/s, Ra 0.06 µm | Liu, Zitzenbacher 등 2014, AIP Conf. Proc. 1593:101–105 [S27] (초록 + 발췌) | 압력이 우리보다 수백 배 높다. 펠릿 모양에 따라 마찰이 달라진다는 근거로만 쓴다 | 하 |
| (a) 측정 | PP 펠릿 한 층이 미끄러지기 시작하는 각: 아연도금 강판 18° / 아크릴 23° (tan ≈ 0.32 / 0.42, 환산) | 자중 하중, 미끄럼과 구름이 섞임 | Tsai 1991, Texas Tech MS thesis, Table 3.1 [S28] | 순수 μs 가 아니다. **경사판법으로 벽 마찰을 재는 절차 예시**로 쓴다 | 중 |
| (a) PLA (그랩 재질) | PLA–강철 0.08–0.17 (두 논문) **vs** 0.30–0.35 (한 논문) · PLA–PLA 0.29–0.36 · PLA–강철 정지 ≈0.30 (마찰력으로 환산) | FDM 시편, 하중·속도 각기 다름 | Brăileanu 2023 [S29a] · Crăciun 2024 [S29b] · SERBIATRIB'25 [S29c] (이상 발췌) · Stoimenov 2024, Tribology in Industry 46(1) [S29d] (원문, 환산값) | **PP–PLA 는 없다.** PLA 쪽 값조차 출처끼리 2배 이상 차이 난다 | 하 |
| (a) 종이 | 골판지–골판지 정지 0.38 / 운동 0.35 · 종이 마찰은 받침·반복 미끄럼에 민감 | 클램프 트럭 모사 / 수평 썰매법 | earticle A370111 (초록) [S29e] · Johansson 등 1991 USDA FPL [S29f] | **PP–종이는 없다.** 종이 표면의 규모 감각과 측정 절차 참고용 | 하 |
| (a) 구름마찰 | 직접 측정한 PP 값은 **없다**. 역보정값만 있다: PP 구슬 0.01 (Mesnier) · PP 분말 0.3–0.4 (Pourandi) · LDPE 분말 0.149 (García-Montagut) · 목재 펠릿 0.02 (μr 을 0→0.05 로 바꾸면 안식각 20.1°→29.9°, Madrid 2022) · ABS–강철 직접 측정 0.03±0.01 (Hlosta 2020 Table 10) | 코드마다 구름저항 모델 정의가 다르다 | [S21][S23][S25][S26][S51] | μr 은 알 모양 표현(구/다중구)과 코드에 강하게 묶여 있어 **옮겨 쓸 수 없다** | 상(표) / 하(이식) |
| (a) 다른 플라스틱 참고 | ABS–강철 0.49 · ABS–PMMA 0.60 · ABS–ABS 0.35 (기울임 팔, 10회) / PE 펠릿: PE–PE 0.287 · PE–연강 0.208 / LLDPE 펠릿 Jenike 벽마찰각 연강 12.3° (tan 0.22) · 아크릴 18° | — | Hlosta 2020 Table 8–9 [S25] · Hastie 2013 [S29g] · Grima & Wypych 2010 [S29h] (발췌) | 플라스틱–플라스틱이 플라스틱–강철보다 크다는 경향 참고 | 중·하 |
| (c) 표준 | 알–벽: ASTM D6128 (Jenike 전단셀), ASTM D6773 (Schulze 링 전단) · 필름: ASTM D1894 (**2023 폐지**), ISO 8295 · 종이: TAPPI T815 (경사판 미끄럼각), ISO 15359 · 보고 양식: ASTM G115 | — | [S45][S46][S47][S48][S49][S49b][S49c] | **PP 알–종이, PP 알–PLA 전용 표준은 없다.** TAPPI T815 방식에 펠릿 썰매를 얹는 변형이 가능하다(도우미 C 제안, 표준 자체는 종이–종이) | 상(scope) |

### 항목 4. 반발계수

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (a) 측정 | PP 구 **0.83** (표). 본문에는 "PP·아크릴·POM 약 0.9" | Ø15 mm PP 구(0.89 g/cm³), 200 mm 에서 304 스테인리스 판으로 낙하, 충돌음 간격법 | Chai 등 2023, J. Phys. Conf. Ser. 2557 012057, Table 2 [S50] | 재료값. 크기(15 mm)·모양이 우리와 다르다. 논문 안에서 표와 본문이 어긋난다 | 중 |
| (a) 측정 | PP 비드–아크릴: 충돌 속도가 오르면 감소하다 **≈0.85 로 수렴**. 3–6 mm 에서 지름 영향 없음. 온도가 오르면 감소 | 낙하 최대 1.5 m (5.4 m/s), 4000 fps | Yurata 2019 박사논문, Chulalongkorn Univ. [S52] (학술지판 Yurata 등 2021 APT 32(4) 는 유료라 미열람) | 크기는 가깝지만 구다. PP–강철 상온 값은 그림에만 있어 판독하지 못했다 | 중 |
| (a) 보정 | PP–PP **0.55** (보정). 문헌 모음: PP–PP 0.3·0.49·0.9·0.97, PP–강철 0.55·0.71 | — | Sudsawat 2023 [S20] | 보정값은 마찰과 얽혀 있다. 안식각 기여도 분석에서 μs 65.7 %, e 21.8 % | 중 |
| (a) 보정 | e **0.3** (PP 구슬, 모든 쌍) | 드럼 분리지수로 맞춤 | Mesnier 2020 Table 2 [S21] | 강성 축소 묶음 | 상 |
| (a) 모델 입력 | 알–알 **0.81** / 알–외부 **0.85** | PP 펠릿, 조건 미기재 | Thieleke 2021 Table 2 [S22] | 조건을 모른다 | 중 |
| (a) 형상 효과 | LLDPE **원판형** 펠릿: 크기 기준 e 강철 0.611 / PC 0.671, **법선 e 강철 0.350 / PC 0.294**, 접선 0.783 / 0.870 | 0.0453 g, 45° 경사판, 70–93 cm/s | Borsa 등 2019, Lat. Am. Appl. Res. 49(2):143–147, Table 2–3 [S53] (초록은 원문, 표는 전문 레코드) | PE 다. 쌀알·원판 모양이면 법선 e 가 크게 낮아질 수 있다는 근거. 표의 "발사 높이 2/3 cm" 가 속도와 맞지 않아 해석이 미해결이다 | 하 |
| (a) 다른 플라스틱 | ABS–강철 0.72 · ABS–PMMA 0.62 · ABS–ABS 0.87 / LLDPE 펠릿 0.65–0.70 / PE 펠릿 0.62–0.72 | 낙하 / 이중 진자 / 고속카메라 | Hlosta 2020 Table 11–12 [S25] · Grima 2010 [S29h] · Hastie 2013 [S29g] | 경향 참고 | 중·하 |
| (b) 데이터시트 | **없음** | — | INEOS 등 어느 데이터시트에도 항목 없음 | — | — |
| (c) 표준 | 입자 반발계수 ISO·ASTM 표준 **없음**. 실무 관행은 낙하 + 고속카메라, 또는 충돌음법 | — | 도우미 C 검색 결과 | 자체 절차를 정해야 한다 | — |

### 항목 5. 안식각과 시험법

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (a) PP 측정 | **30.18° ± 0.51°** (29.25–30.86) | 고정 깔때기 → 컵, 영상 회귀. 3.00×3.01 mm PP 입자 | Sudsawat 2023 Table 6 [S20] | 크기가 가장 가까운 PP 값. 깔때기 높이·받침 재질은 확인하지 못했다 | 중 |
| (a) PP 측정 | 채움(부어 쌓음) **18°** / 배출(빠진 뒤 남은 면) **21°** | 부피밀도 0.5344 g/cm³ | Tsai 1991 Table 3.1–3.2 [S28] | Sudsawat 과 약 10° 차이. 방법·펠릿이 다르다 | 중 |
| (b) 판매사 문서 | **37°** | 방법·조건 미기재. "계산에 반영 안 함" 이라며 언급만 함 | INEOS "Polypropylene Silo Capacity" 2014 [S19] (워커 원문 대조) | 조건을 모른다 | 하 |
| (a) 보정 논문 실측(다른 재질) | ABS 6 mm: 받침 90 mm 27.4° / 150 mm 37.4° / 드럼 30.3° · LDPE 분말: 원통 들기 35.6°, 배수 38.9° | — | Hlosta 2020 Table 15 [S25] · García-Montagut 2025 Table 7 [S26] | **받침 크기만 바꿔도 10° 차이**(같은 ABS) | 상 |
| (a) 방법 민감도 | 낙하 높이 60–80 mm 이하에서 ≈**0.13°/mm** 감소, 120 mm 이상에서 수렴 · 알을 깐 받침이 강판보다 **≈10° 큼** · ISO 4324 출구 10 mm 는 2–3 mm 초과 알에서 막힘 → 출구 ≥ 6×입경 권장 | 목재 펠릿, 스테인리스 받침 | Madrid 등 2022, Agronomy 12(2):424 [S51] | 우리 측정 설계에 바로 쓰인다(높이·받침 기록 필수) | 상 |
| (a) 방법 민감도 | 동적(회전 드럼) 안식각은 정적보다 보통 **3–10° 작다** | 리뷰 인용 | Müller, Fimbinger, Brand 2021, Powder Technol. 383:598–605 [S54] | 정적과 동적을 구분해야 한다 | 중 |
| (a) 방법 민감도 | 정적(원통 들기)으로 보정한 밀 μs 0.40 · μr 0.092 vs 동적(드럼)으로 보정한 0.20 · 0.048 → **35–100 % 차이**. 둘을 동시에 맞추는 조합은 없었다 | — | Liu & Chen 2017 (초록) [S55] · Rousé 2013 또는 2014 (원통 들기가 35 % 작음, 초록) [S56] | **안식각 하나만으로는 μs 와 μr(과 e)가 분리되지 않는다** | 하 |
| (c) 표준 | ISO 4324:1977 (현행, 고정 높이 깔때기 → 평판 원뿔) · USP ⟨1174⟩ (고정 받침 + 턱에 알 층 유지, 깔때기를 더미 꼭대기 2–4 cm 위로 유지, "안식각은 고유 물성이 아니며 방법에 크게 의존") · ASTM D6393 (**입자 ≤ 2 mm** → 우리 알은 범위 밖) · ASTM C1444 (**2005 폐지**) | — | [S57][S58][S59][S60] | 원리는 USP ⟨1174⟩ 가 가장 잘 맞는다. 깔때기 출구는 표준 치수를 그대로 쓰면 막힐 수 있다 | 상(scope) |

### 항목 6. 부피밀도와 시험법

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (b) 편람 | 펠릿 **513–577** / 플레이크 465–497 | kg/m³ (원문 32–36 lb/ft³), 방법 미기재 | INEOS 2014 p.1 [S01] (워커 원문 대조) | 범위 참고 | 중 |
| (b) 판매사 문서 | 평균 **32–38 lb/ft³** (≈513–609 kg/m³, 환산). 보수값 35, 장기 저장 압밀 시 37. 오차 요인으로 "pellet count" 를 명시 | 방법 미기재 | INEOS Silo Capacity 2014 [S19] (워커 원문 대조) | 같은 회사 두 문서의 상한이 36 과 38 로 다르다. 둘 다 기록한다 | 중 |
| (b) 데이터시트 | **0.525** | g/cm³, 방법란에 "ISO 1183" (고체 밀도 규격이라 오기로 의심) | TotalEnergies PPH 9020 [S05] · PPC 2660 [S06] | 유럽 양식에만 있다 | 중 |
| (a) 논문 | 렌즈형 PP 최대(높은 낙하) **0.531** · 원통형 0.491 · 낙하 5.5 mm 에서 **0.419** / 0.389 | g/cm³. Moplen HP400H, 컵 Ø50 mm, 낙하 높이 2–20 mm + 50 mm (DIN EN ISO 60), 3회 평균, Grünschloß 식 맞춤 | Johann, Mehlich, Laichinger, Bonten 2022, Polymers 14(5):898, **Table 3, 식 (4)** [S61] (워커가 원문 XML 로 대조) | **같은 알이라도 붓는 방식에 따라 약 20 % 차이.** 4 cm 층을 어떻게 까느냐에 따라 시뮬 초기 더미 밀도 기준이 달라진다 | 상 |
| (a) 논문 | 0.5344 | g/cm³, PP 펠릿 | Tsai 1991 [S28] | 참고 | 중 |
| (a) 모델 입력 | 540 (단위 행 깨짐), 평균 펠릿 지름 4.55 mm | PP Borealis RD204CF, 해석 모델 입력 | Brüning & Schöppner 2022, Polymers 14(2):256, Table A4 [S62] | 값 출처 미기재 | 하 |
| (b) 소매 제품(환산) | Fairfield 5.4 oz/cup → ≈**0.65** · Victory 5 oz/cup → ≈0.60 · PEI 1 lb/3 cup → ≈0.64 | g/mL, US cup 236.6 mL 가정, 다짐 상태 불명 | 판매 페이지 [S70–S72][S72b] (도우미 D 환산) | 인형용 poly pellet 은 수지 펠릿보다 높게 나온다. 컵 계량이라 근거로서 약하다 | 하 |
| (c) 표준 | ASTM D1895: A (≡ ISO R 60, 잘 흐르는 과립 → 100 cm³ 컵) / **B (굵은 과립·펠릿 → 400 cm³ 컵)** / C (≡ ISO R 61, 플레이크·섬유, 압축) · ISO 60:2023 (지정 깔때기로 부을 수 있는 재료) · ISO 61:2023 (부을 수 없는 재료) | — | ASTM D1895-17 scope·Note 1 [S63] · Method B 컵 부피는 장비업체 설명 [S64] · ISO 60/61 [S65][S66] | "펠릿 = Method B" 는 ASTM 원문이 아니라 장비업체 설명 기준이다. 깔때기 치수는 원문(유료)을 보지 못했다 | 중 |

### 항목 7. 알 1개 질량·치수 분포 (판매 페이지·제조사 규격)

| 갈래 | 값·범위 | 단위·조건 | 출처 | 적용성 | 신뢰도 |
|---|---|---|---|---|---|
| (b) **국내 — 우리 제품** | 상세 이미지: "사이즈: (긴쪽) **3.8±0.3 mm**", "쌀알 모양", 반투명 백색. NOTICE: "±0.1~0.5 mm 오차". 1 kg / 9,000원. 재질 표기 "플라스틱"(상품명에만 PP). 중국 OEM, 제조·수입 브랜드얀. **개/g·1알 질량·밀도·단축 치수 없음** | 2026-09-30 14:03 열람 | 쿠팡 109307604 [S73] (Playwright 로 본문과 상세 이미지 2장 확인) | 긴 축만 있다. 단축·질량은 **측정해야 한다** | 상(표기 존재) |
| (b) 국내 | "사이즈: 3~5 mm 전후", "플라스틱(폴리프로필렌)", 100 g (±5 %) | 2026-09-30 | 스마일러브 [S74] · 후미 [S75] (**문구·이미지가 스마일러브와 같아 독립 출처가 아님**) | 제품마다 크기가 다르다는 근거 | 중 |
| (b) 국내 | "지름: 3 mm", 100 g / 1 kg | 2026-09-30 | 청송뜨개실 [S76] | 같음 | 중 |
| (b) 국내 | "지름: 약 3 mm", 약 100 g, 원산지 한국 | robots.txt 차단. 크롤 발췌로만 확인 | 썬퀼트 [S77] | 같음 | 하 |
| (b) 국내 | 상품명 "pp볼(펄렛)-3mm" | 상세 이미지 미판독 | 프랜즈얀 [S78] | 같음 | 하 |
| (b) 해외 | Fairfield Poly-Pellets: "about 1/8 (4 mm) in diameter", "oval contour", Polypropylene / 소매점: "approx. 3 mm" | 1/8 in = 3.175 mm (환산). 같은 제품인데 표기가 다르다 | [S70][S71] | 같음 | 중 |
| (a) 논문 | 렌즈형 버진 PP: 두께 **2.28±0.01**, 지름 **3.70±0.07 × 4.35±0.04 mm**, 등가구 지름 3.8 mm, **29.2 mg/알** · 원통형(실험실 절단) PP: 2.88×2.86×4.43 mm, 27.8 mg | 100알 | Johann 2022 **Table 1** [S61] (워커 원문 대조) | 크기가 비슷한 PP 펠릿 질량의 유일한 문헌값. **모양이 다르다**(렌즈형 vs 쌀알형) | 상(값) / 중(적용) |
| (a) 특허 | "about 35 to about 60 pellets per gram" (→ 16.7–28.6 mg/알, 환산) | 회전성형 수지 일반, PP 한정 아님 | US 6,833,410 B2 [S79] | 근거가 약하다 | 하 |
| (b) 제조 규범 | 폴리올레핀 수중 절단 펠릿 크기 2.5–5 mm("mostly spherical") · 스트랜드 절단 표준 길이 3 mm·지름 3 mm(원통) | 장비 브로셔 | MAAG/Automatik ZHULI [S80] · MAAG PEARLO [S81] · ips SGU [S82] (크롤 본문) | 쌀알형이 어느 공정에서 나왔는지는 판단할 수 없다 | 하 |
| (b) 수지사 데이터시트 | 펠릿 크기·형상·개/g **없음** | 한국 7사 15건 + 해외 7건 | [S03–S08][S30–S34][S36][S37] | — | 상(부재 확인) |
| (c) 표준 | ASTM D1921 = 체거름 입도(Method B 가 "pellets and cubes" 용). **개/g 규정이 아니다.** 개/g 를 정한 ASTM·ISO 규정은 찾지 못함 | — | ASTM D1921-18 scope [S83] | 참고 | 상(scope) |
| 참고(내부) | 기존 실측 프로토콜: 100알 × 5회 묶음 계량 권고(0.01 g 저울로 1알을 재면 ±17–50 %), 알 1개 ≈0.06 g "미확정 읽음" | — | `~/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/MEASUREMENT_PROTOCOL_20260909.md` §0·§M1 | 문헌값(29.2 mg)과 약 2배 차이다. 모양·크기 차이인지 읽음 오류인지 **측정으로 가려야 한다** | — |

### 항목 8. 플라스틱 펠릿 DEM 보정 논문 → §③ 에 따로 표로 정리

### 항목 9. 고체 밀도 측정법 비교 → §④ 에 따로 표로 정리

### 항목 10. 국내 PP 원료사 데이터시트에 펠릿 규격·부피밀도가 있는가

| 회사 | 열람한 문서 | 밀도 (방법) | 펠릿 크기·형상·개/g·부피밀도 | 출처 | 신뢰도 |
|---|---|---|---|---|---|
| 대한유화 KPIC | YUHWA POLYPRO 4018 (날짜 없음) | 0.90 (ASTM D1505) | **없음** | [S30] | 상 |
| 한화토탈에너지스 | HY301 · HJ730 · HI828 · RJ970Z · CI571 (PDF 생성 2022–2026) | 0.9–0.91 (ASTM D1505) | **없음** | [S31] | 상 |
| LG화학 | H1500 (2022-01-27) · M1500 (2021-12-07) | 0.9 (ASTM D1505, density-gradient) | **없음** | [S32] | 상 |
| 효성화학 | Topilene J801 · R601 · 한국공장 물성 일람표 (2026-03-18) | 0.9 (ASTM D792) | **없음** (J801 에 보관·예비건조 조건만 있음) | [S33][S36] | 상 |
| 폴리미래 | Adstif HA5034 · EA648P · HA5029 | 0.9 (ASTM D1505) / HA5029 는 밀도 행 없음 | **없음** | [S34] | 상 |
| 롯데케미칼 | RANPELEN J-550N (2015-06, **베트남 유통사 게시본**, 문서에 "보안문서" 표기) | 0.9 (ASTM D792 / ISO 1183) | **없음** | [S37] | 중 |
| 롯데케미칼 공식 사이트 | — | — | **확인 불가** (robots.txt `Disallow: /`, 우회하지 않음) | — | — |
| SK지오센트릭 | — | — | **확인 불가** (robots.txt 차단, SpecialChem 등은 로그인 필요) | — | — |
| HD현대케미칼 | — | — | **확인 불가** (공개 PP 데이터시트를 찾지 못함) | — | — |
| 해외 비교 | Borealis BE52 · TotalEnergies 3281·1251·PPH 9020·PPC 2660 · Moplen HP456J · Mosten MT230 · SABIC STAMAX | — | TotalEnergies **유럽 양식만 부피밀도 0.525 g/cm³** 기재. 나머지는 없음 | [S03–S08][S11][S13] | 상 |

사실만 정리하면 다음과 같다.
- 열람한 한국 7사 공개 문서 15건 가운데 펠릿 크기·형상·개/g·부피밀도를 적은 문서는 **0건** 이다.
- 한국사 굴곡탄성률의 표준 양식은 kgf/cm² 단위의 ASTM D790 이다.

---

## ③ 플라스틱 펠릿 DEM 보정 논문 표

원문 표를 직접 확인한 7편만 넣었다. 열지 못한 논문은 표에 넣지 않고 §⑦ 에 적었다.
**공통 주의**
- 우리 조합(쌀알형 PP 펠릿 + 종이 벽 + PLA 벽)을 보정한 논문은 **0편** 이다. 벽 재질은 모두 강철·유리·PMMA·폴리카보네이트·아크릴이다.
- 대부분 강성을 낮춘 상태에서 마찰·구름을 맞췄고, 알 모양 효과를 μr 에 흡수시켰다.
- 따라서 **값을 하나씩 떼어 DEME 에 넣으면 안 된다.**
- DEME 로 보정한 플라스틱 세트는 0건 이다. 모두 EDEM·LIGGGHTS·MercuryDPM 이다.

| # | 논문 | 재질·펠릿 규격·표현 | 보정 실험 | 최종 파라미터 (표/쪽) | 실측 부피밀도·안식각 |
|---|---|---|---|---|---|
| P1 | Sudsawat, Chongchitpaisan, Arunyanart 2023, "Calibrating polypropylene particle model parameters with upscaling and repose surface method", *EUREKA: Physics and Engineering* (6):34–46, doi:10.21303/2461-4262.2023.002968 [S20] | **PP**, 3.00×3.01 mm(논문 표기 "spherical", 장구형 근사), 2배 업스케일(6 mm) | 고정 깔때기 안식각 + 영상 → Plackett–Burman → 반응표면법. EDEM Hertz–Mindlin, 강철 벽 | μs PP–PP **0.52**, e PP–PP **0.55** (p.44). 고정값: E **1.3 GPa (축소 안 함)**, ν 0.36, μs PP–강철 0.26, 운동 PP–PP 0.05 / PP–강철 0.246 (Table 1·2) | 입자밀도 910 kg/m³, 안식각 **30.18±0.51°**. DEM 29.14° (3 mm) / 29.67° (6 mm) (Table 6) |
| P2 | García-Montagut, Paz, Monzón 2025, "Development of a Non-Spherical Polymeric Particles Calibration Procedure…", *Polymers* 17(20):2748, doi:10.3390/polym17202748 [S26] | **LDPE** 3 mm 펠릿을 갈아 만든 **분말**(D[4,3] 579 µm). 구 2개 겹침 | 부피밀도, 원통 들기 안식각, 선반 상자(ledge box), 배수(draw-down). 유전 알고리즘 + Kriging 85회. EDEM + 히스테리시스·응집, **아크릴** 벽 | Table 11: ρ 664(가상값, 실제 918), ν 0.20, e pp 0.662, μs pp 0.526, μr pp 0.149, e pw 0.463, μs pw **0.950 (탐색 상한에 붙음)**, μr pw 0.107. G **1.0×10⁷ Pa 고정** (Table 4) | 부피 366 kg/m³ · 탭 428 (Table 6). 원통 들기 35.6°, 배수 38.9° (Table 7) |
| P3 | Thieleke & Bonten 2021, "Enhanced Processing of Regrind as Recycling Material in Single-Screw Extruders", *Polymers* 13(10):1540, doi:10.3390/polym13101540 [S22] | **PP** 버진 펠릿(Ducor DuPure G 72 TF) + HDPE + 분쇄품. 구 / 초이차곡면(superquadric) | 부피밀도(DIN EN ISO 60, 컵 51 mm) 시뮬–실험 비교, 압축 시험으로 새 접촉 모델 조정. LIGGGHTS 3.8.0 | Table 2 (PP): ρ 0.91, E **1850 MPa**(데이터시트), ν 0.399, e 알–알 **0.81** / 알–외부 **0.85**, 마찰 알–알 **0.432** / 알–외부 **0.304**. 마찰·반발은 저자 박사논문에서 가져옴. 벽 재질 미기재(압출기라 강철로 보이나 **미확인**). μr 미기재 | 버진 펠릿은 구 모델과 실험 부피밀도 차이가 <1 % (그림으로만 제시) |
| P4 | Hlosta, Jezerská, Rozbroj, Žurovec, Nečas, Zegzulka 2020, "DEM Investigation of the Influence of Particulate Properties and Operating Conditions on the Mixing Process in Rotary Drums: Part 1", *Processes* 8(2):222, doi:10.3390/pr8020222 [S25] | **ABS** 구슬 6 mm (+ 나무·강철) | 직접 측정: 기울임 팔(정지·구름마찰), 낙하(e 알–벽), 이중 진자(e 알–알). 보정: 충전 높이, 쌓기 안식각(받침 90/150 mm), 회전 드럼, 호퍼. EDEM, 모듈러스 ×0.001 | 측정값: μs ABS–강철 0.49±0.08 · –PMMA 0.60 · –Al 0.22 · –유리 0.43 (Table 8), ABS–ABS 0.35 (Table 9), μr ABS–강철 0.03 (Table 10), e ABS–강철 0.72 · –PMMA 0.62 · –Al 0.67 · –유리 0.71 (Table 11), ABS–ABS 0.87 (Table 12). DEM 입자밀도 1768 kg/m³ (질량 맞춤, Table 3) | 충전 높이 실험 112 vs DEM 115 mm (Table 7). 안식각 실험/DEM: 90 mm 받침 27.4/26.5°, 150 mm 받침 37.4/34.0°, 드럼 30.3/32.8° (Table 15) |
| P5 | Pourandi, van der Sande, Ostanin, Weinhart 2025, "Calibration of a DEM contact model for wet industrial granular materials", arXiv:2512.08685 [S23] | **PP 반응기 분말** (D50 737 / 905 µm). 구, 업스케일 3·5배 | 회전 드럼 동적 안식각(Ø14 cm, 폴리카보네이트 옆벽, 5 rpm). MercuryDPM, 강성 ×1e-3, Δt = Rayleigh 시간의 20 % | Table 2: E 1325 MPa, G 400 MPa, e 0.5 (문헌). Table 4 (건조): μs **0.8**, μr 0.3–0.4. 벽 마찰 미기재 | 폭기 부피밀도 393 / 368 kg/m³ |
| P5b | Pourandi, Ostanin, Weinhart 2026, "Granular mixing and flow dynamics in horizontal stirred bed reactors", arXiv:2604.07082 [S24] | 같은 PP 분말, 업스케일 3배 | 안식각으로 재보정 (p.6) | μs **0.8**, μr **0.4**, e 0.5. **벽 접촉 = 알–알과 같은 값** (p.7) | — |
| P6 | Mesnier, Peczalski, Mollon, Vessot-Crastes 2020, "Mixing of Bi-Dispersed Milli-Beads in a Rotary Drum…", *Processes* 8(9):1166, doi:10.3390/pr8091166 [S21] | **PP 구슬** (본문 3 mm, 표 1 은 2 mm — 논문 안에서 불일치) + 셀룰로오스 아세테이트 | 알–알 μs: 평판 위 정적 안식각에 맞춤. 알–벽 μs·μr·e: 드럼 분리지수에 맞춤. EDEM, Δt ≤ Rayleigh 의 40 % | Table 2: ρ 910, ν 0.42, **E 2.84 MPa, G 1 MPa**(속도용 축소), μr **0.01**, e **0.3**. Table 3: μs PP–강철 **0.3**, PP–유리 0.2, PP–PP **0.6** | PP 안식각은 그림(Fig 2)에만 있다 |

보정은 아니지만 참고로 연 문헌: Brüning & Schöppner 2022 (PP 스크루 마찰 0.112, 배럴 0.28, 내부 0.5, 해석 모델 입력) [S62].
직접 측정 논문(비보정)은 항목 3~7 표에 나눠 넣었다: Johann 2022 [S61] · Borsa 2019 [S53] · Chai 2023 [S50] · Yurata 2019 [S52] · Madrid 2022 [S51] · Tsai 1991 [S28] · Han 2011 KONA [S84] · Stoimenov 2024 [S29d].

---

## ④ 고체 밀도 측정법 비교표

| 방법 | 원리 | 정밀도 (출처 표기) | 필요한 장비 | 시료 | PP 처럼 물에 뜨는 시료 처리 | 표준 | 출처 |
|---|---|---|---|---|---|---|---|
| **① 아르키메데스 수중 칭량** (ASTM D792 Method A / ISO 1183-1 Method A) | 공기 중 질량과 물 속 겉보기 질량의 차이가 곧 부피. D792 식: sp gr = a / (a + w − b) | D792: 밀도 1 미만·10 g 미만 시료는 **0.1 mg 저울**(그 밖은 1 mg), 유효숫자 3자리, 온도계 0.1 °C. ISO 1183-1 A: ±0.1 mg | 분석저울 + 걸이(pan straddle), 와이어, 싱커, 비커, 탈기수 | D792: **한 덩어리 1–50 g, ≥1 cm³** → 알 1개(수십 mg)는 규격 밖. ISO 1183-1 A: 분말을 뺀 무공극 고체 | **싱커**(비중 ≥7.0)로 가라앉힘. 탈기수, 안 젖으면 습윤제 몇 방울. 그래도 안 되면 Method B(물 이외 액체) | ASTM D792-20 · ISO 1183-1:2019 | [S43] (scope 원문 + §8–12 공개 미리보기, 워커 원문 대조) · [S40] |
| (변형) 밀도 키트 + 에탄올 | 같은 원리. 보조 액체를 물보다 가벼운 에탄올로 바꿈 | Mettler: 와이어에 붙는 액체로 최대 3 mg 겉보기 증가, 온도 영향 0.1–1 ‰/°C, **지름 1 mm 기포 = 부력 0.5 mg**, "오차의 압도적 최대 원인은 시료의 젖음성 부족" | 저울 + 밀도 키트(뒤집을 수 있는 바스켓) | 알 여러 개를 한꺼번에 | 에탄올 0.7893 g/cm³ (20 °C) < PP ≈0.9 이므로 **싱커 없이 가라앉는다**(추론). 또는 바스켓을 뒤집어 위에서 눌러 가둠 / 윗 접시에 추 추가 | (ISO 1183-1 A 계열) | Mettler Toledo 설명서·응용 페이지 [S90] (워커 원문 대조) · PubChem 에탄올 [S91] |
| **② 액체 비중병** (ISO 1183-1 Method B) | 액체만 채운 병 질량과 시료+액체를 채운 병 질량의 차이로 시료 부피를 구함 | 표준 원문(유료)을 보지 못해 **미확인** | 비중병, 분석저울, 항온조, 탈기용 진공 | **"particles, powders, flakes, granules or small pieces"** — 펠릿에 맞는 방법. EN 판 NOTE "pellets 도 무공극이면 적용 가능" | 병 안에 갇히므로 떠도 원리상 문제없다(표준 원문은 미확인). 기포 제거가 핵심 | ISO 1183-1:2019 B · (Sudsawat 2023 은 ASTM D854 비중병으로 PP 910 측정) | [S40] · [S20] |
| (참고) 적정법 (ISO 1183-1 Method C) | 두 액체의 혼합비를 바꿔 시료가 뜨지도 가라앉지도 않는 점의 액체 밀도를 잰다 | 미확인 | 뷰렛, 액체 2종 | 무공극 시료 | 원리상 뜨는 시료에 적합 | ISO 1183-1:2019 C | [S40] |
| **③ 기체 비중병** (ISO 1183-3) | 보일 법칙. 기준실에서 시료실로 기체를 팽창시킬 때의 압력비로 시료의 골격 부피를 구함 | Anton Paar Ultrapyc: 셀 135/50 cm³ 정확도 0.02 %·반복성 0.01 % … 1.8 cm³ 셀 0.30 % … 0.25 cm³ 셀 1.00 %. Micromeritics AccuPyc II 1345: 재현성 ±0.01 % (보증 ±0.02 %, 셀 부피 기준) | 기체 비중병 + He(또는 N₂), 분석저울(질량은 따로 잰다) | 건조 시료. 셀을 채울 만큼 넣어야 정확도가 나온다 | 액체가 없어 **뜨는 문제 자체가 없다.** 단 He 가 스며드는 저밀도 폴리머는 N₂ 사용 권고(Quantachrome 매뉴얼) | ISO 1183-3:1999 (닫힌 기공이 없는 모든 형태). ※ **ASTM D6226 은 발포체 개방셀 시험이지 고체 밀도 표준이 아니다** | [S42] · Anton Paar [S92] · Micromeritics [S93] (둘 다 워커 원문 대조) · ASTM D6226 [S94] |
| **④ 밀도 구배관** (ASTM D1505 / ISO 1183-2) | 두 액체로 위→아래 밀도가 연속 증가하는 기둥을 만들고, 시료가 멈춘 높이를 보정 유리구와 비교 | D1505: **0.05 % 보다 좋게**. 감도 0.0001 g/cm³·mm | 유리 기둥, 항온조, 보정 유리 플로트, 액체 2종 | ISO 1183-2: "moulded or extruded plastics **or pellets**" → **알 1개씩 잴 수 있다**(알마다 분포를 볼 수 있다) | 액체쌍을 0.9 근처로 고르면 된다(액체표는 부록이라 미열람) | ASTM D1505 (≡ ISO 1183-2) · 한국 수지사 대부분이 이 방법으로 표기 | [S44][S41] |
| **(참고 영상) 전용 전자 비중계** (DH-300 계열) | ① 과 같은 아르키메데스식. 공기 중 → 물 속 2단계 칭량 후 밀도를 바로 표시 | DH-300: 분해능 **0.001 g/cm³**, 최대 300 g, 최소 0.005 g. 같은 회사 AU-120S/200S 는 0.0001 g/cm³ | 본체 + 수조 + 0.5 mm 스테인리스 걸이줄 + 방풍 커버 | 과립·필름. 제조사가 "floating" 을 명시 | 표준 부속에 "**floating body accessories**", "grain accessories". "solution compensation" = 물 이외 액체 사용 가능 | 제조사 주장: ASTM D792, ISO 1183, GB/T 1033 | [S95][S96][S97] |

**비용**: 가격을 직접 확인한 출처가 없다(**미확인**).
장비 구성만 보고 추정한 서열은 다음과 같다(추론). 주방·보석 저울 + 자작 싱커 < 분석저울 + 밀도 키트 < 전용 비중계 < 구배관(액체·플로트·항온) < 기체 비중병.

**우리 저울(Riwonas, 0.01 g)로 에탄올·물 칭량을 할 때 분해능만 따진 값** (도우미 D 계산, 기포·온도 오차 제외)
- 시료 10 g 이면 밀도 분해능 기여는 **≈ ±0.001 g/cm³** 수준이다.
- 다만 ASTM D792 가 요구하는 저울(0.1 mg)은 아니다. **규격을 따른 측정이 아니라는 점**을 기록해야 한다.

---

## ⑤ 참고 영상 정리 — "Solids Density Meter for Measuring Plastic Granules" (youtube yO-HlXIG898)

- **메타데이터** (yt-dlp `--skip-download`, 영상은 받지 않음)
  - 채널 Hongtuo Instrument, 업로드 **2017-08-03**, 길이 **42초**, 자막·자동자막 **없음**.
  - 설명문에는 회사 연락처만 있다(DongGuan HongTuo Instrument Co., Ltd). **모델명·사양·표준 언급 없음.** [S98]
- **화면에서 읽은 것** (썸네일 4장)
  - 자막 문구 순서: "Take glass cup on the testing board" → "Press ENTER / The screen is showing the specific gravity of the granules" → "Clean it by ethanol".
  - 검은 방풍 커버 본체, 투명 수조, 유리 컵(과립 담는 용기)이 보인다. 과립은 어두운 색이라 PP 로 보이지 않는다.
  - 본체의 모델 문자는 해상도가 낮아 **판독할 수 없다.**
- **원리**: 공기 중 칭량 → 수조 칭량 → ENTER 로 비중 표시. **아르키메데스식 전자 비중계**와 일치한다. 분류상 §④ 의 ① 에 속한다.
- **같은 회사 과립용 제품** (영상 속 기기와 같은지는 **미검증**)
  - **DH-300** "Plastic Granules Digital Density Meter": 분해능 0.001 g/cm³, 최대 300 g, 최소 0.005 g. 뜨는 시료·알갱이용 부속이 표준으로 들어 있다. ASTM D792·ISO 1183·GB/T 1033 준수를 주장한다. [S95][S96]
  - **AU-300S** (QUARRZ 브랜드): "Archimedes buoyancy method", "밀도가 1 보다 작아도 커도 측정", 약 5초. PP/PE/PVC 과립을 명시한다. [S97]
- **우리 경우와의 관계**
  - 이 기기도 ASTM D792 원리다. D792 Method A 는 1–50 g 한 덩어리 시료 기준이므로, 과립은 유리 컵이나 전용 부속에 담아 재는 방식이다.
  - 뜨는 PP 는 싱커형 부속이나 에탄올로 처리해야 한다.
  - 분해능(0.001 g/cm³)은 우리 0.01 g 저울 + 10 g 시료 수중칭량의 분해능 계산값과 같은 자릿수다. 실제 오차는 기포·온도가 좌우한다(Mettler).

---

## ⑥ 문헌만으로 정할 수 있는 항목 vs 이 봉지를 직접 재야 하는 항목 (근거 나열)

판정이 아니라 근거만 나열한다. 결정은 메인·사용자가 한다.

| 항목 | 문헌 쪽 근거 | 측정 쪽 근거 |
|---|---|---|
| 고체 밀도 | 미충전 PP 는 데이터시트 0.898–0.91 로 범위가 좁다(여러 제조사, 여러 규격) | 판매 페이지 재질이 "플라스틱" 뿐이고 버진·재생·충전 여부를 모른다. 충전 PP 는 1.03–1.12 로 범위 밖이다. 에탄올에 뜨는지 가라앉는지만 봐도 0.79 기준으로 일부를 가를 수 있다(추론) |
| 알 1개 질량·단축 치수·모양 | 가장 가까운 문헌값은 렌즈형 PP 29.2 mg (Johann 2022) 하나다. 판매 페이지는 긴 축 3.8 mm 만 적었다 | 어느 판매 페이지·데이터시트에도 개/g·질량·단축 치수가 없다. 내부 프로토콜의 "≈0.06 g 미확정 읽음" 은 문헌값의 약 2배다. 기존 프로토콜 M1(100알 × 5회)·M2(n=30, 3축)가 이미 설계돼 있다 |
| 부피밀도 | PP 펠릿 문헌값 0.513–0.609 (INEOS·TotalEnergies·Johann) | 같은 알도 붓는 높이에 따라 0.419–0.531 로 변한다(Johann). 우리 4 cm 층을 까는 방식 그대로 재야 시뮬 초기 상태와 비교할 수 있다. 인형용 poly pellet 소매 컵 환산값은 0.60–0.65 로 수지 펠릿보다 높다 |
| 영률·푸아송비 | 재료값: E 1.0–1.9 GPa, ν 0.36–0.42 (여러 출처). DEM 에서는 축소가 관행이고, 안식각·배출에는 영향이 작다는 근거가 있다(Yan, Rackl) | 알 하나의 강성을 재는 것은 의미가 작다(어차피 축소해서 쓴다). 다만 **관입·압축에는 축소의 영향이 있다**는 인용(Lommen via Yan)이 있다. 그랩 관입력은 축소 강성에서 달라질 수 있다 |
| PP–PP 마찰 | 0.432 / 0.52 / 0.6 / 0.76 / 0.8 로 흩어져 있다. 모두 모양·강성·μr 과 한 묶음이다 | 모양(쌀알형)이 다르고 출처 간 편차가 크다. 안식각만으로는 μs–μr–e 가 분리되지 않는다(Sudsawat 기여도, Liu & Chen) |
| PP–종이 벽 마찰 | **문헌 없음** | 측정 말고는 방법이 없다. TAPPI T815 형 경사판에 펠릿 썰매를 얹는 방법이 있다. 종이 마찰은 받침·반복 미끄럼에 민감하다(Johansson 1991) |
| PP–PLA 그랩 마찰 | **문헌 없음.** PLA–강철조차 0.08–0.17 과 0.30–0.35 로 어긋난다 | 같음. FDM 층 두께에 따라 달라진다는 보고가 있다(Crăciun 2024) |
| 반발계수 | PP 구 재료값 0.83–0.85 (Chai, Yurata). 3–6 mm 에서 크기 영향이 없다(Yurata) | 원판형 PE 펠릿은 법선 e 가 0.29–0.35 로 낮았다(Borsa). 쌀알형 PP 값은 없다. Yan 2015 초록은 반발계수와 영률이 배출 예측에 "insignificant" 하다고 했다(호퍼 조건) |
| 구름마찰 | 직접 측정 PP 값은 없다. 역보정값 0.01–0.4 는 코드·모양 표현마다 다르다 | 옮겨 쓸 근거가 없다. 안식각 등으로 역보정하는 수밖에 없다 |
| 안식각 | PP 18°·21°·30°·37° | 방법에 따라 수 °~10° 가 달라진다(Madrid, Hlosta, Müller). 우리 알·우리 받침(종이)·정해진 높이로 재야 보정 목표값이 된다. 표준 깔때기 출구 10 mm 는 막힐 수 있다 |

---

## ⑦ 검색어·소스 로그와 한계

### 7-1. 검색어 (도우미별, 항목당 3개 이상)
- **A (항목 1·2·6·10)**: WebSearch 18건, Brave 3건(1건은 429), exa 4건, Crossref·OpenAlex·Semantic Scholar DOI 조회 8건, 제조사 사이트 직접 탐색(kpic.co.kr, htpchem.com AJAX, polymirae downloadPdf, lgchemon, hyosungchemical, 각 사 robots.txt).
  - 대표 검색어: "polypropylene homopolymer technical data sheet density ISO 1183 flexural modulus ISO 178 pdf", "Lommen Schott Lodewijks DEM speedup stiffness…", "ASTM D1895 … method B pellets", "롯데케미칼 PP 폴리프로필렌 grade TDS 밀도 굴곡탄성률 pdf", "효성화학 TOPILENE PP grade 물성표" 등.
- **B (항목 8)**: 38건. WebSearch, Consensus, Exa, Europe PMC REST, arXiv MCP, HAL, OpenAlex.
  - 대표 검색어: "DEM calibration polypropylene pellets angle of repose", "HDPE pellets discrete element calibration parameters restitution friction", "塑料颗粒 离散元 参数标定 休止角 聚丙烯", "friction coefficient polypropylene against paperboard cardboard measured inclined plane", "PLA 3D printed surface friction coefficient against polypropylene".
- **C (항목 3·4·5)**: WebSearch 14건, exa 13건, Semantic Scholar 3건.
  - 대표 검색어: "coefficient of friction polypropylene steel static kinetic table", "polypropylene pellet coefficient of restitution drop test measurement impact velocity", "\"angle of repose\" \"polypropylene pellets\" degrees measured", "ISO 4324 … scope", "wall friction angle plastic pellets Jenike shear tester".
- **D (항목 7·9·영상)**: 25건. Brave, WebSearch, Exa, Playwright(쿠팡 검색 "인형용 pp 펠릿"), yt-dlp 메타데이터, PubChem API.
  - 대표 검색어: "인형용 PP 펠릿 3.8mm", "PP펠렛 인형 무게 3.8mm 쿠팡", "Hongtuo solids density meter plastic granules", "DahoMeter plastic granules density meter floating sample sinker DH-300", "\"pellets per gram\" polypropylene pellet size".
- 전체 검색어 원문은 도우미 노트의 "queries run / 검색 로그" 절에 있다(아래 7-4 의 scratchpad 경로). 이 폴더에는 복사하지 않았다.

### 7-2. 접근 실패·제약
- ScienceDirect(403·캡차), Wiley(403), ResearchGate(timeout·403), EUREKA 출판사(429 → exa 크롤 사본으로 확인), Twente 저장소(Cloudflare), AIP PDF(403).
- 쿠팡은 일반 요청을 403 으로 막아 Playwright 로 열었다. 롯데케미칼·SK지오센트릭·썬퀼트는 robots.txt 차단이라 **우회하지 않았다**.
- **미열람 핵심 문헌** (값 인용 금지, 필요하면 기관 접속으로 확인)
  - Landgraeber & Brüning 2025, *Adv. Powder Technol.* 36(8):104968 (여러 폴리머 펠릿 보정, 오픈 액세스지만 캡차) — **가장 우선**
  - Yurata 등 2021 *APT* 32(4) (PP 구슬 판 충돌 e)
  - Cho, Bhushan, Dyess 2016 *Tribol. Int.* (PP–PP 마찰)
  - Pourandi 2024 *Powder Technol.* 446:120176
  - Martignoni 2024 *Granular Matter*
  - Hastie 2013 *CES*
  - Moysey & Thompson 2007 *CES*
  - Chavez-Sagarnaga 2004 (PP 펠릿 사일로 벽마찰)
  - Lommen 2014, Coetzee 2017, Paulick 2015 본문
  - ISO 1183-1 B·ISO 1183-2·ISO 1183-3·ISO 4324·ASTM D1895 본문(유료)
- **도구 부작용**: Playwright MCP 가 worktree 루트에 `.playwright-mcp/` 를 자동 생성했다. 도우미가 scratchpad 로 옮겼고, 워커가 `git status` 가 깨끗하고 해당 폴더가 없음을 확인했다.

### 7-3. 워커 원문 대조 (도우미 값 재확인)
- INEOS Engineering Properties PDF: E 1,300/1,100 MPa, ν 0.42, 마찰 0.30/0.28 · 0.76/0.44, 펠릿 부피 513–577 kg/m³ — **일치**.
- INEOS Silo Capacity PDF: 32–38 lb/ft³, 보수 35·압밀 37, 안식각 37° "not considered", "Pellet count" 오차 요인 — **일치**.
- Rackl & Hanley PDF: "down by two orders of magnitude", YM 5.06×10⁸ Pa, "YM significantly influences neither the bulk density nor the angle of repose" — **일치**.
- Johann 2022 (Europe PMC XML): Table 1 렌즈형 PP 2.28 / 3.70 / 4.35 mm, 3.8, 29.2 mg · Table 3 ρ0 0.531, 5.5 mm 에서 0.419 · 식 (4) "ρ0 = maximum bulk density at high dumping height" — **일치**.
  - 도우미 노트의 "낙하 5.5 mm" 는 표 머리글 "Bulk Density at 5.5 mm Dumping Height" 와 같다.
- 측정법 수치 중 도우미가 검색 발췌만 봤던 것은 워커가 원문을 다시 받아 대조했다.
  - Mettler 설명서 RM PDF: 온도 영향 0.1–1 ‰/°C, 와이어 부착 최대 3 mg, 뜨는 시료는 윗 접시에 추 추가 — **일치**.
  - Mettler 응용 페이지(exa fetch): "A bubble with a 1 mm diameter causes a buoyancy of 0.5 mg", "By far, the biggest source of error … is the limited wettability", "use a different reference liquid with a lower density" — **일치**.
  - Anton Paar Ultrapyc 브로셔 PDF: 셀 135/50 cm³ 0.02 %/0.01 %, 10 cm³ 0.03/0.015, 4.5 cm³ 0.10/0.05, 1.8 cm³ 0.30/0.15, 0.25 cm³ 1.00/0.50 — **일치**.
  - Micromeritics AccuPyc II 1345 사양 PDF: 재현성 ±0.01 % (보증 ±0.02 %, 셀 부피 기준), 정확도 0.03 % of reading + 0.03 % of capacity — **일치**.
  - ASTM D792-20 공개 미리보기(exa fetch): §8.1 "one-piece specimen of 1 to 50 g … sinker", §9.1 "0.1 mg or better … densities less than 1.00 g/cm³ and sample weights less than 10 grams", §9.3 싱커 비중 ≥7.0, §11.1 부피 ≥1 cm³ — **일치**.
- 도우미 C 가 URL 을 다시 적으면서 스스로 정정한 것:
  - Rousé 의 연도는 2013/2014 가 불확실하다. sources.json S56 에 표시했다.
  - 노트에 있던 Anaraki 2008 의 소속, Liu Z. 논문의 URL 은 확인되지 않아 보고서에 쓰지 않았다.
- 나머지 값은 도우미의 원문 열람 기록(opened 표시)에 의존한다. 신뢰도 열에 반영했다.

### 7-4. 원자료 위치 (repo 밖, 세션 scratchpad)
`/tmp/claude-1000/-home-cgxr-orca-workspaces-RoArm-Project-w26-pellet-priors/76284312-cd41-4ca2-81ed-0fb914c62480/scratchpad/`
- `notes_A.md` ~ `notes_D.md`: 도우미 원 노트. 검색어 전체와 원문별 메모가 들어 있다.
- `pdf/`: 받은 데이터시트·논문 PDF.
- 이 파일들은 세션이 끝나면 사라질 수 있다. 보존이 필요하면 메인이 결정한다.
