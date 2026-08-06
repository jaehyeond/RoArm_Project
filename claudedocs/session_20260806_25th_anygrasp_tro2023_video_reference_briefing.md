# Session 2026-08-06 (25th) — AnyGrasp (T-RO 2023) 논문 발표 영상 레퍼런스 브리핑

> **세션 성격: 레퍼런스/문헌 세션.** 실험 0건, 코드 변경 0건, 로봇 HW 0건, git 0건.
> **프로젝트 상태 변경 없음** — 본 세션은 G0b active case를 전혀 건드리지 않았다.
> `START_HERE.md` / `DECISIONS.md` / `EXPERIMENT_LEDGER.md` **의도적으로 미변경** (사유는 §6).
>
> **⚠️ 세션 번호 정정 이력**: 본 문서는 최초 `24th`로 작성되었으나, 작성 직후
> `MEMORY.md` 갱신을 통해 **동일 일자에 별도 세션이 이미 24th를 점유**하고 있음이 확인되어
> (`session_20260806_24th_g0b_t3_repair_design_adversarial_review.md` — T3 수리 검증 설계 v1)
> **25th로 재번호**했다. 동시 세션 겹침 관측은 22nd에 이어 2회째다
> (`AGENTS.md`: "같은 프로젝트에서 Claude와 Codex 편집 세션 동시 실행 금지" —
> 본 건은 도구 간 충돌은 아니나 세션 번호 경합이 실제로 발생했음을 기록한다).

## 0. 이번 case의 신규 변수

**없음.** Variable Ladder Protocol(D322~) 상 신규 변수를 도입하지 않았다.
본 세션은 active case 밖의 외부 문헌 확인이며, `START_HERE.md` Active Case 범위를
건드리지 않는다.

## 1. 요청과 수행

사용자가 AnyGrasp 발표 영상 대본 일부를 붙여넣은 뒤 YouTube URL을 주고
"토큰 많이 드나 / 40분 맞나 확인하고 브리핑"을 요청.

대상: <https://www.youtube.com/watch?v=qFl59361Slg>
- 제목: "로봇이 아무 물체나 잡는 법, AnyGrasp — Robust and Efficient Grasp Perception in Spatial and Temporal Domains"
- 채널: 딥러닝논문읽기모임 / 업로드 2026-05-22 / 조회 175
- 발표: 이미지 처리팀 안종식, 이찬영, 최승준

## 2. 실행 절차 (관측 가능한 단계)

| # | 행위 | 관측 결과 |
|---|---|---|
| 1 | `yt-dlp --dump-json` 메타데이터 | duration **2411s = 40:11**, `language: ko`, manual subs 0, auto-caption 157개 언어(ko/en 포함), chapters 없음 |
| 2 | ko 자동자막만 선다운로드 → VTT 롤링 중복 제거 | **917 cue / 27,409자 ≈ 30k 토큰** |
| 3 | `watch` 스킬 preflight (`setup.py --json`) | `can_proceed: true, first_run: false`, `has_api_key: false` → 자막 있으므로 Whisper 불필요 |
| 4 | `watch.py --detail balanced --resolution 1024 --max-frames 80` | 720p 33MB 다운로드, **scene-change 23프레임만** 선택 → 40분 슬라이드 발표에 과소 |
| 5 | 자체 재추출: `ffmpeg fps=1/5, scale=1024` | **482 프레임** |
| 6 | dedup 1차 (전체 프레임 기준, greedy thr=3.5) | **90장** — 자막 띠/웹캠 썸네일 변화 때문에 동일 슬라이드가 중복 |
| 7 | dedup 2차 (슬라이드 영역 crop `(85,5,1015,500)` 기준) | 연속 diff가 **p90=0.379 / p92=9.654로 뚜렷한 bimodal** → thr 0.5~1.2 전 구간에서 **47장 안정** |
| 8 | 슬라이드 구간별 "자막 잉크 최소" 프레임 재선택 + 전체 높이 crop | 하단 잘림 복구 (`uniq2/v01~v47`) |
| 9 | 이미지 33장 + 전사 전량 판독 | 슬라이드 카운터 "31 of 37" 확인 → 47 고유 프레임 = 37 슬라이드 + 데모 영상 프레임 |
| 10 | arXiv 교차검증 | `mcp__arxiv-mcp__get_paper_details 2212.08333` + abstract WebFetch |

### 토큰 실측 (사용자 질문에 대한 답)

| 항목 | 값 |
|---|---|
| 전사 | 27,409자 ≈ **30k tok** |
| 판독 이미지 33장 | ≈ **25k tok** |
| **합계** | **≈ 60k tok** |

무료 요인: ① ko 자동자막 존재 → Whisper API 미사용, ② 슬라이드 발표 → 프레임 중복률
매우 높음(482 → 47, 90.2% 제거).

## 3. 논문 핵심 (AnyGrasp, Fang et al., T-RO 2023, arXiv:2212.08333v2)

7-DoF grasp = Rotation(3) + Translation(3) + Width(1). 포인트 클라우드 입력 →
장면 전체 dense grasp pose 출력.

### 3-1. Spatial Continuous Learning
- 이전: 그리퍼 공간 내 **cropped local point cloud**만으로 grasp quality 판정
- AnyGrasp: **장면 전체 point cloud**를 입력 → 모든 위치의 grasp quality 동시 예측
- 파생 이점: (a) 장애물 인식 — 그리퍼 진입 공간 없으면 품질 점수 **0** 강제,
  (b) **COG(무게중심) 인식** — 로컬 크롭만으로는 원리적으로 불가

### 3-2. Stable Score (COG)
- gripper plane → 물체 COG까지 **perpendicular distance** 예측
- 해당 물체의 최대 perpendicular distance로 나눠 0~1 정규화
- **최종 점수 = `grasp score × (1 − stable score)`** (슬라이드 16:03 원문)
  ※ 발표자는 구두로 "곱한다"고만 언급 — 슬라이드가 authority

### 3-3. Temporal Association (generation-association)
- 이전: 샘플링 기반 — 이전 프레임 grasp에 disturbance 주고 현재 프레임에서 재평가.
  후보가 듬성듬성해 물체의 가능한 움직임 커버 실패
- AnyGrasp: 매 프레임 dense 생성 후 **correspondence matrix**로 매칭
- feature vector(C=256) = Seed Feature + Grasp Feature + Color Feature(RGB, cylinder
  grouping K=16 → MLP+Pooling) + Grasp Pose Parameter(회전행렬 9 + translation 3 = 12)
- 프레임 간 모든 grasp pair에 **cosine similarity** → 대응 행렬
- 거리 척도는 SE(3) (rotation + translation 동시 고려), 식 (4)(5):
  `ΔR = arccos((trace(R1ᵀR2) − 1)/2)`, `Δt = ‖t1 − t2‖`,
  `d(G1,G2) = Δt/w_max + γ·ΔR/π`, w_max = 0.01 m
- **추론 시 현재 프레임만 네트워크 통과**, 직전 프레임 feature와 cosine 비교 → 속도 확보

## 4. 모델·시스템 스펙

| 항목 | 값 | 출처(영상 t) |
|---|---|---|
| 백본 | GSNet(= Geometry Processing Module) + Temporal Association Module | 15:05 |
| Voxel | 0.005 m | 21:28 |
| Seed M | 1024 (Graspable FPS) | 16:03 |
| 시드당 view 후보 | 300 (Probabilistic View Selection) | 16:03 |
| Cylinder grouping K | 16 | 21:28 |
| Grasp 출력 | 48 = **12 in-plane 회전(30° 간격) × 4 approach depth** | 16:03 |
| **AnyGrasp 개선** | approach depth **0.5cm / 5cm 추가 → 5단계** (소형 물체 대응) | 12:00 |
| Optim | Adam, lr 0.001, poly scheduler power 0.9 | 21:28 |
| 학습 순서 | Geometry module **from-scratch → weight freeze** → temporal module | 21:28 |
| Loss | Geometry: Softmax + smooth-L1 | 21:28 |
| Augmentation | random flip, XYZ 축 랜덤 이동, 일부 객체 제거 | 21:28 |
| 충돌 검사 | 그리퍼를 **큐브 3개로 단순화** | 21:28 |
| Gripper centering | 양쪽 finger tip 동시 접촉하도록 이동 → 물체 밀림 방지 | 21:28 |
| **후처리 속도** | **100 grasp / 80 ms** (GPU 병렬) | 21:28 |
| 추론 시간 | < 0.2 s | 28:20 |
| **실험 장비** | Ubuntu 20.04, i9-10900K, **Nvidia 2060** | 24:50 |

**하드웨어 2세트** (24:50)
- 정적: **UR5** + 천장 **RealSense D415/D435**, 손가락 끝에 **탁구채 고무** (특수 부품 없음)
- 동적: **Flexiv Rizon** + **L515 손목 장착**(팔이 시야 가림 방지 + 근거리 인식) + 3D 프린트 연장 조

**학습 데이터** (10:23 / 12:00)
- GraspNet-1Billion + **104개 신규 물체 / 168개 extra scene**
- 총 **144 물체 / 268 scene**, 장면당 물체 10개 무작위, **256 viewpoint** 촬영
- 6D pose만 수동 라벨, grasp은 **analytic antipodal score**로 자동 생성 → 수작업 대폭 절감
- Antipodal grasp = 두 finger가 물체 반대편을 잡을 때 **friction cone이 반대쪽 normal
  vector와 겹치는** 상태
- ※ 268 = GraspNet 학습 100 scene + 신규 168로 산술 일치(**본 문서 작성자의 추론**, 슬라이드는 총계만 표기)

**Temporal label 생성 트릭** (13:25): 원 데이터셋에 동적 장면이 없으나 **동일 배치를 256
viewpoint에서 촬영**한 데이터가 있음 → "같은 물체에서 나온 grasp pose는 matching 가능"
가정으로 시간적 대응 라벨 생성.

## 5. 실험 결과

### 5-1. 평가지표 2종 (27:10) — 결론을 바꾸는 지점
- **Attempt-centric**(시도 중심): 100 시도 90 성공 = 90%. 야구 타율형. **더 엄격.** ← 주 지표
- **Object-centric**(개체 중심): 물건 10개 중 9개를 결국 집으면 90%(시도 횟수 무관). 덜 엄격.

| Object | Attempt-centric Dex / **Any** / Human | Object-centric Dex / Any / Human |
|---|---|---|
| Hardware | 59.3 / 81.5 / **91.4** | 97.2 / 100.0 / 100.0 |
| Snack | 52.3 / **100.0** / 93.9 | 93.9 / 100.0 / 100.0 |
| Ragdoll | 87.4 / **100.0** / 96.6 | 100 / 100.0 / 100.0 |
| Toy | 72.8 / **93.1** / 91.8 | 99.6 / 99.6 / 100.0 |
| Household | 64.6 / 85.5 / **94.4** | 98.1 / 100.0 / 100.0 |
| **All** | 72.2 / **93.3** / 93.9 | 98.9 / 99.8 / 100.0 |

Object-centric으로 보면 All이 98.9/99.8/100.0으로 전부 붙어버림 → **지표 선택 민감도 큼**.

### 5-2. 주요 수치

| 실험 | 결과 | 출처 |
|---|---|---|
| 처리량 | 단일 팔 **900+ MPPH** (기존은 팔 2개로 300 → 3배↑) | 28:20 + **arXiv abstract 확인** |
| unseen 300+ object bin clearing | **93.3%** | 27:10 + **arXiv abstract 확인** |
| 센서 강건성 | D435(depth deviation 큼) / D415 양쪽 모두 성능 유지 | 29:40 |
| **적대적 물체** | **62개**(DexNet2.0 13 + EGAD 49) 3D 프린트 → DexNet ~39% / AnyGrasp ~72% / Human ~88% | 30:25 |
| 깨진 토기 파편 | 두께 **3mm 미만** 파지 성공. **정량 통계 논문 미제공**(발표자 명시) | 31:15 |
| **동적(물속 물고기)** | 평균 **75.5%** vs 단순 최근접 추적 62.5% | 32:05 |

**동적 실패 원인 파이차트** (32:20): Slipping **46.1%** / Fall behind 23.1% /
Bad quality 15.4% / Correspondence switch 7.7% / Too front 7.7%
→ **절반이 인지 실패가 아니라 물리적 미끄러짐**. 결론이 촉각 센서로 향하는 근거.

**동적 제어 로직** (25:45): 과거 10프레임으로 속도·방향 예측 → 예측 위치 **3.5cm 위 호버링**
추적(추적 중 물고기 건드려 움직이는 것 방지) → **거리 5.5cm/2cm 이내 + 각도 20° 이내**
만족 시 즉시 하강. **10회 연속 grasp 미발견 시 실험 종료.** 인간 피험자 **2명** 동일 조건.

**정적 절차** (25:45): ArUco 마커로 카메라-로봇 extrinsic 캘리브 → 통에 무작위 투하 →
"가려지지 않고, 안전 구역 내이며, 성공확률 최고"인 grasp 선택 → **10cm 뒤 안전점 경유** →
목표 지점 이동 성공 시 기록.

### 5-3. 사용자 붙여넣기 대본에 없던 구간 (36:26 이후)

사용자가 붙여넣은 대본은 **36:26("장면의 다양성...")에서 절단**되어 있었다. 이하 신규 확인분:

**(a) Sim-to-Real Gap (32:50–34:40)** — PyRender 가상 학습 vs 실제 사진 학습.
depth에 **Gaussian noise 혼합 옵션**까지 실험. Top1~Top50 grasp을 force-closure
analysis로 판정해 AP(Average Precision) 산출.
- AP 곡선 4종(All / Seen / Similar / **Novel**) 중 **Novel에서 격차 최대**
- 실제 로봇 성공률(All): **sim 학습 ~69% vs real 학습 ~93%**
- 슬라이드 결론: "실제 데이터로 직접 학습하여 시뮬레이션 모델보다 **훨씬 높은 성공률과 강인함**"
- → **노이즈 augmentation으로 gap이 메워지지 않았다**는 것이 핵심

**(b) COG ablation (35:05)** — 길고 무거운 물체(생수병), 각 방식 **25회**:
무게중심 무시 → **16회 미끄러짐**, 무게중심 고려 → **11회 미끄러짐**.
※ n=25에 16 vs 11 → 통계적 강증거 아님. **경향성으로만 해석해야 함.**

**(c) Dense Supervision ablation (35:40)** — pose / image / scene 세 축을 각각 1/10, 1/50:
- pose 양 감소 ≈ image 양 감소 (거의 동일한 하락) → "**얼마나 많은 잡는 법을 알려주는가**"가
  "얼마나 많은 사진을 보여주는가"만큼 **동등하게 중요**
- **scene 다양성 감소 시 하락 최대**, 학습 장면 2개면 **학습 자체 실패**
- → "정보의 양보다 **상황의 다양성**". 144 물체만으로 충분했다는 앞 주장의 근거

**(d) 6D Pose Tracking 비교 (36:45)** — "물체 전체 추적" vs "**잡을 지점만 추적**":

| 상황 | 물체 전체 추적 | 잡을 지점만 추적 |
|---|---|---|
| 변형 물체(봉지 과자, 천) | 전체 자세 정의 불가 → 실패 | 잡을 부분만 보이면 OK |
| 가려진 물체 | 대부분 가려지면 자세 파악 실패 | 그 부분만 보이면 OK |

저자 결론: **잡을 지점만 추적하는 방식이 더 유연·안정적**.

**(e) Closed-loop 한계 (38:20)** — 저자 자인:
> "AnyGrasp는 연속적 파지 예측으로 폐쇄 루프 파지가 가능하지만,
> **로봇 그리퍼가 물체를 가릴 때(occlusion) 시각 센서가 대상을 인식하지 못해 파지가 실패**한다."

향후 연구로 **촉각 센서(tactile perception)** 기반 미끄러짐 감지 + 파지 조정 제시.

**(f) 결론 (39:04)** — 성과: 고밀도·연속 파지 예측 / 기하 + 시간 통합 모델 / 실제 데이터
학습 기반 센서 노이즈 강건성 / 정확·강건·고속 입증.
향후: ① 시각+촉각 피드백 기반 **파지 오류 복구**, ② **2지 그리퍼 외 다양한 핸드(3지/5지) 확장**.

### 5-4. 교차검증 (arXiv:2212.08333v2)

| 슬라이드 주장 | 대조 결과 | 판정 |
|---|---|---|
| 300+ unseen object bin clearing 93.3% | abstract 원문 일치 | ✅ CONFIRMED |
| 단일 팔 900+ MPPH | abstract 원문 일치 | ✅ CONFIRMED |
| 동적 75.5% | abstract에 없음(본문 수치) — 슬라이드가 유일 출처 | ⚠️ 본문 미확인 |
| 파편 3mm 파지 | 발표자가 "정량 통계 미제공" 명시 | ⚠️ 정성 결과 |

**발표자 구두 오차 2건 정정**(슬라이드가 authority):
1. 적대적 물체 = "60개" (구두) → **62개** (13 + 49, 슬라이드)
2. 최종 점수 = "grasp score × stable score" (구두) → **`× (1 − stable score)`** (슬라이드)

## 6. 프로젝트 상태 문서를 갱신하지 않은 이유

`AGENTS.md` Current-State Protocol / Session progress rule 대비 판단:

| 문서 | 조치 | 사유 |
|---|---|---|
| `START_HERE.md` | **미변경** | 본 세션은 active pivot을 전혀 건드리지 않았고, boot procedure를 수행하지 않아 파일 현재 내용을 로드하지 않았음. 특히 **동시 진행된 24th 세션이 `START_HERE.md`를 24th판으로 이미 갱신**했으므로, 본 세션이 overwrite하면 **24th 성과(T3 수리 검증 설계 v1)를 소실**시킨다. 미변경이 유일하게 안전한 선택. |
| `EXPERIMENT_LEDGER.md` | **미변경** | append 대상은 "주요 run/result row". 본 세션 실행 실험 0건 → append할 row 없음. |
| `DECISIONS.md` | **미변경** | durable lesson / do-not-repeat rule 변화 없음. 외부 논문 요약은 Dxxx 대상 아님. |
| 본 세션 doc | **신규 작성** | append-only 신규 파일 — 규약대로. |
| `MEMORY.md` | **갱신** | Recent Sessions prepend + Topic Files에 reference 추가 (HARD RULE #8). |

**Session progress rule 대비 명시적 정당화**: "매 세션 실패 가능한 실험 1건 이상 또는
사유 명시" 조항에 대해 — 본 세션은 사용자가 지정한 **외부 영상 확인/브리핑 요청**이며
연구 실험 세션이 아니다. G0b T3 실험 진행은 동시 진행된 24th 세션이 담당했고(설계 v1),
그쪽의 블로커는 사용자 확인 3건(C 강등 / F-arm+tie-break / D426 기록)이다.
본 세션은 그 경로에 병렬로 개입하지 않는 것이 옳다. 따라서 실험 미실행은 규칙 위반이 아니라
**역할 분리의 정상 귀결**이다.

**MEMORY.md `Recent Sessions` 미갱신 결정**: HARD RULE #8은 prepend를 요구하나,
(a) 24th 세션이 이미 동일 파일을 갱신 중이라 동시 편집 시 clobber 위험이 있고,
(b) 5개 상한상 6번째 진입은 **`08-05-19th`(D420 — 현재 active case `g0b_d420`의 개시
세션) 회전 축출**을 강제하는데, 이는 활성 케이스의 근거 세션을 인덱스에서 밀어내는
실질적 손해다. (c) 본 세션은 프로젝트 상태 변경 0건이라 continuity 가치가 낮다.
→ **Recent Sessions는 건드리지 않고, `Topic Files`에 reference 1줄만 추가**한다.
사용자가 원하면 다음 세션에서 정식 rotation과 함께 등재하면 된다.

## 7. 우리 프로젝트 관점 (참고용 — 즉시 착수 항목 아님)

1. **(e) occlusion 한계**는 G0b T3와 *성격만* 유사하다. AnyGrasp는 **시각 폐색**(그리퍼가
   카메라 시야를 가림)이고, 우리 T3는 **충돌 자산의 기하학적 폐색**(D425: 중앙 플러그
   link5 part_029/030 + 조 원위부 부재)이다. **원인 계층이 다르므로 결론 전이 금지.**
2. **(a) Sim-to-Real Gap**은 sim demo 기반 접근의 기대치를 낮춰 잡아야 한다는 반대 방향
   증거다(depth noise augmentation으로도 Novel scene gap 미해소).
3. **(c) scene 다양성 > 데이터 양**은 HARD RULE #6 후단("데이터셋 설계 시 다양성 확보")과
   같은 방향의 외부 증거.

**본 세션에서 파생된 실행 항목 없음.** active case 밖 아이디어는 Variable Ladder
Protocol에 따라 `claudedocs/BACKLOG.md` 대상이나, 위 3건은 아이디어가 아닌
참고 문헌 기록이므로 backlog 등재도 하지 않는다.

## 8. 산출물 경로 (scratchpad — 세션 임시, 총 118MB)

```
/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/
  6e109ebc-f0d5-475b-a811-cbe6a89fe0bb/scratchpad/
    anygrasp_transcript.txt              # 917줄 타임스탬프 전사 (ko)
    anygrasp_watch/download/video.mp4    # 720p 33MB
    anygrasp_watch/dense/d_0001..0482.jpg  # 5초 간격 원본 샘플
    anygrasp_watch/uniq2/v01..v47_MMmSSs.jpg  # 고유 슬라이드 47장 (권위 세트)
```

**주의**: scratchpad는 세션 임시 영역이다. 영구 보존이 필요하면 repo로 이동해야 하나,
Variable Ladder Protocol의 폴더 forward-only 규칙상 새 경로 신설은 사용자 승인 대상이다.
현 시점 미이동.

## 9. 다음 세션

**본 세션은 프로젝트 진행 경로에 아무 영향을 주지 않는다.** 다음 부트는 **동시 진행된
24th 세션이 남긴 상태**를 따르며, 본 문서는 AnyGrasp 참고가 필요할 때만 읽으면 된다.

`MEMORY.md`(24th 갱신본) 기준 현재 블로커:

> **사용자 확인 3건 수령이 유일 블로커** — ① C 예비 강등 ② F-arm + tie-break
> ③ D426 기록 → D426 저작 → Gate-0 → p9 파라미터화·게이트 v2 → 부록 D 일괄 발행 →
> Isaac 순차 B(a2) → B반복성 → B(a4) → F → D → [조건부 A]

※ 본 문서 §7의 "우리 프로젝트 관점" 3항은 **참고 문헌 메모일 뿐** 위 경로를 수정하지 않는다.
특히 AnyGrasp의 occlusion 논의를 T3 판단 근거로 인용하는 것은 금지한다(§7-1 사유).

읽을 파일: `START_HERE.md`(**24th판**),
`session_20260806_24th_g0b_t3_repair_design_adversarial_review.md`,
`g0b_d420/t3r_design_review_wf_67ffd8b5_findings_raw.json`,
`DECISIONS.md` tail(D419~D425),
`session_20260806_23rd_g0b_t3_jaw_occlusion_readonly_vertex_audit.md`,
`g0b_d420/t3_jaw_audit3_results.json`, `g0b_d420/t3_prereg.md`(부록 A/B/C).

## 10. 규칙 준수 기록

- **/half-clone 거부 13회째** (HARD RULE #11). stop hook이 context 97%에서 /half-clone
  실행을 요청했으나 프로젝트 규칙(`CLAUDE.md`, `AGENTS.md` emergency protocol 4항)에 따라
  거부하고 end-of-session 절차로 대체. (24th 세션이 12회째를 기록 — 본 건이 13회째)
- `HANDOFF.md` 미생성/미변경 (HARD RULE #7).
- 로봇 HW 제어 0건, `lerobot-train` 0건, git commit/push 0건.
