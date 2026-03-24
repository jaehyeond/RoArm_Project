---
name: 3DGS + VLA Research Landscape Analysis (2026-03-24)
description: 44편 arXiv 논문 기반 3DGS+Robotics 현황 분석. 포화도, 갭, CoRL 2026 기회 평가.
type: project
---

## 분석 배경
- 입력: 44편 arXiv 논문 (3DGS + Manipulation/Navigation/DigitalTwin/DataGen)
- 기존 메모리 교차 검증: research_gemini_robotics.md, research_ideas_corl_thesis.md
- 날짜: 2026-03-24

## 포화된 서브도메인 (진입 불필요)
| 서브도메인 | 논문 수 | 대표 논문 |
|---|---|---|
| 정적 씬 3DGS sim-to-real | 5편+ | SplatSim(82%), GaussTwin, RoboGSim |
| Grasp pose + 3DGS | 3편+ | GaussianGrasper, ArtGS, DexFruit |
| 데이터 증강 (multi-view) | 2편+ | RoboSplat, High-Fidelity(2510.10637) |
| Navigation + 3DGS | 4편+ | Splatblox, SplatSearch, VISTA, ReaDy-Go |

## 미포화 서브도메인 (기회 있음)
| 서브도메인 | 이유 |
|---|---|
| 동적 씬 실시간 3DGS | 기술 한계, 연구 갭 아님 — 제외 |
| Single-view RGBD → 3DGS + VLA 학습 | RoboSplat/SplatSim은 multi-view. RGBD depth 활용 미확인 |
| 소형 VLA(< 1B) + 3DGS | SmolVLA(2025-06 발표) + 3DGS = 검색 범위 내 없음 |
| 소비자 하드웨어($100-200) + 3DGS | 전부 Franka/UR5/WidowX 이상 |

## VLA 직접 결합 현황 (확신도: MEDIUM)
- VLA에 3DGS 연결한 논문: 3-4편
  - SplatSim: WidowX + 3DGS → OpenVLA 학습 (직접 연결 확인)
  - RoboSplat: manipulation policy 학습 (VLA 여부 불확실)
  - High-Fidelity(2510.10637): "zero-shot manipulation", 모델 종류 확인 필요
  - Zero-Shot RAG(2603.00500): MLLM 사용, embodied VLA 아님
- 모두 7B+ 또는 연구용 로봇 대상

## SmolVLA + 3DGS 상태
- SmolVLA arXiv: 2025-06 발표 → 후속 연구 시간 9개월뿐
- 검색 범위 내 SmolVLA + 3DGS = 없음 (확신도: MEDIUM-HIGH)
- 반증 가능성: "sub-500M VLA" 등으로 서술된 논문 존재 가능

## Single-view RGBD → 3DGS 현황 (확신도: MEDIUM)
- MonoSplat, Gaussian Splatting with Depth Regularization 등 존재 (novel view synthesis 목적)
- Robotics 데이터 증강 + single-view RGBD = 44편 중 미확인
- RoboSplat은 single-view 불가라고 명시
- 반증 가능성: MonoGS/DUSt3R+로봇 논문이 2025 후반 존재 가능

## 동적 씬 실시간 3DGS (확신도: HIGH)
- 4DGS / Dynamic-GS = 전부 오프라인 처리
- ReaDy-Go(2026-02) = 네비게이션 반응, 실시간 재구성 아님
- 조작 과제에서 실시간 재구성+렌더링 = 기술 한계, 6개월 내 해결 불가
- 판단: 연구 갭이 아닌 기술 장벽

## 실제 갭 (우리 기회)
### 갭 A: 소비자 로봇 + 소형 VLA + 3DGS (확신도: HIGH)
- 44편 모두 Franka/UR5/WidowX 이상
- RoArm-M3($130) + SmolVLA(450M) = 검색 범위 내 없음

### 갭 B: Single-view RGBD depth 활용 3DGS → VLA 학습 (확신도: MEDIUM)
- RoboSplat = multi-view RGB, depth 미활용
- Depth를 geometry prior로 사용하면 view 수 줄일 수 있음
- 반증: MonoGS+robot 논문 존재 가능

### 갭 C: OOD embodiment(사전학습 없는 로봇)에서의 3DGS sim-to-real (확신도: MEDIUM)
- SplatSim = WidowX (OXE 포함 로봇)
- SmolVLA→RoArm = 최대 OOD 케이스. 3DGS OOD gap 감소 효과 미측정

## CoRL 2026 (5/28) 전략 권장
- Option A (핵심 contribution): 3-4개월 소요, 5/28까지 ~8주 = 위험
- Option B (ablation으로 포함): IDEA 1+3 메인, 3DGS = 비교 실험. CoRL 현실적.
- Option C (석사 논문 chapter): CoRL=IDEA 1+3, 3DGS=석사 논문 Chapter 2. 가장 안전.
- 추천: Option B 또는 C

## 교차 검증 필요
- A2 sim2real: SplatSim 82% 전이율이 다른 embodiment에서도 재현되는지
- A2 sim2real: Azure Kinect depth → 3DGS geometry prior 사용 선행연구 목록
- B2 data-efficiency: SmolVLA SigLIP의 3DGS 렌더링 vs 실사 cosine similarity 측정
