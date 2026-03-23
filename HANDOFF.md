# HANDOFF — RoArm-M3 SmolVLA Project

> Written: 2026-03-19 | Context: 93% exhausted, continuing in fresh conversation

---

## Goal

**CoRL 2026 논문 제출 (마감 2026-05-28, D-70)**

논문 제목 (working): "Data-Efficient VLA Adaptation on Consumer Hardware"

핵심 질문: "새 로봇에 소형 VLA를 적용할 때, 얼마나 많은/좋은 데이터가 필요한가?"

4가지 기여:
1. OOD Scaling Laws: episodes(25-150) × quality × steps → 성공률 곡선
2. Data Quality Methodology: 7단계 검증, FK depth, gripper phase, static frame
3. Multi-Object Transfer: 4물체(sponge/cup/box/tool) × 50ep, cross-task transfer
4. Self-Improving Loop (Seed2Scale-lite): 배포→VLM 판별→성공 rollout 재활용→재학습

보험: IROS 2026 LBR (마감 7/31)

---

## Current Progress

### Step 1 완료 (이번 세션)

**Agent-Team v2 구축**:
- `claudedocs/AGENT_PERSONAS.md` — 9개 페르소나 정의서 + 운영 규칙
- `.claude/agents/` — 9개 에이전트 파일 생성:
  - Team A (Robotics): robotics-manipulation.md, robotics-sim2real.md, robotics-hardware.md
  - Team B (Physical AI): pai-vla-model.md, pai-data-efficiency.md, pai-deployment.md
  - Team C (Research): research-experiment.md, research-analysis.md, research-writing.md
- 기존 3개 에이전트(data-agent, pipeline-agent, deploy-agent)는 그대로 유지

**문서 업데이트**:
- `CLAUDE.md` — Agent Team 섹션에 Research Agents v2 추가
- `ResearchPlan.md` — Bimanual → Data-Centric 방향 전환 요약 추가 (기존 내용 아카이브)
- Memory: `project_corl2026_direction.md`, `project_agent_team_v2.md` 생성, MEMORY.md 업데이트

**계획서**: `.claude/plans/radiant-waddling-wilkes.md` — 전체 전략 + 70일 타임라인

### 이전 세션 성과 (참고)

- SmolVLA + RoArm-M3 스펀지 pick 100% 성공 (74ep, open-loop 4-chunk, 50K steps)
- 데이터 품질 도구: data_episode_quality.py, data_distribution_simple.py
- Isaac Lab 설치 완료 (conda env isaaclab, URDF→USD 변환 성공)

---

## What Worked

1. **SmolVLA 유지 결정**: 70일 안에 모델 전환은 불가능. SmolVLA 생태계가 얇다 = 블루오션
2. **Data-Centric 전환**: Bimanual은 SO-101 미보유 + 시간 부족. Data-Centric은 기존 자산 즉시 활용
3. **Agent-Team v2**: 상황별 2-3개만 소환하는 구조 (9개 전체 동시 사용 금지)
4. **비판적 분석**: Projector-VLA 아이디어를 체계적으로 검증 → 기각 (소프트웨어 대안도 SmolVLA에 적용 불가)

---

## What Didn't Work / 주의사항

1. **Projector-VLA**: 물리 프로젝터 + VLA 학습 아이디어 — 기각됨. SigLIP OOD 리스크, 소프트웨어 대안(AimBot, TraceVLA)도 SmolVLA에 직접 적용 불가. 프로젝터는 데이터 수집 보조로만 가치 있으나 마스킹 테이프로 대체 가능
2. **MEM/RECAP/RD-VLA 접목**: 3개 최신 논문을 Projector-VLA에 접목하려는 시도 — 기각됨. MEM(비디오 메모리)은 SmolVLA 아키텍처 비호환, RECAP(RL)은 규모 차이, RD-VLA(latent iteration)는 아키텍처 비호환
3. **Gemini의 분석**: "비판을 장점으로 뒤집기" 수사 기법 사용. debate technique이지 engineering analysis가 아님. 자기 모순 다수 (Step 1에서 "유니버설 솔루션" 주장 → Step 4에서 스스로 부정)
4. **"소프트웨어 대안이 우월" 단정**: 내가 초기에 과도하게 단정. 실제로는 SmolVLA에 적용 가능한 소프트웨어 visual prompting도 0개 — 공정 비교 불가

---

## Key Decisions Made

| 결정 | 선택 | 이유 |
|------|------|------|
| 연구 방향 | Data-Centric Multi-Object | 70일 내 완료 가능, 기존 자산 활용 |
| 모델 | SmolVLA 유지 | 파이프라인 완성, 전환 비용 치명적 |
| OpenVLA | 비교 대상으로만 | 클라우드 GPU($50-100/월)로 LoRA fine-tune |
| Agent 구현 | 단계적 (문서→정식) | 문서 완료, .claude/agents/ 구현 완료 |
| Bimanual | 졸업논문 후속 연구 | SO-101 미보유, CoRL 시간 부족 |

---

## Next Steps (Step 2부터)

### Step 2: Multi-Object 데이터 수집 준비 (D-68~D-56)

**즉시 해야 할 코드 작업**:
1. `collect_data_manual.py` — 물체명 파라미터 추가 (--object cup/box/tool)
2. `convert_to_lerobot_v3.py` — multi-object 태스크 텍스트 지원 ("Pick up the [object]\n")
3. 물체 확보: cup, box, tool (실제 물체)
4. 5-zone 배치 계획: LEFT_FAR(10), LEFT(10), CENTER(10), RIGHT(10), RIGHT_FAR(10) per object

### Step 3: Scaling 실험 매트릭스 (D-56~D-46)

**새로 작성할 파일**:
- `experiment_matrix.py` — 배치 실험 자동화 (에피소드 서브샘플링 + 학습 + 평가)
- 실험: 5(episodes) × 2(quality) × 4(steps) = 40 학습 runs

### Step 4-8: (계획서 참조)

- Step 4: 배포 평가 + `eval_deployment.py`
- Step 5: Self-improving loop + `self_improve_loop.py` + `vlm_success_detector.py`
- Step 6: Multi-task transfer
- Step 7: 논문 작성
- Step 8: 제출 (5/28)

---

## Critical Files

| 파일 | 역할 |
|------|------|
| `CLAUDE.md` | 프로젝트 규칙 + agent team 정의 |
| `ResearchPlan.md` | 연구 방향 (Data-Centric으로 전환됨) |
| `claudedocs/AGENT_PERSONAS.md` | 9개 페르소나 상세 정의 |
| `.claude/plans/radiant-waddling-wilkes.md` | 전체 전략 계획서 |
| `collect_data_manual.py` | 데이터 수집 (수정 필요) |
| `convert_to_lerobot_v3.py` | LeRobot 포맷 변환 (수정 필요) |
| `run_official_train.py` | 학습 래퍼 (수정 불필요) |
| `deploy_smolvla.py` | 배포 (수정 불필요) |
| `data_episode_quality.py` | 품질 검증 도구 |

---

## 70일 타임라인 요약

```
D-70~D-68: Agent personas + 연구 계획 ✅ 완료
D-68~D-56: Multi-object 데이터 수집 ← NEXT
D-56~D-46: Scaling 실험 매트릭스
D-46~D-38: 배포 평가
D-38~D-30: Self-improving loop
D-30~D-24: Multi-task transfer
D-24~D-10: 논문 작성
D-10~D-0:  제출 (5/28)
```
