---
name: Robot Self-Improvement Gap Analysis (2026-03-25)
description: VLA self-improvement 분야 비판적 갭 분석. 5개 핵심 논문 검토, 3개 갭 후보 도출.
type: project
---

## 분석 배경
2026-03-25 세션. SmolVLA + RoArm-M3 기반 robot self-improvement 방향 탐색.
파일: `/home/cgxr/Documents/Robotics/RoArm_Project/model_self_improvement_gap_analysis.py`

**Why:** 기존 AR-Guided 방향에 Self-Improvement를 추가하거나 대체할지 결정하기 위해.

**How to apply:** 갭 주장 전 CRITICAL verification checklist 완료 필수.

---

## 핵심 논문 분석 결과

| 논문 | 로봇 | VLA 타입 | 로컬 실행? | 우리와 차이 |
|------|------|----------|-----------|------------|
| SOAR (CoRL'24) | WidowX (~$2K) | CLIP 기반 | 클라우드 | 고가 로봇, 클라우드 |
| SimpleVLA-RL (2509.09674) | 미확인 | 미확인 | 미확인 | **CRITICAL: 로봇 확인 필수** |
| VLAC (2509.15937) | 미확인 | VLM critic | 대형 VLM 가정 | local judge 갭 |
| On-the-Fly VLA (2601.06748) | 미확인 | test-time RL | 미확인 | flow-matching 호환 미확인 |

---

## 3개 갭 후보 (검증 전 상태)

### GAP A: Consumer-grade VLA self-improvement
- 주장: SOAR/$2K 로봇 vs 우리/$130 로봇
- Confidence: MEDIUM (SimpleVLA-RL 로봇 확인 전)
- **검증 필수: SimpleVLA-RL (2509.09674) 로봇이 consumer-grade인가?**

### GAP B: Local VLM judge (edge-deployable, 3B 이하)
- 주장: SOAR/VLAC 모두 클라우드 or 대형 VLM. Qwen2.5-VL 3B 로컬 = 없음
- Confidence: MEDIUM
- 기술 실현: RTX 4090에서 SmolVLA (3-5GB) + Qwen 3B INT4 (3-4GB) 순차 실행 가능
- **검증 필수: "edge VLM reward labeling robot" 검색**

### GAP C: Flow-matching denoising ensemble variance as uncertainty proxy
- 주장: SmolVLA 10-step ODE에서 N=5 샘플 variance = uncertainty signal
- 소스 검증 (HIGH confidence): sample_actions() loop이 random noise에서 시작 → 재샘플 시 다른 trajectory
- Confidence: HIGH (mechanism) / MEDIUM (선행연구 완전 부재 확인 전)
- **검증 필수: "diffusion policy uncertainty active learning" + "flow matching robot uncertainty"**

---

## SmolVLA 기술 검증 (HIGH confidence, 소스 직접 확인)

```python
# modeling_smolvla.py line 392-396
if reduction == 'none':
    per_sample_loss = losses.mean(dim=(1, 2))  # shape (B,)
    return per_sample_loss, loss_dict
```

- `forward(reduction='none')` → per-sample loss (B,) 반환 이미 구현됨
- reward-weighted BC 코드 추가: ~50줄
- denoising variance 계산: sample_actions() 를 N=5회 호출 → std 계산 (~20줄)

---

## Verification Checklist (다음 세션에서 완료)

1. **[CRITICAL]** SimpleVLA-RL (2509.09674): 로봇 + VLA 타입 확인
2. **[CRITICAL]** VLAC (2509.15937): VLM critic 크기 + 로컬 실행 가능성
3. **[CRITICAL]** 2303.01488: SOAR과 동일 논문인가?
4. **[MEDIUM]** "diffusion policy active learning uncertainty" 검색 (3개 키워드)
5. **[MEDIUM]** "edge VLM reward labeling robot" 검색

---

## CoRL 2026 권장

- **메인 방향: AR-Guided Collection + Quality Oracle (기존)** — 유지
- Self-Improvement: Chapter 5 확장 or ablation (보조)
- 방향 전환 결정 데드라인: 2026-03-28 (verification 완료 후)
- **단, GAP C (denoising variance)는 AR-Guided 논문에 이미 포함 가능: "quality oracle의 uncertainty signal"**
