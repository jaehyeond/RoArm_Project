---
name: feedback-analysis-style
description: User prefers read-only safety analysis with specific guard recommendations; no code modification outside monitor_/safety_ prefix
metadata:
  type: feedback
---

User requests read-only analysis for safety verification tasks (P0 calibration, deployment scripts). Deliver findings as structured report text, not new code files, unless explicitly asked to create monitor_* or safety_* scripts.

**Why:** File ownership constraints — deploy-agent owns deploy_*.py. B3 agent may only create/modify monitor_* and safety_* files.

**How to apply:** For analysis-only requests, output findings inline. When guard implementations are needed, describe the pattern precisely so deploy-agent can integrate, or create a standalone safety_*.py helper that deploy scripts can import.


---

# [Merged 2026-07-12] 구 디렉토리 "Deployment & Safety Specialist"의 feedback_analysis_style.md 원문 (2026-03월대, 이름 sanitization 변경으로 분열됐던 내용)

---
name: feedback_analysis_style
description: User wants brutally honest skeptical analysis, not hedged summaries — especially for industry claims
type: feedback
---

Be brutally honest and skeptical, especially for large company robotics claims. User explicitly requests critical analysis ("be SKEPTICAL", "look for critical reviews not just announcements").

**Why:** User has experienced first-hand how deployment reality (60%) differs from lab claims (100%). They are aware of the 2026-03-10 false research gap incident and want to avoid hype-driven decisions for their CoRL 2026 paper positioning.

**How to apply:** When analyzing industry demos or deployment claims: (1) separate what was shown vs what was claimed, (2) check if there is independent verification, (3) compare to companies that have verifiable deployments (Boston Dynamics Spot, Amazon Sparrow), (4) cite Google's prior robotics project history as a base rate for skepticism.
