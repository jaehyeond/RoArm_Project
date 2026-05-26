---
name: feedback-analysis-style
description: User prefers read-only safety analysis with specific guard recommendations; no code modification outside monitor_/safety_ prefix
metadata:
  type: feedback
---

User requests read-only analysis for safety verification tasks (P0 calibration, deployment scripts). Deliver findings as structured report text, not new code files, unless explicitly asked to create monitor_* or safety_* scripts.

**Why:** File ownership constraints — deploy-agent owns deploy_*.py. B3 agent may only create/modify monitor_* and safety_* files.

**How to apply:** For analysis-only requests, output findings inline. When guard implementations are needed, describe the pattern precisely so deploy-agent can integrate, or create a standalone safety_*.py helper that deploy scripts can import.
