---
name: project_applied_scenarios
description: Applied research direction analysis for thesis — 4 scenarios evaluated, multi-arm coordination selected as primary
type: project
---

Investigated 4 applied research scenarios for master's thesis (2026-03-23).

**Winner: Scenario 4 — Multi-Arm Parallel Coordination (3x RoArm-M3)**

Research question: "Can three independently-trained VLA models coordinate a parallel sorting task without explicit inter-robot communication, using only visual observations?"

Key finding: Zero papers exist on consumer multi-arm VLA without explicit communication (verified via 6 search terms). This is the clearest gap of all 4 scenarios AND it directly solves the student's data collection bottleneck (3x throughput from shared workspace setup).

Paper title: "Three VLAs, Zero Messages: Implicit Coordination in Consumer Multi-Robot Manipulation via Visual Grounding Alone"

**Application narrative: Lab Reset (Scenario 3 framing)**
Wrap the multi-arm experiment in the "lab bench reset" story. Objects: ruler, pen, bottle, notebook, eraser. Each arm handles 2 objects. More compelling story than "sort 3 bins."

**Thesis chapter 2: Digital Twin (Scenario 2)**
Build Isaac Lab + Unity dashboard. Test: "Does joint-state divergence predict task failure for a $130 arm?" Both positive and negative results publishable. HIGH novelty, HIGH null-result risk. Do NOT put this in CoRL.

**Why Scenarios 1 and 3 alone are rejected:**
Both require 5 classes × 50 episodes = 250 hand-guiding episodes, making the data collection pain WORSE. Only viable if combined with multi-arm (which enables parallel collection).

**Why:** Advisor said "stop just grasping sponges." Needs a broader, more demonstrable system. 3 arms doing coordinated work is visually compelling and uses all hardware already owned.
**How to apply:** Design all experiments around the multi-arm baseline (3 arms × 1 class each) before scaling to more complex tasks. Always frame in terms of "what replaces explicit communication."
