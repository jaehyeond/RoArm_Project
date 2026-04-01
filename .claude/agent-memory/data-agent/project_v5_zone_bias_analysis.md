---
name: V5 Zone System Design Flaw Analysis
description: Root cause analysis of v5 data collection bias — 5-zone system was actually 3-zone in base angle space, causing 69.7% of frames to cluster in ±20° of center
type: project
---

# V5 Zone System Design Flaw (2026-03-31)

## Finding

The v5 zone classification was structurally biased. Three of five zones (NEAR, FAR_CENTER, OVERHEAD)
all permit base angle in the range -30° to +30°. Only MID_LEFT (base < -30°) and MID_RIGHT (base > +30°)
enforce lateral diversity. Completing all zone quotas would produce 84 center-zone + 50 lateral episodes.

**Why:** Zone was designed with 3 axes (base angle, distance, elevation) but SmolVLA conditions
primarily on base angle + visual appearance. Distance-at-same-angle is a weaker cue.

**How to apply:** Any future zone redesign must use base angle as the primary axis for all 5 zones.

## Actual Frame Distribution (v5, 136 eps, 13,470 frames)

| Bucket | Frames | % |
|--------|--------|---|
| base < -30° | 1,710 | 12.7% |
| -30° to -5° | 735 | 5.5% |
| -5° to +15° | 7,431 | 55.2% |
| +15° to +30° | 1,683 | 12.5% |
| +30° to +50° | 180 | 1.3% |
| +50° to +100° | 1,731 | 12.9% |

80% of frames fall in [-3°, +23°] window (from stats.json quantiles q10/q90).
Dataset mean base = +9.93° (systematically right-of-center), std=31.0°.
MID_RIGHT bimodal: dense near +70° (mean_max=71.6°), sparse in +30° to +50°.

## Quota Enforcement

ZONE_TARGETS = {NEAR:30, MID_LEFT:25, MID_RIGHT:25, FAR_CENTER:35, OVERHEAD:15}
Total = 130, actual collected = 136.

The quota system is **advisory only**. No code blocks or FAILs an episode for being over-quota.
The OSD shows zone counts and recommends the most under-quota zone, but the user must choose
to place the sponge at a laterally offset position. The system never refused to save a CENTER episode.

## Zone Actual Coverage

| Zone | Episodes | Base range |
|------|---------|------------|
| NEAR | 30 | -9.8 to +28.4° |
| FAR_CENTER | 39 | -26.4 to +28.8° |
| OVERHEAD | 15 | -5.2 to +32.0° |
| MID_LEFT | 25 | -49.1 to -33.6° |
| MID_RIGHT | 27 | +16.6 to +88.4° |

All three center zones overlap almost perfectly in base angle space.

## Fix Required

Replace classify_zone() with base-angle-only classification:

```python
def classify_zone(base_angle, fk_dist, fk_z):
    if base_angle < -40:   return "FAR_LEFT"
    if base_angle < -15:   return "LEFT"
    if base_angle >  40:   return "FAR_RIGHT"
    if base_angle >  15:   return "RIGHT"
    return "CENTER"

ZONE_TARGETS = {
    "FAR_LEFT": 27, "LEFT": 27, "CENTER": 27, "RIGHT": 27, "FAR_RIGHT": 27,
}
```

Add soft-block OSD warning when zone is >5 episodes over-quota.

## Ideal Distribution for 135 episodes

5 zones × 27 episodes each = 2,700 frames/zone (roughly equal).
Expected base mean ≈ 0°, std ≈ 35°. No systematic rightward offset.

## User Note

This was a design failure, not a user collection failure. The user followed the zone system correctly.
