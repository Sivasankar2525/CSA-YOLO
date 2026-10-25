"""Worker-PPE Association Module (WPAM).

Associates PPE items (helmet, vest, etc.) with individual workers using
geometric proximity and containment rules.

Algorithm (see docs/MODULES.md):
  1. Head region (top 25% of box, widened) ↔ helmet/hat detections
  2. Torso region (20%–75% of box height) ↔ vest/safety vest detections
  3. Spatial gate: a PPE centre must lie inside the region expanded by
     ``WPAMConfig.gate_margin_ratio`` × box height (provisional default)
  4. Cost: normalised distance from PPE centre to region centre
  5. Solver: Hungarian algorithm (scipy) for one-to-one matching per class
"""

from .associate import associate_ppe_to_workers

__all__ = ["associate_ppe_to_workers"]
