"""Worker-PPE Association Module (WPAM).

Associates PPE items (helmet, vest, etc.) with individual workers using
geometric proximity and containment rules.

Algorithm: For each person box, match nearby PPE items using:
  1. Head region (top 20% of box) ↔ helmet/hat detections
  2. Torso region (middle 60%) ↔ vest/safety vest detections
  3. Cost: negative IoU + center distance
  4. Solver: Hungarian algorithm for clean one-to-one matching
"""

from .associate import associate_ppe_to_workers

__all__ = ["associate_ppe_to_workers"]
