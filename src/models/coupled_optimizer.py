"""
Coupled Consistency Optimizer.
Layer 5 Business Rule & Economic Constraint Module.
Solves Research Gap 3 (Inconsistent multi-task predictions between Arrivals and Nights).
Enforces:
  1. Physical Lower Bound: Nights / Arrivals >= 1.0
  2. Historical Realistic Range: [Ratio_base - delta, Ratio_base + delta]
Performs convex projection to eliminate business contradictions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class CoupledConsistencyOptimizer:
    """
    Guarantees economic domain consistency between predicted tourist arrivals
    and overnight stays.
    """

    def __init__(self, delta: float = 1.0, min_ratio: float = 1.05):
        self.delta = delta
        self.min_ratio = min_ratio

    def optimize_pair(
        self,
        y_arr_pred: np.ndarray,
        y_nig_pred: np.ndarray,
        base_ratios: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects (y_arr, y_nig) into the mathematically consistent domain:
          Ratio = y_nig / max(y_arr, 1.0)
          Feasible region: [max(min_ratio, base_ratio - delta), base_ratio + delta]
        Returns:
          - y_arr_adj: Adjusted arrivals
          - y_nig_adj: Adjusted nights
          - violations_flag: Boolean array indicating where adjustments occurred
        """
        y_arr_adj = np.maximum(y_arr_pred.copy(), 1.0)
        y_nig_adj = np.maximum(y_nig_pred.copy(), 1.0)

        # Current ratios
        current_ratios = y_nig_adj / y_arr_adj

        # Feasible bounds per country
        r_min = np.maximum(self.min_ratio, base_ratios - self.delta)
        r_max = base_ratios + self.delta

        violations_low = current_ratios < r_min
        violations_high = current_ratios > r_max
        violations_flag = violations_low | violations_high

        # 1. Adjust under-ratio (nights too small compared to arrivals)
        # Pull nights up and slightly adjust arrivals to hit r_min
        if np.any(violations_low):
            target_r = r_min[violations_low]
            # Geometric mean adjustment to distribute correction smoothly
            geom = np.sqrt(y_arr_adj[violations_low] * y_nig_adj[violations_low])
            y_arr_adj[violations_low] = geom / np.sqrt(target_r)
            y_nig_adj[violations_low] = geom * np.sqrt(target_r)

        # 2. Adjust over-ratio (nights too large compared to arrivals)
        if np.any(violations_high):
            target_r = r_max[violations_high]
            geom = np.sqrt(y_arr_adj[violations_high] * y_nig_adj[violations_high])
            y_arr_adj[violations_high] = geom / np.sqrt(target_r)
            y_nig_adj[violations_high] = geom * np.sqrt(target_r)

        return (
            np.round(y_arr_adj).astype(np.float32),
            np.round(y_nig_adj).astype(np.float32),
            violations_flag,
        )
