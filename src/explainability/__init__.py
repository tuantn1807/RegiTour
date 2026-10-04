"""
Explainable AI (XAI) Package for Regime-Aware Tourism Econometrics.
Includes:
- Feature Attribution (TreeSHAP, Permutation Importance)
- Cross-Temporal Attention Map Extractor & Visualizer
- Counterfactual What-If Policy Simulation Engine
"""

from src.explainability.feature_attribution import (
    FeatureAttributionAnalyzer,
    compute_tree_shap,
    compute_permutation_importance,
)
from src.explainability.attention_visualizer import (
    AttentionVisualizer,
    extract_cross_attention_weights,
)
from src.explainability.counterfactual_engine import (
    CounterfactualSimulator,
    simulate_hicp_policy_scenarios,
)

__all__ = [
    "FeatureAttributionAnalyzer",
    "compute_tree_shap",
    "compute_permutation_importance",
    "AttentionVisualizer",
    "extract_cross_attention_weights",
    "CounterfactualSimulator",
    "simulate_hicp_policy_scenarios",
]
