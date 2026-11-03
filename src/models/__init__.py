"""
Models module for the RA-HE-BiLSTM Architecture.
"""

from src.models.regime_hmm import RegimeHMMNavigator
from src.models.catboost_regime import CatBoostRegimeForecaster
from src.models.m_ra_bilstm import MRABiLSTM, CoupledConsistencyLoss
from src.models.ra_exo_dlinear import (
    RAExoDLinearNet,
    RAExoDLinearForecaster,
    VanillaDLinearNet,
    CoupledConsistencyLinearLoss,
)
from src.models.trmf_spatial import TRMFSpatialSpillover
from src.models.dcqf_conformal import (
    DeepConformalQuantileForecaster,
    ResidualConformalForecaster,
)
from src.models.moe_gating import DynamicMoEGatingNetwork
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.transformer_baselines import (
    PatchTSTForecaster,
    iTransformerForecaster,
    TimesNetForecaster,
)

__all__ = [
    "RegimeHMMNavigator",
    "CatBoostRegimeForecaster",
    "MRABiLSTM",
    "CoupledConsistencyLoss",
    "RAExoDLinearNet",
    "RAExoDLinearForecaster",
    "VanillaDLinearNet",
    "CoupledConsistencyLinearLoss",
    "TRMFSpatialSpillover",
    "DeepConformalQuantileForecaster",
    "ResidualConformalForecaster",
    "DynamicMoEGatingNetwork",
    "CoupledConsistencyOptimizer",
    "PatchTSTForecaster",
    "iTransformerForecaster",
    "TimesNetForecaster",
]

