"""Construction Risk Assessment Module (CRAM).

Transparent, rule-based risk scoring based on PPE violations and temporal state.
THIS IS A PROTOTYPE RULE-BASED MODEL, NOT A MACHINE-LEARNING RISK PREDICTOR.
"""

from .risk_assessment import RiskAssessmentEngine, RiskResult

__all__ = ["RiskAssessmentEngine", "RiskResult"]
