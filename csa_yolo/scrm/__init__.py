"""Safety Compliance Reasoning Module (SCRM).

Determines worker-level compliance status based on confirmed PPE status
and configurable safety rules.
"""

from .compliance import ComplianceEngine, ComplianceResult

__all__ = ["ComplianceEngine", "ComplianceResult"]
