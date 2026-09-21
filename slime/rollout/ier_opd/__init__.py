"""Rollout-side IER metrics for budgeted OPD token selection.

``compute_ier_score`` exposes the reverse-KL Fisher SNR used by the ``ier``
selector. ``compute_position`` also supplies diagnostics for IER rank fusion.
"""

from .ier_metrics import compute_ier_score, compute_position

__all__ = [
    "compute_ier_score",
    "compute_position",
]
