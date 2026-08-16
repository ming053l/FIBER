"""Spectrum estimation.

`certified` holds the operator used for discovery and reporting; `observability`
holds the teacher-output covariance, which is a diagnostic only (P0-1).
"""
from .certified import (CertifiedObservabilityOperator, CertifiedSpectrum, fit_certified,
                        project_operator, quadratic_form, subspace_certificate,
                        teacher_validity, variance_form, zero_tolerance)
from .observability import (Spectrum, fit_gram, fit_randomized, fit_spectrum,
                            subspace_alignment, trace_teacher_covariance)

__all__ = [
    # certified (P0-1): the operator that is a valid lower bound on C_obs
    "CertifiedObservabilityOperator", "CertifiedSpectrum", "fit_certified",
    "quadratic_form", "variance_form", "teacher_validity",
    "project_operator", "subspace_certificate", "zero_tolerance",
    # diagnostic only: Cov(f(Y)) == C_obs only for an exact conditional mean
    "Spectrum", "fit_spectrum", "fit_gram", "fit_randomized",
    "trace_teacher_covariance", "subspace_alignment",
]
