"""Registration harness exports."""

from __future__ import annotations

from langslice_harness.harness.registration.agent import build_registration_review_agent
from langslice_harness.harness.registration.image_gen_registration import (
    generate_registration_candidate,
)
from langslice_harness.harness.registration.runner import run_registration_review_session
from langslice_harness.harness.registration.types import (
    GeneratedSegmentation,
    RegistrationCandidate,
    candidate_to_registration_result,
)

__all__ = [
    "GeneratedSegmentation",
    "RegistrationCandidate",
    "build_registration_review_agent",
    "candidate_to_registration_result",
    "generate_registration_candidate",
    "run_registration_review_session",
]
