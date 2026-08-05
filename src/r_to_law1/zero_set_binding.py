"""Bidirectional zero-set binding certificate interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .source_binding import source_hash_matches
from .protocol import threshold


@dataclass(frozen=True)
class ZeroSetBindingCertificate:
    physical_cost_source_path: str
    physical_cost_source_hash: str
    signed_representative_source_path: str
    signed_representative_source_hash: str
    selected_plane: str
    domain: str
    mapping_frozen_before_outcomes: bool
    sources_independent: bool
    forward_inclusion: bool
    reverse_inclusion: bool
    completeness_scope: str
    forward_violation_rate: float = 0.0
    reverse_violation_rate: float = 0.0
    construction_kind: str = "independent"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZeroSetBindingCertificate":
        return cls(
            physical_cost_source_path=str(data.get("physical_cost_source_path", "")),
            physical_cost_source_hash=str(data.get("physical_cost_source_hash", "")),
            signed_representative_source_path=str(
                data.get("signed_representative_source_path", "")
            ),
            signed_representative_source_hash=str(
                data.get("signed_representative_source_hash", "")
            ),
            selected_plane=str(data.get("selected_plane", "")),
            domain=str(data.get("domain", "")),
            mapping_frozen_before_outcomes=bool(
                data.get("mapping_frozen_before_outcomes", False)
            ),
            sources_independent=bool(data.get("sources_independent", False)),
            forward_inclusion=bool(data.get("forward_inclusion", False)),
            reverse_inclusion=bool(data.get("reverse_inclusion", False)),
            forward_violation_rate=float(data.get("forward_violation_rate", 0.0)),
            reverse_violation_rate=float(data.get("reverse_violation_rate", 0.0)),
            completeness_scope=str(data.get("completeness_scope", "")),
            construction_kind=str(data.get("construction_kind", "independent")),
        )


def _record(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def is_circular_negative_control(certificate: ZeroSetBindingCertificate) -> bool:
    normalized = certificate.construction_kind.strip().lower()
    return normalized in {
        "f_abs_q",
        "abs_q",
        "f=q_squared",
        "q_squared",
        "q_g_squared",
        "target_g_constructed",
    }


def audit_zero_set_binding_certificate(
    certificate: ZeroSetBindingCertificate | dict[str, Any] | None,
    *,
    base_dir: str | Path = ".",
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if certificate is None:
        certificate = {}
    if isinstance(certificate, dict):
        certificate = ZeroSetBindingCertificate.from_dict(certificate)

    circular = is_circular_negative_control(certificate)
    physical_source_bound = source_hash_matches(
        _record(
            certificate.physical_cost_source_path,
            certificate.physical_cost_source_hash,
        ),
        path_key="path",
        hash_key="sha256",
        base_dir=base_dir,
    )
    signed_source_bound = source_hash_matches(
        _record(
            certificate.signed_representative_source_path,
            certificate.signed_representative_source_hash,
        ),
        path_key="path",
        hash_key="sha256",
        base_dir=base_dir,
    )
    if protocol is None:
        forward_tol = 0.0
        reverse_tol = 0.0
    else:
        forward_tol = threshold(protocol, "zero_set_forward_violation_tol")
        reverse_tol = threshold(protocol, "zero_set_reverse_violation_tol")
    forward_gate = (
        bool(certificate.forward_inclusion)
        and certificate.forward_violation_rate <= forward_tol
    )
    reverse_gate = (
        bool(certificate.reverse_inclusion)
        and certificate.reverse_violation_rate <= reverse_tol
    )
    gates = {
        "physical_cost_source_bound": physical_source_bound,
        "signed_representative_source_bound": signed_source_bound,
        "sources_independent": bool(certificate.sources_independent) and not circular,
        "mapping_frozen_before_outcomes": bool(
            certificate.mapping_frozen_before_outcomes
        ),
        "selected_plane_declared": bool(certificate.selected_plane),
        "domain_declared": bool(certificate.domain),
        "completeness_scope_declared": bool(certificate.completeness_scope),
        "forward_inclusion_ZF_subset_Zq": forward_gate,
        "reverse_inclusion_Zq_subset_ZF": reverse_gate,
    }
    equivalence = all(gates.values())
    return {
        "gates": gates,
        "forward_inclusion_certified": forward_gate and physical_source_bound,
        "reverse_inclusion_certified": reverse_gate and signed_source_bound,
        "zero_set_equivalence_certified": equivalence,
        "circular_negative_control": circular,
        "gate": equivalence,
    }
