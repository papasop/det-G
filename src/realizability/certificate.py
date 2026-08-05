"""Source-bound zero-mode certificate data structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r_to_law1.source_binding import source_hash_matches


@dataclass(frozen=True)
class ZeroModeCertificate:
    schema_version: str
    protocol_sha256: str
    cost_source_path: str
    cost_source_sha256: str
    path_source_path: str
    path_source_sha256: str
    admissible_class_source_path: str
    admissible_class_source_sha256: str
    vacuum_sector_source_path: str
    vacuum_sector_source_sha256: str
    protocol_predeclared: bool
    cost_nonnegative: bool
    contraction_family_certified: bool
    zero_infimum_certified: bool
    path_nonconstant: bool
    zero_total_cost: bool
    local_zero_mode_positive_measure: bool
    same_meter_positive_control: bool
    witness_not_constructed_from_target_G: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZeroModeCertificate":
        return cls(
            schema_version=str(data.get("schema_version", "")),
            protocol_sha256=str(data.get("protocol_sha256", "")),
            cost_source_path=str(data.get("cost_source_path", "")),
            cost_source_sha256=str(data.get("cost_source_sha256", "")),
            path_source_path=str(data.get("path_source_path", "")),
            path_source_sha256=str(data.get("path_source_sha256", "")),
            admissible_class_source_path=str(
                data.get("admissible_class_source_path", "")
            ),
            admissible_class_source_sha256=str(
                data.get("admissible_class_source_sha256", "")
            ),
            vacuum_sector_source_path=str(data.get("vacuum_sector_source_path", "")),
            vacuum_sector_source_sha256=str(
                data.get("vacuum_sector_source_sha256", "")
            ),
            protocol_predeclared=bool(data.get("protocol_predeclared", False)),
            cost_nonnegative=bool(data.get("cost_nonnegative", False)),
            contraction_family_certified=bool(
                data.get("contraction_family_certified", False)
            ),
            zero_infimum_certified=bool(data.get("zero_infimum_certified", False)),
            path_nonconstant=bool(data.get("path_nonconstant", False)),
            zero_total_cost=bool(data.get("zero_total_cost", False)),
            local_zero_mode_positive_measure=bool(
                data.get("local_zero_mode_positive_measure", False)
            ),
            same_meter_positive_control=bool(
                data.get("same_meter_positive_control", False)
            ),
            witness_not_constructed_from_target_G=bool(
                data.get("witness_not_constructed_from_target_G", False)
            ),
        )


def load_zero_mode_certificate(path: str | Path) -> ZeroModeCertificate:
    import json

    return ZeroModeCertificate.from_dict(json.loads(Path(path).read_text()))


def source_bound_gates(
    certificate: ZeroModeCertificate,
    *,
    base_dir: str | Path,
) -> dict[str, bool]:
    return {
        "cost_source_bound": source_hash_matches(
            {
                "path": certificate.cost_source_path,
                "sha256": certificate.cost_source_sha256,
            },
            path_key="path",
            hash_key="sha256",
            base_dir=base_dir,
        ),
        "path_source_bound": source_hash_matches(
            {
                "path": certificate.path_source_path,
                "sha256": certificate.path_source_sha256,
            },
            path_key="path",
            hash_key="sha256",
            base_dir=base_dir,
        ),
        "admissible_class_source_bound": source_hash_matches(
            {
                "path": certificate.admissible_class_source_path,
                "sha256": certificate.admissible_class_source_sha256,
            },
            path_key="path",
            hash_key="sha256",
            base_dir=base_dir,
        ),
        "vacuum_sector_source_bound": source_hash_matches(
            {
                "path": certificate.vacuum_sector_source_path,
                "sha256": certificate.vacuum_sector_source_sha256,
            },
            path_key="path",
            hash_key="sha256",
            base_dir=base_dir,
        ),
    }
