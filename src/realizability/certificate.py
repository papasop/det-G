"""Source-bound zero-mode certificate data structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r_to_law1.source_binding import source_hash_matches

SCHEMA_VERSION = "zero-mode-certificate-v0.1.0"
HEX_DIGITS = set("0123456789abcdef")
BOOLEAN_FIELDS = {
    "protocol_predeclared",
    "cost_nonnegative",
    "contraction_family_certified",
    "zero_infimum_certified",
    "path_nonconstant",
    "zero_total_cost",
    "local_zero_mode_positive_measure",
    "same_meter_positive_control",
    "witness_not_constructed_from_target_G",
}
SHA_FIELDS = {
    "protocol_sha256",
    "cost_source_sha256",
    "path_source_sha256",
    "admissible_class_source_sha256",
    "vacuum_sector_source_sha256",
    "path_data_source_sha256",
}
PATH_FIELDS = {
    "cost_source_path",
    "path_source_path",
    "admissible_class_source_path",
    "vacuum_sector_source_path",
    "path_data_source_path",
}


class ZeroModeCertificateError(ValueError):
    """Raised when a zero-mode certificate is malformed."""


def _require_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ZeroModeCertificateError(
            f"certificate field {field!r} must be a nonempty string"
        )
    return value


def _require_sha256(data: dict[str, Any], field: str) -> str:
    value = _require_string(data, field)
    if len(value) != 64 or any(character not in HEX_DIGITS for character in value):
        raise ZeroModeCertificateError(
            f"certificate field {field!r} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_bool(data: dict[str, Any], field: str) -> bool:
    value = data.get(field)
    if type(value) is not bool:
        raise ZeroModeCertificateError(
            f"certificate field {field!r} must be a JSON boolean"
        )
    return value


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
    path_data_source_path: str
    path_data_source_sha256: str
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
        if not isinstance(data, dict):
            raise ZeroModeCertificateError("certificate must be a JSON object")
        schema_version = _require_string(data, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ZeroModeCertificateError(
                f"unsupported zero-mode certificate schema_version: {schema_version!r}"
            )
        for field in PATH_FIELDS:
            _require_string(data, field)
        for field in SHA_FIELDS:
            _require_sha256(data, field)
        for field in BOOLEAN_FIELDS:
            _require_bool(data, field)
        return cls(
            schema_version=schema_version,
            protocol_sha256=data["protocol_sha256"],
            cost_source_path=data["cost_source_path"],
            cost_source_sha256=data["cost_source_sha256"],
            path_source_path=data["path_source_path"],
            path_source_sha256=data["path_source_sha256"],
            admissible_class_source_path=data["admissible_class_source_path"],
            admissible_class_source_sha256=data["admissible_class_source_sha256"],
            vacuum_sector_source_path=data["vacuum_sector_source_path"],
            vacuum_sector_source_sha256=data["vacuum_sector_source_sha256"],
            path_data_source_path=data["path_data_source_path"],
            path_data_source_sha256=data["path_data_source_sha256"],
            protocol_predeclared=data["protocol_predeclared"],
            cost_nonnegative=data["cost_nonnegative"],
            contraction_family_certified=data["contraction_family_certified"],
            zero_infimum_certified=data["zero_infimum_certified"],
            path_nonconstant=data["path_nonconstant"],
            zero_total_cost=data["zero_total_cost"],
            local_zero_mode_positive_measure=data["local_zero_mode_positive_measure"],
            same_meter_positive_control=data["same_meter_positive_control"],
            witness_not_constructed_from_target_G=data[
                "witness_not_constructed_from_target_G"
            ],
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
        "path_data_source_bound": source_hash_matches(
            {
                "path": certificate.path_data_source_path,
                "sha256": certificate.path_data_source_sha256,
            },
            path_key="path",
            hash_key="sha256",
            base_dir=base_dir,
        ),
    }
