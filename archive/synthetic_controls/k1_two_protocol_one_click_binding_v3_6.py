"""SUPERSEDED SYNTHETIC CONTROL.

Its kernel-pair construction is not the native K=1 null-flow theorem.

This archived file is retained only to classify the old evidence level:
SYNTHETIC_CONTROL. It may be useful for checking that an audit can recognize a
model-internal double cone, but it is not evidence for a native Principle-R or
K=1 physical zero-set binding.
"""

from __future__ import annotations

import json


def main() -> None:
    print(
        json.dumps(
            {
                "title": "Archived v3.6 synthetic control",
                "status": "SYNTHETIC_CONTROL",
                "superseded_by": "audits/downstream_k1/k1_self_contained_nullflow_theorem_audit_v3_8.py",
                "claim_boundary": (
                    "The old kernel-pair construction is not the native K=1 "
                    "null-flow theorem and is not physical binding evidence."
                ),
                "all_scientific_gates_pass": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
