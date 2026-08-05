"""SUPERSEDED TRANSITIONAL AUDIT.

Law III selects Bc=JG-dcI; the native two-line structure is
{ker Bc, Im Bc}, not two response-composed A± branches.

This archived file records the corrected object boundary. It is not part of
the public R-to-Law-I evidence chain.
"""

from __future__ import annotations

import json


def main() -> None:
    print(
        json.dumps(
            {
                "title": "Archived v3.7 transitional audit",
                "status": "SUPERSEDED",
                "superseded_by": "audits/downstream_k1/k1_self_contained_nullflow_theorem_audit_v3_8.py",
                "corrected_native_structure": "N(G)=ker(B_c) union Im(B_c), B_c=JG-d_cI",
                "claim_boundary": (
                    "The native K=1 object is the kernel/image pair of B_c, "
                    "not two response-composed A+/- branches."
                ),
                "all_scientific_gates_pass": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
