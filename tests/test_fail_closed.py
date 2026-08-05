from __future__ import annotations

import unittest

from audits.r_law1_two_channel_origin_audit_v3_1 import audit_cert, template


class FailClosedAuditTests(unittest.TestCase):
    def test_two_channel_template_fails_closed(self) -> None:
        result = audit_cert(template())
        self.assertFalse(result["gate"])
        self.assertFalse(result["gates"]["plus_source_bound"])
        self.assertTrue(result["gates"]["channels_linearly_independent"])


if __name__ == "__main__":
    unittest.main()
