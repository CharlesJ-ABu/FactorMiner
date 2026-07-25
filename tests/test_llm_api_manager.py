import unittest

from core.miner.paradigms.llm_api_manager import LLMAPIManager


class LLMAPIManagerRedactionTests(unittest.TestCase):
    def test_diagnostics_remove_configured_and_common_credentials(self):
        manager = object.__new__(LLMAPIManager)
        configured_key = "sk-" + "live-configured-secret"
        other_key = "sk-" + "other-secret"
        manager.api_keys = [configured_key]

        redacted = manager._redact_sensitive_text(
            f'Bearer abc.def {other_key} '
            '"api_key":"plain-secret" '
            f"configured={configured_key}"
        )

        self.assertNotIn("abc.def", redacted)
        self.assertNotIn(other_key, redacted)
        self.assertNotIn("plain-secret", redacted)
        self.assertNotIn(configured_key, redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_diagnostics_are_bounded_and_single_line(self):
        manager = object.__new__(LLMAPIManager)
        manager.api_keys = []
        redacted = manager._redact_sensitive_text("line 1\n" + ("x" * 1000), 80)
        self.assertEqual(len(redacted), 80)
        self.assertNotIn("\n", redacted)


if __name__ == "__main__":
    unittest.main()
