import unittest

from arxiv_daily.llm import _extract_json_object


class ExtractJsonObjectTests(unittest.TestCase):
    def test_extracts_plain_json_object(self):
        self.assertEqual(
            _extract_json_object('{"tldr": "A short summary."}'),
            {"tldr": "A short summary."},
        )

    def test_extracts_json_inside_code_fence(self):
        self.assertEqual(
            _extract_json_object('```json\n{"tldr": "A short summary."}\n```'),
            {"tldr": "A short summary."},
        )

    def test_extracts_json_with_prefix_and_suffix_text(self):
        self.assertEqual(
            _extract_json_object(
                'Here is the result:\n{"tldr": "A short summary."}\nThank you.'
            ),
            {"tldr": "A short summary."},
        )

    def test_ignores_braces_inside_string_values(self):
        self.assertEqual(
            _extract_json_object('{"tldr": "Uses {tokens} safely in output."}'),
            {"tldr": "Uses {tokens} safely in output."},
        )


if __name__ == "__main__":
    unittest.main()
