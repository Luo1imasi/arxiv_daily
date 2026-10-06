import unittest
from types import SimpleNamespace

from arxiv_daily.llm import (
    LLMCallError,
    _create_completion,
    _extract_json_object,
    make_llm_cache_key,
    message_text,
    resolve_model,
)


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


def _status_error(status: int, message: str, headers: dict[str, str] | None = None):
    import httpx
    from openai import APIStatusError

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request, text=message)
    return APIStatusError(message, response=response, body={"error": message})


class MessageTextTests(unittest.TestCase):
    def test_empty_content_falls_back_to_reasoning_content(self):
        message = SimpleNamespace(
            content="",
            reasoning_content='{"tldr": "中文总结"}',
            model_extra=None,
        )

        self.assertEqual(message_text(message), '{"tldr": "中文总结"}')

    def test_visible_content_wins_over_reasoning(self):
        message = SimpleNamespace(
            content='{"tldr": "可见内容"}',
            reasoning_content="hidden",
            model_extra=None,
        )

        self.assertEqual(message_text(message), '{"tldr": "可见内容"}')


class ModelRoutingTests(unittest.TestCase):
    def test_cache_key_includes_model_role_and_prompt_version(self):
        config = {
            "llm": {
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "fallback",
                "language": "Chinese",
                "models": {
                    "extract": "deepseek-flash",
                    "summarize": "deepseek-flash",
                    "judge": "grok-4.7",
                },
            }
        }

        summarize_key = make_llm_cache_key(config, "summarize")
        judge_key = make_llm_cache_key(config, "judge")

        self.assertIn("deepseek-flash", summarize_key)
        self.assertIn("tldr-v2", summarize_key)
        self.assertIn("summarize", summarize_key)
        self.assertIn("grok-4.7", judge_key)
        self.assertIn("judge-v1", judge_key)
        self.assertNotEqual(summarize_key, judge_key)
        self.assertEqual(resolve_model(config, "profile"), "grok-4.7")
        self.assertEqual(resolve_model(config, "qa"), "deepseek-flash")


class CompletionRetryTests(unittest.TestCase):
    def _client(self, create):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_auth_failure_is_not_retried_inside_parameter_fallback(self):
        calls = {"count": 0}

        def create(**kwargs):
            del kwargs
            calls["count"] += 1
            raise _status_error(401, "invalid api key")

        with self.assertRaises(LLMCallError) as caught:
            _create_completion(
                self._client(create),
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                temperature=0,
                response_format={"type": "json_object"},
            )

        self.assertEqual(calls["count"], 1)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.status_code, 401)

    def test_rate_limit_exposes_retry_after_and_stops(self):
        calls = {"count": 0}

        def create(**kwargs):
            del kwargs
            calls["count"] += 1
            raise _status_error(429, "slow down", {"Retry-After": "7"})

        with self.assertRaises(LLMCallError) as caught:
            _create_completion(
                self._client(create),
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                temperature=0,
                response_format=None,
            )

        self.assertEqual(calls["count"], 1)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.retry_after, 7.0)

    def test_max_tokens_rejection_retries_with_max_completion_tokens(self):
        calls: list[str] = []

        def create(**kwargs):
            if "max_tokens" in kwargs:
                calls.append("max_tokens")
                raise _status_error(400, "Unsupported parameter: max_tokens")
            calls.append("max_completion_tokens")
            return SimpleNamespace()

        _create_completion(
            self._client(create),
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=32,
            temperature=0,
            response_format=None,
        )

        self.assertEqual(calls, ["max_tokens", "max_completion_tokens"])


if __name__ == "__main__":
    unittest.main()
