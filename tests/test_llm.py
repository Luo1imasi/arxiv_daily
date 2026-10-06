import unittest
from types import SimpleNamespace

from unittest.mock import patch

from arxiv_daily.llm import (
    LLMCallError,
    _call_json,
    _create_completion,
    _extract_json_object,
    _extract_json_value,
    _record_usage,
    generate_tldrs_batch,
    get_llm_usage,
    make_llm_cache_key,
    message_text,
    reset_llm_usage,
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

    def test_strips_think_block_and_mid_text_code_fence(self):
        content = (
            '<think>We need {"tldr": "decoy"}</think>\n'
            '结果如下：\n```json\n{"tldr": "中文总结"}\n```\n完毕'
        )
        self.assertEqual(_extract_json_object(content), {"tldr": "中文总结"})

    def test_skips_broken_prefix_and_reads_first_valid_object(self):
        self.assertEqual(
            _extract_json_object('We need {not json}\n{"tldr": "好的总结"} trailing'),
            {"tldr": "好的总结"},
        )

    def test_extracts_first_complete_array(self):
        self.assertEqual(
            _extract_json_value('note [{"id": "a", "tldr": "甲"}] {"id": "b"}'),
            [{"id": "a", "tldr": "甲"}],
        )

    def test_object_extractor_skips_a_leading_array(self):
        self.assertEqual(
            _extract_json_object('note [1, 2] {"tldr": "后面的对象"}'),
            {"tldr": "后面的对象"},
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
        self.assertIn("tldr-v3", summarize_key)
        self.assertIn("summarize", summarize_key)
        self.assertIn("grok-4.7", judge_key)
        self.assertIn("judge-v1", judge_key)
        self.assertNotEqual(summarize_key, judge_key)
        self.assertEqual(resolve_model(config, "profile"), "grok-4.7")
        self.assertEqual(resolve_model(config, "qa"), "deepseek-flash")
        self.assertEqual(resolve_model(config, "coarse"), "fallback")
        config["llm"]["models"]["coarse"] = "grok-4.3"
        config["llm"]["models"]["profile"] = "grok-4.6"
        self.assertEqual(resolve_model(config, "coarse"), "grok-4.3")
        self.assertEqual(resolve_model(config, "profile"), "grok-4.6")


class UsageAccountingTests(unittest.TestCase):
    def test_usage_is_split_by_model(self):
        reset_llm_usage()
        _record_usage(3, 4, "grok-4.3")
        _record_usage(5, 6, "grok-4.7")

        usage = get_llm_usage()

        self.assertEqual(usage["llm_prompt_tokens"], 8)
        self.assertEqual(usage["llm_completion_tokens"], 10)
        self.assertEqual(usage["llm_api_requests"], 2)
        self.assertEqual(usage["llm_usage_by_model"]["grok-4.3"]["completion_tokens"], 4)
        self.assertEqual(usage["llm_usage_by_model"]["grok-4.7"]["prompt_tokens"], 5)


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


class CallJsonTests(unittest.TestCase):
    def test_requests_json_object_and_unwraps_think_fence(self):
        captured = {}

        def fake(config, messages, max_tokens=4096, *, role, temperature, response_format):
            del config, messages, max_tokens, role
            captured["temperature"] = temperature
            captured["response_format"] = response_format
            return '<think>We need to answer</think>\n```json\n{"tldr": "中文总结"}\n```'

        with patch("arxiv_daily.llm._call_llm_api_with_retry", side_effect=fake):
            parsed = _call_json(
                {},
                role="summarize",
                system="只返回 JSON",
                user="写短评",
                max_tokens=32,
            )

        self.assertEqual(parsed, {"tldr": "中文总结"})
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(captured["temperature"], 0)

    def test_wraps_a_bare_json_array_as_items(self):
        def fake(config, messages, max_tokens=4096, *, role, temperature, response_format):
            del config, messages, max_tokens, role, temperature, response_format
            return '[{"id": "http://example.test/a", "tldr": "甲"}]'

        with patch("arxiv_daily.llm._call_llm_api_with_retry", side_effect=fake):
            parsed = _call_json({}, role="summarize", system="s", user="u", max_tokens=32)

        self.assertEqual(parsed, {"items": [{"id": "http://example.test/a", "tldr": "甲"}]})


def _tldr_paper(url: str):
    return SimpleNamespace(title=f"Title {url}", abstract="Abstract", url=url)


def _ids(prompt: str) -> list[str]:
    import re

    return re.findall(r"(?m)^id: (\S+)", prompt)


class BatchTldrRetryTests(unittest.TestCase):
    def test_failed_batch_retries_each_paper_once(self):
        papers = [_tldr_paper("http://example.test/a"), _tldr_paper("http://example.test/b")]
        calls: list[str] = []

        def fake(config, *, role, system, user, max_tokens):
            del config, role, system, max_tokens
            calls.append(user)
            ids = _ids(user)
            if len(ids) != 1:
                return None
            return {
                "items": [
                    {
                        "id": ids[0],
                        "tldr": "这是一篇中文短评",
                        "method": "方法",
                        "evidence": "证据",
                        "why_for_me": "相关",
                    }
                ]
            }

        with patch("arxiv_daily.llm._call_json", side_effect=fake):
            result = generate_tldrs_batch(papers, None, {"llm": {"language": "Chinese"}})

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(set(result), {"http://example.test/a", "http://example.test/b"})
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(_ids(calls[0])), 2)
        self.assertEqual([len(_ids(call)) for call in calls[1:]], [1, 1])

    def test_partial_batch_retries_only_missing_paper(self):
        papers = [_tldr_paper("http://example.test/a"), _tldr_paper("http://example.test/b")]
        calls: list[list[str]] = []

        def fake(config, *, role, system, user, max_tokens):
            del config, role, system, max_tokens
            ids = _ids(user)
            calls.append(ids)
            if len(ids) > 1:
                return {
                    "items": [
                        {
                            "id": ids[0],
                            "tldr": "第一篇中文短评",
                            "method": "方法",
                            "evidence": "证据",
                            "why_for_me": "相关",
                        }
                    ]
                }
            return {
                "items": [
                    {
                        "id": ids[0],
                        "tldr": "第二篇中文短评",
                        "method": "方法",
                        "evidence": "证据",
                        "why_for_me": "相关",
                    }
                ]
            }

        with patch("arxiv_daily.llm._call_json", side_effect=fake):
            result = generate_tldrs_batch(papers, None, {"llm": {"language": "Chinese"}})

        self.assertEqual(calls, [
            ["http://example.test/a", "http://example.test/b"],
            ["http://example.test/b"],
        ])
        assert result is not None
        self.assertEqual(result["http://example.test/a"]["tldr"], "第一篇中文短评")
        self.assertEqual(result["http://example.test/b"]["tldr"], "第二篇中文短评")

    def test_complete_batch_does_not_retry(self):
        papers = [_tldr_paper("http://example.test/a"), _tldr_paper("http://example.test/b")]
        calls = {"count": 0}

        def fake(config, *, role, system, user, max_tokens):
            del config, role, system, max_tokens
            calls["count"] += 1
            return {
                "items": [
                    {
                        "id": item_id,
                        "tldr": "中文短评",
                        "method": "方法",
                        "evidence": "证据",
                        "why_for_me": "相关",
                    }
                    for item_id in _ids(user)
                ]
            }

        with patch("arxiv_daily.llm._call_json", side_effect=fake):
            result = generate_tldrs_batch(papers, None, {"llm": {"language": "Chinese"}})

        self.assertEqual(calls["count"], 1)
        assert result is not None
        self.assertEqual(len(result), 2)

    def test_batch_token_budget_grows_with_paper_count(self):
        from arxiv_daily.llm import _tldr_max_tokens

        self.assertLess(_tldr_max_tokens(1), _tldr_max_tokens(10))
        self.assertEqual(_tldr_max_tokens(10), 3500)
        self.assertEqual(_tldr_max_tokens(40), 4500)

    def test_single_paper_failure_is_not_retried_inside_batch(self):
        calls = {"count": 0}

        def fake(config, *, role, system, user, max_tokens):
            del config, role, system, user, max_tokens
            calls["count"] += 1
            return None

        with patch("arxiv_daily.llm._call_json", side_effect=fake):
            result = generate_tldrs_batch(
                [_tldr_paper("http://example.test/a")],
                None,
                {"llm": {"language": "Chinese"}},
            )

        self.assertIsNone(result)
        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
