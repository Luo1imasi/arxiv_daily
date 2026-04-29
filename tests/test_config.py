import tempfile
import unittest

import yaml

from arxiv_daily.config import build_override_config, get_default_config, save_config


class BuildOverrideConfigTests(unittest.TestCase):
    def test_only_persists_values_different_from_default(self):
        default = {
            "webdav": {"url": "", "path": "/papers/"},
            "executor": {"schedule_hour": 8, "schedule_minute": 0},
        }
        current = {
            "webdav": {"url": "https://dav.example", "path": "/papers/"},
            "executor": {"schedule_hour": 9, "schedule_minute": 0},
        }

        self.assertEqual(
            build_override_config(default, current),
            {
                "webdav": {"url": "https://dav.example"},
                "executor": {"schedule_hour": 9},
            },
        )


class SaveConfigTests(unittest.TestCase):
    def test_save_config_writes_only_override_payload(self):
        overrides = {
            "webdav": {"url": "https://dav.example"},
            "executor": {"schedule_hour": 9},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_config(overrides, f"{tmpdir}/custom.yaml")
            with open(path, encoding="utf-8") as file:
                saved = yaml.safe_load(file)

        self.assertEqual(saved, overrides)


class DefaultConfigTests(unittest.TestCase):
    def test_executor_worker_defaults_match_used_keys(self):
        executor = get_default_config()["executor"]

        self.assertIn("corpus_workers", executor)
        self.assertIn("llm_workers", executor)
        self.assertIn("retriever_workers", executor)
        self.assertIn("arxiv_id_batch_size", executor)

    def test_default_max_paper_num_matches_ui_fallback(self):
        config = get_default_config()

        self.assertEqual(config["executor"]["max_paper_num"], 20)

    def test_default_llm_concurrency_limit_exists(self):
        config = get_default_config()

        self.assertEqual(config["llm"]["max_concurrent_requests"], 2)


if __name__ == "__main__":
    unittest.main()
