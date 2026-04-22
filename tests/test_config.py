import tempfile
import unittest

import yaml

from arxiv_daily.config import build_override_config, save_config


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


if __name__ == "__main__":
    unittest.main()
