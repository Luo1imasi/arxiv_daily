from typing import Optional
import yaml
from pathlib import Path
import copy

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "default.yaml"
_CUSTOM_CONFIG_PATH = _CONFIG_DIR / "custom.yaml"
_DEFAULT_CONFIG_CACHE: dict | None = None
_MISSING = object()


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_override_config(default: dict, current: dict) -> dict:
    result = {}
    for key, value in current.items():
        if key not in default:
            result[key] = copy.deepcopy(value)
            continue

        default_value = default[key]
        if isinstance(default_value, dict) and isinstance(value, dict):
            nested = build_override_config(default_value, value)
            if nested:
                result[key] = nested
        elif value != default_value:
            result[key] = copy.deepcopy(value)

    return result


def load_config() -> dict:
    config = get_default_config()

    if _CUSTOM_CONFIG_PATH.exists():
        with open(_CUSTOM_CONFIG_PATH, encoding="utf-8") as f:
            custom = yaml.safe_load(f)
        if custom:
            config = deep_merge(config, custom)

    return config


def get_default_config() -> dict:
    global _DEFAULT_CONFIG_CACHE
    if _DEFAULT_CONFIG_CACHE is None:
        with open(_DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            _DEFAULT_CONFIG_CACHE = yaml.safe_load(f) or {}
    return copy.deepcopy(_DEFAULT_CONFIG_CACHE)


def _get_nested_value(data: dict, path: str, default=_MISSING):
    current = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_config_value(config: dict, path: str, default=_MISSING):
    value = _get_nested_value(config, path, _MISSING)
    if value is not _MISSING:
        return value

    value = _get_nested_value(get_default_config(), path, _MISSING)
    if value is not _MISSING:
        return value

    if default is not _MISSING:
        return default
    raise KeyError(f"Missing config value: {path}")


def save_config(config: dict, path: Optional[str] = None) -> str:
    target = Path(path) if path else _CUSTOM_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return str(target)
