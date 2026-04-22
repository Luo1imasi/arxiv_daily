from typing import Any, Optional
import yaml
from pathlib import Path
import copy

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "default.yaml"
_CUSTOM_CONFIG_PATH = _CONFIG_DIR / "custom.yaml"
_default_config_cache: dict[str, Any] | None = None
_MISSING = object()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_override_config(default: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
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


def load_config() -> dict[str, Any]:
    config = get_default_config()

    if _CUSTOM_CONFIG_PATH.exists():
        with open(_CUSTOM_CONFIG_PATH, encoding="utf-8") as f:
            custom = yaml.safe_load(f)
        if custom:
            config = deep_merge(config, custom)

    return config


def get_default_config() -> dict[str, Any]:
    global _default_config_cache
    if _default_config_cache is None:
        with open(_DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            _default_config_cache = yaml.safe_load(f) or {}
    if _default_config_cache is None:
        return {}
    return copy.deepcopy(_default_config_cache)


def _get_nested_value(data: dict[str, Any], path: str, default: Any = _MISSING) -> Any:
    current = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_config_value(config: dict[str, Any], path: str, default: Any = _MISSING) -> Any:
    value = _get_nested_value(config, path, _MISSING)
    if value is not _MISSING:
        return value

    value = _get_nested_value(get_default_config(), path, _MISSING)
    if value is not _MISSING:
        return value

    if default is not _MISSING:
        return default
    raise KeyError(f"Missing config value: {path}")


def save_config(config: dict[str, Any], path: Optional[str] = None) -> str:
    target = Path(path) if path else _CUSTOM_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return str(target)
