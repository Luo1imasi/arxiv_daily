import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from arxiv_daily import main


def make_request(password: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if password is not None:
        headers.append((b"x-admin-password", password.encode()))
    return Request({"type": "http", "method": "POST", "headers": headers})


@pytest.mark.anyio
async def test_admin_password_rejects_missing_password(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(main.ADMIN_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException) as exc_info:
        await main._require_admin_password(make_request())

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_admin_password_rejects_wrong_password(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(main.ADMIN_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException) as exc_info:
        await main._require_admin_password(make_request("wrong"))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_admin_password_accepts_env_password(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(main.ADMIN_PASSWORD_ENV, "secret")

    await main._require_admin_password(make_request("secret"))


@pytest.mark.anyio
async def test_admin_password_falls_back_to_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(main.ADMIN_PASSWORD_ENV, raising=False)
    monkeypatch.setattr(main, "_app_config", {"server": {"admin_password": "configured"}})

    await main._require_admin_password(make_request("configured"))


def test_public_config_hides_secrets_and_paths():
    view = main.public_config(
        {
            "webdav": {"local_path": "/var/private/library", "url": "https://dav.example", "password": "secret"},
            "llm": {
                "api_key": "secret-key",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "deepseek-flash",
                "language": "Chinese",
                "models": {"extract": "deepseek-flash", "summarize": "deepseek-flash", "judge": "grok-4.7"},
            },
            "source": {"arxiv": {"category": ["cs.AI"], "include_cross_list": False}},
            "executor": {"max_paper_num": 10, "timezone": "Asia/Shanghai", "schedule_hour": 8, "schedule_minute": 0},
            "reranker": {"model": "jina"},
            "server": {"admin_password": "operator-secret"},
        }
    )
    blob = json.dumps(view)

    assert "secret-key" not in blob
    assert "operator-secret" not in blob
    assert "/var/private/library" not in blob
    assert "8080" not in blob
    assert view["llm"]["configured"] is True
    assert view["llm"]["models"]["judge"] == "grok-4.7"


def test_admin_config_keeps_paths_and_drops_secrets():
    view = main.admin_config_view(
        {
            "webdav": {"local_path": "/var/private/library", "password": "secret"},
            "llm": {"api_key": "secret-key", "base_url": "http://127.0.0.1:8080/v1"},
            "server": {"admin_password": "configured"},
        }
    )

    assert view["webdav"]["local_path"] == "/var/private/library"
    assert "password" not in view["webdav"]
    assert view["webdav"]["password_set"] is True
    assert "api_key" not in view["llm"]
    assert view["llm"]["api_key_set"] is True
    assert view["llm"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert "admin_password" not in view["server"]
