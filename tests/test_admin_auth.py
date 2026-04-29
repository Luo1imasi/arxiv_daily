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
