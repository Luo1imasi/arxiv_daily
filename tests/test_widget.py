import json

import pytest
from starlette.requests import Request

from arxiv_daily import database as db
from arxiv_daily import main


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_widget_public_fields_and_allowed_origin(monkeypatch):
    async def dates():
        return ["2026-10-08"]

    async def papers(day):
        assert day == "2026-10-08"
        return [
            {
                "id": i,
                "title": f"Paper {i}",
                "tldr": "a" * 400,
                "judge_relevance": 0,
                "score": 8,
                "abstract": "private-field",
                "feedback_vote": "irrelevant",
            }
            for i in range(6)
        ]

    monkeypatch.setattr(db, "get_all_dates", dates)
    monkeypatch.setattr(db, "get_papers_by_date", papers)
    monkeypatch.setattr(main, "_app_config", {})
    request = Request(
        {"type": "http", "headers": [(b"origin", b"https://luolimasi.xyz")]}
    )
    response = await main.latest_widget(request, limit=3)
    data = json.loads(response.body)
    assert data["total"] == 6
    assert len(data["papers"]) == 3
    assert data["papers"][0]["score"] == 0
    assert data["papers"][0]["score_max"] == 5
    assert len(data["papers"][0]["tldr"]) == 320
    assert "private-field" not in response.body.decode()
    assert "feedback_vote" not in response.body.decode()
    assert response.headers["access-control-allow-origin"] == "https://luolimasi.xyz"
    assert response.headers["vary"] == "Origin"


@pytest.mark.anyio
async def test_empty_widget_and_untrusted_origin(monkeypatch):
    async def dates():
        return []

    monkeypatch.setattr(db, "get_all_dates", dates)
    monkeypatch.setattr(main, "_app_config", {})
    request = Request(
        {"type": "http", "headers": [(b"origin", b"https://other.example")]}
    )
    response = await main.latest_widget(request, limit=3)
    assert json.loads(response.body) == {"date": None, "total": 0, "papers": []}
    assert "access-control-allow-origin" not in response.headers
