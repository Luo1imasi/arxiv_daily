import json

import pytest
from starlette.requests import Request

from arxiv_daily import database as db
from arxiv_daily import main


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_terminal_search_and_private_fields(tmp_path, monkeypatch):
    path = str(tmp_path / "test.db")
    await db.init_db(path)
    async with db._connect(path) as conn:
        for index, title in enumerate(
            ["Humanoid control", "Other research", "Humanoid planning"]
        ):
            await conn.execute(
                "INSERT INTO papers (date, source, title, authors, url, pdf_url, abstract, tldr) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "2026-10-08",
                    "arxiv",
                    title,
                    '["Author"]',
                    f"https://arxiv.org/abs/2601.1234{index}",
                    f"https://arxiv.org/pdf/2601.1234{index}",
                    "Abstract",
                    "Summary",
                ),
            )
        await conn.commit()
    result = await db.search_recommended_papers("HUMANOID", 1, path)
    assert len(result) == 1
    assert result[0]["matched_count"] == 2
    assert await db.search_recommended_papers("%", 10, path) == []

    async def dates():
        return ["2026-10-08"]

    search_original = db.search_recommended_papers

    async def search(query, limit):
        return await search_original(query, limit, path)

    monkeypatch.setattr(db, "get_all_dates", dates)
    monkeypatch.setattr(db, "search_recommended_papers", search)
    request = Request(
        {"type": "http", "headers": [(b"origin", b"https://luolimasi.xyz")]}
    )
    response = await main.terminal_papers(request, date=None, q="humanoid", limit=1)
    data = json.loads(response.body)
    assert data["total"] == 2
    assert data["papers"][0]["authors"] == ["Author"]
    assert data["papers"][0]["pdf_url"].startswith("https://arxiv.org/pdf/")
    assert "feedback_vote" not in response.body.decode()
    assert response.headers["access-control-allow-origin"] == "https://luolimasi.xyz"
    missing = await main.terminal_papers(request, date="2000-01-01", q=None, limit=5)
    assert missing.status_code == 404
    invalid = await main.terminal_papers(request, date=None, q="   ", limit=5)
    assert invalid.status_code == 400
