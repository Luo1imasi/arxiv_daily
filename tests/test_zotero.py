import hashlib
import json
import zipfile
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

import pytest

from arxiv_daily import database as db
from arxiv_daily import main, zotero
from arxiv_daily.protocol import CorpusPaper
from tests.test_admin_auth import make_request


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def export_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("ARXIV_DAILY_ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ARXIV_DAILY_ZOTERO_USER_ID", raising=False)
    storage = tmp_path / "webdav"
    storage.mkdir()
    config = {
        "zotero": {"user_id": "123", "api_key": "secret"},
        "webdav": {"local_path": str(storage)},
    }
    paper = {
        "title": "A paper",
        "abstract": "An abstract",
        "authors": json.dumps(["A. Author"]),
        "url": "https://arxiv.org/abs/2601.12345v2",
    }
    items = {}

    class Client:
        def __init__(self, cfg):
            self.session = type("Session", (), {"close": lambda self: None})()

        def find_paper(self, identifier):
            return None

        def ensure_item(self, data):
            items.setdefault(data["key"], data)

        def finish_attachment(self, key, md5, mtime):
            items[key].update(md5=md5, mtime=mtime)

        def prepare_parent(self, key, collection):
            pass

        def find_attachment(self, parent, md5):
            return None

        def validate_attachment(self, record):
            pass

    def download(url, target):
        target.write_bytes(b"%PDF-1.4\nexample")

    monkeypatch.setattr(zotero, "ZoteroClient", Client)
    monkeypatch.setattr(zotero, "download_pdf", download)
    return config, paper, items, str(tmp_path / "test.db"), storage


@pytest.mark.anyio
async def test_export_publishes_matching_zip_prop_and_survives_reload(export_setup):
    config, paper, items, path, storage = export_setup
    await db.init_db(path)
    result = await zotero.export_paper(paper, config, path)
    assert result["status"] == "saved"
    assert len(items) == 2
    attachment = items[result["attachment_key"]]
    with zipfile.ZipFile(storage / (result["attachment_key"] + ".zip")) as archive:
        assert archive.namelist() == [attachment["filename"]]
        assert (
            hashlib.md5(archive.read(attachment["filename"])).hexdigest()
            == attachment["md5"]
        )
    prop = ET.parse(storage / (result["attachment_key"] + ".prop")).getroot()
    assert prop.findtext("hash") == attachment["md5"]
    assert int(prop.findtext("mtime")) == attachment["mtime"]
    assert len(await db.load_corpus_cache(path)) == 1
    scanned = CorpusPaper(
        title="PDF extracted title",
        abstract="different extraction",
        added_date=datetime.now(UTC),
        source_path=result["source_path"],
    )
    await db.save_corpus_cache([scanned], path)
    corpus = await db.load_corpus_cache(path)
    assert len(corpus) == 1
    assert corpus[0].abstract == paper["abstract"]
    assert corpus[0].authors == ["A. Author"]
    await db.save_corpus_cache([], path)
    assert await db.get_corpus_count(path) == 1
    result2 = await zotero.export_paper(
        {**paper, "url": "https://arxiv.org/abs/2601.12345"}, config, path
    )
    assert result2["attachment_key"] == result["attachment_key"]
    assert len(items) == 2


@pytest.mark.anyio
async def test_retry_after_webdav_failure_reuses_keys(export_setup, monkeypatch):
    config, paper, items, path, _storage = export_setup
    await db.init_db(path)
    publish = zotero.publish_attachment
    monkeypatch.setattr(
        zotero,
        "publish_attachment",
        lambda *args: (_ for _ in ()).throw(zotero.ExportError("disk full")),
    )
    with pytest.raises(zotero.ExportError, match="disk full"):
        await zotero.export_paper(paper, config, path)
    failed = (await db.load_zotero_exports(path))[0]
    assert failed["status"] == "failed"
    assert await db.get_corpus_count(path) == 0
    monkeypatch.setattr(zotero, "publish_attachment", publish)
    result = await zotero.export_paper(paper, config, path)
    assert result["paper_key"] == failed["paper_key"]
    assert result["attachment_key"] == failed["attachment_key"]
    assert len(items) == 2


@pytest.mark.anyio
async def test_existing_paper_reused(export_setup, monkeypatch):
    config, paper, items, path, _storage = export_setup
    monkeypatch.setattr(zotero.ZoteroClient, "find_paper", lambda *args: "ABCDEFGH")
    await db.init_db(path)
    result = await zotero.export_paper(paper, config, path)
    assert result["paper_key"] == "ABCDEFGH"
    assert len(items) == 1
    assert items[result["attachment_key"]]["parentItem"] == "ABCDEFGH"


@pytest.mark.anyio
async def test_existing_identical_attachment_reused(export_setup, monkeypatch):
    config, paper, items, path, _storage = export_setup
    attachment = {"key": "ABCDEFGH", "filename": "existing.pdf", "mtime": 1700000000000}
    items["ABCDEFGH"] = attachment
    storage = config["webdav"]["local_path"]
    with zipfile.ZipFile(storage + "/ABCDEFGH.zip", "w") as archive:
        archive.writestr("existing.pdf", b"%PDF-1.4\nexample")
        archive.writestr("metadata.json", '{"keep": true}')
    monkeypatch.setattr(
        zotero.ZoteroClient, "find_attachment", lambda *args: attachment
    )
    await db.init_db(path)
    result = await zotero.export_paper(paper, config, path)
    assert result["attachment_key"] == "ABCDEFGH"
    assert result["filename"] == "existing.pdf"
    assert result["mtime"] == 1700000000000
    assert len(items) == 2
    with zipfile.ZipFile(storage + "/ABCDEFGH.zip") as archive:
        assert archive.read("metadata.json") == b'{"keep": true}'


@pytest.mark.anyio
async def test_recommendation_lookup_returns_fields(export_setup):
    _config, paper, _items, path, _storage = export_setup
    await db.init_db(path)
    async with db._connect(path) as conn:
        await conn.execute(
            "INSERT INTO papers (id, date, source, title, abstract, authors, url) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "2026-10-08",
                "arxiv",
                paper["title"],
                paper["abstract"],
                paper["authors"],
                paper["url"],
            ),
        )
        await conn.commit()
    assert (await db.get_paper_by_id(1, path))["url"] == paper["url"]
    assert await db.get_paper_by_id(999, path) is None


@pytest.mark.parametrize(
    "url,identifier",
    [
        ("https://arxiv.org/abs/2601.12345v2", "2601.12345"),
        ("https://arxiv.org/pdf/2601.12345v2.pdf", "2601.12345"),
        ("http://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_arxiv_normalization(url, identifier):
    assert zotero.arxiv_id(url) == identifier


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/abs/2601.12345",
        "http://localhost/a",
        "https://arxiv.org/abs/../../etc/passwd",
    ],
)
def test_reject_other_download_sources(url):
    with pytest.raises(zotero.ExportError):
        zotero.arxiv_id(url)


def test_config_redaction():
    view = main.admin_config_view({"zotero": {"api_key": "secret", "user_id": "123"}})
    assert "secret" not in json.dumps(view)
    assert view["zotero"]["api_key_set"] is True


@pytest.mark.anyio
async def test_endpoint_requires_password(monkeypatch):
    monkeypatch.setenv(main.ADMIN_PASSWORD_ENV, "secret")
    with pytest.raises(main.HTTPException) as info:
        await main.save_to_zotero(1, make_request())
    assert info.value.status_code == 401


@pytest.mark.anyio
async def test_missing_configuration_does_not_write(export_setup):
    config, paper, _items, path, storage = export_setup
    with pytest.raises(zotero.ExportError, match="Configure"):
        await zotero.export_paper(
            paper, {"zotero": {}, "webdav": config["webdav"]}, path
        )
    assert list(storage.iterdir()) == []
