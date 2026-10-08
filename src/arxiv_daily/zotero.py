"""Export recommendations to a personal Zotero library and its WebDAV storage."""

import asyncio
import fcntl
import hashlib
import json
import os
import re
import secrets
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit
from xml.etree import ElementTree as ET

import requests

from . import database as db


class ExportError(ValueError):
    pass


def arxiv_id(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        raise ExportError("Only arXiv papers can be saved to Zotero")
    match = re.fullmatch(
        r"/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?:\.pdf)?",
        parsed.path,
    )
    if not match:
        raise ExportError("Invalid arXiv URL")
    return match[1]


def settings(config: dict[str, Any]) -> dict[str, str]:
    cfg = config.get("zotero") or {}
    result = {
        "api_key": os.environ.get("ARXIV_DAILY_ZOTERO_API_KEY")
        or str(cfg.get("api_key") or ""),
        "user_id": os.environ.get("ARXIV_DAILY_ZOTERO_USER_ID")
        or str(cfg.get("user_id") or ""),
        "collection_key": str(cfg.get("collection_key") or ""),
    }
    return result


class ZoteroClient:
    def __init__(self, cfg: dict[str, str]):
        self.base = f"https://api.zotero.org/users/{cfg['user_id']}"
        self.session = requests.Session()
        self.session.headers.update(
            {"Zotero-API-Key": cfg["api_key"], "Zotero-API-Version": "3"}
        )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.session.request(
                method, self.base + path, timeout=(10, 60), **kwargs
            )
        except requests.RequestException:
            raise ExportError("Zotero connection failed; retry to continue") from None
        if response.status_code == 404 and method == "GET":
            return None
        if not response.ok:
            raise ExportError(f"Zotero request failed (HTTP {response.status_code})")
        return response.json() if response.content else None

    def find_paper(self, identifier: str) -> str | None:
        start = 0
        while True:
            items = self.request(
                "GET",
                "/items/top",
                params={"q": identifier, "limit": 100, "start": start},
            )
            for item in items or []:
                data = item["data"]
                try:
                    if arxiv_id(data.get("url", "")) == identifier:
                        return data["key"]
                except ExportError:
                    if f"arXiv: {identifier}" in data.get("extra", "").splitlines():
                        return data["key"]
            if not items or len(items) < 100:
                return None
            start += 100

    def ensure_item(self, data: dict[str, Any]) -> None:
        existing = self.request("GET", f"/items/{data['key']}")
        if existing:
            current = existing["data"]
            if current.get("itemType") != data["itemType"] or current.get(
                "url"
            ) != data.get("url"):
                raise ExportError("Stored Zotero key conflicts with another item")
            return
        result = self.request("POST", "/items", json=[data])
        if result.get("failed") or not (
            result.get("successful") or result.get("success") or result.get("unchanged")
        ):
            raise ExportError(
                "Zotero rejected the item; check library permissions and collection"
            )

    def prepare_parent(self, key: str, collection: str) -> None:
        item = self.request("GET", f"/items/{key}")
        if not item or item["data"].get("itemType") in {"attachment", "note"}:
            raise ExportError("Zotero parent paper no longer exists")
        collections = item["data"].get("collections", [])
        if collection and collection not in collections:
            self.request(
                "PATCH",
                f"/items/{key}",
                json={"collections": [*collections, collection]},
                headers={"If-Unmodified-Since-Version": str(item["version"])},
            )

    def find_attachment(self, parent: str, md5: str) -> dict[str, Any] | None:
        start = 0
        while True:
            items = self.request(
                "GET",
                f"/items/{parent}/children",
                params={"limit": 100, "start": start},
            )
            for item in items or []:
                data = item["data"]
                if (
                    data.get("itemType") == "attachment"
                    and data.get("contentType") == "application/pdf"
                    and data.get("linkMode") in {"imported_file", "imported_url"}
                    and data.get("md5") == md5
                ):
                    return data
            if not items or len(items) < 100:
                return None
            start += 100

    def finish_attachment(self, key: str, md5: str, mtime: int) -> None:
        item = self.request("GET", f"/items/{key}")
        if not item:
            raise ExportError("Zotero attachment no longer exists")
        self.request(
            "PATCH",
            f"/items/{key}",
            json={"md5": md5, "mtime": mtime},
            headers={"If-Unmodified-Since-Version": str(item["version"])},
        )

    def validate_attachment(self, record: dict[str, Any]) -> None:
        item = self.request("GET", f"/items/{record['attachment_key']}")
        if not item:
            raise ExportError("Zotero attachment no longer exists")
        data = item["data"]
        if (
            data.get("parentItem") != record["paper_key"]
            or data.get("filename") != record["filename"]
            or data.get("md5") not in {None, "", record["md5"]}
        ):
            raise ExportError(
                "Zotero attachment changed; saving stopped to preserve it"
            )


def new_key() -> str:
    return "".join(
        secrets.choice("23456789ABCDEFGHIJKLMNPQRSTUVWXYZ") for _ in range(8)
    )


def download_pdf(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    try:
        for _ in range(5):
            response = requests.get(
                url, stream=True, timeout=(10, 60), allow_redirects=False
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            location = urljoin(url, response.headers.get("Location", ""))
            response.close()
            if urlsplit(location).scheme != "https" or arxiv_id(location) != arxiv_id(
                url
            ):
                raise ExportError("Unexpected arXiv PDF redirect")
            url = location
        else:
            raise ExportError("Too many arXiv PDF redirects")
        with response:
            if response.status_code != 200:
                raise ExportError(
                    f"arXiv PDF download failed (HTTP {response.status_code})"
                )
            size = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise ExportError("PDF exceeds the 100 MB download limit")
                    stream.write(chunk)
        with temporary.open("rb") as stream:
            if not stream.read(1024).lstrip().startswith(b"%PDF-"):
                raise ExportError("arXiv returned an invalid PDF")
        os.replace(temporary, target)
    except requests.RequestException:
        raise ExportError("arXiv PDF download failed; retry to continue") from None
    finally:
        temporary.unlink(missing_ok=True)


def publish_attachment(webdav: dict[str, Any], record: dict[str, Any]) -> str:
    key = record["attachment_key"]
    root = ET.Element("properties", version="1")
    ET.SubElement(root, "mtime").text = str(record["mtime"])
    ET.SubElement(root, "hash").text = record["md5"]
    prop = ET.tostring(root, encoding="utf-8")
    local = webdav.get("local_path")
    if local and not Path(local).is_dir():
        raise ExportError("Configured WebDAV local directory does not exist")
    with tempfile.TemporaryDirectory(dir=local or None, prefix=".arxiv-daily-") as temp:
        archive = Path(temp) / f"{key}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            output.write(record["file_path"], record["filename"])
        if local:
            prop_path = Path(temp) / f"{key}.prop"
            prop_path.write_bytes(prop)
            destination = Path(local) / archive.name
            if destination.exists():
                try:
                    with zipfile.ZipFile(destination) as existing:
                        md5 = hashlib.md5(existing.read(record["filename"])).hexdigest()
                except (KeyError, zipfile.BadZipFile):
                    raise ExportError(
                        "Existing WebDAV archive is incompatible; saving stopped"
                    ) from None
                if md5 != record["md5"]:
                    raise ExportError(
                        "WebDAV attachment changed; saving stopped to preserve it"
                    )
            else:
                os.replace(archive, destination)
            os.replace(prop_path, Path(local) / prop_path.name)
            return str(destination)
        base = str(webdav.get("url") or "").rstrip("/")
        if not base.startswith("https://"):
            raise ExportError(
                "Configure a local WebDAV directory or an HTTPS WebDAV URL"
            )
        path = str(webdav.get("path") or "/papers").strip("/")
        auth = (webdav.get("username", ""), webdav.get("password", ""))
        try:
            if record.get("reused_attachment"):
                prefix = f"{base}/{quote(path, safe='/')}/{key}"
                response = requests.get(
                    prefix + ".prop", auth=auth, timeout=(10, 60), allow_redirects=False
                )
                head = requests.head(
                    prefix + ".zip", auth=auth, timeout=(10, 60), allow_redirects=False
                )
                if response.status_code == 200 and head.status_code == 200:
                    try:
                        existing = ET.fromstring(response.content)
                    except ET.ParseError:
                        raise ExportError(
                            "Invalid WebDAV attachment metadata"
                        ) from None
                    if existing.findtext("hash") != record["md5"] or existing.findtext(
                        "mtime"
                    ) != str(record["mtime"]):
                        raise ExportError(
                            "WebDAV attachment changed; saving stopped to preserve it"
                        )
                    return f"/{path}/{key}.zip"
                if response.status_code != 404 or head.status_code != 404:
                    raise ExportError(
                        "Existing WebDAV attachment could not be verified"
                    )
            for suffix, content_type in (
                ("zip", "application/zip"),
                ("prop", "text/xml"),
            ):
                url = f"{base}/{quote(path, safe='/')}/{key}.{suffix}"
                with archive.open("rb") as stream:
                    response = requests.put(
                        url,
                        data=stream if suffix == "zip" else prop,
                        auth=auth,
                        headers={"Content-Type": content_type},
                        timeout=(10, 120),
                        allow_redirects=False,
                    )
                if response.status_code not in {200, 201, 204}:
                    raise ExportError(
                        f"WebDAV upload failed (HTTP {response.status_code})"
                    )
        except requests.RequestException:
            raise ExportError("WebDAV upload failed; retry to continue") from None
        return f"/{path}/{key}.zip"


async def export_paper(
    paper: dict[str, Any], config: dict[str, Any], db_path: str | None = None
) -> dict[str, Any]:
    cfg = settings(config)
    if not cfg["api_key"] or not cfg["user_id"].isdigit():
        raise ExportError("Configure zotero.api_key and zotero.user_id before saving")
    if cfg["collection_key"] and not re.fullmatch(
        r"[A-Z0-9]{8}", cfg["collection_key"]
    ):
        raise ExportError("Invalid Zotero collection key")
    identifier = arxiv_id(paper["url"])
    directory = db._get_db_path(db_path).parent / "zotero_exports"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.sha256(f"{cfg['user_id']}:{identifier}".encode()).hexdigest()
    # A process-level lock also prevents duplicates across multiple server workers.
    with (directory / f"{prefix}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExportError(
                "This paper is already being saved; retry shortly"
            ) from None
        record = next(
            (
                r
                for r in await db.load_zotero_exports(db_path)
                if r["library_id"] == cfg["user_id"] and r["arxiv_id"] == identifier
            ),
            None,
        )
        if record and record["status"] == "saved":
            return record
        if record is None:
            authors = paper.get("authors") or []
            if isinstance(authors, str):
                authors = json.loads(authors)
            record = {
                "library_id": cfg["user_id"],
                "arxiv_id": identifier,
                "paper_key": new_key(),
                "attachment_key": new_key(),
                "title": paper["title"],
                "abstract": paper.get("abstract") or "",
                "authors": authors,
                "added_date": datetime.now(UTC).isoformat(),
                "filename": identifier.replace("/", "_") + ".pdf",
                "pdf_url": "https://arxiv.org/pdf/"
                + re.sub(
                    r"^/(?:abs|pdf)/", "", urlsplit(paper["url"]).path
                ).removesuffix(".pdf"),
                "file_path": str(directory / f"{prefix}.pdf"),
                "status": "pending",
                "error": "",
            }
            await db.save_zotero_export(record, db_path)
        client = ZoteroClient(cfg)
        try:
            if not Path(record["file_path"]).is_file():
                await asyncio.to_thread(
                    download_pdf, record["pdf_url"], Path(record["file_path"])
                )
            record.setdefault("mtime", int(time.time() * 1000))
            record["md5"] = await asyncio.to_thread(
                lambda: hashlib.md5(Path(record["file_path"]).read_bytes()).hexdigest()
            )
            if cfg["collection_key"] and not await asyncio.to_thread(
                client.request, "GET", f"/collections/{cfg['collection_key']}"
            ):
                raise ExportError("Zotero collection not found")
            if not record.get("parent_ready"):
                existing = await asyncio.to_thread(client.find_paper, identifier)
                if existing:
                    record["paper_key"] = existing
                await db.save_zotero_export(record, db_path)
                if not existing:
                    await asyncio.to_thread(
                        client.ensure_item,
                        {
                            "key": record["paper_key"],
                            "itemType": "preprint",
                            "title": record["title"],
                            "abstractNote": record["abstract"],
                            "url": f"https://arxiv.org/abs/{identifier}",
                            "repository": "arXiv",
                            "archiveID": identifier,
                            "creators": [
                                {"creatorType": "author", "name": a}
                                for a in record["authors"]
                            ],
                            "tags": [{"tag": "arxiv-daily"}],
                            "collections": [cfg["collection_key"]]
                            if cfg["collection_key"]
                            else [],
                        },
                    )
                record["parent_ready"] = True
                await db.save_zotero_export(record, db_path)
            await asyncio.to_thread(
                client.prepare_parent, record["paper_key"], cfg["collection_key"]
            )
            if not record.get("attachment_ready"):
                existing_attachment = await asyncio.to_thread(
                    client.find_attachment, record["paper_key"], record["md5"]
                )
                if existing_attachment:
                    filename = existing_attachment["filename"]
                    if Path(filename).name != filename or "\\" in filename:
                        raise ExportError(
                            "Unsafe filename in existing Zotero attachment"
                        )
                    record.update(
                        attachment_key=existing_attachment["key"],
                        filename=filename,
                        reused_attachment=True,
                    )
                    record["mtime"] = int(
                        existing_attachment.get("mtime") or record["mtime"]
                    )
                    await db.save_zotero_export(record, db_path)
                else:
                    await asyncio.to_thread(
                        client.ensure_item,
                        {
                            "key": record["attachment_key"],
                            "itemType": "attachment",
                            "parentItem": record["paper_key"],
                            "linkMode": "imported_url",
                            "title": "PDF",
                            "url": record["pdf_url"],
                            "contentType": "application/pdf",
                            "filename": record["filename"],
                        },
                    )
                record["attachment_ready"] = True
                await db.save_zotero_export(record, db_path)
            await asyncio.to_thread(client.validate_attachment, record)
            record["source_path"] = await asyncio.to_thread(
                publish_attachment, config["webdav"], record
            )
            await asyncio.to_thread(
                client.finish_attachment,
                record["attachment_key"],
                record["md5"],
                record["mtime"],
            )
            record.update(status="saved", error="")
            await db.save_zotero_export(record, db_path)
            return record
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            requests.RequestException,
        ) as error:
            message = (
                str(error)
                if isinstance(error, ExportError)
                else "Save failed; retry to continue"
            )
            record.update(status="failed", error=message)
            await db.save_zotero_export(record, db_path)
            raise ExportError(message) from None
        finally:
            client.session.close()
