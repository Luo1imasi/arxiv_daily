import os
import re
import uuid
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .protocol import CorpusPaper
from .utils import parallel_execute

try:
    import pymupdf4llm
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
except ImportError:
    pymupdf4llm = None
    pymupdf = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


CACHE_DIR = Path.home() / ".arxiv_daily" / "papers"
_META_DIR = Path.home() / ".arxiv_daily" / "meta"
_webdav_client_cache: dict = {}


def _get_cached_webdav_client(
    url: str, username: str, password: str, base_path: str
) -> "WebDAVClient":
    """获取缓存的WebDAV客户端"""
    key = (url, username, password, base_path)
    if key not in _webdav_client_cache:
        _webdav_client_cache[key] = WebDAVClient(url, username, password, base_path)
    return _webdav_client_cache[key]


def _get_cache_path(source_path: str, name: str) -> Path:
    h = hashlib.md5(source_path.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{h}_{name}"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _META_DIR.mkdir(parents=True, exist_ok=True)


def _meta_cache_path(pdf_path: str) -> Path:
    h = hashlib.md5(pdf_path.encode()).hexdigest()
    return _META_DIR / f"{h}.json"


def _manifest_cache_path(webdav_config: dict) -> Path:
    local_path = webdav_config.get("local_path", "")
    source_key = local_path or "|".join(
        [
            webdav_config.get("url", ""),
            webdav_config.get("username", ""),
            webdav_config.get("path", "/papers"),
        ]
    )
    source_hash = hashlib.md5(source_key.encode("utf-8")).hexdigest()
    return _META_DIR / f"manifest_{source_hash}.json"


def load_corpus_manifest(webdav_config: dict) -> Optional[dict]:
    path = _manifest_cache_path(webdav_config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_corpus_manifest(webdav_config: dict, manifest: dict):
    _ensure_cache_dir()
    path = _manifest_cache_path(webdav_config)
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _build_manifest_from_files(files: list[dict]) -> dict:
    entries = []
    for file_info in files:
        modified = file_info.get("modified")
        if isinstance(modified, datetime):
            modified = modified.isoformat()
        entries.append(
            {
                "path": file_info.get("path", ""),
                "name": file_info.get("name", ""),
                "size": file_info.get("size", 0),
                "modified": modified or "",
            }
        )
    entries.sort(key=lambda item: item["path"])
    return {
        "count": len(entries),
        "generated_at": datetime.now().isoformat(),
        "files": entries,
    }


def _manifest_entry_key(entry: dict) -> tuple[str, int, str]:
    return (
        entry.get("path", ""),
        int(entry.get("size", 0) or 0),
        str(entry.get("modified", "") or ""),
    )


def build_corpus_manifest(webdav_config: dict) -> dict:
    local_source_path = webdav_config.get("local_path", "")
    use_local = local_source_path and os.path.isdir(local_source_path)

    files = []
    if use_local:
        _walk_local(local_source_path, files, [])
    else:
        webdav_client = _get_cached_webdav_client(
            url=webdav_config["url"],
            username=webdav_config["username"],
            password=webdav_config["password"],
            base_path=webdav_config.get("path", "/papers"),
        )
        files = webdav_client.list_files()

    return _build_manifest_from_files(files)


def _load_meta_cache(pdf_path: str) -> Optional[dict]:
    p = _meta_cache_path(pdf_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_meta_cache(pdf_path: str, data: dict):
    p = _meta_cache_path(pdf_path)
    try:
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _extract_pdf_from_zip(
    zip_path: str, output_dir: Path, final_path: Path
) -> Optional[str]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
            if not pdf_names:
                return None
            pdf_name = pdf_names[0]
            pdf_data = zf.read(pdf_name)
            final_path.write_bytes(pdf_data)
            return str(final_path)
    except Exception as e:
        logger.warning(f"Failed to extract zip {zip_path}: {e}")
        return None


class WebDAVClient:
    def __init__(
        self, url: str, username: str, password: str, base_path: str = "/papers"
    ):
        self.url = url.rstrip("/")
        self.username = username
        self.password = password
        self.base_path = base_path if base_path.endswith("/") else base_path + "/"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from webdav4.client import Client

            self._client = Client(
                base_url=self.url,
                auth=(self.username, self.password),
            )
        return self._client

    def test_connection(self) -> bool:
        try:
            client = self._get_client()
            client.ls(self.base_path)
            return True
        except Exception as e:
            logger.error(f"WebDAV connection test failed: {e}")
            return False

    def list_files(self) -> list[dict]:
        client = self._get_client()
        files = []
        self._walk(client, self.base_path, files, [])
        return files

    def _walk(self, client, path: str, files: list, current_path_parts: list[str]):
        try:
            items = client.ls(path)
        except Exception as e:
            logger.warning(f"Failed to list {path}: {e}")
            return

        for item in items:
            is_dir = item.get("type") == "directory"
            if is_dir:
                name = item["name"].rstrip("/").split("/")[-1]
                if name.startswith("."):
                    continue
                child_path_parts = current_path_parts + [name]
                child_path = f"{path}/{name}"
                self._walk(client, child_path, files, child_path_parts)
            else:
                name = item["name"].rstrip("/").split("/")[-1]
                lower_name = name.lower()
                if lower_name.endswith(".pdf") or lower_name.endswith(".zip"):
                    files.append(
                        {
                            "name": name,
                            "path": item["name"],
                            "size": item.get("content_length", 0),
                            "modified": item.get("modified", ""),
                            "collection_path": "/".join(current_path_parts),
                        }
                    )

    def download_file(self, remote_path: str, local_path: str):
        client = self._get_client()
        client.download_file(remote_path, local_path)


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    if pymupdf4llm is None:
        logger.warning("pymupdf4llm not installed, cannot extract PDF text")
        return None
    try:
        text = pymupdf4llm.to_markdown(
            file_path, use_ocr=False, header=False, footer=False, ignore_code=True
        )
        return text
    except Exception as e:
        logger.warning(f"Failed to extract text from {file_path}: {e}")
        return None


def extract_metadata_from_pdf(file_path: str) -> dict:
    if pymupdf is None:
        return {}
    try:
        doc = pymupdf.open(file_path)
        meta = doc.metadata or {}
        result = {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
        }
        doc.close()
        return result
    except Exception as e:
        logger.warning(f"Failed to extract metadata from {file_path}: {e}")
        return {}


def extract_abstract_from_text(text: str) -> str:
    if not text:
        return ""
    text = _clean_extracted_text(text)
    patterns = [
        r"(?i)abstract[:\s]*\n?(.*?)(?=\n\s*\n(?:introduction|keywords|1[\s\.]|I[\.\s]))",
        r"(?i)abstract[:\s]*(.*?)(?=\n\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:
                return _trim_abstract(abstract)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for paragraph in paragraphs[:6]:
        if len(paragraph) < 80:
            continue
        if _looks_like_noise(paragraph):
            continue
        return _trim_abstract(paragraph)
    return ""


def _clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _trim_abstract(text: str, max_chars: int = 1800) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or text[:max_chars].strip()


def _looks_like_noise(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in ["copyright", "figure ", "table "]):
        return True
    alpha_chars = sum(ch.isalpha() for ch in text)
    return alpha_chars < max(40, len(text) * 0.45)


def _select_best_abstract(title: str, metadata_subject: str, text: Optional[str]) -> str:
    subject = _trim_abstract(metadata_subject)
    if subject and len(subject) >= 80 and not _looks_like_noise(subject):
        return subject

    extracted = extract_abstract_from_text(text or "")
    if extracted and extracted.lower() != (title or "").strip().lower():
        return extracted

    return title


def _walk_local(path: str, files: list, current_path_parts: list[str]):
    try:
        items = os.listdir(path)
    except Exception as e:
        logger.warning(f"Failed to list {path}: {e}")
        return

    for name in items:
        if name.startswith("."):
            continue
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            child_path_parts = current_path_parts + [name]
            _walk_local(full_path, files, child_path_parts)
        else:
            lower_name = name.lower()
            if lower_name.endswith(".pdf") or lower_name.endswith(".zip"):
                stat = os.stat(full_path)
                files.append(
                    {
                        "name": name,
                        "path": full_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "collection_path": "/".join(current_path_parts),
                    }
                )


def _process_file(
    file_info: dict, cache_dir: Path, webdav_client=None
) -> Optional[tuple[str, datetime]]:
    _ensure_cache_dir()
    name = file_info["name"]
    source_path = file_info["path"]
    lower_name = name.lower()

    cache_pdf_name = name.rsplit(".", 1)[0] + ".pdf"
    cache_path = _get_cache_path(source_path, cache_pdf_name)

    if cache_path.exists():
        logger.debug(f"Cache hit: {cache_path.name}")
        return str(cache_path), file_info.get("modified", datetime.now())

    logger.debug(f"Cache miss: {cache_path.name}")

    if lower_name.endswith(".pdf"):
        if webdav_client:
            webdav_client.download_file(source_path, str(cache_path))
        else:
            shutil.copy2(source_path, str(cache_path))
        return str(cache_path), file_info.get("modified", datetime.now())

    elif lower_name.endswith(".zip"):
        final_name = name.rsplit(".", 1)[0] + ".pdf"
        final_path = _get_cache_path(source_path, final_name)

        temp_zip = cache_dir / f"temp_{uuid.uuid4().hex[:8]}_{name}"
        try:
            if webdav_client:
                webdav_client.download_file(source_path, str(temp_zip))
                zip_path = str(temp_zip)
            else:
                zip_path = source_path

            result = _extract_pdf_from_zip(zip_path, cache_dir, final_path)
            if result:
                return str(result), file_info.get("modified", datetime.now())
        finally:
            if temp_zip.exists():
                temp_zip.unlink()

    return None


def _process_single(args) -> Optional[CorpusPaper]:
    file_info, cache_dir, webdav_client = args
    result = _process_file(file_info, cache_dir, webdav_client)
    if not result:
        return None

    pdf_path, modified = result

    cached = _load_meta_cache(pdf_path)
    if cached:
        title = cached.get("title", "") or file_info["name"].rsplit(".", 1)[0]
        abstract = cached.get("abstract", "")
        if not abstract or abstract == title:
            abstract = title
    else:
        metadata = extract_metadata_from_pdf(pdf_path)
        text = extract_text_from_pdf(pdf_path)

        title = metadata.get("title", "") or file_info["name"].rsplit(".", 1)[0]
        abstract = _select_best_abstract(title, metadata.get("subject", ""), text)

        _save_meta_cache(pdf_path, {"title": title, "abstract": abstract})

    if isinstance(modified, str):
        try:
            modified = datetime.fromisoformat(modified.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            modified = datetime.now()

    return CorpusPaper(
        title=title,
        abstract=abstract,
        added_date=modified,
        file_path=pdf_path,
        source_path=file_info["path"],
        paths=[file_info["collection_path"]]
        if file_info.get("collection_path")
        else [],
    )


def fetch_corpus(
    webdav_config: dict,
    executor_config: dict = None,
    previous_corpus: Optional[list[CorpusPaper]] = None,
    previous_manifest: Optional[dict] = None,
) -> list[CorpusPaper]:
    _ensure_cache_dir()

    local_source_path = webdav_config.get("local_path", "")
    use_local = local_source_path and os.path.isdir(local_source_path)

    files = []
    webdav_client = None

    if use_local:
        logger.info(f"Scanning local path: {local_source_path}")
        _walk_local(local_source_path, files, [])
    else:
        webdav_client = _get_cached_webdav_client(
            url=webdav_config["url"],
            username=webdav_config["username"],
            password=webdav_config["password"],
            base_path=webdav_config.get("path", "/papers"),
        )
        logger.info("Connecting to WebDAV...")
        files = webdav_client.list_files()

    logger.info(f"Found {len(files)} files (PDF + ZIP)")
    current_manifest = _build_manifest_from_files(files)

    reusable_paths = set()
    if previous_corpus and previous_manifest:
        previous_entries = {
            entry.get("path", ""): _manifest_entry_key(entry)
            for entry in previous_manifest.get("files", [])
        }
        current_entries = {
            entry.get("path", ""): _manifest_entry_key(entry)
            for entry in current_manifest.get("files", [])
        }
        reusable_paths = {
            path
            for path, signature in current_entries.items()
            if previous_entries.get(path) == signature
        }

    cached_by_source = {
        paper.source_path: paper
        for paper in (previous_corpus or [])
        if getattr(paper, "source_path", "") and paper.file_path and os.path.exists(paper.file_path)
    }

    reused = []
    files_to_process = []
    for file_info in files:
        source_path = file_info.get("path", "")
        cached = cached_by_source.get(source_path)
        if source_path in reusable_paths and cached is not None:
            collection_path = file_info.get("collection_path")
            cached.paths = [collection_path] if collection_path else []
            reused.append(cached)
        else:
            files_to_process.append(file_info)

    save_corpus_manifest(webdav_config, current_manifest)

    args_list = [(f, CACHE_DIR, webdav_client) for f in files_to_process]

    max_workers = (
        (executor_config or {}).get("corpus_workers", 8)
        if executor_config
        else min(8, os.cpu_count() or 4)
    )
    logger.info(
        f"Processing with {max_workers} threads... (reused={len(reused)}, changed={len(files_to_process)})"
    )

    results = parallel_execute(
        _process_single, args_list, max_workers=max_workers, desc="Processing"
    )
    corpus = reused + [r for r in results if r is not None]

    logger.info(f"Loaded {len(corpus)} papers into corpus")
    return corpus
