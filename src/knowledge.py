import csv
import io
import re
import shutil
import tempfile
import time
import zipfile
from urllib.parse import urlparse
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ACTIVE_DB_FILE, DB_ROOT, DEFAULT_DB_DIR, EMBED_MODEL, TOP_K

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def clear_knowledge_base():
    if DB_ROOT.exists():
        shutil.rmtree(DB_ROOT, ignore_errors=True)


def _read_text_file(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def _docx_to_text(path: Path):
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        namespace = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }
        paragraphs = []
        for node in root.findall(".//w:p", namespace):
            text = "".join(node.itertext()).strip()
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)
    except Exception:
        return ""


def _load_local_file(path: Path):
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".log"}:
        return [
            Document(page_content=_read_text_file(path), metadata={"source": path.name})
        ]
    if ext == ".csv":
        text = _read_text_file(path)
        reader = csv.reader(io.StringIO(text))
        lines = [", ".join(row) for row in reader if row]
        return [Document(page_content="\n".join(lines), metadata={"source": path.name})]
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".docx":
        text = _docx_to_text(path)
        return (
            [Document(page_content=text, metadata={"source": path.name})]
            if text
            else []
        )
    return [
        Document(page_content=_read_text_file(path), metadata={"source": path.name})
    ]


def load_uploaded_files(uploaded_files):
    documents = []
    temp_paths = []
    try:
        for uploaded in uploaded_files or []:
            suffix = Path(uploaded.name).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = Path(tmp.name)
            temp_paths.append(temp_path)

            for doc in _load_local_file(temp_path):
                doc.metadata = {**getattr(doc, "metadata", {}), "source": uploaded.name}
                if doc.page_content.strip():
                    documents.append(doc)
        return documents
    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def load_urls(url_text: str):
    urls = [line.strip() for line in (url_text or "").splitlines() if line.strip()]
    documents = []
    for url in urls:
        try:
            response = requests.get(url, timeout=20, headers=BROWSER_HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(
                ["script", "style", "noscript", "header", "footer", "nav", "aside"]
            ):
                tag.decompose()

            text = ""
            page_path = urlparse(url).path.lower()
            if "our-institutions" in page_path:
                candidates = soup.find_all(
                    lambda tag: tag.name in {"h1", "h2", "h3"}
                    and tag.get_text(" ", strip=True).lower() == "our institutions"
                )
                heading = candidates[-1] if candidates else None
                if heading:
                    section = heading.find_parent("div", class_="col-lg-12")
                    items = []
                    if section:
                        raw_lines = [
                            re.sub(r"\s+", " ", line).strip()
                            for line in section.get_text("\n", strip=True).splitlines()
                            if line.strip()
                        ]
                        keywords = (
                            "college",
                            "school",
                            "academy",
                            "university",
                            "institute",
                            "hospital",
                            "pharmacy",
                            "nursing",
                            "physiotherapy",
                            "medical lab",
                            "technology",
                            "law",
                        )
                        for line in raw_lines:
                            if re.fullmatch(r"view", line, flags=re.I):
                                continue
                            if any(key in line.lower() for key in keywords):
                                items.append(line)
                    cleaned = []
                    seen = set()
                    for item in items:
                        if len(item) < 3:
                            continue
                        if item.lower() in seen:
                            continue
                        seen.add(item.lower())
                        cleaned.append(item)
                    if cleaned:
                        text = "\n".join(cleaned)
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": url,
                                    "page_type": "institutions",
                                    "item_count": str(len(cleaned)),
                                },
                            )
                        )
                        continue

            text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
            if text:
                documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception:
            continue
    return documents


def _set_active_db_dir(db_dir: Path):
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_DB_FILE.write_text(str(db_dir.resolve()), encoding="utf-8")


def _get_active_db_dir():
    if ACTIVE_DB_FILE.exists():
        try:
            path = Path(ACTIVE_DB_FILE.read_text(encoding="utf-8").strip())
            if path.exists():
                return path
        except Exception:
            pass
    return DEFAULT_DB_DIR


def build_vector_db(documents, replace_existing=True):
    if not documents:
        return 0

    if replace_existing and DB_ROOT.exists():
        shutil.rmtree(DB_ROOT, ignore_errors=True)

    db_dir = DB_ROOT / (
        f"session_{int(time.time() * 1000)}" if replace_existing else "current"
    )
    db_dir.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(db_dir),
    )
    _set_active_db_dir(db_dir)
    return len(chunks)


def ingest_sources(uploaded_files, url_text, replace_existing=True):
    documents = []
    documents.extend(load_uploaded_files(uploaded_files))
    documents.extend(load_urls(url_text))
    return build_vector_db(documents, replace_existing=replace_existing)


def get_vectorstore():
    db_dir = _get_active_db_dir()
    if not db_dir.exists():
        return None
    try:
        return Chroma(
            persist_directory=str(db_dir), embedding_function=get_embeddings()
        )
    except Exception:
        return None


def get_retriever():
    store = get_vectorstore()
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": TOP_K})


def unique_sources(documents):
    sources = []
    for doc in documents:
        source = (doc.metadata or {}).get("source")
        if source and source not in sources:
            sources.append(source)
    return sources
