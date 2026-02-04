"""
Document Ingestion Pipeline for Trading Knowledge Base.

Supports PDF, EPUB, Markdown, and TXT files.
Uses smart chunking with paragraph/section boundary respect.
Stores embeddings in ChromaDB for efficient retrieval.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "knowledge_layer" / "config" / "settings.yaml"
INGESTION_LOG_PATH = PROJECT_ROOT / "knowledge_layer" / "embeddings" / "ingestion_log.json"


def load_config() -> Dict:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_file_hash(file_path: str) -> str:
    """Generate hash of file for change detection."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_ingestion_log() -> Dict:
    """Load the ingestion log to track what's been processed."""
    if INGESTION_LOG_PATH.exists():
        with open(INGESTION_LOG_PATH, "r") as f:
            return json.load(f)
    return {"documents": {}, "last_updated": None}


def save_ingestion_log(log: Dict):
    """Save the ingestion log."""
    log["last_updated"] = datetime.now().isoformat()
    INGESTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INGESTION_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        raise
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise


def extract_text_from_epub(file_path: str) -> str:
    """Extract text from EPUB file."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(file_path)
        text = ""

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text += soup.get_text() + "\n\n"

        return text
    except ImportError:
        logger.error("ebooklib or beautifulsoup4 not installed.")
        raise
    except Exception as e:
        logger.error(f"Error extracting EPUB: {e}")
        raise


def extract_text_from_markdown(file_path: str) -> str:
    """Extract text from Markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Extract text from any supported file type."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    extractors = {
        ".pdf": extract_text_from_pdf,
        ".epub": extract_text_from_epub,
        ".md": extract_text_from_markdown,
        ".markdown": extract_text_from_markdown,
        ".txt": extract_text_from_txt,
    }

    if suffix not in extractors:
        raise ValueError(f"Unsupported file type: {suffix}")

    return extractors[suffix](file_path)


def smart_chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    respect_boundaries: bool = True
) -> List[str]:
    """
    Chunk text intelligently, respecting paragraph and section boundaries.

    Args:
        text: The text to chunk
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Overlap between chunks in tokens
        respect_boundaries: Try to break at paragraph/section boundaries

    Returns:
        List of text chunks
    """
    # Approximate tokens as words * 1.3
    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    chunks = []

    if respect_boundaries:
        # Split by double newlines (paragraphs) first
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    else:
        paragraphs = [text]

    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If single paragraph is too large, split it further
        if para_tokens > chunk_size:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sent_tokens = estimate_tokens(sentence)
                if current_tokens + sent_tokens > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap
                    overlap_text = " ".join(current_chunk.split()[-chunk_overlap:])
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = estimate_tokens(current_chunk)
                else:
                    current_chunk += " " + sentence
                    current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                overlap_text = " ".join(current_chunk.split()[-chunk_overlap:])
                current_chunk = overlap_text + "\n\n" + para
                current_tokens = estimate_tokens(current_chunk)
            else:
                current_chunk += "\n\n" + para
                current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def get_chroma_client():
    """Get or create ChromaDB client."""
    try:
        import chromadb
        from chromadb.config import Settings

        config = load_config()
        persist_dir = PROJECT_ROOT / config["chroma"]["persist_directory"]
        persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        return client
    except ImportError:
        logger.error("chromadb not installed. Run: pip install chromadb")
        raise


def get_embedding_function():
    """Get the embedding function based on config."""
    try:
        from chromadb.utils import embedding_functions

        config = load_config()
        model_name = config["embedding"]["model"]

        # Use sentence-transformers embedding
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    except ImportError:
        logger.error("sentence-transformers not installed.")
        raise


def get_or_create_collection():
    """Get or create the ChromaDB collection."""
    config = load_config()
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=config["chroma"]["collection_name"],
        embedding_function=embedding_fn,
        metadata={"description": "Trading knowledge base"}
    )
    return collection


def ingest_document(
    file_path: str,
    category: str,
    metadata: Dict = None,
    force: bool = False
) -> Dict:
    """
    Ingest a single document into the knowledge base.

    Args:
        file_path: Path to the document
        category: One of the predefined categories
        metadata: Additional metadata to store
        force: Re-ingest even if already processed

    Returns:
        Dict with ingestion results
    """
    config = load_config()
    valid_categories = config["categories"]

    if category not in valid_categories:
        raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if already ingested
    log = load_ingestion_log()
    file_hash = get_file_hash(str(file_path))
    file_key = str(file_path.absolute())

    if file_key in log["documents"] and not force:
        existing = log["documents"][file_key]
        if existing.get("hash") == file_hash:
            logger.info(f"Skipping {file_path.name} - already ingested (unchanged)")
            return {"status": "skipped", "reason": "already_ingested"}

    logger.info(f"Ingesting: {file_path.name} [{category}]")

    # Extract text
    text = extract_text(str(file_path))

    # Chunk the text
    chunks = smart_chunk_text(
        text,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        respect_boundaries=config["chunking"]["respect_boundaries"]
    )

    logger.info(f"Created {len(chunks)} chunks from {file_path.name}")

    # Prepare metadata
    base_metadata = {
        "source": file_path.name,
        "source_path": str(file_path.absolute()),
        "category": category,
        "ingested_at": datetime.now().isoformat(),
        "file_type": file_path.suffix.lower(),
    }
    if metadata:
        base_metadata.update(metadata)

    # Get collection
    collection = get_or_create_collection()

    # Generate IDs and prepare for insertion
    doc_ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_hash}_{i}"
        doc_ids.append(chunk_id)
        documents.append(chunk)
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_index"] = i
        chunk_metadata["total_chunks"] = len(chunks)
        metadatas.append(chunk_metadata)

    # Delete existing chunks for this document (if re-ingesting)
    try:
        existing_ids = collection.get(where={"source_path": str(file_path.absolute())})
        if existing_ids and existing_ids["ids"]:
            collection.delete(ids=existing_ids["ids"])
            logger.info(f"Removed {len(existing_ids['ids'])} existing chunks")
    except Exception:
        pass

    # Add to collection
    collection.add(
        ids=doc_ids,
        documents=documents,
        metadatas=metadatas
    )

    # Update log
    log["documents"][file_key] = {
        "hash": file_hash,
        "category": category,
        "chunks": len(chunks),
        "ingested_at": datetime.now().isoformat(),
        "metadata": metadata
    }
    save_ingestion_log(log)

    logger.info(f"Successfully ingested {file_path.name}: {len(chunks)} chunks")

    return {
        "status": "success",
        "file": file_path.name,
        "category": category,
        "chunks": len(chunks)
    }


def ingest_directory(
    dir_path: str,
    category: str,
    recursive: bool = True,
    force: bool = False
) -> List[Dict]:
    """
    Batch ingest all documents in a directory.

    Args:
        dir_path: Path to directory
        category: Category for all documents
        recursive: Process subdirectories
        force: Re-ingest existing documents

    Returns:
        List of ingestion results
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    supported_extensions = {".pdf", ".epub", ".md", ".markdown", ".txt"}
    results = []

    pattern = "**/*" if recursive else "*"
    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                result = ingest_document(str(file_path), category, force=force)
                results.append(result)
            except Exception as e:
                logger.error(f"Error ingesting {file_path}: {e}")
                results.append({
                    "status": "error",
                    "file": file_path.name,
                    "error": str(e)
                })

    return results


def get_ingestion_status() -> Dict:
    """Return stats on what's been ingested."""
    log = load_ingestion_log()

    # Count by category
    by_category = {}
    total_chunks = 0

    for doc_info in log["documents"].values():
        cat = doc_info.get("category", "unknown")
        chunks = doc_info.get("chunks", 0)

        if cat not in by_category:
            by_category[cat] = {"documents": 0, "chunks": 0}

        by_category[cat]["documents"] += 1
        by_category[cat]["chunks"] += chunks
        total_chunks += chunks

    # Get collection stats
    try:
        collection = get_or_create_collection()
        collection_count = collection.count()
    except Exception:
        collection_count = 0

    return {
        "total_documents": len(log["documents"]),
        "total_chunks": total_chunks,
        "collection_count": collection_count,
        "by_category": by_category,
        "last_updated": log.get("last_updated")
    }


def delete_document(file_path: str) -> bool:
    """Remove a document from the knowledge base."""
    file_path = Path(file_path)
    file_key = str(file_path.absolute())

    log = load_ingestion_log()

    if file_key not in log["documents"]:
        logger.warning(f"Document not found in log: {file_path}")
        return False

    # Remove from ChromaDB
    try:
        collection = get_or_create_collection()
        existing = collection.get(where={"source_path": file_key})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info(f"Removed {len(existing['ids'])} chunks from collection")
    except Exception as e:
        logger.error(f"Error removing from collection: {e}")

    # Remove from log
    del log["documents"][file_key]
    save_ingestion_log(log)

    logger.info(f"Deleted document: {file_path}")
    return True


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    parser.add_argument("path", nargs="?", help="File or directory path")
    parser.add_argument("-c", "--category", required=False, help="Document category")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-ingestion")
    parser.add_argument("-s", "--status", action="store_true", help="Show ingestion status")

    args = parser.parse_args()

    if args.status:
        status = get_ingestion_status()
        print("\n=== Knowledge Base Status ===")
        print(f"Total Documents: {status['total_documents']}")
        print(f"Total Chunks: {status['total_chunks']}")
        print(f"Collection Count: {status['collection_count']}")
        print(f"\nBy Category:")
        for cat, info in status.get("by_category", {}).items():
            print(f"  {cat}: {info['documents']} docs, {info['chunks']} chunks")
        print(f"\nLast Updated: {status.get('last_updated', 'Never')}")
    elif args.path:
        path = Path(args.path)
        if not args.category:
            print("Error: --category is required")
            exit(1)

        if path.is_file():
            result = ingest_document(str(path), args.category, force=args.force)
            print(f"Result: {result}")
        elif path.is_dir():
            results = ingest_directory(str(path), args.category, force=args.force)
            success = sum(1 for r in results if r.get("status") == "success")
            print(f"\nIngested {success}/{len(results)} documents")
        else:
            print(f"Path not found: {path}")
    else:
        parser.print_help()
