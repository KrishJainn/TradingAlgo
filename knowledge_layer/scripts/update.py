"""
Incremental Update Script for Knowledge Base.

Scans source directories for new or modified documents and updates
the knowledge base accordingly.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .ingest import (
    load_config,
    load_ingestion_log,
    save_ingestion_log,
    get_file_hash,
    ingest_document,
    delete_document,
    get_ingestion_status,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCES_DIR = PROJECT_ROOT / "knowledge_layer" / "sources"


def scan_for_changes() -> Dict[str, List[Path]]:
    """
    Scan source directories for new, modified, or deleted documents.

    Returns:
        Dict with 'new', 'modified', 'deleted' file lists
    """
    log = load_ingestion_log()
    existing_docs = set(log.get("documents", {}).keys())

    supported_extensions = {".pdf", ".epub", ".md", ".markdown", ".txt"}

    changes = {
        "new": [],
        "modified": [],
        "deleted": []
    }

    # Define category mappings based on directory
    category_dirs = {
        "books": "trading_books",
        "notes": "notes",
        "strategies": "strategies",
        "risk_management": "risk_management",
        "technical_analysis": "technical_analysis",
        "market_psychology": "market_psychology",
        "fundamental_analysis": "fundamental_analysis",
    }

    found_files = set()

    # Scan each source directory
    for dir_name, category in category_dirs.items():
        dir_path = SOURCES_DIR / dir_name
        if not dir_path.exists():
            continue

        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_extensions:
                continue

            file_key = str(file_path.absolute())
            found_files.add(file_key)

            if file_key not in existing_docs:
                # New file
                changes["new"].append((file_path, category))
            else:
                # Check if modified
                current_hash = get_file_hash(str(file_path))
                existing_hash = log["documents"][file_key].get("hash")
                if current_hash != existing_hash:
                    changes["modified"].append((file_path, category))

    # Check for deleted files
    for doc_key in existing_docs:
        if doc_key not in found_files:
            # Check if file actually existed in our source dirs
            doc_path = Path(doc_key)
            if str(SOURCES_DIR) in doc_key:
                changes["deleted"].append(doc_path)

    return changes


def update_knowledge_base(
    dry_run: bool = False,
    force: bool = False
) -> Dict:
    """
    Update the knowledge base with any changes.

    Args:
        dry_run: If True, only report what would change
        force: Re-ingest all files regardless of changes

    Returns:
        Summary of changes made
    """
    if force:
        # Force re-ingest everything
        results = force_reingest_all()
        return results

    changes = scan_for_changes()

    summary = {
        "new": [],
        "modified": [],
        "deleted": [],
        "errors": [],
        "dry_run": dry_run
    }

    # Process new files
    for file_path, category in changes["new"]:
        if dry_run:
            summary["new"].append(f"{file_path.name} [{category}]")
        else:
            try:
                result = ingest_document(str(file_path), category)
                if result["status"] == "success":
                    summary["new"].append(f"{file_path.name}: {result['chunks']} chunks")
            except Exception as e:
                summary["errors"].append(f"{file_path.name}: {str(e)}")

    # Process modified files
    for file_path, category in changes["modified"]:
        if dry_run:
            summary["modified"].append(f"{file_path.name} [{category}]")
        else:
            try:
                result = ingest_document(str(file_path), category, force=True)
                if result["status"] == "success":
                    summary["modified"].append(f"{file_path.name}: {result['chunks']} chunks")
            except Exception as e:
                summary["errors"].append(f"{file_path.name}: {str(e)}")

    # Process deleted files
    for file_path in changes["deleted"]:
        if dry_run:
            summary["deleted"].append(str(file_path.name))
        else:
            try:
                delete_document(str(file_path))
                summary["deleted"].append(str(file_path.name))
            except Exception as e:
                summary["errors"].append(f"Delete {file_path.name}: {str(e)}")

    return summary


def force_reingest_all() -> Dict:
    """Force re-ingest all documents in source directories."""
    from .ingest import ingest_directory

    summary = {
        "categories": {},
        "total_docs": 0,
        "total_chunks": 0,
        "errors": []
    }

    category_dirs = {
        "books": "trading_books",
        "notes": "notes",
        "strategies": "strategies",
        "risk_management": "risk_management",
        "technical_analysis": "technical_analysis",
        "market_psychology": "market_psychology",
        "fundamental_analysis": "fundamental_analysis",
    }

    for dir_name, category in category_dirs.items():
        dir_path = SOURCES_DIR / dir_name
        if not dir_path.exists():
            continue

        logger.info(f"Re-ingesting {dir_name} -> {category}")

        try:
            results = ingest_directory(str(dir_path), category, force=True)

            success_count = sum(1 for r in results if r.get("status") == "success")
            chunk_count = sum(r.get("chunks", 0) for r in results if r.get("status") == "success")

            summary["categories"][category] = {
                "documents": success_count,
                "chunks": chunk_count
            }
            summary["total_docs"] += success_count
            summary["total_chunks"] += chunk_count

            for r in results:
                if r.get("status") == "error":
                    summary["errors"].append(f"{r.get('file')}: {r.get('error')}")

        except Exception as e:
            summary["errors"].append(f"{dir_name}: {str(e)}")

    return summary


def watch_for_changes(interval: int = 60):
    """
    Watch source directories for changes and auto-update.

    Args:
        interval: Check interval in seconds
    """
    import time

    logger.info(f"Watching for changes every {interval} seconds...")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            changes = scan_for_changes()

            total_changes = (
                len(changes["new"]) +
                len(changes["modified"]) +
                len(changes["deleted"])
            )

            if total_changes > 0:
                logger.info(f"Found {total_changes} changes, updating...")
                summary = update_knowledge_base()

                if summary["new"]:
                    logger.info(f"Added: {len(summary['new'])} documents")
                if summary["modified"]:
                    logger.info(f"Modified: {len(summary['modified'])} documents")
                if summary["deleted"]:
                    logger.info(f"Deleted: {len(summary['deleted'])} documents")
                if summary["errors"]:
                    logger.warning(f"Errors: {len(summary['errors'])}")

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Watch stopped")


def print_status():
    """Print current knowledge base status."""
    status = get_ingestion_status()

    print("\n" + "=" * 50)
    print("KNOWLEDGE BASE STATUS")
    print("=" * 50)
    print(f"Total Documents: {status['total_documents']}")
    print(f"Total Chunks: {status['total_chunks']}")
    print(f"ChromaDB Collection: {status['collection_count']} vectors")
    print(f"Last Updated: {status.get('last_updated', 'Never')}")

    if status.get("by_category"):
        print("\nBy Category:")
        for cat, info in sorted(status["by_category"].items()):
            print(f"  {cat}: {info['documents']} docs, {info['chunks']} chunks")

    print("=" * 50)


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Update knowledge base")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Show what would change without updating")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force re-ingest all documents")
    parser.add_argument("-w", "--watch", action="store_true",
                        help="Watch for changes continuously")
    parser.add_argument("-i", "--interval", type=int, default=60,
                        help="Watch interval in seconds")
    parser.add_argument("-s", "--status", action="store_true",
                        help="Show knowledge base status")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.watch:
        watch_for_changes(args.interval)
    else:
        print("\nScanning for changes...")
        summary = update_knowledge_base(dry_run=args.dry_run, force=args.force)

        if args.dry_run:
            print("\n[DRY RUN] Would make the following changes:")
        else:
            print("\nUpdate complete:")

        if summary.get("categories"):
            # Force reingest summary
            print(f"\nTotal: {summary['total_docs']} documents, {summary['total_chunks']} chunks")
            for cat, info in summary["categories"].items():
                print(f"  {cat}: {info['documents']} docs, {info['chunks']} chunks")
        else:
            # Incremental update summary
            if summary["new"]:
                print(f"\nNew ({len(summary['new'])}):")
                for item in summary["new"]:
                    print(f"  + {item}")

            if summary["modified"]:
                print(f"\nModified ({len(summary['modified'])}):")
                for item in summary["modified"]:
                    print(f"  ~ {item}")

            if summary["deleted"]:
                print(f"\nDeleted ({len(summary['deleted'])}):")
                for item in summary["deleted"]:
                    print(f"  - {item}")

            if not (summary["new"] or summary["modified"] or summary["deleted"]):
                print("\nNo changes detected.")

        if summary.get("errors"):
            print(f"\nErrors ({len(summary['errors'])}):")
            for err in summary["errors"]:
                print(f"  ! {err}")
