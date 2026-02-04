"""
Knowledge Layer Scripts.

Provides document ingestion, querying, and update functionality.
"""

from .ingest import (
    ingest_document,
    ingest_directory,
    get_ingestion_status,
    delete_document,
)

from .query import (
    query_knowledge,
    query_by_category,
    get_all_sources,
    search_by_source,
)

from .update import (
    scan_for_changes,
    update_knowledge_base,
    force_reingest_all,
)

__all__ = [
    # Ingestion
    "ingest_document",
    "ingest_directory",
    "get_ingestion_status",
    "delete_document",
    # Query
    "query_knowledge",
    "query_by_category",
    "get_all_sources",
    "search_by_source",
    # Update
    "scan_for_changes",
    "update_knowledge_base",
    "force_reingest_all",
]
