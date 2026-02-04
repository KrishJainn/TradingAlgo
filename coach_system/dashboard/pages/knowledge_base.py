"""
Knowledge Base Management Page for the 5-Player Coach Dashboard.

Allows users to:
- View ingested documents and their status
- Ingest new documents
- Query the knowledge base
- See knowledge base statistics
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def render_knowledge_base():
    """Render the knowledge base management page."""
    st.header("📚 Knowledge Base Management")

    # Try to import knowledge layer
    try:
        from knowledge_layer import KnowledgeContext
        from knowledge_layer.scripts import (
            get_ingestion_status,
            get_all_sources,
            query_knowledge,
            ingest_document,
            ingest_directory,
            update_knowledge_base,
        )
        knowledge_available = True
    except ImportError as e:
        st.error(f"Knowledge layer not available: {e}")
        st.info("Install required packages: pip install chromadb sentence-transformers pypdf")
        knowledge_available = False
        return

    # Tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Status", "🔍 Query", "📥 Ingest", "🔄 Update"])

    with tab1:
        render_status_tab(get_ingestion_status, get_all_sources)

    with tab2:
        render_query_tab(query_knowledge)

    with tab3:
        render_ingest_tab(ingest_document, ingest_directory)

    with tab4:
        render_update_tab(update_knowledge_base)


def render_status_tab(get_ingestion_status, get_all_sources):
    """Render the status overview tab."""
    st.subheader("Knowledge Base Status")

    try:
        status = get_ingestion_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Documents", status.get("total_documents", 0))

        with col2:
            st.metric("Total Chunks", status.get("total_chunks", 0))

        with col3:
            st.metric("Vector Count", status.get("collection_count", 0))

        # Last updated
        last_updated = status.get("last_updated", "Never")
        st.caption(f"Last updated: {last_updated}")

        # By category breakdown
        if status.get("by_category"):
            st.subheader("Documents by Category")

            for category, info in sorted(status["by_category"].items()):
                with st.expander(f"📁 {category.replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Documents", info["documents"])
                    col2.metric("Chunks", info["chunks"])

        # Sources list
        st.subheader("Ingested Sources")
        sources = get_all_sources()

        if sources:
            for src in sorted(sources, key=lambda x: x.get("category", "")):
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"📄 {src['source']}")
                col2.write(src.get("category", "unknown"))
                col3.write(f"{src.get('chunks', 0)} chunks")
        else:
            st.info("No documents ingested yet. Go to the Ingest tab to add documents.")

    except Exception as e:
        st.error(f"Error getting status: {e}")


def render_query_tab(query_knowledge):
    """Render the query interface tab."""
    st.subheader("Query Knowledge Base")

    query = st.text_input("Enter your query:", placeholder="e.g., How to size positions in volatile markets?")

    col1, col2 = st.columns(2)

    with col1:
        category_filter = st.multiselect(
            "Filter by category (optional):",
            ["trading_books", "notes", "strategies", "risk_management",
             "technical_analysis", "market_psychology", "fundamental_analysis"]
        )

    with col2:
        top_k = st.slider("Number of results:", 1, 10, 5)

    use_mmr = st.checkbox("Use MMR diversity", value=True, help="Diversify results to avoid repetition")

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                try:
                    results = query_knowledge(
                        query=query,
                        categories=category_filter if category_filter else None,
                        top_k=top_k,
                        use_mmr=use_mmr
                    )

                    if results:
                        st.success(f"Found {len(results)} results")

                        for i, result in enumerate(results, 1):
                            with st.expander(
                                f"**{i}. [{result['category']}] {result['source']}** "
                                f"(score: {result['relevance_score']:.3f})"
                            ):
                                st.markdown(result["content"])
                                st.caption(f"Chunk {result['metadata'].get('chunk_index', '?')} of {result['metadata'].get('total_chunks', '?')}")
                    else:
                        st.warning("No results found. Try a different query or check if documents are ingested.")

                except Exception as e:
                    st.error(f"Search error: {e}")
        else:
            st.warning("Please enter a query")


def render_ingest_tab(ingest_document, ingest_directory):
    """Render the document ingestion tab."""
    st.subheader("Ingest Documents")

    st.markdown("""
    Add documents to the knowledge base. Supported formats:
    - **PDF** (`.pdf`)
    - **EPUB** (`.epub`)
    - **Markdown** (`.md`)
    - **Text** (`.txt`)
    """)

    # Category selection
    category = st.selectbox(
        "Select category for documents:",
        ["trading_books", "notes", "strategies", "risk_management",
         "technical_analysis", "market_psychology", "fundamental_analysis"]
    )

    # Get the sources directory path
    project_root = Path(__file__).parent.parent.parent.parent
    sources_dir = project_root / "knowledge_layer" / "sources"

    # Ingest method selection
    ingest_type = st.radio(
        "Ingest method:",
        ["📤 Drag & Drop Upload", "📁 File Path", "📂 Directory"],
        horizontal=True
    )

    if ingest_type == "📤 Drag & Drop Upload":
        # Drag and drop file uploader
        st.markdown("**Drag and drop files here or click to browse:**")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "epub", "md", "txt"],
            accept_multiple_files=True,
            help="Drag and drop files here, or click to browse",
            label_visibility="collapsed"
        )

        force = st.checkbox("Force re-ingest (even if already processed)", key="upload_force")

        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) selected")

            # Show preview of files
            for f in uploaded_files:
                col1, col2 = st.columns([3, 1])
                col1.write(f"📄 {f.name}")
                col2.write(f"{f.size / 1024:.1f} KB")

            if st.button("📥 Ingest Uploaded Files", type="primary"):
                # Save uploaded files to sources directory and ingest
                target_dir = sources_dir / category
                target_dir.mkdir(parents=True, exist_ok=True)

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    # Save to disk
                    file_path = target_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Ingest
                    try:
                        result = ingest_document(str(file_path), category, force=force)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "status": "error",
                            "file": uploaded_file.name,
                            "error": str(e)
                        })

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.empty()
                progress_bar.empty()

                # Show results
                success = sum(1 for r in results if r.get("status") == "success")
                skipped = sum(1 for r in results if r.get("status") == "skipped")
                errors = sum(1 for r in results if r.get("status") == "error")

                if success > 0:
                    st.success(f"✅ Successfully ingested {success} file(s)")
                if skipped > 0:
                    st.info(f"⏭️ Skipped {skipped} file(s) (already ingested)")
                if errors > 0:
                    st.error(f"❌ {errors} file(s) failed")
                    with st.expander("Show errors"):
                        for r in results:
                            if r.get("status") == "error":
                                st.error(f"{r.get('file')}: {r.get('error')}")

                # Show chunk details
                total_chunks = sum(r.get("chunks", 0) for r in results if r.get("status") == "success")
                if total_chunks > 0:
                    st.caption(f"Total chunks created: {total_chunks}")

    elif ingest_type == "📁 File Path":
        # File path input
        file_path = st.text_input(
            "File path:",
            placeholder=str(sources_dir / category / "example.pdf")
        )

        force = st.checkbox("Force re-ingest (even if already processed)", key="path_force")

        if st.button("📥 Ingest File", type="primary"):
            if file_path:
                with st.spinner(f"Ingesting {Path(file_path).name}..."):
                    try:
                        result = ingest_document(file_path, category, force=force)

                        if result["status"] == "success":
                            st.success(f"✅ Ingested: {result['file']} ({result['chunks']} chunks)")
                        elif result["status"] == "skipped":
                            st.info(f"⏭️ Skipped: Already ingested (use force to re-ingest)")
                        else:
                            st.warning(f"Result: {result}")

                    except FileNotFoundError:
                        st.error(f"File not found: {file_path}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a file path")

    else:  # Directory
        # Directory path input
        dir_path = st.text_input(
            "Directory path:",
            value=str(sources_dir / category),
            help="Will ingest all supported files in this directory"
        )

        col1, col2 = st.columns(2)
        with col1:
            recursive = st.checkbox("Include subdirectories", value=True)
        with col2:
            force = st.checkbox("Force re-ingest all", key="dir_force")

        if st.button("📥 Ingest Directory", type="primary"):
            if dir_path:
                with st.spinner(f"Ingesting directory..."):
                    try:
                        results = ingest_directory(dir_path, category, recursive=recursive, force=force)

                        success = sum(1 for r in results if r.get("status") == "success")
                        skipped = sum(1 for r in results if r.get("status") == "skipped")
                        errors = sum(1 for r in results if r.get("status") == "error")

                        st.success(f"✅ Processed {len(results)} files: {success} ingested, {skipped} skipped, {errors} errors")

                        if errors > 0:
                            with st.expander("Show errors"):
                                for r in results:
                                    if r.get("status") == "error":
                                        st.error(f"{r.get('file')}: {r.get('error')}")

                    except FileNotFoundError:
                        st.error(f"Directory not found: {dir_path}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a directory path")

    # Show existing source directories
    st.markdown("---")
    st.subheader("📂 Source Directories")

    for cat_dir in sources_dir.iterdir():
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            files = list(cat_dir.glob("*.*"))
            supported = [f for f in files if f.suffix.lower() in {".pdf", ".epub", ".md", ".txt"}]
            st.write(f"**{cat_dir.name}/**: {len(supported)} documents")


def render_update_tab(update_knowledge_base):
    """Render the update/sync tab."""
    st.subheader("Update Knowledge Base")

    st.markdown("""
    Scan source directories for changes and update the knowledge base:
    - **New files** will be ingested
    - **Modified files** will be re-ingested
    - **Deleted files** will be removed from the database
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Scan for Changes (Dry Run)", help="Preview changes without applying them"):
            with st.spinner("Scanning..."):
                try:
                    summary = update_knowledge_base(dry_run=True)

                    st.info("**Dry Run Results** (no changes made)")

                    if summary.get("new"):
                        st.markdown("**New files to ingest:**")
                        for item in summary["new"]:
                            st.write(f"  + {item}")

                    if summary.get("modified"):
                        st.markdown("**Modified files to re-ingest:**")
                        for item in summary["modified"]:
                            st.write(f"  ~ {item}")

                    if summary.get("deleted"):
                        st.markdown("**Files to remove:**")
                        for item in summary["deleted"]:
                            st.write(f"  - {item}")

                    if not (summary.get("new") or summary.get("modified") or summary.get("deleted")):
                        st.success("No changes detected!")

                except Exception as e:
                    st.error(f"Error scanning: {e}")

    with col2:
        if st.button("🔄 Apply Updates", type="primary", help="Scan and apply all changes"):
            with st.spinner("Updating..."):
                try:
                    summary = update_knowledge_base(dry_run=False)

                    if summary.get("new"):
                        st.success(f"✅ Ingested {len(summary['new'])} new documents")

                    if summary.get("modified"):
                        st.success(f"🔄 Re-ingested {len(summary['modified'])} modified documents")

                    if summary.get("deleted"):
                        st.success(f"🗑️ Removed {len(summary['deleted'])} deleted documents")

                    if not (summary.get("new") or summary.get("modified") or summary.get("deleted")):
                        st.info("No changes to apply")

                    if summary.get("errors"):
                        st.warning(f"⚠️ {len(summary['errors'])} errors occurred")
                        with st.expander("Show errors"):
                            for err in summary["errors"]:
                                st.error(err)

                except Exception as e:
                    st.error(f"Error updating: {e}")

    st.markdown("---")

    if st.button("⚠️ Force Re-ingest All", help="Re-process all documents from scratch"):
        with st.spinner("Re-ingesting all documents..."):
            try:
                summary = update_knowledge_base(force=True)

                st.success(f"✅ Re-ingested {summary.get('total_docs', 0)} documents with {summary.get('total_chunks', 0)} chunks")

                if summary.get("categories"):
                    with st.expander("Details by category"):
                        for cat, info in summary["categories"].items():
                            st.write(f"  {cat}: {info['documents']} docs, {info['chunks']} chunks")

            except Exception as e:
                st.error(f"Error: {e}")


# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Knowledge Base", layout="wide")
    render_knowledge_base()
