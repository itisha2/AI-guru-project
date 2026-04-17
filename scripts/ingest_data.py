"""
Data ingestion script — run once before starting the app.

Usage:
    python scripts/ingest_data.py          # first-time ingestion
    python scripts/ingest_data.py --force  # re-ingest even if data exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data_loader import load_all_documents
from backend.vector_store import collection_exists, create_vector_store


def main():
    parser = argparse.ArgumentParser(description="Ingest Gita datasets into ChromaDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the vector store already exists",
    )
    args = parser.parse_args()

    if collection_exists() and not args.force:
        print(
            "Vector store already populated.\n"
            "Run with --force to re-index from scratch."
        )
        return

    print("=" * 60)
    print("  AI Guru — Data Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/2] Loading documents from both datasets …")
    documents = load_all_documents(force=args.force)
    print(f"      Total documents: {len(documents)}")

    print("\n[2/2] Embedding and indexing into ChromaDB …")
    print("      (This may take several minutes on first run)")
    create_vector_store(documents)

    print("\n✓ Ingestion complete. You can now start the app:")
    print("  streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
