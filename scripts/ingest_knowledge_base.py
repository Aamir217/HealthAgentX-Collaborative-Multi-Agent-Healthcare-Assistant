#!/usr/bin/env python
"""CLI entrypoint to (re)build the RAGWire vector index from the local
medical knowledge base at data/knowledge_base/.

Usage:
    python scripts/ingest_knowledge_base.py [--reset]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.ragwire.ingest import ingest_knowledge_base  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="Drop and rebuild the vector collection from scratch."
    )
    args = parser.parse_args()

    count = ingest_knowledge_base(reset=args.reset)
    print(f"Ingested {count} chunks into the RAGWire vector store.")


if __name__ == "__main__":
    main()
