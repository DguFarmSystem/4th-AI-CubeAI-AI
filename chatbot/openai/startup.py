#!/usr/bin/env python3
import os
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_service(url: str, timeout: int = 60) -> bool:
    """Wait for FastAPI service to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("FastAPI service is ready")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False

def auto_ingest_docs():
    """Auto-ingest documents on startup if they exist"""
    docs_path = "/app/docs"
    service_url = "http://localhost:80"
    
    if not os.path.exists(docs_path):
        logger.info(f"Docs directory {docs_path} not found, skipping auto-ingest")
        return
    
    # Check if there are any .txt files
    txt_files = list(Path(docs_path).rglob("*.txt"))
    if not txt_files:
        logger.info(f"No .txt files found in {docs_path}, skipping auto-ingest")
        return
    
    # Wait for FastAPI service to be ready
    if not wait_for_service(service_url):
        logger.error("FastAPI service not ready, skipping auto-ingest")
        return
    
    # Perform ingestion
    try:
        logger.info(f"Auto-ingesting {len(txt_files)} documents from {docs_path}")
        response = requests.post(
            f"{service_url}/ingest",
            json={
                "path": docs_path,
                "pattern": "**/*.txt",
                "chunk_size": 800,
                "chunk_overlap": 120
            },
            timeout=300  # 5 minutes timeout for large document sets
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Auto-ingest completed: {result['files_processed']} files → {result['chunks_created']} chunks")
    except requests.RequestException as e:
        logger.error(f"Auto-ingest failed: {e}")

if __name__ == "__main__":
    auto_ingest_docs()