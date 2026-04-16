import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)

# Load environment variables
load_dotenv()

# SSL Fix
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768,
)

# Vector Store
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

# Tavily
tavily_crawl = TavilyCrawl()

# 🔒 Concurrency limit (VERY IMPORTANT)
semaphore = asyncio.Semaphore(2)


async def add_batch_with_retry(batch: List[Document], batch_num: int, total_batches: int):
    """Add batch with retry + exponential backoff"""
    async with semaphore:
        for attempt in range(5):
            try:
                await vectorstore.aadd_documents(batch)

                log_success(
                    f"✅ Batch {batch_num}/{total_batches} added ({len(batch)} docs)"
                )
                return True

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 2 ** attempt
                    log_warning(
                        f"⚠️ Batch {batch_num} rate-limited. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    log_error(f"❌ Batch {batch_num} failed: {e}")
                    return False

        log_error(f"❌ Batch {batch_num} failed after retries")
        return False


async def index_documents_async(documents: List[Document], batch_size: int = 5):
    """Index documents safely with batching + concurrency control"""

    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"📚 Preparing to index {len(documents)} documents",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i: i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(f"📦 Created {len(batches)} batches (size={batch_size})")

    # Run tasks
    tasks = [
        add_batch_with_retry(batch, i + 1, len(batches))
        for i, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)

    success_count = sum(results)

    if success_count == len(batches):
        log_success("🎉 All batches indexed successfully!")
    else:
        log_warning(
            f"⚠️ {success_count}/{len(batches)} batches succeeded"
        )


async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    # Crawl
    log_info(
        "🌐 Crawling https://python.langchain.com/",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
    })

    all_docs = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]}
        )
        for result in res["results"]
    ]

    log_success(f"✅ Crawled {len(all_docs)} pages")

    # Split
    log_header("DOCUMENT CHUNKING PHASE")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # 🔥 safer size
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"✂️ Created {len(split_docs)} chunks from {len(all_docs)} docs"
    )

    # Index
    await index_documents_async(split_docs, batch_size=5)

    # Summary
    log_header("PIPELINE COMPLETE")

    log_success("🎉 Ingestion completed successfully!")
    log_info(f"URLs: {len(res['results'])}")
    log_info(f"Documents: {len(all_docs)}")
    log_info(f"Chunks: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())