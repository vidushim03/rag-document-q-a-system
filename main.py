import atexit
import os
import time
from pathlib import Path

from src.chunker import chunk_data
from src.data_loader import load_pdf
from src.embedding import get_embeddings
from src.agent import run_agent
from src.knowledge_base import QdrantKnowledgeBase
from src.ocr_loader import load_pdf_with_ocr
from src.retriever import retrieve_docs


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def load_documents_from_pdf(pdf_path: Path):
    try:
        docs = load_pdf(str(pdf_path))
        if not docs or len(docs[0].page_content.strip()) < 20:
            print(f"{pdf_path.name} seems scanned. Using OCR.")
            docs = load_pdf_with_ocr(str(pdf_path))
    except Exception as exc:
        print(f"Normal loading failed for {pdf_path.name}. Using OCR. Error: {exc}")
        docs = load_pdf_with_ocr(str(pdf_path))

    for doc in docs:
        doc.metadata["source"] = pdf_path.name

    return docs


def build_knowledge_base():
    embeddings = get_embeddings(mode="tfidf")
    configured_path = os.getenv("QDRANT_PATH", "knowledge_base").strip()
    base_dir = Path(configured_path)
    if not base_dir.is_absolute():
        base_dir = APP_DIR / base_dir

    return QdrantKnowledgeBase(
        embeddings=embeddings,
        embedding_mode="tfidf",
        base_dir=base_dir,
        url=os.getenv("QDRANT_URL") or None,
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


def ingest_data_folder(kb: QdrantKnowledgeBase):
    if not DATA_DIR.exists():
        print(f"Data folder not found: {DATA_DIR}")
        return

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    for pdf_path in pdf_files:
        docs = load_documents_from_pdf(pdf_path)
        if not docs:
            print(f"Skipping {pdf_path.name}: no text extracted.")
            continue

        chunks = chunk_data(docs)
        result = kb.add_documents(chunks, source_name=pdf_path.name)
        print(
            f"Indexed {result['source_name']} with {result['chunk_count']} chunks."
        )


def main():
    try:
        kb = build_knowledge_base()
    except Exception as exc:
        print(f"Error: {exc}")
        return

    atexit.register(kb.close)
    ingest_data_folder(kb)

    if kb.source_count() == 0:
        print("No sources are available in the knowledge base.")
        return

    print("\nQdrant knowledge base ready.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() == "exit":
            break

        start = time.time()

        try:
            agent_result = run_agent(query, kb)
            answer = agent_result.get("generation", "Sorry, I could not generate an answer.")
        except Exception as exc:
            print(f"\nError: {exc}")
            continue

        end = time.time()
        print("\nAnswer:", answer)
        print("Response Time:", round(end - start, 2), "seconds\n")


if __name__ == "__main__":
    main()
