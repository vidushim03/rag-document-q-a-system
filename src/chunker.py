from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_data(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # increased from 300 → more context per chunk
        chunk_overlap=150  # increased from 50 → better continuity between chunks
    )
    chunks = splitter.split_documents(docs)
    print("Chunks Created:", len(chunks))
    return chunks