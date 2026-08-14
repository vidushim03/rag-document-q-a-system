import uuid
from typing import List, Optional

from langchain_core.documents import Document


class SessionVectorStore:
    """In-memory, session-scoped vector store.

    Documents live only for the current session and disappear when the app
    restarts or redeploys. Mirrors the QdrantKnowledgeBase interface used by
    the app: add_documents, similarity_search, list_sources, delete_source,
    source_count, clear.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._entries = []
        self._sources = {}
        self._source_order = []

    def add_documents(self, documents, source_name: str) -> dict:
        docs = list(documents)
        if not docs:
            return {
                "source_id": None,
                "source_name": source_name,
                "chunk_count": 0,
            }

        for existing in list(self._sources.values()):
            if existing["source_name"].lower() == source_name.lower():
                self.delete_source(existing["source_id"])

        source_id = str(uuid.uuid4())
        page_numbers = set()

        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata.setdefault("source", source_name)
            page = metadata.get("page")
            if page is not None:
                page_numbers.add(page)
            vector = self.embeddings.embed_query(doc.page_content)
            self._entries.append(
                (
                    Document(page_content=doc.page_content, metadata=metadata),
                    vector,
                    source_id,
                )
            )

        self._sources[source_id] = {
            "source_id": source_id,
            "source_name": source_name,
            "chunk_count": len(docs),
            "page_count": len(page_numbers),
            "ingested_at": source_id,
        }
        self._source_order.append(source_id)
        return self._sources[source_id]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        scored = []
        for doc, vector, _source_id in self._entries:
            dot = sum(a * b for a, b in zip(query_vector, vector))
            scored.append((dot, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _dot, doc in scored[:k]]

    def list_sources(self) -> List[dict]:
        return [self._sources[source_id] for source_id in reversed(self._source_order)]

    def delete_source(self, source_id: str):
        if source_id not in self._sources:
            return
        self._sources.pop(source_id, None)
        self._source_order = [sid for sid in self._source_order if sid != source_id]
        self._entries = [entry for entry in self._entries if entry[2] != source_id]

    def source_count(self) -> int:
        return len(self._sources)

    def clear(self):
        self._entries.clear()
        self._sources.clear()
        self._source_order.clear()
