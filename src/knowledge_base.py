import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    return str(value)


def _safe_metadata(metadata: dict) -> dict:
    return {str(key): _safe_value(value) for key, value in metadata.items()}


def build_collection_name(mode: str) -> str:
    return f"knowledge_base_{mode}"


class QdrantKnowledgeBase:
    def __init__(
        self,
        embeddings,
        embedding_mode: str,
        base_dir: Optional[Path] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.embeddings = embeddings
        self.embedding_mode = embedding_mode
        self.base_dir = Path(base_dir or os.getenv("QDRANT_PATH", "knowledge_base"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = build_collection_name(embedding_mode)
        self.manifest_path = self.base_dir / "manifest.json"
        self.qdrant_storage_path = self.base_dir / "qdrant_storage"

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant is not installed. Run `python -m pip install qdrant-client`."
            ) from exc

        self._models = __import__(
            "qdrant_client.http.models",
            fromlist=["Distance", "VectorParams", "PointStruct", "Filter"],
        )

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=str(self.qdrant_storage_path))

        self.vector_size = len(self.embeddings.embed_query("knowledge base probe"))
        self.distance = Distance.COSINE
        self.vector_params = VectorParams(size=self.vector_size, distance=self.distance)

        self._ensure_collection()
        self._ensure_manifest_collection()

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"collections": {}}

        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"collections": {}}

    def _save_manifest(self, manifest: dict):
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _ensure_manifest_collection(self):
        manifest = self._load_manifest()
        manifest.setdefault("collections", {})
        manifest["collections"].setdefault(
            self.collection_name,
            {
                "embedding_mode": self.embedding_mode,
                "sources": {},
                "updated_at": _utc_now(),
            },
        )
        self._save_manifest(manifest)

    def _collection_manifest(self, manifest: Optional[dict] = None) -> dict:
        manifest = manifest or self._load_manifest()
        manifest.setdefault("collections", {})
        manifest["collections"].setdefault(
            self.collection_name,
            {
                "embedding_mode": self.embedding_mode,
                "sources": {},
                "updated_at": _utc_now(),
            },
        )
        return manifest["collections"][self.collection_name]

    def _ensure_collection(self):
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_params,
        )

    def _search_points(self, query_vector: List[float], limit: int = 5):
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return getattr(response, "points", response)
        except AttributeError:
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

    def _delete_points_by_source_id(self, source_id: str):
        condition = self._models.FieldCondition(
            key="source_id",
            match=self._models.MatchValue(value=source_id),
        )
        selector = self._models.FilterSelector(
            filter=self._models.Filter(must=[condition])
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=selector,
            wait=True,
        )

    def _find_existing_source(self, source_name: str) -> Optional[dict]:
        for source in self.list_sources():
            if source["source_name"].lower() == source_name.lower():
                return source
        return None

    def add_documents(self, documents: Iterable[Document], source_name: str) -> dict:
        docs = list(documents)
        if not docs:
            return {"source_id": None, "source_name": source_name, "chunk_count": 0}

        existing_source = self._find_existing_source(source_name)
        if existing_source:
            self.delete_source(existing_source["source_id"])

        source_id = str(uuid.uuid4())
        vectors = self.embeddings.embed_documents([doc.page_content for doc in docs])
        point_struct = self._models.PointStruct
        points = []
        page_numbers = set()

        for index, (doc, vector) in enumerate(zip(docs, vectors)):
            metadata = _safe_metadata(doc.metadata)
            page_number = metadata.get("page")
            if page_number is not None:
                page_numbers.add(page_number)

            payload = {
                "source_id": source_id,
                "source_name": source_name,
                "text": doc.page_content,
                "chunk_index": index,
                "page": page_number,
                "metadata": metadata,
                "ingested_at": _utc_now(),
            }
            points.append(
                point_struct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        manifest = self._load_manifest()
        collection_manifest = self._collection_manifest(manifest)
        collection_manifest["sources"][source_id] = {
            "source_id": source_id,
            "source_name": source_name,
            "chunk_count": len(docs),
            "page_count": len(page_numbers),
            "ingested_at": _utc_now(),
        }
        collection_manifest["updated_at"] = _utc_now()
        self._save_manifest(manifest)

        return collection_manifest["sources"][source_id]

    def similarity_search(self, query: str, k: int = 4):
        query_vector = self.embeddings.embed_query(query)
        results = self._search_points(query_vector, limit=k)
        documents = []

        for result in results:
            payload = result.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("source", payload.get("source_name", "Unknown"))
            if payload.get("page") is not None:
                metadata.setdefault("page", payload.get("page"))
            metadata["source_id"] = payload.get("source_id")

            documents.append(
                Document(
                    page_content=payload.get("text", ""),
                    metadata=metadata,
                )
            )

        return documents

    def list_sources(self) -> List[dict]:
        """Derive the source list from indexed points.

        Works with both local and remote Qdrant, so source tracking does not
        depend on the manifest file (which is ephemeral on hosted platforms).
        """
        sources = {}
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=200,
                offset=next_offset,
                with_payload=True,
            )

            for point in points:
                payload = point.payload or {}
                source_id = payload.get("source_id")
                if not source_id:
                    continue

                entry = sources.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "source_name": payload.get("source_name", "Unknown"),
                        "chunk_count": 0,
                        "page_count": 0,
                        "ingested_at": payload.get("ingested_at", ""),
                        "_pages": set(),
                    },
                )
                entry["chunk_count"] += 1
                page = payload.get("page")
                if page is not None:
                    entry["_pages"].add(page)

            if next_offset is None:
                break

        result = []
        for entry in sources.values():
            entry["page_count"] = len(entry.pop("_pages"))
            result.append(entry)

        return sorted(
            result,
            key=lambda item: item.get("ingested_at", ""),
            reverse=True,
        )

    def delete_source(self, source_id: str):
        self._delete_points_by_source_id(source_id)

        manifest = self._load_manifest()
        collection_manifest = self._collection_manifest(manifest)
        collection_manifest.get("sources", {}).pop(source_id, None)
        collection_manifest["updated_at"] = _utc_now()
        self._save_manifest(manifest)

    def clear(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()

        manifest = self._load_manifest()
        collection_manifest = self._collection_manifest(manifest)
        collection_manifest["sources"] = {}
        collection_manifest["updated_at"] = _utc_now()
        self._save_manifest(manifest)

    def source_count(self) -> int:
        return len(self.list_sources())

    def chunk_count(self) -> int:
        return sum(source.get("chunk_count", 0) for source in self.list_sources())

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
