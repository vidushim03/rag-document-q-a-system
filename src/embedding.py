import hashlib
import math
from typing import List

from langchain.embeddings.base import Embeddings


class TFIDFEmbeddings(Embeddings):
    def __init__(self, dim=384):
        self.dim = dim

    def _text_to_vector(self, text: str) -> List[float]:
        text = text.lower().strip()
        words = text.split()

        vector = [0.0] * self.dim
        word_counts = {}

        for word in words:
            word = "".join(char for char in word if char.isalnum())
            if len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            tf = count / max(len(words), 1)
            idf = math.log(1 + 1 / (1 + count))
            vector[index] += tf * idf

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > 0:
            vector = [value / magnitude for value in vector]

        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)


def get_embeddings(mode="tfidf"):
    if mode == "tfidf":
        print("Using TF-IDF embeddings")
        return TFIDFEmbeddings(dim=384)

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("Using Transformer embeddings")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as exc:
        print(
            "Transformer embeddings are unavailable, falling back to TF-IDF. "
            f"Reason: {exc}"
        )
        return TFIDFEmbeddings(dim=384)
