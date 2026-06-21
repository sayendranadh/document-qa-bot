import unittest

from src.embeddings import HashEmbeddingProvider
from src.models import DocumentChunk
from src.vector_store import InMemoryVectorStore


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_most_similar_chunk_first(self) -> None:
        chunks = [
            DocumentChunk("a", "doc.txt", "alpha beta project", {}),
            DocumentChunk("b", "doc.txt", "invoice payment finance", {}),
            DocumentChunk("c", "doc.txt", "neural retrieval embeddings", {}),
        ]
        provider = HashEmbeddingProvider(dimensions=128)
        vectors = provider.embed_documents([chunk.text for chunk in chunks])

        store = InMemoryVectorStore()
        store.add(chunks, vectors)
        results = store.search(provider.embed_query("semantic retrieval"), top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, "c")
        self.assertEqual(len(results), 2)

    def test_add_requires_matching_vectors(self) -> None:
        store = InMemoryVectorStore()
        with self.assertRaises(ValueError):
            store.add([DocumentChunk("a", "doc.txt", "text", {})], [])


if __name__ == "__main__":
    unittest.main()
