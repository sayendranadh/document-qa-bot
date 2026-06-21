import unittest

from src.chunker import chunk_documents
from src.models import LoadedDocument


class ChunkerTests(unittest.TestCase):
    def test_short_document_creates_single_chunk(self) -> None:
        documents = [
            LoadedDocument(
                source_id="doc1",
                name="policy.txt",
                text="Short policy text.",
                metadata={"type": "txt"},
            )
        ]

        chunks = chunk_documents(documents, chunk_size=400, chunk_overlap=40)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short policy text.")
        self.assertEqual(chunks[0].metadata["chunk"], 1)

    def test_long_document_uses_overlap(self) -> None:
        text = " ".join(f"sentence{i}" for i in range(250))
        documents = [
            LoadedDocument(source_id="doc1", name="long.txt", text=text, metadata={})
        ]

        chunks = chunk_documents(documents, chunk_size=400, chunk_overlap=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in chunks))

    def test_invalid_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_size=400, chunk_overlap=400)


if __name__ == "__main__":
    unittest.main()
