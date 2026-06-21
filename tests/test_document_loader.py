import tempfile
import unittest
from pathlib import Path

from src.document_loader import documents_to_records, scan_document_folder


class DocumentLoaderTests(unittest.TestCase):
    def test_scan_folder_loads_text_with_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            document_path = folder / "notes.txt"
            document_path.write_text("Line one.\n\nLine two.", encoding="utf-8")

            documents = scan_document_folder(folder)
            records = documents_to_records(documents)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source"], "notes.txt")
            self.assertEqual(records[0]["type"], "txt")
            self.assertIn("Line one.", records[0]["text"])


if __name__ == "__main__":
    unittest.main()
