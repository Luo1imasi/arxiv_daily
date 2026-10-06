import subprocess
import sys
import unittest


class RuntimeImportTests(unittest.TestCase):
    def test_runtime_modules_do_not_import_torch_or_sentence_transformers(self):
        script = """
import sys
import arxiv_daily.coarse
import arxiv_daily.executor
import arxiv_daily.lexical
import arxiv_daily.llm
import arxiv_daily.main
import arxiv_daily.reranker.local
import arxiv_daily.retriever.arxiv_retriever
assert "torch" not in sys.modules, sorted(name for name in sys.modules if "torch" in name)
assert "sentence_transformers" not in sys.modules
print("ok")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
