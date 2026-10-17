from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.training.results import ResultsPaths


class ResultsTests(unittest.TestCase):
    def test_relative_root_is_anchored_and_creates_only_agreed_directories(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = ResultsPaths.resolve("results", base_dir=base)
            self.assertEqual(paths.root, (base / "results").resolve())
            self.assertFalse(paths.root.exists())
            paths.create()
            paths.create()
            self.assertEqual({p.name for p in paths.root.iterdir()}, {"logs", "runs", "files"})
            self.assertTrue(all(p.is_dir() for p in paths.root.iterdir()))
            self.assertTrue(all(not list(p.iterdir()) for p in paths.root.iterdir()))

    def test_absolute_root_is_preserved(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            paths = ResultsPaths.resolve(str(root), base_dir=Path("/unused"))
            self.assertEqual(paths.root, root.resolve())
