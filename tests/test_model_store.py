import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from deepdoc.common import model_store as ms


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _create_combined_repo_layout(root: Path) -> None:
    # Vision bundle
    for name in ms.BUNDLES["vision"].required_files:
        _touch(root / "vision" / name)

    # XGB bundle
    for name in ms.BUNDLES["xgb"].required_files:
        _touch(root / "xgb" / name)

    # Tokenizer bundle
    for name in ms.BUNDLES["tokenizer"].required_files:
        _touch(root / "tokenizer" / name)


class TestModelStoreSharedRepo(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_resolve_from_single_combined_repo(self) -> None:
        calls: list[dict[str, str | None]] = []

        def snapshot_download(
            model_id: str | None = None,
            repo_id: str | None = None,
            revision: str | None = None,
            cache_dir: str | None = None,
            local_dir: str | None = None,
            local_dir_use_symlinks: bool | None = None,
            local_files_only: bool | None = None,
        ) -> str:
            resolved_repo = model_id or repo_id
            calls.append(
                {
                    "repo": resolved_repo,
                    "revision": revision,
                    "cache_dir": cache_dir,
                    "local_dir": local_dir,
                }
            )
            root = Path(local_dir) if local_dir else Path(cache_dir or ".")
            root.mkdir(parents=True, exist_ok=True)
            _create_combined_repo_layout(root)
            return str(root)

        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ms.GLOBAL_MODELSCOPE_REPO_ENV] = "Xorbits/deepdoc"
            os.environ[ms.GLOBAL_MODELSCOPE_REVISION_ENV] = "v1"

            with patch.object(ms, "_import_modelscope_snapshot_download", return_value=snapshot_download):
                vision_dir = Path(ms.resolve_bundle_dir("vision", model_home=tmp, provider="modelscope", offline=False))
                xgb_dir = Path(ms.resolve_bundle_dir("xgb", model_home=tmp, provider="modelscope", offline=False))
                tok_dir = Path(ms.resolve_bundle_dir("tokenizer", model_home=tmp, provider="modelscope", offline=False))

            expected_root = (Path(tmp) / "modelscope" / "Xorbits__deepdoc" / "v1").resolve()
            self.assertEqual(vision_dir.resolve(), (expected_root / "vision").resolve())
            self.assertEqual(xgb_dir.resolve(), (expected_root / "xgb").resolve())
            self.assertEqual(tok_dir.resolve(), (expected_root / "tokenizer").resolve())

            self.assertGreaterEqual(len(calls), 1)
            for call in calls:
                self.assertEqual(call["repo"], "Xorbits/deepdoc")
                self.assertEqual(call["revision"], "v1")
                self.assertEqual(Path(call["local_dir"]).resolve(), expected_root)

    def test_bundle_specific_repo_does_not_use_shared_dir(self) -> None:
        calls: list[dict[str, str | None]] = []

        def snapshot_download(
            model_id: str | None = None,
            repo_id: str | None = None,
            revision: str | None = None,
            cache_dir: str | None = None,
            local_dir: str | None = None,
            local_dir_use_symlinks: bool | None = None,
            local_files_only: bool | None = None,
        ) -> str:
            resolved_repo = model_id or repo_id
            calls.append(
                {
                    "repo": resolved_repo,
                    "revision": revision,
                    "cache_dir": cache_dir,
                    "local_dir": local_dir,
                }
            )
            root = Path(local_dir) if local_dir else Path(cache_dir or ".")
            root.mkdir(parents=True, exist_ok=True)
            _create_combined_repo_layout(root)
            return str(root)

        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ms.GLOBAL_MODELSCOPE_REPO_ENV] = "Xorbits/deepdoc"
            os.environ[ms.BUNDLES["vision"].repo_env] = "OtherOrg/vision-only"

            with patch.object(ms, "_import_modelscope_snapshot_download", return_value=snapshot_download):
                _ = ms.resolve_bundle_dir("vision", model_home=tmp, provider="modelscope", offline=False)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["repo"], "OtherOrg/vision-only")
            self.assertEqual(Path(calls[0]["local_dir"]).resolve(), (Path(tmp) / "vision").resolve())
