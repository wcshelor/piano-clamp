import tempfile
import unittest
from pathlib import Path

from piano_clamp.clamp_backend import ClampBackend, ClampBackendError
from piano_clamp.embeddings import C2_CHECKPOINT, SAAS_CHECKPOINT


C2_CONFIG_TEXT = '''CLAMP3_WEIGHTS_PATH = (
    "weights_clamp3_c2" +
    "_h_size_768_test.pth"
)
'''

SAAS_CONFIG_TEXT = '''CLAMP3_WEIGHTS_PATH = (
    "weights_clamp3_saas" +
    "_h_size_768_test.pth"
)
'''

NEUTRAL_CONFIG_TEXT = '''CLAMP3_WEIGHTS_PATH = (
    "weights_other" +
    "_h_size_768_test.pth"
)
'''

DUMMY_EXTRACTOR = '''checkpoint_path = "dummy"
x = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint['model'])'''

DUMMY_UTILS = '''class Foo:
    def bar(self):
        self.text_model = AutoModel.from_pretrained(text_model_name) # Load the text model
'''


def _make_vendor(root: Path, config_text: str) -> Path:
    vendor = root / "vendor"
    code = vendor / "code"
    code.mkdir(parents=True)
    (code / "extract_clamp3.py").write_text(DUMMY_EXTRACTOR, encoding="utf-8")
    (code / "utils.py").write_text(DUMMY_UTILS, encoding="utf-8")
    (code / "config.py").write_text(config_text, encoding="utf-8")
    return vendor


def _make_checkpoint(root: Path, name: str) -> Path:
    # create a dummy file as checkpoint
    p = root / name
    p.write_bytes(b"dummy checkpoint")
    return p


def _backend_config(vendor: Path, c2_path: Path, saas_path: Path, *, model_space: str, tmp: Path):
    # Use cpu and disable low_memory patching to keep vendor files minimal,
    # but also test that rewriting works regardless of patching flag.
    base = {
        "vendor_root": str(vendor),
        "checkpoint_path": str(c2_path),
        "saas_checkpoint_path": str(saas_path),
        "device": "cpu",
        "low_memory_checkpoint_loading": False,
        "verify_checkpoint_sha256": False,
    }
    return base


class ClampBackendRuntimeConfigTests(unittest.TestCase):
    def test_c2_runtime_keeps_c2_when_vendor_is_c2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, C2_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="c2", tmp=root)
            backend = ClampBackend(cfg, model_space="c2")
            dest = root / "runtime_c2_from_c2"
            backend._runtime_code(dest)
            text = (dest / "config.py").read_text(encoding="utf-8")
            self.assertIn('"weights_clamp3_c2"', text)
            self.assertNotIn('"weights_clamp3_saas"', text)
            # symlink should point to C2 checkpoint with correct name
            self.assertTrue((dest / C2_CHECKPOINT).is_symlink())
            self.assertEqual((dest / C2_CHECKPOINT).resolve(), c2.resolve())

    def test_c2_runtime_rewrites_saas_vendor_to_c2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, SAAS_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="c2", tmp=root)
            backend = ClampBackend(cfg, model_space="c2")
            dest = root / "runtime_c2_from_saas"
            backend._runtime_code(dest)
            text = (dest / "config.py").read_text(encoding="utf-8")
            self.assertIn('"weights_clamp3_c2"', text)
            self.assertNotIn('"weights_clamp3_saas"', text)
            self.assertTrue((dest / C2_CHECKPOINT).is_symlink())

    def test_saas_runtime_keeps_saas_when_vendor_is_saas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, SAAS_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="saas", tmp=root)
            backend = ClampBackend(cfg, model_space="saas")
            dest = root / "runtime_saas_from_saas"
            backend._runtime_code(dest)
            text = (dest / "config.py").read_text(encoding="utf-8")
            self.assertIn('"weights_clamp3_saas"', text)
            self.assertNotIn('"weights_clamp3_c2"', text)
            self.assertTrue((dest / SAAS_CHECKPOINT).is_symlink())
            self.assertEqual((dest / SAAS_CHECKPOINT).resolve(), saas.resolve())

    def test_saas_runtime_rewrites_c2_vendor_to_saas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, C2_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="saas", tmp=root)
            backend = ClampBackend(cfg, model_space="saas")
            dest = root / "runtime_saas_from_c2"
            backend._runtime_code(dest)
            text = (dest / "config.py").read_text(encoding="utf-8")
            self.assertIn('"weights_clamp3_saas"', text)
            self.assertNotIn('"weights_clamp3_c2"', text)
            self.assertTrue((dest / SAAS_CHECKPOINT).is_symlink())

    def test_c2_runtime_with_neither_marker_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, NEUTRAL_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="c2", tmp=root)
            backend = ClampBackend(cfg, model_space="c2")
            dest = root / "runtime_c2_neutral"
            with self.assertRaisesRegex(ClampBackendError, "cannot derive C2 runtime"):
                backend._runtime_code(dest)

    def test_saas_runtime_with_neither_marker_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, NEUTRAL_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="saas", tmp=root)
            backend = ClampBackend(cfg, model_space="saas")
            dest = root / "runtime_saas_neutral"
            with self.assertRaisesRegex(ClampBackendError, "cannot derive SAAS runtime"):
                backend._runtime_code(dest)

    def test_c2_runtime_never_attempts_to_reference_saas_weights(self):
        # Simulate the bug: vendor is SAAS, C2 job must not keep SAAS config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, SAAS_CONFIG_TEXT)
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = _backend_config(vendor, c2, saas, model_space="c2", tmp=root)
            backend = ClampBackend(cfg, model_space="c2")
            dest = root / "runtime_c2_no_saas"
            backend._runtime_code(dest)
            text = (dest / "config.py").read_text(encoding="utf-8")
            # C2 runtime must not contain SAAS string, which would trigger HF download attempt
            self.assertNotIn("weights_clamp3_saas", text)
            self.assertIn("weights_clamp3_c2", text)

    def test_saas_runtime_preserves_saas_audio_behavior(self):
        # SAAS audio jobs must always reference SAAS, regardless of vendor starting point
        for initial in (C2_CONFIG_TEXT, SAAS_CONFIG_TEXT):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                vendor = _make_vendor(root, initial)
                c2 = _make_checkpoint(root, "c2.pth")
                saas = _make_checkpoint(root, "saas.pth")
                cfg = _backend_config(vendor, c2, saas, model_space="saas", tmp=root)
                backend = ClampBackend(cfg, model_space="saas")
                dest = root / "runtime_saas_audio"
                backend._runtime_code(dest)
                text = (dest / "config.py").read_text(encoding="utf-8")
                self.assertIn('"weights_clamp3_saas"', text)

    def test_checkpoint_validation_remains_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = _make_vendor(root, C2_CONFIG_TEXT)
            # No checkpoint file -> should raise at construction
            missing = root / "missing.pth"
            saas = _make_checkpoint(root, "saas.pth")
            cfg = {
                "vendor_root": str(vendor),
                "checkpoint_path": str(missing),
                "saas_checkpoint_path": str(saas),
                "device": "cpu",
                "verify_checkpoint_sha256": False,
            }
            with self.assertRaisesRegex(ClampBackendError, "checkpoint is unavailable"):
                ClampBackend(cfg, model_space="c2")

    def test_runtime_with_low_memory_patch_still_rewrites_config(self):
        # Ensure config rewriting works even when low_memory patching is enabled
        full_extractor = '''import torch
x = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint['model'])
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)'''
        full_utils = 'self.text_model = AutoModel.from_pretrained(text_model_name) # Load the text model'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor"
            code = vendor / "code"
            code.mkdir(parents=True)
            (code / "extract_clamp3.py").write_text(full_extractor, encoding="utf-8")
            (code / "utils.py").write_text(full_utils, encoding="utf-8")
            (code / "config.py").write_text(SAAS_CONFIG_TEXT, encoding="utf-8")
            c2 = _make_checkpoint(root, "c2.pth")
            saas = _make_checkpoint(root, "saas.pth")
            cfg = {
                "vendor_root": str(vendor),
                "checkpoint_path": str(c2),
                "saas_checkpoint_path": str(saas),
                "device": "cpu",
                "verify_checkpoint_sha256": False,
                "low_memory_checkpoint_loading": True,
            }
            backend = ClampBackend(cfg, model_space="c2")
            dest = root / "runtime_lowmem"
            backend._runtime_code(dest)
            text = (dest / "config.py").read_text(encoding="utf-8")
            self.assertIn('"weights_clamp3_c2"', text)
            self.assertNotIn('"weights_clamp3_saas"', text)
            # verify low-memory patches applied
            extractor_text = (dest / "extract_clamp3.py").read_text(encoding="utf-8")
            self.assertIn("mmap=True", extractor_text)
            self.assertIn("local_files_only=True", extractor_text)
            utils_text = (dest / "utils.py").read_text(encoding="utf-8")
            self.assertIn("local_files_only=True", utils_text)


if __name__ == "__main__":
    unittest.main()
