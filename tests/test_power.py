import argparse
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from piano_clamp.power import add_caffeinate_arguments, maybe_caffeinate, validate_caffeinate_flags


class CaffeinateTests(unittest.TestCase):
    def test_parser_accepts_caffeinate_arguments(self):
        parser = argparse.ArgumentParser()
        add_caffeinate_arguments(parser)
        args = parser.parse_args(["--caffeinate", "--caffeinate-flags", "dis"])
        self.assertTrue(args.caffeinate)
        self.assertEqual(args.caffeinate_flags, "dis")

    def test_validate_flags_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "unsupported caffeinate flag"):
            validate_caffeinate_flags("xyz")

    def test_non_macos_caffeinate_is_ignored_with_note(self):
        buffer = io.StringIO()
        with patch("piano_clamp.power.sys.platform", "linux"):
            with redirect_stderr(buffer):
                with maybe_caffeinate(True, flags="is"):
                    pass
        self.assertIn("ignored because this is not macOS", buffer.getvalue())

    def test_macos_caffeinate_spawns_helper_and_cleans_up(self):
        process = Mock()
        process.poll.return_value = None
        with patch("piano_clamp.power.sys.platform", "darwin"):
            with patch("piano_clamp.power.shutil.which", return_value="/usr/bin/caffeinate"):
                with patch("piano_clamp.power.subprocess.Popen", return_value=process) as popen:
                    with maybe_caffeinate(True, flags="is"):
                        pass
        popen.assert_called_once()
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
