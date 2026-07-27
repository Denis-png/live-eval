import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from framework.pipeline import _render_plots

_PATHS = {"session_dir": "/fake/session", "plots_dir": "/fake/session/plots"}


class RenderPlotsTests(unittest.TestCase):
    def test_plots_disabled_skips_render_session(self):
        config = {"output": {"plots": False}}
        with patch("framework.plotting.session.render_session") as mock_render:
            _render_plots(config, _PATHS)
        mock_render.assert_not_called()

    def test_plots_enabled_calls_render_session_with_known_dirs(self):
        config = {"output": {"plots": True}}
        with patch("framework.plotting.session.render_session") as mock_render:
            _render_plots(config, _PATHS)
        mock_render.assert_called_once_with(_PATHS["session_dir"], _PATHS["plots_dir"])

    def test_render_session_failure_does_not_propagate(self):
        # Plotting must never cost a run whose results are already on disk —
        # an exception from render_session must be swallowed (and warned), not raised.
        config = {"output": {"plots": True}}
        stderr = io.StringIO()
        with patch("framework.plotting.session.render_session",
                   side_effect=RuntimeError("boom")):
            try:
                with redirect_stderr(stderr):
                    _render_plots(config, _PATHS)  # must not raise
            except Exception as e:  # pragma: no cover - failure path
                self.fail(f"_render_plots propagated an exception: {e!r}")
        self.assertIn("[WARN]", stderr.getvalue())
        self.assertIn("boom", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
