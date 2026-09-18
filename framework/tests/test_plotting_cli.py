"""python -m framework.plotting: a session, or every session under a folder."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework.plotting import __main__ as cli


def _session(root, *parts):
    path = os.path.join(root, *parts)
    os.makedirs(os.path.join(path, "generated"))
    os.makedirs(os.path.join(path, "plots"))
    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump({"meta": {}, "results": {}}, f)
    return path


class FindSessionsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def test_a_session_is_itself(self):
        s = _session(self.root, "gec", "a")
        self.assertEqual(cli.find_sessions(s), [s])

    def test_a_folder_yields_every_session_beneath_it_sorted(self):
        b = _session(self.root, "spam", "b")
        a = _session(self.root, "gec", "a")
        os.makedirs(os.path.join(self.root, "spam", "comparison"))
        with open(os.path.join(self.root, "spam", "comparison", "results.json"), "w") as f:
            f.write("{}")
        self.assertEqual(cli.find_sessions(self.root), [a, b])


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.a = _session(self.root, "gec", "a")
        self.b = _session(self.root, "spam", "b")

    def _run(self, *argv):
        with mock.patch.object(cli, "render_session", return_value=["x.png"]) as render, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main(list(argv))
        return [c.args[0] for c in render.call_args_list]

    def test_a_runs_root_renders_every_session(self):
        self.assertEqual(self._run(self.root), [self.a, self.b])

    def test_out_is_refused_for_several_sessions(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run(self.root, "--out", os.path.join(self.root, "figs"))
        self.assertIn("2 sessions", str(ctx.exception.code))

    def test_a_folder_with_no_session_is_an_error(self):
        empty = os.path.join(self.root, "empty")
        os.makedirs(empty)
        with self.assertRaises(SystemExit) as ctx:
            self._run(empty)
        self.assertIn("no session", str(ctx.exception.code))


if __name__ == "__main__":
    unittest.main()
