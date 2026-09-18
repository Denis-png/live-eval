"""Standalone plotting CLI.

    python -m framework.plotting <session_dir> [--out DIR]
    python -m framework.plotting framework/data/runs          # every session under it

A path holding results.json is a session; any other directory is searched for
the sessions beneath it, so refreshing a whole archive is one command in any
shell, with no loop to write.
"""
import argparse
import os
import sys

from framework.plotting.session import render_session


def find_sessions(path: str) -> list[str]:
    """`path` itself when it is a session, else every session beneath it, sorted.
    A compare_models `comparison/` folder is not a session."""
    if os.path.isfile(os.path.join(path, "results.json")):
        return [path]
    found = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in ("plots", "generated", "comparison"))
        if "results.json" in files:
            found.append(root)
            dirs[:] = []           # a session's own subfolders hold no sessions
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m framework.plotting",
        description="Render figures for run sessions: a session directory "
                    "(framework/data/runs/<task>/<session>/), or any directory "
                    "to render every session beneath it.",
    )
    parser.add_argument("paths", nargs="+",
                        help="Session directories, or directories to search for sessions")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: <session_dir>/plots/); one session only")
    args = parser.parse_args(argv)

    sessions = [s for p in args.paths for s in find_sessions(p)]
    if not sessions:
        sys.exit(f"[ERROR] no session (a directory with results.json) in: "
                 f"{', '.join(args.paths)}")
    if args.out and len(sessions) > 1:
        sys.exit(f"[ERROR] --out names one directory, but {len(sessions)} sessions "
                 "were found; drop --out to write each into its own plots/.")

    failed = []
    for i, session in enumerate(sessions, 1):
        if len(sessions) > 1:
            print(f"[{i}/{len(sessions)}] {session}", flush=True)
        try:
            written = render_session(session, args.out)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            written = []
        if not written:
            failed.append(session)
    if failed:
        sys.exit(f"[ERROR] no figures for {len(failed)} of {len(sessions)} session(s): "
                 + ", ".join(failed))


if __name__ == "__main__":
    main()
