"""Standalone plotting CLI: python -m framework.plotting <session_dir> [--out DIR]"""
import argparse
import sys

from framework.plotting.session import render_session


def main():
    parser = argparse.ArgumentParser(
        prog="python -m framework.plotting",
        description="Render figures for a run session "
                    "(framework/data/runs/<task>/<session>/).",
    )
    parser.add_argument("session_dir", help="Path to a run session directory")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: <session_dir>/plots/)")
    args = parser.parse_args()
    try:
        written = render_session(args.session_dir, args.out)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    if not written:
        sys.exit("[ERROR] no figures were produced — see the warnings above.")


if __name__ == "__main__":
    main()
