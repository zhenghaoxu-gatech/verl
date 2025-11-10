#!/usr/bin/env python3

"""Block until a local release marker exists (or a timeout elapses)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a local file to appear before exiting.",
    )
    parser.add_argument(
        "--signal-path",
        required=True,
        help="File path to watch for.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between file-system checks (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=0.0,
        help="Maximum seconds to wait; 0 disables the timeout (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    marker = Path(args.signal_path)
    marker.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    while True:
        if marker.exists():
            return 0
        if args.timeout_secs > 0 and (time.time() - start) > args.timeout_secs:
            print(
                f"Timed out waiting for {marker} after {args.timeout_secs} seconds.",
                file=sys.stderr,
            )
            return 1
        time.sleep(max(args.poll_interval, 0.5))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted while waiting for release signal.", file=sys.stderr)
        sys.exit(130)
