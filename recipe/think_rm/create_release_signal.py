#!/usr/bin/env python3

"""Fan-out a release signal to every alive Ray node by touching a local file."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local release marker on every alive Ray node.",
    )
    parser.add_argument(
        "--signal-path",
        required=True,
        help="Absolute path of the local file each node should create.",
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Address passed to ray.init (default: %(default)s).",
    )
    parser.add_argument(
        "--payload",
        default=None,
        help="Optional content written into the signal file.",
    )
    return parser.parse_args()


@ray.remote
def _write_signal(path: str, payload: str | None) -> str:
    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        payload = f"released_at={time.time()}\n"
    marker.write_text(payload)
    return str(marker)


def main() -> int:
    args = _parse_args()
    ray.init(address=args.ray_address)
    try:
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        if not nodes:
            print("No alive Ray nodes to signal.", file=sys.stderr)
            return 1

        futures = []
        for node in nodes:
            node_id = node.get("NodeID")
            if not node_id:
                continue
            resources = {f"node:{node_id}": 0.001}
            futures.append(
                _write_signal.options(resources=resources).remote(
                    args.signal_path,
                    args.payload,
                )
            )
        if not futures:
            print("Unable to submit release tasks for any node.", file=sys.stderr)
            return 1
        ray.get(futures)
        return 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
