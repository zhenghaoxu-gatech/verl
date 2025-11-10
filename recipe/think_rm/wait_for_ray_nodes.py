#!/usr/bin/env python3
"""Wait until the expected number of Ray nodes are registered."""

from __future__ import annotations

import argparse
import os
import sys
import time

import ray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-nodes",
        type=int,
        default=None,
        help="Expected total Ray nodes (head + workers). "
        "Defaults to $RAY_EXPECTED_NODES or 1.",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=None,
        help="Max seconds to wait. Defaults to $RAY_WAIT_FOR_NODES_TIMEOUT or 1800.",
    )
    parser.add_argument(
        "--address",
        default=os.environ.get("RAY_ADDRESS", "auto"),
        help='Ray address to join. Defaults to "auto" or $RAY_ADDRESS.',
    )
    parser.add_argument(
        "--namespace",
        default="ray_wait_check",
        help="Ray namespace used for probing (default: ray_wait_check).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between cluster state checks after connection (default: 10).",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=5.0,
        help="Seconds between attempts to connect to the head node (default: 5).",
    )
    return parser.parse_args()


def _resolve_expected(raw: int | None) -> int:
    if raw is not None:
        return raw
    env = os.environ.get("RAY_EXPECTED_NODES")
    if env:
        return int(env)
    return 1


def _resolve_timeout(raw: int | None) -> int:
    if raw is not None:
        return raw
    env = os.environ.get("RAY_WAIT_FOR_NODES_TIMEOUT")
    if env:
        return int(env)
    return 1800


def main() -> int:
    args = _parse_args()
    expected = max(_resolve_expected(args.expected_nodes), 1)
    timeout = _resolve_timeout(args.timeout_secs)
    poll_interval = max(args.poll_interval, 1.0)
    retry_interval = max(args.retry_interval, 1.0)
    address = args.address
    namespace = args.namespace

    deadline = time.time() + timeout if timeout > 0 else None
    remaining_repr = lambda: "inf" if deadline is None else f"{int(deadline - time.time())}s"

    print(
        f"[wait-for-ray] Waiting for {expected} Ray nodes to register "
        f"(timeout: {'no limit' if deadline is None else f'{timeout}s'})",
        flush=True,
    )

    while True:
        try:
            ray.init(address=address, namespace=namespace)
            break
        except Exception as exc:  # pylint: disable=broad-except
            if deadline is not None and time.time() >= deadline:
                msg = (
                    f"Timed out after {timeout}s while trying to connect to Ray head "
                    f"at {address}: {exc}"
                )
                print(f"[wait-for-ray] {msg}", flush=True)
                return 1
            print(
                f"[wait-for-ray] Ray head not ready yet ({exc!r}); "
                f"retrying in {retry_interval}s (remaining {remaining_repr()})",
                flush=True,
            )
            time.sleep(retry_interval)

    try:
        while True:
            nodes = [
                node
                for node in ray.nodes()
                if node.get("Alive") or node.get("alive")
            ]
            if len(nodes) >= expected:
                print(
                    f"[wait-for-ray] Detected {len(nodes)} Ray nodes "
                    f"(expected {expected}). Proceeding.",
                    flush=True,
                )
                return 0
            if deadline is not None and time.time() >= deadline:
                print(
                    f"[wait-for-ray] Timed out after {timeout}s waiting for Ray nodes "
                    f"(have {len(nodes)}, expected {expected}).",
                    flush=True,
                )
                return 1
            print(
                f"[wait-for-ray] Waiting for Ray nodes: "
                f"{len(nodes)}/{expected} alive. "
                f"Checking again in {poll_interval}s (remaining {remaining_repr()}).",
                flush=True,
            )
            time.sleep(poll_interval)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
