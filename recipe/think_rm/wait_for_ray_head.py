#!/usr/bin/env python3
import argparse
import sys
import time
import ray

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-addr", required=True)
    parser.add_argument("--port", default="6379")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    address = f"{args.head_addr}:{args.port}"
    print(f"Waiting for Ray head at {address} ...", flush=True)

    while True:
        try:
            ray.init(address=address, namespace="health_check", ignore_reinit_error=True, log_to_driver=False)
            # forces a round-trip to head / GCS
            _ = ray.cluster_resources()
            print("Head is healthy.", flush=True)
            return 0
        except Exception as e:
            print(f"Head not ready yet ({e}), retrying...", flush=True)
            time.sleep(args.interval)
        finally:
            if ray.is_initialized():
                ray.shutdown()

if __name__ == "__main__":
    raise SystemExit(main())
