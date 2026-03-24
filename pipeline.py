#!/usr/bin/env python3
import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from covxplore.api_client import AkaUTClient


FUNCTIONS = [
    "_readMLString",
    "_readKeyname",
    "_readString",
    "_readValueBegin",
    "_quoteForComment"
]

TARGET_FILE = Path("target_functions.txt")


def fetch_paths():
    """Queries AkaUT to find the absolute paths for the target functions."""
    paths = []
    print("Fetching function paths from AkaUT...")
    with AkaUTClient() as client:
        # Loop over requested function names to get their full paths
        for func_name in FUNCTIONS:
            try:
                nodes = client.search_nodes(query=func_name)
                # Take the first match
                if nodes:
                    paths.append(nodes[0].absolute_path)
                    print(f"[*] Found {func_name} -> {nodes[0].absolute_path}")
                else:
                    print(f"[!] Warning: Could not find node for {func_name}")
            except Exception as e:
                print(f"[!] Error fetching {func_name}: {e}")

    if paths:
        TARGET_FILE.write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"\nSaved {len(paths)} paths to {TARGET_FILE.name}")
    else:
        print("\nNo paths were found. Check if AkaUT is running.")
    return paths


def run_ablation(func_path: str, out_dir: Path):
    """Runs a single variant ablation for a function via CLI."""
    # Build safe name for output folder routing
    safe_name = func_path.split("::")[-1].split("(")[0]
    if not safe_name:
        safe_name = "func_unknown"
        
    out_dir_func = out_dir / safe_name
    
    # We execute covxplore-ablate on each target
    cmd = [
        "covxplore-ablate",
        "--path", func_path,
        "--out", str(out_dir_func),
        "--repeat", "1"  # Do 1 repeat per variant to start with in parallel
    ]
    print(f"🚀 Starting ablation for {safe_name}")
    try:
        # Use subprocess to run isolated CrewAI agents in separate CLI invocations
        subprocess.run(cmd, check=True)
        print(f"✅ Completed ablation for {safe_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed ablation for {safe_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parallel Ablation Pipeline")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch paths and exit")
    parser.add_argument("--run-only", action="store_true", help="Only run ablation (requires target_functions.txt)")
    parser.add_argument("--workers", type=int, default=3, help="Max parallel workers (default: 3)")
    parser.add_argument("--out", default="results", help="Output directory for results")
    args = parser.parse_args()

    # Step 1: Fetch Paths
    if not args.run_only:
        paths = fetch_paths()
    else:
        if not TARGET_FILE.exists():
            print(f"Error: {TARGET_FILE} not found. Run without --run-only first.")
            return
        paths = [p.strip() for p in TARGET_FILE.read_text(encoding="utf-8").splitlines() if p.strip()]
    
    if args.fetch_only or not paths:
        return

    # Step 2: Parallel ThreadPool
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting parallel ablation with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for p in paths:
            executor.submit(run_ablation, p, out_dir)


if __name__ == "__main__":
    main()
