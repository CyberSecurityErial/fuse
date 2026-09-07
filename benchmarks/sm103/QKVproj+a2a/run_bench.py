#!/usr/bin/env python3
"""SM103 training QKV GEMM -> Ulysses A2A baseline entry point."""
from pathlib import Path
import runpy
import sys

if any(arg == "--directions" or arg.startswith("--directions=") for arg in sys.argv[1:]):
    raise SystemExit("direction is fixed to qkv by this entry point")
sys.argv.extend(("--directions", "qkv"))
runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench.py"), run_name="__main__")
