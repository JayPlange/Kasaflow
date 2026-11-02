#!/usr/bin/env python3
"""
Run the KasaFlow model-call baseline (scripts/bench/baseline.py).

Sends real requests to the real API, using the OPENAI_API_KEY and
OPENAI_MODEL from .env, exactly like the app. Default plan is 45 model
requests: 5 first requests, a 1/2/4/8 concurrency ladder of 8 requests
each, 4 production-path requests and 4 end-to-end turns.

Usage (from the repo root):
    python3 scripts/run_llm_baseline.py --dry-run
    python3 scripts/run_llm_baseline.py --probe-one --show-events
    python3 scripts/run_llm_baseline.py --note "laptop, home wifi"
"""

import sys
from pathlib import Path

# Same bootstrap as run_evaluator.py: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from scripts.bench.baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
