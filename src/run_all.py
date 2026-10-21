#!/usr/bin/env python3
"""
Master Reproduction Pipeline
============================
Reproduces all experimental results, metric tables, error taxonomies,
and Decision Impact Rate (DIR) evaluations reported in the paper:

"Measuring the Downstream Decision Impact of Predicate Extraction Errors in Symbolic Legal Engines"
(JURIX 2026)

Usage:
    python src/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("Step 1: Neuro-Symbolic Error Collection & CoT Context", SRC_DIR / "step1_error_extraction.py"),
    ("Step 2: Error Taxonomy & Decision Impact Rate (DIR)", SRC_DIR / "step2_error_taxonomy.py"),
    ("Step 3: Standalone Predicate Extraction Evaluation", SRC_DIR / "evaluate_extraction.py"),
]


def run_pipeline():
    print("=" * 80)
    print("REPRODUCING ALL RESULTS FOR JURIX 2026 PAPER")
    print("Measuring Downstream Decision Impact of Predicate Extraction Errors")
    print("=" * 80)
    print()

    for title, script in SCRIPTS:
        print(f"\n▶ Running {title} ({script.name})...")
        cmd = [sys.executable, str(script)]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"❌ Error encountered while running {script.name}. Exiting.")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("🎉 ALL EXPERIMENTS AND ANALYSES REPRODUCED SUCCESSFULLY!")
    print("All output tables and reports are saved under:")
    print("  - results/summary_metrics/")
    print("  - results/error_taxonomy/")
    print("  - results/reports/")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
