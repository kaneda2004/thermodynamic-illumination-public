#!/usr/bin/env python3
"""
RES-291 Sweep: Anneal Quantile x PCA Step End
=============================================

Runs a grid over anneal_quantile and pca_step_scale_end to stabilize stage-2.
"""

import json
import sys
import time
from pathlib import Path

# Project root resolution (local + GCP)
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from experiments import res_291_cppn_task_generalization as res291


def run_sweep():
    anneal_values = [0.1, 0.2, 0.3, 0.4]
    step_end_values = [0.1, 0.2, 0.3]

    sweep = {
        "experiment": "RES-291 Anneal + Step Sweep",
        "grid": {
            "anneal_quantile": anneal_values,
            "pca_step_scale_end": step_end_values,
        },
        "runs": [],
    }

    for anneal_q in anneal_values:
        for step_end in step_end_values:
            config = res291.ExperimentConfig()
            config.anneal_quantile = anneal_q
            config.pca_step_scale_end = step_end

            started = time.time()
            result = res291.run_experiment(config)
            elapsed = time.time() - started

            run_summary = {}
            for obj, payload in result.get("results", {}).items():
                summary = payload.get("summary", {})
                variants = summary.get("variants", {})
                run_summary[obj] = {
                    "speedup_mean": summary.get("speedup_mean"),
                    "speedup_std": summary.get("speedup_std"),
                    "speedup_min": summary.get("speedup_min"),
                    "speedup_max": summary.get("speedup_max"),
                    "variants": {
                        key: {
                            "stage2_used_count": stats.get("stage2_used_count"),
                            "stage2_speedup_mean": stats.get("stage2_speedup_mean"),
                            "speedup_mean": stats.get("speedup_mean"),
                        }
                        for key, stats in variants.items()
                    },
                }

            sweep["runs"].append({
                "anneal_quantile": anneal_q,
                "pca_step_scale_end": step_end,
                "elapsed_seconds": elapsed,
                "config": result.get("config", {}),
                "calibration": result.get("calibration", {}),
                "summary": run_summary,
            })

            print(f"Done anneal={anneal_q} step_end={step_end} in {elapsed:.1f}s")

    out_path = Path("results/cppn_task_generalization/res_291_anneal_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(sweep, f, indent=2)

    print(f"Sweep saved to {out_path}")


if __name__ == "__main__":
    run_sweep()
