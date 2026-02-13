#!/usr/bin/env python3
"""
Remote batch executor for corrected experiments on GCP.
Runs a list of experiments sequentially, skipping those that have already completed.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_execution.log')
    ]
)
logger = logging.getLogger(__name__)

# List of corrected experiments to run
# Note: These paths are relative to the project root on the VM
EXPERIMENTS = [
    {
        "script": "experiments/res_220_manifold_sampling.py",
        "result_file": "results/manifold_aware_sampling/res_220_results.json"
    },
    {
        "script": "experiments/res_224_multi_stage_sampling.py",
        "result_file": "results/multi_stage_sampling/res_224_results.json"
    },
    {
        "script": "experiments/res_225_low_d_initialization.py",
        "result_file": "results/initialization_dimensionality_control/res_225_results.json"
    },
    {
        "script": "experiments/res_226_adaptive_manifold.py",
        "result_file": "results/adaptive_manifold_sampling/res_226_results.json"
    },
    {
        "script": "experiments/res_232_three_stage_sampling.py",
        "result_file": "results/progressive_manifold_sampling/res_232_results.json"
    },
    {
        "script": "experiments/res_230_dual_channel.py",
        "result_file": "results/dual_channel_architecture/res_230_results.json"
    },
    {
        "script": "experiments/res_233_hybrid_manifold.py",
        "result_file": "results/hybrid_manifold_sampling/res_233_results.json"
    },
    {
        "script": "experiments/res_245_nonlinear_two_stage.py",
        "result_file": "results/interaction_two_stage_sampling/res_245_results.json"
    },
    {
        "script": "experiments/res_283_variance_decomposition_full.py",
        "result_file": "results/variance_decomposition/res_283_full_results.json"
    },
    {
        "script": "experiments/res_284_speedup_prediction.py",
        "result_file": "results/variance_decomposition/res_284_results.json"
    },
    {
        "script": "experiments/res_285_early_predictability.py",
        "result_file": "results/early_predictability/res_285_results.json"
    }
]

def run_experiment(exp_config):
    script_path = Path(exp_config["script"])
    result_path = Path(exp_config["result_file"])
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    if result_path.exists():
        # Check if result is valid JSON
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Skipping {script_path.name}: Result already exists.")
            return True
        except json.JSONDecodeError:
            logger.warning(f"Result file corrupted for {script_path.name}, re-running.")

    logger.info(f"Running {script_path.name}...")
    start_time = time.time()
    
    # Define log paths relative to CWD
    log_dir = Path("logs") / script_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    
    logger.info(f"  -> Saving logs to: {stdout_log} and {stderr_log}")
    
    try:
        # Run the script and stream output
        # Using subprocess.run with capture_output=False (default) streams to stdout/stderr
        # But we want to capture AND stream. 
        # Since the main goal is "GET LOGS", we will just let it stream to stdout/stderr directly
        # and also capture it via shell redirection in the wrapper script if possible.
        # But here, we want to save files.
        
        with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Real-time streaming to both file and stdout
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                if output == '' and error == '' and process.poll() is not None:
                    break
                if output:
                    sys.stdout.write(output)
                    sys.stdout.flush()
                    out_f.write(output)
                    out_f.flush()
                if error:
                    sys.stderr.write(error)
                    sys.stderr.flush()
                    err_f.write(error)
                    err_f.flush()
            
            returncode = process.poll()

        duration = time.time() - start_time

        if returncode == 0:
            logger.info(f"✓ {script_path.name} completed successfully in {duration:.1f}s")
            return True
        else:
            logger.error(f"✗ {script_path.name} failed with code {returncode}")
            logger.error(f"  -> Check stderr log: {stderr_log}")
            return False

    except Exception as e:
        logger.error(f"✗ Error running {script_path.name}: {str(e)}")
        return False

def main():
    logger.info("Starting batch execution of corrected experiments...")
    logger.info(f"Total experiments: {len(EXPERIMENTS)}")
    
    # Ensure results directories exist
    for exp in EXPERIMENTS:
        Path(exp["result_file"]).parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    
    for i, exp in enumerate(EXPERIMENTS):
        logger.info(f"\n[{i+1}/{len(EXPERIMENTS)}] Processing {Path(exp['script']).name}")
        if run_experiment(exp):
            success_count += 1
        else:
            fail_count += 1
            
    logger.info("\n" + "="*50)
    logger.info("BATCH EXECUTION COMPLETE")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed:     {fail_count}")
    logger.info("="*50)
    
    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
