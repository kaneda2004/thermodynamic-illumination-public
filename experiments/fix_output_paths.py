
import sys
import re
from pathlib import Path

FILES_TO_FIX = [
    "experiments/res_220_manifold_sampling.py",
    "experiments/res_224_multi_stage_sampling.py",
    "experiments/res_225_low_d_initialization.py",
    "experiments/res_226_adaptive_manifold.py",
    "experiments/res_232_three_stage_sampling.py",
    "experiments/res_230_dual_channel.py",
    "experiments/res_233_hybrid_manifold.py",
    "experiments/res_245_nonlinear_two_stage.py",
    "experiments/res_283_variance_decomposition_full.py",
    "experiments/res_284_speedup_prediction.py"
]

# Regex to find: Path('/Users/matt/Development/monochrome_noise_converger/results/XXX')
# Replacement: project_root / "results" / "XXX"

pattern = r"Path\('/Users/matt/Development/monochrome_noise_converger/results/([^']+)'\)"

for file_path in FILES_TO_FIX:
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"Skipping {file_path} (not found)")
            continue
            
        content = path.read_text()
        
        # Check for matches
        matches = re.findall(pattern, content)
        if matches:
            print(f"Fixing {len(matches)} paths in {file_path}")
            # Replace logic
            # We want to replace: Path('/Users/.../results/subdir')
            # With: project_root / "results" / "subdir"
            
            def replace_match(m):
                subdir = m.group(1)
                return f'project_root / "results" / "{subdir}"'
                
            new_content = re.sub(pattern, replace_match, content)
            path.write_text(new_content)
        else:
            print(f"No hardcoded result paths found in {file_path}")
            
            # Fallback: check for slight variations or direct string usage without Path()
            if "/Users/matt/Development/monochrome_noise_converger/results" in content:
                print(f"  WARNING: Found string path in {file_path} that wasn't caught by regex.")
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
