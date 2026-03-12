
import sys
from pathlib import Path

FILES_TO_FIX = [
    "experiments/res_220_manifold_sampling.py",
    "experiments/res_224_multi_stage_sampling.py",
    "experiments/res_225_low_d_initialization.py",
    "experiments/res_226_adaptive_manifold.py",
    "experiments/res_229_three_stage_sampling.py",
    "experiments/res_230_dual_channel.py",
    "experiments/res_233_hybrid_manifold.py",
    "experiments/res_245_nonlinear_two_stage.py"
]

OLD_BLOCK = """# Ensure project root is in path
project_root = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(project_root))

# Set working directory
os.chdir(project_root)"""

NEW_BLOCK = """# Ensure project root is in path (works on both local and GCP)
local_path = Path('/Users/matt/Development/monochrome_noise_converger')
if local_path.exists():
    project_root = local_path
else:
    # On GCP, use current working directory (should be ~/repo)
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))
os.chdir(project_root)"""

for file_path in FILES_TO_FIX:
    try:
        path = Path(file_path)
        content = path.read_text()
        
        # Check if already fixed
        if "if local_path.exists():" in content:
            print(f"Skipping {file_path} (already fixed)")
            continue
            
        # Replace the block
        if OLD_BLOCK in content:
            new_content = content.replace(OLD_BLOCK, NEW_BLOCK)
            path.write_text(new_content)
            print(f"Fixed {file_path}")
        else:
            # Fallback: try finding just the path line if the comments/spacing differ slightly
            print(f"Warning: Exact block match failed for {file_path}, trying fuzzy replacement...")
            
            # Simple fuzzy replace for the critical path line
            target_line = "project_root = Path('/Users/matt/Development/monochrome_noise_converger')"
            if target_line in content:
                # Find the surrounding context to replace safely
                start_idx = content.find(target_line)
                end_idx = content.find("os.chdir(project_root)", start_idx) + len("os.chdir(project_root)")
                
                original_segment = content[start_idx:end_idx]
                # Reconstruct imports if they were part of the replaced block (usually sys/os/pathlib are earlier)
                
                # Actually, let's just replace the specific hardcoded line logic with the dynamic one
                # We need to be careful about imports.
                pass
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
