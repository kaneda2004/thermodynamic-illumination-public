# Phase 3: Find or Recreate Missing Code

**Purpose**: For entries with results but no experiment code, search for the code or recreate it from methodology.

## Environment Setup

- **Working directory**: `/Users/matt/Development/monochrome_noise_converger`
- **Files to modify**: `research_system/research_log.yaml`
- **Experiment location**: `experiments/`

## Input

You will receive RES-IDs that have results but no experiment_code link.

## Process for Each Entry

### Strategy 1: Search for Existing Code (Fastest)

#### Step 1: Check experiment_code_mapping.json

```python
import json
with open('results/audit/experiment_code_mapping.json') as f:
    mapping = json.load(f)

res_id = 'RES-XXX'
if res_id in mapping:
    code_file = mapping[res_id]  # Use this!
```

#### Step 2: Grep in experiments/

```bash
cd experiments
grep -r "RES-XXX" *.py */*.py 2>/dev/null
grep -r "hypothesis_keyword" *.py */*.py 2>/dev/null
ls *domain_name*.py 2>/dev/null
```

#### Step 3: Check git history for deleted files

```bash
git log --all --full-history --diff-filter=D -- "experiments/*RES*" | grep "RES-XXX"
git show <commit>:experiments/filename.py > recovered_file.py
```

### Strategy 2: Recreate Code from Results (If Not Found)

#### Step 1: Load the results JSON

```python
import json
res_path = entry['links']['results_json']
with open(res_path) as f:
    results = json.load(f)
```

#### Step 2: Extract methodology clues

Look for:
- `results['method']` or `results['methodology']`
- `results['hypothesis']`
- `results['parameters']` - tells you what was tested
- Test names in result summary: "Mann-Whitney", "ANOVA", "correlation", etc.

#### Step 3: Create minimal experiment code

Generate a Python script at `experiments/res_XXX_domain.py` that:
- Implements the methodology described in results
- Generates output matching the metrics in the results JSON
- Includes docstring: `RES-XXX: [hypothesis]`

Write code that can be executed standalone and produces compatible results.

#### Step 4: Test the code

Execute it locally to verify it produces similar metrics as the stored results.json.

### Step 5: Update research_log.yaml

```python
entry['links']['experiment_code'] = 'experiments/res_XXX_domain.py'
```

## Output Format (CONCISE - 2-3 lines max)

**Success Finding Code**:
```
✓ Found code for N entries
  RES-XXX → experiments/existing_file.py
  RES-YYY → experiments/recovered_from_git.py
```

**Success Recreating Code**:
```
✓ Recreated code for N entries
  RES-XXX: New file created at experiments/res_XXX_domain.py
```

**Partial Success**:
```
✓ Found code for 3, recreated 2, failed 1
✗ RES-ZZZ: Could not locate or infer methodology
```

## Error Handling

- **Code found**: Report location, update YAML
- **Code not found, methodology clear**: Create minimal version
- **Code not found, methodology unclear**: Report and skip
- **Multiple code files match**: Pick most recent, report
- **Partial success**: Report each success/failure concisely

## Key Notes

- Do NOT include full file contents or code dumps
- Do NOT explain methodology—just link or create
- Return ONLY: which entries were successful and where code is
- Update YAML after finding/creating code
- For recreated code, save to `experiments/res_XXX_domain.py`
