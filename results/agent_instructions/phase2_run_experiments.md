# Phase 2: Run Experiments with Existing Code

**Purpose**: Execute experiment Python files that already exist and capture their results.

## Environment Setup

- **Working directory**: `/Users/matt/Development/monochrome_noise_converger`
- **Python execution**: Use `uv run python experiments/filename.py` to execute
- **Files to modify**: `research_system/research_log.yaml`
- **Results location**: Check experiment output for where JSON is saved

## Input

You will receive entries with RES-IDs and their corresponding experiment files:
- Format: "RES-XXX → experiments/domain/experiment_file.py"

## Process for Each Entry

### Step 1: Verify experiment file exists

```bash
ls experiments/path/to/file.py
# If doesn't exist: fail, report filename
```

### Step 2: Execute the experiment

```bash
cd /Users/matt/Development/monochrome_noise_converger
uv run python experiments/path/to/file.py
```

Capture STDOUT and STDERR. Keep running for up to 30 minutes.

### Step 3: Look for results JSON file

Experiments typically save to one of:
- `results/{domain}/results.json`
- `results/{domain}/RES_XXX_results.json`
- `results/{filename}/results.json`

Check the domain from research_log.yaml to find the right location.

### Step 4: Verify results file is valid

```python
import json
with open(results_file) as f:
    data = json.load(f)  # Should not raise error
```

### Step 5: Update research_log.yaml

Load the log, find the entry by RES-ID:

```python
entry['links']['results_json'] = 'results/domain/results.json'
```

Extract metrics from the JSON and populate `entry['result']['metrics']`:

```python
entry['result']['metrics'] = {
    'p_value': data.get('p_value'),
    'effect_size': data.get('effect_size') or data.get('cohens_d'),
    # ... extract other metrics
}
```

### Step 6: Save research_log.yaml

Same as Phase 1 - write the modified YAML.

## Output Format (CONCISE - 2-3 lines max)

**Success**:
```
✓ Executed N experiments: M success, K failed, Z timeout
  Runtime: 15 min total
```

**Errors**:
```
✗ RES-XXX: Code execution failed - [error type]
✗ RES-YYY: Timeout after 30 minutes
✗ RES-ZZZ: No results file found after execution
```

## Error Handling

- **Code execution fails**: Report error type, skip, continue
- **Timeout (>30 min)**: Kill process, report, skip
- **No results file generated**: Report, skip
- **Results file malformed**: Report, skip
- **Partial success**: Report successes and failures separately

## Key Notes

- Do NOT include verbose execution output
- Do NOT explain—just report results
- Return ONLY: count of successes/failures, runtime estimate
- Update YAML after EACH successful run (don't batch updates)
- If an experiment takes 30+ minutes, kill it and report timeout
