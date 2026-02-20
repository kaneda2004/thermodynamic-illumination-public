# Phase 1: Extract Metrics from Unverified Results

**Purpose**: Read existing results JSON files and populate the `result.metrics` field in research_log.yaml for entries marked as "unverified".

## Environment Setup

- **Working directory**: `/Users/matt/Development/monochrome_noise_converger`
- **Python execution**: ALWAYS use `uv run python` (not plain `python`)
- **Files to modify**: `research_system/research_log.yaml`

## Input

You will receive a comma-separated list of RES-IDs to process (e.g., "RES-037, RES-038, RES-039")

## Process for Each Entry

### Step 1: Load research_log.yaml

```python
import yaml
from pathlib import Path

log_path = Path('/Users/matt/Development/monochrome_noise_converger/research_system/research_log.yaml')
with open(log_path, 'r') as f:
    log = yaml.safe_load(f)
```

### Step 2: Find the entry by RES-ID

```python
entry = next(e for e in log['entries'] if e['id'] == 'RES-XXX')
```

### Step 3: Verify it's unverified

```python
if entry.get('verification_status') != 'unverified':
    skip_with_note("Not unverified")
```

### Step 4: Get results file path

```python
results_path = entry.get('links', {}).get('results_json')
if not results_path:
    fail_with_note("No results_json link")
```

### Step 5: Read results JSON

```python
import json
full_path = Path('/Users/matt/Development/monochrome_noise_converger') / results_path
with open(full_path, 'r') as f:
    results_json = json.load(f)
```

### Step 6: Extract metrics

Look for these fields in the JSON (check multiple possible locations):
- `result['metrics']` (direct metrics)
- `result['summary']` (may have inline metrics)
- Root level: `p_value`, `effect_size`, `correlation`, etc.

Extract any of:
- `p_value` (p < 0.01 for significance)
- `effect_size` / `cohens_d` / `d` (d > 0.5 for substantial)
- `correlation` / `rho` / `r` (correlation strength)
- `r_squared` / `R²` (variance explained)
- `chi_squared` (categorical test)
- `f_statistic` (ANOVA)
- Domain-specific: `alpha`, `beta`, `gamma`, `slope`, `intercept`, etc.

### Step 7: Populate metrics field

```python
entry['result']['metrics'] = {
    'p_value': extracted_p_value,
    'effect_size': extracted_effect_size,
    'correlation': extracted_correlation,
    # ... other metrics
}
```

### Step 8: Update verification status

```python
entry['verification_status'] = 'verified'
if 'notes' not in entry:
    entry['notes'] = ''
entry['notes'] += f"\n[2025-12-19] Metrics extracted from results file {results_path}"
```

### Step 9: Save research_log.yaml

```python
with open(log_path, 'w') as f:
    yaml.safe_dump(log, f, default_flow_style=False, sort_keys=False)
```

## Output Format (CONCISE - 2-3 lines max)

**Success**:
```
✓ Processed N entries: M success, K failed
  Updated metrics for RES-XXX, RES-YYY, RES-ZZZ
```

**Errors**:
```
✗ RES-041: Results file not found at results/xxx/results.json
✗ RES-042: JSON malformed - skipped
```

## Error Handling

- **Results file doesn't exist**: Log filename, skip, continue
- **JSON malformed**: Log entry and error, skip, continue
- **Can't extract metrics**: Log entry and reason, skip, continue
- **All 8 entries fail**: Return that fact concisely

## Key Notes

- Do NOT be verbose
- Do NOT dump JSON outputs
- Do NOT explain what you're doing - just do it
- Return ONLY: success count, failure count, specific errors
- Save the modified YAML file (this is critical)
