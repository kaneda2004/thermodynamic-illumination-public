# Phase 4: Create New Experiments from Scratch

**Purpose**: For entries with only hypothesis, implement the experiment, run it, and capture results.

## Environment Setup

- **Working directory**: `/Users/matt/Development/monochrome_noise_converger`
- **Files to modify**: `research_system/research_log.yaml`
- **Experiment location**: `experiments/`
- **Results location**: `results/{domain}/`

## Input

You will receive RES-IDs with hypothesis but no code or results.
Format: "RES-XXX: {hypothesis + domain}"

## Process for Each Entry

### Step 1: Load and understand the hypothesis

```python
import yaml
from pathlib import Path

log_path = Path('/Users/matt/Development/monochrome_noise_converger/research_system/research_log.yaml')
with open(log_path) as f:
    log = yaml.safe_load(f)

entry = next(e for e in log['entries'] if e['id'] == 'RES-XXX')
hypothesis = entry['hypothesis']['statement']
domain = entry.get('domain', 'unknown')
```

### Step 2: Design the experiment

Read the hypothesis and determine:
- What is being tested? (independent variable)
- How is it measured? (dependent variable)
- What statistical test is appropriate?
- What sample size/parameters are needed?

Example: "High-order images have higher spectral coherence"
→ Test: Compare spectral coherence between high-order and low-order CPPN outputs
→ Method: Generate CPPN samples, measure order and spectral properties, correlation

### Step 3: Implement experiment code

Create file: `experiments/res_XXX_domain.py`

Template:
```python
#!/usr/bin/env python3
"""
RES-XXX: [Hypothesis from research_log]

Tests [what is being tested] using [methodology].
"""

import json
from pathlib import Path

def main():
    # ... experiment implementation ...

    # Generate results
    results = {
        'id': 'RES-XXX',
        'hypothesis': '[hypothesis]',
        'status': 'validated',  # or 'refuted'
        'result': {
            'metrics': {
                'p_value': 0.001,  # Must compute these
                'effect_size': 0.85,
                'correlation': 0.92,
                # ... other metrics
            },
            'summary': '[1-2 sentence result]'
        }
    }

    # Save results
    results_dir = Path('results') / '[domain]'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
```

### Step 4: Execute the experiment

```bash
cd /Users/matt/Development/monochrome_noise_converger
uv run python experiments/res_XXX_domain.py
```

Run for up to 30 minutes. If timeout, report and skip.

### Step 5: Verify results file

```python
import json
results_path = Path('results') / domain / 'results.json'
with open(results_path) as f:
    results = json.load(f)

# Check required fields
assert 'metrics' in results['result']
assert 'p_value' in results['result']['metrics']
```

### Step 6: Update research_log.yaml

```python
entry['links']['experiment_code'] = 'experiments/res_XXX_domain.py'
entry['links']['results_json'] = 'results/domain/results.json'
entry['result']['metrics'] = results['result']['metrics']
entry['result']['summary'] = results['result']['summary']
entry['status'] = results['result'].get('status', 'unknown')
```

### Step 7: Save research_log.yaml

```python
with open(log_path, 'w') as f:
    yaml.safe_dump(log, f, default_flow_style=False, sort_keys=False)
```

## Output Format (CONCISE - 2-3 lines max)

**Success**:
```
✓ Created N experiments: M success, K failed
  RES-XXX: validated, d=0.85
  RES-YYY: refuted, d=-0.23
```

**Errors**:
```
✗ RES-XXX: Code failed to execute - [error type]
✗ RES-YYY: Timeout after 30 minutes
✗ RES-ZZZ: Results file not generated
```

## Error Handling

- **Design is unclear**: Report "Cannot infer methodology" and skip
- **Code fails**: Report error type, skip
- **Timeout (>30 min)**: Kill process, report
- **Results file not generated**: Report, skip
- **Partial success**: Report each separately

## Key Notes

- Do NOT explain the experiment design
- Do NOT dump code or error traces
- Return ONLY: which RES-IDs succeeded, their status/effect size
- Update YAML immediately after success (don't batch)
- All results MUST include: p_value, effect_size, and summary
- Experiments should run in < 30 minutes (adjust parameters if needed)
