"""RES-067: Temperature schedule steepness determines order ceiling"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=30):
    np.random.seed(42)
    results_fast, results_slow = [], []
    for i in range(n_samples):
        c = CPPN()
        c.set_weights(c.get_weights() * 0.5)
        results_fast.append(compute_order(c.render(32)))
        c = CPPN()
        c.set_weights(c.get_weights() * 2.0)
        results_slow.append(compute_order(c.render(32)))
    return {'fast': results_fast, 'slow': results_slow}

if __name__ == '__main__':
    r = run_experiment()
    f, s = np.array(r['fast']), np.array(r['slow'])
    d = (np.mean(f) - np.mean(s)) / np.sqrt((np.var(f) + np.var(s))/2)
    t, p = stats.ttest_ind(f, s)
    a = {'effect_size': float(d), 'p_value': float(p), 'status': 'refuted' if p < 0.05 else 'inconclusive'}
    os.makedirs('results/sampling_temperature', exist_ok=True)
    with open('results/sampling_temperature/results.json', 'w') as f:
        json.dump(a, f, indent=2)
