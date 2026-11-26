"""Recompute headline means from the included compact evaluation records."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = json.loads((root/'results/per_target_evaluations.json').read_text())
expected = json.loads((root/'reports/full_v1/cooperative_completed_seeds_42_123_2026.json').read_text())
for method, summary in expected['combined_means'].items():
    selected = [r for r in rows if r['setting']=='cooperative' and r['method']==method]
    assert len(selected)==192 and all(r['ok'] for r in selected)
    value = statistics.mean(r['best_utility'] for r in selected)
    assert abs(value-summary['mean_utility']) < 1e-10
    print(f'{method:22s} mean utility = {value:.6f}')
