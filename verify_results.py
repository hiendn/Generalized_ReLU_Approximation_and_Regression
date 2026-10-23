#!/usr/bin/env python3
"""Check the recorded experiment and recompute its summaries; no simulations run."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
from adaptive_regression import design


def verify(data_dir: Path) -> dict:
    config = json.loads((data_dir/'experiment_config.json').read_text())
    expected_config = {'seed':20261021, 'n_min':100, 'n_max':100000,
                       'n_grid':100, 'repetitions':20, 'test_size':10000,
                       'noise_scale':0.25}
    for key, value in expected_config.items():
        assert config[key] == value, (key,config[key],value)
    rows = list(csv.DictReader((data_dir/'adaptive_replications.csv').open()))
    summary = list(csv.DictReader((data_dir/'adaptive_summary.csv').open()))
    expected_sizes = np.unique(np.rint(np.geomspace(100,100000,100)).astype(int))
    assert len(rows) == 2000 and len(summary) == 100
    assert [int(row['n']) for row in summary] == expected_sizes.tolist()
    seen = set()
    max_ratio = 0.
    for row in rows:
        n, rep = int(row['n']), int(row['rep'])
        assert n in expected_sizes and 0 <= rep < 20
        assert (n,rep) not in seen
        seen.add((n,rep))
        cfg = design(n)
        for key, value in {'m':cfg.m,'v':cfg.v,'N':len(cfg.scales),
                           'requested_blocks':cfg.requested_blocks,
                           'blocks':cfg.blocks,'block_size':cfg.block_size,
                           'block_cap_active':int(cfg.blocks<cfg.requested_blocks)}.items():
            assert int(row[key]) == value, (n,rep,key)
        assert math.isclose(float(row['T']),cfg.clip,rel_tol=1e-14)
        assert math.isclose(float(row['epsilon']),cfg.tolerance,rel_tol=1e-14)
        err = float(row['max_normaliser_error'])
        assert math.isfinite(err) and 0 <= err <= cfg.tolerance + 1e-7
        max_ratio = max(max_ratio,err/cfg.tolerance)
        for key in ('net','thr','ul'):
            risk = float(row['risk_'+key])
            assert math.isfinite(risk) and risk >= 0
            assert int(row['scale_'+key]) in cfg.scales
    for entry in summary:
        n = int(entry['n'])
        group = [row for row in rows if int(row['n']) == n]
        assert len(group) == 20
        for key in ('m','requested_blocks','blocks','block_size','block_cap_active'):
            assert all(row[key] == entry[key] for row in group), (n,key)
        for key in ('net','thr','ul'):
            risks=np.array([float(row['risk_'+key]) for row in group])
            scales=np.array([int(row['scale_'+key]) for row in group])
            computed={'mean':risks.mean(),'se':risks.std(ddof=1)/math.sqrt(20),
                      'median_scale':np.median(scales)}
            for stat, val in computed.items():
                assert math.isclose(float(entry[key+'_'+stat]),float(val),
                                    rel_tol=1e-13,abs_tol=1e-15), (n,key,stat)
        assert math.isclose(float(entry['max_normaliser_error']),
                            max(float(row['max_normaliser_error']) for row in group),
                            rel_tol=1e-14)
        assert all(row['epsilon'] == entry['epsilon'] for row in group)
    run_report=json.loads((data_dir/'run_summary.json').read_text())
    assert run_report['replications']==len(rows)
    assert run_report['capped_sample_sizes']==[int(r['n']) for r in summary if int(r['block_cap_active'])]
    assert run_report['first_uncapped_grid_n']==next(int(r['n']) for r in summary if not int(r['block_cap_active']))
    assert math.isclose(run_report['maximum_error_tolerance_ratio'],max_ratio,rel_tol=1e-14)
    for key, value in run_report['last_summary'].items():
        assert math.isclose(float(summary[-1][key]),value,rel_tol=1e-14), key
    return {'status':'passed','replications_checked':len(rows),
            'summary_rows_recomputed':len(summary),
            'sample_size_range':[int(expected_sizes[0]),int(expected_sizes[-1])],
            'maximum_recorded_normaliser_error_over_tolerance':max_ratio,
            'last_summary':run_report['last_summary'],
            'full_simulation_rerun':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    print(json.dumps(verify(args.data_dir),indent=2))
