#!/usr/bin/env python3
"""Adaptive generalized-ReLU regression experiment.

The three procedures use independent training and validation samples, a dyadic
scale grid, clipping T=m**(1/8), and pairwise median-of-means validation with
a feasible odd block count at every sample size n >= 2. Only the
two density-oracle procedures receive q(X_i).  The normalised procedure evaluates
the finite piecewise-affine ReLU normaliser, not the exact ratio.

Run this file with --run to reproduce the manuscript experiment.  --check runs
local algebra and implementation checks.  --plot creates the figure from stored
replication results.  NumPy and Matplotlib are the only dependencies.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import NamedTuple
import numpy as np


class Design(NamedTuple):
    n: int
    m: int
    v: int
    scales: np.ndarray
    requested_blocks: int
    blocks: int
    block_size: int
    clip: float
    tolerance: float


def design(n: int, d: int = 1) -> Design:
    if n < 2 or d < 1:
        raise ValueError('n must be at least 2 and d must be positive.')
    m, v = n // 2, n - n // 2
    # Integer arithmetic prevents rounding errors at dyadic endpoints.
    j_max = (m.bit_length() - 1) // d
    scales = np.array([2**j for j in range(j_max + 1)], dtype=np.int64)
    k = math.ceil(8 * math.log(2 * len(scales)**2 * n**3))
    if k % 2 == 0:
        k += 1
    requested_blocks = k
    k = min(k, 2*((v-1)//2)+1)
    return Design(n, m, v, scales, requested_blocks, k, v // k,
                  m**0.125, m**-0.5)


def single_observation_predictor(y: float, x: np.ndarray) -> np.ndarray:
    """The manuscript convention for n=1, independent of the query location."""
    return np.full_like(np.asarray(x, dtype=float), np.clip(y, -1, 1))


class Normaliser(NamedTuple):
    T: float
    eta: float
    M: float
    epsilon: float
    rho: float
    log_gamma: float
    K: int
    J: int


def normaliser_parameters(T: float, eta: float, M: float,
                          epsilon: float) -> Normaliser:
    if not (T > 0 and M >= 1 and 0 < eta <= M and epsilon > 0):
        raise ValueError('Normaliser arguments are outside the admissible domain.')
    eps = min(epsilon, 1.0)
    R = T*M
    rho = min(1.0, math.sqrt(eps*eta/(2*R)))
    log_gamma = 2*math.log1p(rho)
    K = 0 if M == eta else math.ceil(math.log(M/eta)/log_gamma)
    J = math.ceil(math.sqrt(6*R/(eta*eps)))
    return Normaliser(T, eta, M, epsilon, rho, log_gamma, K, J)


def reciprocal_interpolant(z: np.ndarray, p: Normaliser) -> np.ndarray:
    """Evaluate the geometric-grid linear interpolant a_K exactly algebraically."""
    z = np.clip(np.asarray(z, dtype=float), 0, p.M)
    if p.K == 0:
        return np.full_like(z, 1/p.eta)
    z_floor = np.maximum(z, p.eta)
    j = np.floor(np.log(z_floor/p.eta)/p.log_gamma).astype(np.int64)
    j = np.clip(j, 0, p.K-1)
    a = p.eta*np.exp(j*p.log_gamma)
    b = np.minimum(p.eta*np.exp((j+1)*p.log_gamma), p.M)
    z_local = np.minimum(np.maximum(z_floor, a), b)
    out = (a+b-z_local)/(a*b)
    return np.where(z <= p.eta, 1/p.eta, out)


def square_interpolation_error(t: np.ndarray, J: int) -> np.ndarray:
    """S_J(t)-t**2 on [0,2], evaluated without summing J hinge functions."""
    t = np.clip(np.asarray(t, dtype=float), 0, 2)
    h = 2/J
    j = np.minimum(np.floor(t/h).astype(np.int64), J-1)
    a = j*h
    return np.maximum(0, (t-a)*((j+1)*h-t))


def relu_normaliser(r: np.ndarray, z: np.ndarray, p: Normaliser) -> np.ndarray:
    """Evaluate exactly the PWA function specified by the manuscript network.

    The stable formula below is algebraically identical to the three square
    interpolants and affine readout.  It does not substitute division by z.
    """
    R = p.T*p.M
    rc = np.clip(np.asarray(r, dtype=float), -R, R)
    aK = reciprocal_interpolant(z, p)
    u = (rc+R)/(2*R)
    v = np.clip(p.eta*aK, 0, 1)
    correction = (R/p.eta)*(square_interpolation_error(u+v, p.J)
                              - square_interpolation_error(u, p.J)
                              - square_interpolation_error(v, p.J))
    return np.clip(rc*aK + correction, -p.T, p.T)


class TentSums:
    """One-dimensional tent sums for several coefficient vectors.

    Long-double prefix sums reduce cancellation for the finest candidate scales.
    Only the final prediction arrays are converted to float64.
    """
    def __init__(self, X: np.ndarray, coefficients: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        coeff = np.asarray(coefficients, dtype=np.longdouble)
        if X.ndim != 1 or coeff.ndim != 2 or coeff.shape[1] != X.size:
            raise ValueError('Expected X of shape (m,) and coefficients of shape (p,m).')
        order = np.argsort(X)
        self.x = X[order]
        coeff = coeff[:, order]
        self.prefix = np.c_[np.zeros(coeff.shape[0], dtype=np.longdouble),
                            np.cumsum(coeff, axis=1, dtype=np.longdouble)]
        self.prefix_x = np.c_[np.zeros(coeff.shape[0], dtype=np.longdouble),
                              np.cumsum(coeff*self.x, axis=1, dtype=np.longdouble)]

    def evaluate(self, x: np.ndarray, nu: int) -> np.ndarray:
        if nu <= 0:
            raise ValueError('nu must be positive.')
        x = np.asarray(x, dtype=float)
        left = np.searchsorted(self.x, x-1/nu, side='left')
        mid = np.searchsorted(self.x, x, side='right')
        right = np.searchsorted(self.x, x+1/nu, side='right')
        cl = self.prefix[:, mid]-self.prefix[:, left]
        cr = self.prefix[:, right]-self.prefix[:, mid]
        xl = self.prefix_x[:, mid]-self.prefix_x[:, left]
        xr = self.prefix_x[:, right]-self.prefix_x[:, mid]
        distance_sum = (x*cl-xl)+(xr-x*cr)
        result = nu*(cl+cr) - np.longdouble(nu)**2*distance_sum
        # Disjoint support must produce zero, including after prefix subtraction.
        result[:, left == right] = 0
        return np.asarray(result, dtype=float)


def pairwise_selector(predictions: np.ndarray, Y: np.ndarray,
                      k: int) -> tuple[int, np.ndarray]:
    """Minimise maximum median block loss difference; choose first tie.

    Each median is applied to blockwise differences, NOT to separately computed
    medians of individual losses.  No true regression values enter selection.
    """
    predictions = np.asarray(predictions, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != len(Y):
        raise ValueError('Prediction array must have shape (candidates, validation).')
    if k < 1 or k % 2 != 1 or k > len(Y):
        raise ValueError('The number of blocks must be odd and at most validation size.')
    if predictions.shape[0] == 1:
        return 0, np.zeros((1, 1), dtype=float)
    ell = len(Y)//k
    used = k*ell
    scores = (predictions[:, :used]**2 - 2*predictions[:, :used]*Y[:used])
    block_means = scores.reshape(predictions.shape[0], k, ell).mean(axis=2)
    differences = np.median(block_means[:, None, :] - block_means[None, :, :], axis=2)
    worst = differences.max(axis=1)
    return int(np.argmin(worst)), differences


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5*np.asarray(x)**2)/math.sqrt(2*math.pi)


def target(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.asarray(x)**2)


def replication(n: int, rng: np.random.Generator, test_size: int = 10000,
                noise_scale: float = 0.25) -> dict[str, float | int]:
    cfg = design(n)
    X = rng.normal(size=n)
    Y = target(X)*(1 + noise_scale*rng.normal(size=n))
    xt = rng.normal(size=test_size)
    ft = target(xt)
    row: dict[str, float | int] = dict(n=n, m=cfg.m, v=cfg.v, N=len(cfg.scales),
                                      requested_blocks=cfg.requested_blocks,
                                      blocks=cfg.blocks, block_size=cfg.block_size,
                                      block_cap_active=int(cfg.blocks < cfg.requested_blocks),
                                      T=cfg.clip, epsilon=cfg.tolerance)
    m = cfg.m
    # The oracle density is used only in coefficient rows 2 and 3.
    qx = normal_pdf(X[:m])
    weighted_Y = Y[:m]/qx
    coefficients = np.vstack((np.ones(m), Y[:m],
                              weighted_Y*(qx >= 1/m), weighted_Y))/m
    sums = TentSums(X[:m], coefficients)
    val = {key: np.empty((len(cfg.scales), cfg.v)) for key in ('net','thr','ul')}
    max_error = 0.0
    for j, nu in enumerate(cfg.scales):
        z, r, thr, ul = sums.evaluate(X[m:], int(nu))
        z = np.clip(z, 0, float(nu))
        eta = float(nu)/m
        p = normaliser_parameters(cfg.clip, eta, float(nu), cfg.tolerance)
        val['net'][j] = relu_normaliser(r, z, p)
        val['thr'][j] = np.clip(thr, -cfg.clip, cfg.clip)
        val['ul'][j] = np.clip(ul, -cfg.clip, cfg.clip)
        ideal = np.clip(r/np.maximum(z, eta), -cfg.clip, cfg.clip)
        max_error = max(max_error, float(np.max(np.abs(ideal-val['net'][j]))))
    selected = {key: pairwise_selector(pred, Y[m:], cfg.blocks)[0]
                for key, pred in val.items()}
    cache = {}
    for key, j in selected.items():
        nu = int(cfg.scales[j])
        if nu not in cache:
            cache[nu] = sums.evaluate(xt, nu)
        z, r, thr, ul = cache[nu]
        if key == 'net':
            z = np.clip(z, 0, float(nu))
            p = normaliser_parameters(cfg.clip, nu/m, float(nu), cfg.tolerance)
            prediction = relu_normaliser(r, z, p)
            ideal = np.clip(r/np.maximum(z, nu/m), -cfg.clip, cfg.clip)
            max_error = max(max_error, float(np.max(np.abs(ideal-prediction))))
        else:
            prediction = np.clip(thr if key == 'thr' else ul, -cfg.clip, cfg.clip)
        row['risk_'+key] = float(np.mean((prediction-ft)**2))
        row['scale_'+key] = nu
    row['max_normaliser_error'] = max_error
    if max_error > cfg.tolerance + 1e-7:
        raise ArithmeticError('Normaliser approximation error exceeds tolerance.')
    return row


def check(output_dir: Path) -> dict:
    rng = np.random.default_rng(1729)
    checks = {}
    # Compare the fast normaliser with the explicit hinge-network computation.
    worst_hinge_difference = 0.0
    worst_approx_ratio = 0.0
    for T, eta, M, epsilon in [(1.,.2,2.,.05), (.1,1.,1.,.8),
                               (2.,.7,3.,.2), (1.,1.,1.,.01)]:
        p = normaliser_parameters(T,eta,M,epsilon)
        z = np.r_[0.,eta,M,rng.uniform(0,M,500)]
        r = np.r_[0.,-T*M,T*M,rng.uniform(-2*T*M,2*T*M,500)]
        if p.K:
            nodes = eta*np.exp(np.arange(p.K)*p.log_gamma)
            nodes = np.r_[nodes,M]
            slopes = np.diff(1/nodes)/np.diff(nodes)
            b = np.r_[slopes[0],np.diff(slopes)]
            aK = 1/eta + np.maximum(z[:,None]-nodes[:-1],0)@b
        else:
            aK = np.full_like(z,1/eta)
        rc = np.clip(r,-T*M,T*M)
        u,v = (rc+T*M)/(2*T*M),eta*aK
        h = 2/p.J
        knots = np.arange(1,p.J)*h
        def square(t):
            return h*t+2*h*np.maximum(t[:,None]-knots,0).sum(axis=1)
        product = (square(u+v)-square(u)-square(v))/2
        explicit = np.clip((2*T*M/eta)*product-(T*M/eta)*v,-T,T)
        fast = relu_normaliser(r,z,p)
        assert np.all(np.abs(fast) <= T)
        worst_hinge_difference = max(worst_hinge_difference,float(np.max(abs(explicit-fast))))
        ideal = np.clip(r/np.maximum(z,eta),-T,T)
        worst_approx_ratio = max(worst_approx_ratio,float(np.max(abs(fast-ideal)))/epsilon)
    assert worst_hinge_difference < 1e-9
    assert worst_approx_ratio <= 1+1e-8
    checks['normaliser_explicit_hinges_max_difference'] = worst_hinge_difference
    checks['normaliser_max_error_divided_by_tolerance'] = worst_approx_ratio
    X = rng.normal(size=317)
    coeff = rng.normal(size=(4,len(X)))/len(X)
    x = rng.normal(size=239)
    sums = TentSums(X,coeff)
    worst = 0.0
    for nu in (1,2,8,64,256):
        dense = coeff@(nu*np.maximum(1-nu*abs(X[:,None]-x),0))
        fast = sums.evaluate(x,nu)
        worst = max(worst,float(np.max(abs(dense-fast))))
    assert worst < 1e-8
    checks['tent_prefix_vs_direct_max_difference'] = worst
    P = rng.normal(size=(7,87))
    Y = rng.normal(size=87)
    j,D = pairwise_selector(P,Y,9)
    ell=87//9
    direct = np.zeros((7,7))
    for a in range(7):
        for b in range(7):
            dif = ((Y-P[a])**2-(Y-P[b])**2)[:9*ell].reshape(9,ell).mean(axis=1)
            direct[a,b]=np.median(dif)
    assert np.allclose(D,direct,atol=1e-12)
    assert j == int(np.argmin(direct.max(axis=1)))
    assert np.allclose(D,-D.T)
    checks['pairwise_selector_direct_agreement'] = True
    # Both a single block and singleton blocks follow the same comparison rule.
    for k in (1, 3, 87):
        chosen, differences = pairwise_selector(P, Y, k)
        ell = len(Y)//k
        direct = np.empty((len(P), len(P)))
        for a in range(len(P)):
            for b in range(len(P)):
                loss_difference = ((Y-P[a])**2-(Y-P[b])**2)[:k*ell]
                direct[a,b] = np.median(loss_difference.reshape(k,ell).mean(axis=1))
        assert np.allclose(differences,direct,atol=1e-12)
        assert chosen == int(np.argmin(direct.max(axis=1)))
    checks['single_block_and_singleton_blocks_direct_agreement'] = True
    checks['normaliser_range_checked'] = True
    # Exhaustive small-sample block checks, plus large-sample endpoints.
    sample_sizes = list(range(2, 1001)) + [10000, 100000]
    for dimension in (1, 2, 3, 10):
        for n in sample_sizes:
            cfg = design(n, dimension)
            expected = min(cfg.requested_blocks, 2*((cfg.v-1)//2)+1)
            assert cfg.blocks == expected
            assert 1 <= cfg.blocks <= cfg.v and cfg.blocks % 2 == 1
            assert cfg.block_size >= 1
            if cfg.v >= cfg.requested_blocks:
                assert cfg.blocks == cfg.requested_blocks
    checks['block_count_configurations_checked'] = len(sample_sizes)*4
    checks['single_candidate_returned'] = pairwise_selector(
        np.array([[0.3]]), np.array([0.4]), 1)[0] == 0
    assert checks['single_candidate_returned']
    checks['n1_clipped_response_convention'] = bool(np.array_equal(
        single_observation_predictor(2., np.array([-2., 0., 3.])), np.ones(3)))
    assert checks['n1_clipped_response_convention']
    checks['small_sample_replications'] = [replication(n, rng, 127)
                                            for n in (2, 3, 4, 5, 100, 351, 376)]
    assert all(r['scale_net'] >= 1 and r['scale_thr'] >= 1 and r['scale_ul'] >= 1
               for r in checks['small_sample_replications'])
    checks['pilot_replication'] = replication(1000,rng,1000)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'implementation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    return checks


def run(output_dir: Path, n_grid: int = 100, reps: int = 20,
        test_size: int = 10000, seed: int = 20261021) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    sizes = np.unique(np.rint(np.geomspace(100,100000,n_grid)).astype(int))
    config = dict(seed=seed,n_min=100,n_max=100000,n_grid=len(sizes),
                  repetitions=reps,test_size=test_size,noise_scale=.25,
                  estimator='finite PWA normaliser and capped-block pairwise median-of-means selection')
    (output_dir/'experiment_config.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.time()
    rows=[]
    raw_path=output_dir/'adaptive_replications.csv'
    with raw_path.open('w',newline='') as f:
        writer=None
        for i,n in enumerate(sizes):
            for rep in range(reps):
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(n),rep]))
                row=replication(int(n),rng,test_size)
                row={'rep':rep,**row}
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            f.flush()
            if (i+1)%10==0 or i==0:
                print(f'{i+1}/{len(sizes)} sample sizes; n={n}; elapsed={time.time()-start:.1f}s',flush=True)
    summary=[]
    for n in sizes:
        sub=[r for r in rows if r['n']==n]
        row={'n':int(n),'m':sub[0]['m'],'requested_blocks':sub[0]['requested_blocks'],
             'blocks':sub[0]['blocks'],'block_size':sub[0]['block_size'],
             'block_cap_active':sub[0]['block_cap_active']}
        for key in ('net','thr','ul'):
            vals=np.array([r['risk_'+key] for r in sub])
            row[key+'_mean']=float(vals.mean())
            row[key+'_se']=float(vals.std(ddof=1)/math.sqrt(reps))
            row[key+'_median_scale']=float(np.median([r['scale_'+key] for r in sub]))
        row['max_normaliser_error']=max(r['max_normaliser_error'] for r in sub)
        row['epsilon']=sub[0]['epsilon']
        summary.append(row)
    with (output_dir/'adaptive_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    report={'elapsed_seconds':time.time()-start,'replications':len(rows),
            'capped_sample_sizes':[int(r['n']) for r in summary if r['block_cap_active']],
            'first_uncapped_grid_n':int(next(r['n'] for r in summary if not r['block_cap_active'])),
            'maximum_error_tolerance_ratio':max(r['max_normaliser_error']/r['epsilon'] for r in rows),
            'last_summary':summary[-1]}
    (output_dir/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def plot(output_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt
    with (output_dir/'adaptive_summary.csv').open() as f:
        rows=list(csv.DictReader(f))
    ns=np.array([int(r['n']) for r in rows])
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    labels={'net':'Adaptive normalised ReLU network',
            'thr':'Adaptive density-threshold oracle',
            'ul':'Adaptive unthresholded density oracle'}
    for key,style in zip(('net','thr','ul'),('-','--','-.')):
        mean=np.array([float(r[key+'_mean']) for r in rows])
        se=np.array([float(r[key+'_se']) for r in rows])
        ax.errorbar(ns,mean,yerr=1.96*se,label=labels[key],linestyle=style,
                    linewidth=1.4,elinewidth=.6,capsize=0)
    ref=.001*(ns/1000.)**(-2/3)
    ax.plot(ns,ref,':',linewidth=1.3,label=r'Reference $0.001(n/1000)^{-2/3}$')
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlabel('Total sample size $n$')
    ax.set_ylabel(r'Estimated squared $\mathcal{L}^2(\mu)$ risk')
    ax.legend(fontsize=8.5,frameon=False)
    ax.grid(True,which='major',linewidth=.4,alpha=.35)
    fig.tight_layout()
    figure_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure_dir/'adaptive_estimator_risks.pdf',bbox_inches='tight')
    fig.savefig(figure_dir/'adaptive_estimator_risks.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--n-grid',type=int,default=100)
    parser.add_argument('--reps',type=int,default=20)
    parser.add_argument('--test-size',type=int,default=10000)
    args=parser.parse_args()
    if args.check:check(args.output_dir)
    if args.run:run(args.output_dir,args.n_grid,args.reps,args.test_size)
    if args.plot:plot(args.output_dir,args.output_dir)
