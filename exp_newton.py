"""Ablation: metaheuristic start + direct Newton on F(x)=0 (no homotopy),
paired with Linear Newton-HCM from the same start. Seeds 0-49."""
import csv, time, numpy as np
from multiprocessing import Pool
from benchmarks import get_all_systems
from hybrid import META_SOLVERS
from hcm import LinearNewtonHCM
NR = 50; ALGOS = ['Standard', 'PSO', 'GA', 'GWO']; NEWTON_MAX = 100; TOL = 1e-6

def newton(S, x0):
    t0 = time.process_time(); x = np.array(x0, float).copy(); it = 0
    with np.errstate(all='ignore'):
        r = np.max(np.abs(S.F(x)))
        while it < NEWTON_MAX and np.isfinite(r) and r > TOL:
            Jx = S.J(x); Fx = S.F(x)
            try: dx = np.linalg.solve(Jx, Fx)
            except np.linalg.LinAlgError: dx = np.linalg.lstsq(Jx, Fx, rcond=None)[0]
            x = x - dx; it += 1; r = np.max(np.abs(S.F(x)))
    return dict(n=it, cpu=time.process_time() - t0, res=float(r), ok=int(np.isfinite(r) and r <= TOL))

def damped_newton(S, x0):
    """Newton with Armijo backtracking on ||F||_2 (step halved up to 30 times)."""
    t0 = time.process_time(); x = np.array(x0, float).copy(); it = 0
    Fx = S.F(x); f = np.linalg.norm(Fx); r = np.max(np.abs(Fx))
    while it < NEWTON_MAX and np.isfinite(r) and r > TOL:
        Jx = S.J(x)
        try: dx = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError: dx = np.linalg.lstsq(Jx, Fx, rcond=None)[0]
        lam = 1.0
        for _ in range(30):
            xn = x - lam * dx; Fn = S.F(xn); fn = np.linalg.norm(Fn)
            if np.isfinite(fn) and fn <= (1 - 1e-4 * lam) * f: break
            lam *= 0.5
        x, Fx, f = xn, Fn, fn; r = np.max(np.abs(Fx)); it += 1
    return dict(n=it, cpu=time.process_time() - t0, res=float(r), ok=int(np.isfinite(r) and r <= TOL))

def task(name):
    S = {s.name: s for s in get_all_systems()}[name]; lo, hi = S.bounds; out = []
    for algo in ALGOS:
        for seed in range(NR):
            if algo == 'Standard':
                x0 = np.random.default_rng(seed).uniform(lo, hi, size=S.n); cm = 0.0; nm = 0
            else:
                m = META_SOLVERS[algo](seed=seed).solve(S); x0 = m['x_best']; cm = m['cpu_time']; nm = m['n_iter']
            h = LinearNewtonHCM().solve(S, x0)
            with np.errstate(all='ignore'):
                nw = newton(S, x0); dn = damped_newton(S, x0)
            out.append(dict(system=name, n=S.n, algo=algo, seed=seed, n_meta=nm, cpu_meta=cm,
                            hcm_ok=int(h['converged']), hcm_iter=h['n_iter'], hcm_cpu=h['cpu_time'],
                            nwt_ok=nw['ok'], nwt_iter=nw['n'], nwt_cpu=nw['cpu'], nwt_res=nw['res'],dmp_ok=dn['ok'], dmp_iter=dn['n'], dmp_cpu=dn['cpu'], dmp_res=dn['res']))
    return out

if __name__ == '__main__':
    names = [s.name for s in get_all_systems()]; rows = []; t0 = time.time()
    with Pool(2) as p:
        for i, o in enumerate(p.imap_unordered(task, names)):
            rows += o; print(i + 1, len(names), round(time.time() - t0), flush=True)
    with open('results/newton_ablation_raw.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('done', flush=True)
