"""
metaheuristics.py
==================
Fasa 1 (Global Search) bagi sistem hibrid: PSO_Solver, GA_Solver, GWO_Solver.

Fungsi kecergasan (fitness) yang diminimumkan oleh ketiga-tiga algoritma ialah:

    E(x) = ||F(x)||_2 = sqrt( sum_i f_i(x)^2 )

Setiap solver berhenti apabila E(x) <= eps_meta (lalai 1e-2) atau apabila
bilangan iterasi maksimum dicapai. Setiap solver mengembalikan kamus (dict)
berisi:
    x_best      : np.ndarray - titik terbaik dijumpai (akan menjadi x0 bagi HCM)
    E_best      : float      - nilai fitness terbaik
    n_iter      : int        - bilangan iterasi (N_meta) yang digunakan
    history     : list[float]- sejarah E_best setiap iterasi (untuk plot penumpuan)
    time_history: list[float]- masa CPU terkumpul setiap iterasi
    cpu_time    : float      - jumlah masa CPU fasa metaheuristik (saat)
    converged   : bool       - True jika E_best <= eps_meta sebelum max_iter habis
"""

import time
import numpy as np


def _fitness(F, x):
    fx = F(x)
    return np.sqrt(np.sum(fx ** 2))


class PSO_Solver:
    """Particle Swarm Optimization (PSO) piawai dengan inertia weight menyusut."""

    def __init__(self, n_particles=40, max_iter=200, w_max=0.9, w_min=0.4,
                 c1=1.6, c2=1.6, eps_meta=1e-2, seed=None):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.eps_meta = eps_meta
        self.rng = np.random.default_rng(seed)

    def solve(self, system):
        F, n = system.F, system.n
        lo, hi = system.bounds
        rng = self.rng
        t_start = time.process_time()

        X = rng.uniform(lo, hi, size=(self.n_particles, n))
        V = rng.uniform(-abs(hi - lo), abs(hi - lo), size=(self.n_particles, n)) * 0.1
        fit = np.array([_fitness(F, X[i]) for i in range(self.n_particles)])

        pbest_X = X.copy()
        pbest_fit = fit.copy()
        g_idx = np.argmin(pbest_fit)
        gbest_X = pbest_X[g_idx].copy()
        gbest_fit = pbest_fit[g_idx]

        history = [gbest_fit]
        time_history = [time.process_time() - t_start]
        converged = gbest_fit <= self.eps_meta
        it = 0

        while it < self.max_iter and not converged:
            w = self.w_max - (self.w_max - self.w_min) * it / self.max_iter
            r1 = rng.uniform(0, 1, size=(self.n_particles, n))
            r2 = rng.uniform(0, 1, size=(self.n_particles, n))
            V = (w * V
                 + self.c1 * r1 * (pbest_X - X)
                 + self.c2 * r2 * (gbest_X - X))
            X = X + V
            X = np.clip(X, lo, hi)

            fit = np.array([_fitness(F, X[i]) for i in range(self.n_particles)])
            improved = fit < pbest_fit
            pbest_X[improved] = X[improved]
            pbest_fit[improved] = fit[improved]

            g_idx = np.argmin(pbest_fit)
            if pbest_fit[g_idx] < gbest_fit:
                gbest_fit = pbest_fit[g_idx]
                gbest_X = pbest_X[g_idx].copy()

            it += 1
            history.append(gbest_fit)
            time_history.append(time.process_time() - t_start)
            converged = gbest_fit <= self.eps_meta

        cpu_time = time.process_time() - t_start
        return dict(x_best=gbest_X, E_best=gbest_fit, n_iter=it, history=history,
                    time_history=time_history, cpu_time=cpu_time, converged=converged)


class GA_Solver:
    """Real-coded Genetic Algorithm dengan SBX crossover & Polynomial mutation."""

    def __init__(self, pop_size=40, max_iter=200, pc=0.9, pm=0.1,
                 eta_c=15, eta_m=20, eps_meta=1e-2, seed=None):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pc, self.pm = pc, pm
        self.eta_c, self.eta_m = eta_c, eta_m
        self.eps_meta = eps_meta
        self.rng = np.random.default_rng(seed)

    def _sbx_crossover(self, p1, p2, lo, hi):
        rng = self.rng
        n = len(p1)
        c1, c2 = p1.copy(), p2.copy()
        if rng.uniform() <= self.pc:
            for i in range(n):
                if rng.uniform() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    u = rng.uniform()
                    beta = 1.0 + (2.0 * (x1 - lo) / (x2 - x1 + 1e-14))
                    alpha = 2.0 - beta ** (-(self.eta_c + 1))
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (self.eta_c + 1))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1))
                    c1[i] = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
                    c2[i] = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _poly_mutation(self, x, lo, hi):
        rng = self.rng
        n = len(x)
        y = x.copy()
        for i in range(n):
            if rng.uniform() <= self.pm:
                u = rng.uniform()
                delta1 = (y[i] - lo) / (hi - lo + 1e-14)
                delta2 = (hi - y[i]) / (hi - lo + 1e-14)
                if u < 0.5:
                    val = 2 * u + (1 - 2 * u) * (1 - delta1) ** (self.eta_m + 1)
                    deltaq = val ** (1.0 / (self.eta_m + 1)) - 1.0
                else:
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (self.eta_m + 1)
                    deltaq = 1.0 - val ** (1.0 / (self.eta_m + 1))
                y[i] = y[i] + deltaq * (hi - lo)
        return np.clip(y, lo, hi)

    def _tournament(self, pop, fit):
        rng = self.rng
        i, j = rng.integers(0, len(pop), size=2)
        return pop[i].copy() if fit[i] < fit[j] else pop[j].copy()

    def solve(self, system):
        F, n = system.F, system.n
        lo, hi = system.bounds
        rng = self.rng
        t_start = time.process_time()

        pop = rng.uniform(lo, hi, size=(self.pop_size, n))
        fit = np.array([_fitness(F, pop[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fit)
        best_X = pop[best_idx].copy()
        best_fit = fit[best_idx]

        history = [best_fit]
        time_history = [time.process_time() - t_start]
        converged = best_fit <= self.eps_meta
        it = 0

        while it < self.max_iter and not converged:
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fit)
                p2 = self._tournament(pop, fit)
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                new_pop.append(c1)
                new_pop.append(c2)
            new_pop = np.array(new_pop[:self.pop_size])
            new_fit = np.array([_fitness(F, new_pop[i]) for i in range(self.pop_size)])

            # elitism: pastikan individu terbaik lama dikekalkan
            worst_idx = np.argmax(new_fit)
            if best_fit < new_fit[worst_idx]:
                new_pop[worst_idx] = best_X
                new_fit[worst_idx] = best_fit

            pop, fit = new_pop, new_fit
            gen_best_idx = np.argmin(fit)
            if fit[gen_best_idx] < best_fit:
                best_fit = fit[gen_best_idx]
                best_X = pop[gen_best_idx].copy()

            it += 1
            history.append(best_fit)
            time_history.append(time.process_time() - t_start)
            converged = best_fit <= self.eps_meta

        cpu_time = time.process_time() - t_start
        return dict(x_best=best_X, E_best=best_fit, n_iter=it, history=history,
                    time_history=time_history, cpu_time=cpu_time, converged=converged)


class GWO_Solver:
    """Grey Wolf Optimizer dengan hierarki alpha, beta, delta."""

    def __init__(self, n_wolves=40, max_iter=200, eps_meta=1e-2, seed=None):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.eps_meta = eps_meta
        self.rng = np.random.default_rng(seed)

    def solve(self, system):
        F, n = system.F, system.n
        lo, hi = system.bounds
        rng = self.rng
        t_start = time.process_time()

        X = rng.uniform(lo, hi, size=(self.n_wolves, n))
        fit = np.array([_fitness(F, X[i]) for i in range(self.n_wolves)])

        def top3(X, fit):
            idx = np.argsort(fit)[:3]
            return X[idx[0]].copy(), fit[idx[0]], X[idx[1]].copy(), fit[idx[1]], X[idx[2]].copy(), fit[idx[2]]

        Xa, fa, Xb, fb, Xd, fd = top3(X, fit)
        history = [fa]
        time_history = [time.process_time() - t_start]
        converged = fa <= self.eps_meta
        it = 0

        while it < self.max_iter and not converged:
            a = 2.0 - 2.0 * it / self.max_iter  # menyusut linear dari 2 ke 0
            for i in range(self.n_wolves):
                # kemas kini kedudukan berpandukan alpha, beta, delta
                r1, r2 = rng.uniform(size=n), rng.uniform(size=n)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_a = np.abs(C1 * Xa - X[i])
                X1 = Xa - A1 * D_a

                r1, r2 = rng.uniform(size=n), rng.uniform(size=n)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_b = np.abs(C2 * Xb - X[i])
                X2 = Xb - A2 * D_b

                r1, r2 = rng.uniform(size=n), rng.uniform(size=n)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_d = np.abs(C3 * Xd - X[i])
                X3 = Xd - A3 * D_d

                X[i] = np.clip((X1 + X2 + X3) / 3.0, lo, hi)

            fit = np.array([_fitness(F, X[i]) for i in range(self.n_wolves)])
            Xa, fa, Xb, fb, Xd, fd = top3(
                np.vstack([X, Xa, Xb, Xd]),
                np.concatenate([fit, [fa, fb, fd]])
            )

            it += 1
            history.append(fa)
            time_history.append(time.process_time() - t_start)
            converged = fa <= self.eps_meta

        cpu_time = time.process_time() - t_start
        return dict(x_best=Xa, E_best=fa, n_iter=it, history=history,
                    time_history=time_history, cpu_time=cpu_time, converged=converged)


SOLVERS = {
    "PSO": PSO_Solver,
    "GA": GA_Solver,
    "GWO": GWO_Solver,
}


if __name__ == "__main__":
    from benchmarks import System1ExpTrig
    sysobj = System1ExpTrig()
    for name, cls in SOLVERS.items():
        solver = cls(seed=1)
        res = solver.solve(sysobj)
        print(f"{name:5s} E_best={res['E_best']:.4e}  n_iter={res['n_iter']:4d}  "
              f"cpu={res['cpu_time']:.4f}s  x_best={res['x_best']}")
