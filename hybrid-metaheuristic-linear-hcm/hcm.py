"""
hcm.py
======
Phase 2 (Local Solver): Linear Newton Homotopy Continuation Method
(Linear Newton-HCM).

Formulation (classical/standard linear parameter homotopy -- the k=1
special case of Verschelde's (1996) general parameterized homotopy family
H(x,t) = (1-t)^k G(x) + t^k F(x)):

    H(x, t) = (1 - t) G(x) + t F(x) = 0,   t in [0, 1]
    G(x) = x - x0

This is the primary reference formulation used as the "linear" member of
the linear-quadratic-cubic trilogy completed by this study: the quadratic
parameter homotopy of Mohamad Nor & Mohd Yatim (2021) and the cubic
parameter homotopy of Abdul Rahman et al. (2025) are both degree-k
generalizations built on top of this same classical linear form.

Jacobian of H with respect to x:

    J_H(x, t) = (1 - t) * I + t * J_F(x)

Newton-HCM update scheme:

    x_{k+1} = x_k - [J_H(x_k, t)]^{-1} H(x_k, t)

The homotopy path is tracked stepwise with a simple predictor-corrector
scheme: start at t=0 (with x = x0, so H(x0,0) = G(x0) = 0 exactly), then
increase t by delta_t up to t=1, performing several Newton (corrector)
iterations at each t until H is sufficiently small before advancing to
the next t. Final stopping criterion (at t=1):

    ||F(x)||_inf <= tol (default 1e-6)
"""

import time
import numpy as np


class LinearNewtonHCM:
    def __init__(self, n_steps=100, max_newton_iter=25, tol=1e-6,
                 corrector_tol=1e-8, damping=1.0):
        """
        n_steps         : number of t-steps from 0 to 1 (delta_t = 1/n_steps)
        max_newton_iter : cap on Newton (corrector) iterations per t-value
        tol             : final convergence criterion ||F(x)||_inf <= tol (at t=1)
        corrector_tol   : internal convergence criterion at each t-step
        damping         : optional damping factor for Newton-iteration stability
        """
        self.n_steps = n_steps
        self.max_newton_iter = max_newton_iter
        self.tol = tol
        self.corrector_tol = corrector_tol
        self.damping = damping

    @staticmethod
    def _H(x, t, x0, F, G):
        a = (1 - t)
        return a * G(x, x0) + (1 - a) * F(x)

    @staticmethod
    def _JH(x, t, x0, JF, n):
        a = (1 - t)
        return a * np.eye(n) + (1 - a) * JF(x)

    def solve(self, system, x0):
        """
        system : benchmark object with methods F(x), J(x), and attribute n
        x0     : initial guess (typically x_best from the metaheuristic phase)

        Returns dict:
            x_sol        : final solution
            residual_inf : ||F(x_sol)||_inf
            n_iter       : total number of Newton iterations (N_HCM) across all t-steps
            t_path       : list of t values traversed
            x_path       : list of x at each t value (for path-tracking plots)
            residual_path: list of ||F(x)||_inf at each t value
            cpu_time     : CPU time (seconds)
            converged    : True if ||F(x_sol)||_inf <= tol
        """
        F, JF, n = system.F, system.J, system.n
        G = lambda x, x0_: x - x0_

        t_start = time.process_time()
        x = np.array(x0, dtype=float).copy()
        delta_t = 1.0 / self.n_steps

        t_path = [0.0]
        x_path = [x.copy()]
        residual_path = [np.max(np.abs(F(x)))]
        total_newton_iter = 0

        t = 0.0
        for step in range(1, self.n_steps + 1):
            t = step * delta_t
            # Corrector: Newton iterations at the current t
            for _ in range(self.max_newton_iter):
                Hx = self._H(x, t, x0, F, G)
                if np.max(np.abs(Hx)) <= self.corrector_tol:
                    break
                Jx = self._JH(x, t, x0, JF, n)
                try:
                    dx = np.linalg.solve(Jx, Hx)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(Jx, Hx, rcond=None)[0]
                x = x - self.damping * dx
                total_newton_iter += 1

            t_path.append(t)
            x_path.append(x.copy())
            residual_path.append(np.max(np.abs(F(x))))

        cpu_time = time.process_time() - t_start
        residual_inf = np.max(np.abs(F(x)))
        converged = residual_inf <= self.tol

        return dict(x_sol=x, residual_inf=residual_inf, n_iter=total_newton_iter,
                    t_path=t_path, x_path=x_path, residual_path=residual_path,
                    cpu_time=cpu_time, converged=converged)


if __name__ == "__main__":
    from benchmarks import System1ExpTrig, System2BroydenTridiagonal, System3ExtendedRosenbrock

    for sysobj, x0 in [
        (System1ExpTrig(), np.array([0.5, 0.5])),
        (System2BroydenTridiagonal(10), np.zeros(10)),
        (System3ExtendedRosenbrock(8), np.zeros(8)),
    ]:
        hcm = LinearNewtonHCM(n_steps=100, max_newton_iter=25, tol=1e-6)
        res = hcm.solve(sysobj, x0)
        print(f"{sysobj.name:35s} converged={res['converged']!s:5s} "
              f"||F||_inf={res['residual_inf']:.3e}  N_HCM={res['n_iter']:4d}  "
              f"cpu={res['cpu_time']:.4f}s")
