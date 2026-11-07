"""
benchmarks.py
=============
Sistem tanda aras (benchmark systems) bagi Sistem Persamaan Tak Linear
Berdimensi Tinggi F(x) = 0, lengkap dengan Jacobian analitik J_F(x).

Setiap sistem didedahkan sebagai kelas dengan kaedah:
    F(x)      -> np.ndarray, shape (n,)
    J(x)      -> np.ndarray, shape (n, n)
    n         -> dimensi sistem
    bounds    -> (lo, hi) sempadan carian untuk fasa metaheuristik
    name      -> label sistem
"""

import numpy as np


class System1ExpTrig:
    """Sistem 1 (Exp-Trig 2D):
        f1 = e^x1 + x2 - 1 = 0
        f2 = x1^2 + x2^2 - 4 = 0
    """
    name = "System1_ExpTrig2D"

    def __init__(self):
        self.n = 2
        self.bounds = (-5.0, 5.0)

    def F(self, x):
        x1, x2 = x
        return np.array([
            np.exp(x1) + x2 - 1.0,
            x1 ** 2 + x2 ** 2 - 4.0
        ])

    def J(self, x):
        x1, x2 = x
        return np.array([
            [np.exp(x1), 1.0],
            [2.0 * x1, 2.0 * x2]
        ])


class System2BroydenTridiagonal:
    """Sistem 2 (Broyden Tridiagonal), dimensi n:
        f_i(x) = (3 - 2 x_i) x_i - x_{i-1} - 2 x_{i+1} + 1,  i = 1..n
        dengan x_0 = x_{n+1} = 0 (syarat sempadan piawai).
    """
    def __init__(self, n=10):
        self.n = n
        self.bounds = (-2.0, 2.0)
        self.name = f"System2_BroydenTridiagonal_n{n}"

    def F(self, x):
        n = self.n
        xp = np.zeros(n + 2)
        xp[1:n + 1] = x
        f = np.zeros(n)
        for i in range(1, n + 1):
            f[i - 1] = (3 - 2 * xp[i]) * xp[i] - xp[i - 1] - 2 * xp[i + 1] + 1
        return f

    def J(self, x):
        n = self.n
        Jm = np.zeros((n, n))
        for i in range(n):
            Jm[i, i] = 3 - 4 * x[i]
            if i - 1 >= 0:
                Jm[i, i - 1] = -1.0
            if i + 1 < n:
                Jm[i, i + 1] = -2.0
        return Jm


class System3ExtendedRosenbrock:
    """Sistem 3 (Extended Rosenbrock), dimensi n (genap):
        Untuk setiap pasangan i = 1, 3, 5, ..., n-1:
            f_i     = 10 (x_{i+1} - x_i^2)
            f_{i+1} = 1 - x_i
    """
    def __init__(self, n=8):
        assert n % 2 == 0, "n mesti genap untuk Extended Rosenbrock"
        self.n = n
        self.bounds = (-2.5, 2.5)
        self.name = f"System3_ExtendedRosenbrock_n{n}"

    def F(self, x):
        n = self.n
        f = np.zeros(n)
        for k in range(0, n, 2):
            f[k] = 10.0 * (x[k + 1] - x[k] ** 2)
            f[k + 1] = 1.0 - x[k]
        return f

    def J(self, x):
        n = self.n
        Jm = np.zeros((n, n))
        for k in range(0, n, 2):
            Jm[k, k] = -20.0 * x[k]
            Jm[k, k + 1] = 10.0
            Jm[k + 1, k] = -1.0
            Jm[k + 1, k + 1] = 0.0
        return Jm


class System4ExtendedPowellSingular:
    """Sistem 4 (Extended Powell Singular; More, Garbow & Hillstrom 1981,
    problem #13), dimensi n (gandaan 4). Untuk setiap blok 4 pemboleh ubah
    (x_{4k-3},x_{4k-2},x_{4k-1},x_{4k}):
        f_{4k-3} = x_{4k-3} + 10 x_{4k-2}
        f_{4k-2} = sqrt(5) (x_{4k-1} - x_{4k})
        f_{4k-1} = (x_{4k-2} - 2 x_{4k-1})^2
        f_{4k}   = sqrt(10) (x_{4k-3} - x_{4k})^2
    Penyelesaian x* = 0 untuk semua komponen. Jacobian pada x* adalah
    SINGULAR TEPAT (rank separuh sahaja), berbeza daripada Broyden
    Tridiagonal yang hanya hampir-singular pada n tinggi -- ujian tekanan
    yang lebih tajam terhadap penumpuan tempatan Newton berhampiran punca.
    """
    def __init__(self, n=8):
        assert n % 4 == 0, "n mesti gandaan 4 untuk Extended Powell Singular"
        self.n = n
        self.bounds = (-5.0, 5.0)
        self.name = f"System4_ExtendedPowellSingular_n{n}"

    def F(self, x):
        n = self.n
        f = np.zeros(n)
        for k in range(0, n, 4):
            x1, x2, x3, x4 = x[k], x[k+1], x[k+2], x[k+3]
            f[k] = x1 + 10.0 * x2
            f[k+1] = np.sqrt(5.0) * (x3 - x4)
            f[k+2] = (x2 - 2.0 * x3) ** 2
            f[k+3] = np.sqrt(10.0) * (x1 - x4) ** 2
        return f

    def J(self, x):
        n = self.n
        Jm = np.zeros((n, n))
        for k in range(0, n, 4):
            x1, x2, x3, x4 = x[k], x[k+1], x[k+2], x[k+3]
            Jm[k, k] = 1.0
            Jm[k, k+1] = 10.0
            Jm[k+1, k+2] = np.sqrt(5.0)
            Jm[k+1, k+3] = -np.sqrt(5.0)
            Jm[k+2, k+1] = 2.0 * (x2 - 2.0 * x3)
            Jm[k+2, k+2] = -4.0 * (x2 - 2.0 * x3)
            Jm[k+3, k] = 2.0 * np.sqrt(10.0) * (x1 - x4)
            Jm[k+3, k+3] = -2.0 * np.sqrt(10.0) * (x1 - x4)
        return Jm


def get_all_systems():
    """Kembalikan senarai lengkap sistem tanda aras yang digunakan dalam kajian."""
    systems = [System1ExpTrig()]
    for n in (5, 10, 20):
        systems.append(System2BroydenTridiagonal(n))
    for n in (4, 8, 16):
        systems.append(System3ExtendedRosenbrock(n))
    for n in (4, 8, 16):
        systems.append(System4ExtendedPowellSingular(n))
    return systems


if __name__ == "__main__":
    # Ujian pantas: semak kecerunan berangka lawan Jacobian analitik
    rng = np.random.default_rng(0)
    for sysobj in get_all_systems():
        x0 = rng.uniform(-1, 1, sysobj.n)
        Ja = sysobj.J(x0)
        eps = 1e-6
        Jn = np.zeros_like(Ja)
        for j in range(sysobj.n):
            dx = np.zeros(sysobj.n)
            dx[j] = eps
            Jn[:, j] = (sysobj.F(x0 + dx) - sysobj.F(x0 - dx)) / (2 * eps)
        err = np.max(np.abs(Ja - Jn))
        print(f"{sysobj.name:35s} n={sysobj.n:3d}  max|J_analitik - J_berangka| = {err:.2e}")
