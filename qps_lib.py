"""Exact statevector QAOA for k-of-n portfolio selection (generic n).
Used by scaling_robustness.ipynb. Bit i of a basis index = asset i."""
import numpy as np
from scipy.optimize import minimize

class Space:
    def __init__(self, n, k):
        self.n, self.k, self.N = n, k, 2 ** n
        ar = np.arange(self.N)
        self.bits = ((ar[:, None] >> np.arange(n)) & 1).astype(np.float64)
        self.ham = self.bits.sum(1)
        self.feas = self.ham == k
        ring = [(i, (i + 1) % n) for i in range(n)]
        assert n % 2 == 0, 'use even n'
        self.edges = []  # applied one edge at a time (edges in a layer commute)
        for edges in (ring[0::2], ring[1::2]):
            for a, b in edges:
                idx = np.where(((ar >> a) & 1 == 1) & ((ar >> b) & 1 == 0))[0]
                self.edges.append((idx, idx ^ ((1 << a) | (1 << b))))
        self.plus = np.full(self.N, 1 / np.sqrt(self.N), complex)
        self.dicke = self.feas / np.sqrt(self.feas.sum()) + 0j

    def rx(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(self.n):
            v = psi.reshape(-1, 2, 2 ** q)
            a = v[:, 0, :].copy(); b = v[:, 1, :].copy()
            v[:, 0, :] = c * a + s * b; v[:, 1, :] = s * a + c * b
        return psi

    def xy(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for I, J in self.edges:
            a = psi[I]; b = psi[J]
            psi[I] = c * a + s * b; psi[J] = s * a + c * b
        return psi


def objective(sp, mu, S, q):
    b = sp.bits
    return b @ mu - q * np.einsum('si,ij,sj->s', b, S, b)


def scaled(sp, f):
    fe = f[sp.feas]; rng = fe.max() - fe.min()
    return (fe.max() - f) / rng, rng


def p_auto_qiskit(mu, S, q, k, scale):
    """Penalty chosen automatically by qiskit-optimization's LinearEqualityToPenalty
    for min -f(x) s.t. sum x = k, expressed in units of the rescaled cost c = -f/scale."""
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import LinearEqualityToPenalty
    n = len(mu); qp = QuadraticProgram()
    for i in range(n): qp.binary_var(f'x{i}')
    lin = {f'x{i}': -(mu[i]) for i in range(n)}
    quad = {(f'x{i}', f'x{j}'): q * S[i, j] for i in range(n) for j in range(n)}
    qp.minimize(linear=lin, quadratic=quad)
    qp.linear_constraint({f'x{i}': 1 for i in range(n)}, '==', k)
    conv = LinearEqualityToPenalty(); conv.convert(qp)
    return conv.penalty / scale


def p_tight(sp, c):
    nb = np.arange(sp.N)[:, None] ^ (1 << np.arange(sp.n))
    return np.abs(c[:, None] - c[nb]).max() + 0.1


def run(sp, method, c, p, restarts, seed, P=None, maxiter=200, return_theta=False):
    rng = np.random.default_rng(seed)
    if method == 'xy_dicke':
        H, start, mix = c, sp.dicke, sp.xy
    else:
        H, start, mix = c + P * (sp.ham - sp.k) ** 2, sp.plus, sp.rx

    def state(th):
        psi = start.copy()
        for l in range(p):
            psi = mix(psi * np.exp(-1j * th[l] * H), th[p + l])
        return psi

    E = lambda th: float(np.abs(state(th)) ** 2 @ H)
    best, nf = None, 0
    for _ in range(restarts):
        x0 = np.r_[rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)]
        r = minimize(E, x0, method='L-BFGS-B', options={'maxiter': maxiter}); nf += r.nfev
        if best is None or r.fun < best.fun: best = r
    pr = np.abs(state(best.x)) ** 2
    return (pr, nf, best.x) if return_theta else (pr, nf)


def metrics(sp, pr, c):
    pf = pr[sp.feas].sum(); opt = sp.feas & (c <= 1e-12)
    popt = pr[opt].sum()
    return dict(p_feas=pf, p_opt=popt, enhancement=popt * sp.feas.sum(),
                ratio_cond=1 - (pr[sp.feas] @ c[sp.feas]) / pf if pf > 1e-12 else np.nan)
