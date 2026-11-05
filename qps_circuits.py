"""Qiskit circuits matching qps_lib conventions (qubit i = bit i of the basis index)."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate, RYGate

def _scs(qc, m, k):
    """Split-and-cyclic-shift block U_{m,k} of Baertschi & Eidenbenz (2019), on qubits 0..m-1."""
    qc.cx(m - 2, m - 1)
    qc.cry(2 * np.arccos(np.sqrt(1 / m)), m - 1, m - 2)
    qc.cx(m - 2, m - 1)
    for l in range(2, k + 1):
        qc.cx(m - l - 1, m - 1)
        qc.append(RYGate(2 * np.arccos(np.sqrt(l / m))).control(2), [m - 1, m - l, m - l - 1])
        qc.cx(m - l - 1, m - 1)

def dicke_circuit(n, k):
    qc = QuantumCircuit(n)
    for i in range(n - k, n): qc.x(i)
    for m in range(n, k, -1): _scs(qc, m, k)
    for m in range(k, 1, -1): _scs(qc, m, m - 1)
    return qc

def ising(H, n):
    """Exact Walsh decomposition of a diagonal of degree <= 2: H = c0 + sum h_i Z_i + sum J_ij Z_i Z_j."""
    N = 2 ** n; z = 1 - 2 * ((np.arange(N)[:, None] >> np.arange(n)) & 1)
    h = (H[:, None] * z).mean(0)
    J = {(i, j): float((H * z[:, i] * z[:, j]).mean()) for i in range(n) for j in range(i + 1, n)}
    rec = H.mean() + z @ h + sum(v * z[:, i] * z[:, j] for (i, j), v in J.items())
    assert np.allclose(rec, H), 'cost is not quadratic'
    return h, J

def qaoa_circuit(H, n, theta, p, method, k):
    h, J = ising(H, n); qc = QuantumCircuit(n)
    if method == 'xy_dicke': qc.compose(dicke_circuit(n, k), inplace=True)
    else: qc.h(range(n))
    for l in range(p):
        g, b = theta[l], theta[p + l]
        for i in range(n): qc.rz(2 * g * h[i], i)
        for (i, j), v in J.items():
            if abs(v) > 1e-12: qc.rzz(2 * g * v, i, j)
        if method == 'xy_dicke':
            for a, c in [(i, (i + 1) % n) for i in range(0, n, 2)] + [(i, (i + 1) % n) for i in range(1, n, 2)]:
                qc.append(XXPlusYYGate(2 * b), [a, c])
        else:
            for i in range(n): qc.rx(2 * b, i)
    return qc
