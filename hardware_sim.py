"""Point 3: compiled cost on an IBM Heron device model and noisy simulation. Writes results/gatecounts.csv, results/noisy.csv."""
import numpy as np, pandas as pd, time
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qps_lib import *; from qps_circuits import *
SEED = 20261012
ORDER = ['ASML.AS','SAP.DE','SIE.DE','AIR.PA','SAN.MC','ALV.DE','TTE.PA','IBE.MC','MC.PA','OR.PA','SAN.PA','DTE.DE','SU.PA','BBVA.MC','ENEL.MI','ITX.MC']
L = np.log(pd.read_csv('data/prices_monthly.csv', index_col=0, parse_dates=True)).diff().dropna()
MONTHS = [d for d in L.index if (L.index < d).sum() >= 12 and d >= pd.Timestamp('2016-01-01')]
backend = FakeMarrakesh()
def H_for(sp, c, m): return c if m == 'xy_dicke' else c + p_tight(sp, c) * (sp.ham - sp.k) ** 2
def inst(t, n, k):
    h = L.loc[L.index < t, ORDER[:n]].iloc[-12:]; sp = Space(n, k)
    c, _ = scaled(sp, objective(sp, h.mean().values * 12, h.cov().values * 12, 2.0)); return sp, c
rows = []
for n in [8, 12, 16]:
    k = round(n / 3)
    for t in MONTHS[::40][:3]:
        sp, c = inst(t, n, k)
        for p in [1, 2, 3]:
            for m in ['pen_tight', 'xy_dicke']:
                qc = qaoa_circuit(H_for(sp, c, m), n, np.full(2 * p, 0.3), p, m, k); qc.measure_all()
                for s in range(3):
                    tq = transpile(qc, backend, optimization_level=3, seed_transpiler=SEED + s)
                    ops = tq.count_ops()
                    rows.append(dict(n=n, k=k, p=p, month=t, method=m, seed=s, twoq=ops.get('cz', 0) + ops.get('ecr', 0) + ops.get('cx', 0),
                                     depth=tq.depth(lambda x: x.operation.num_qubits == 2)))
        print('gates n', n, time.strftime('%H:%M:%S'), flush=True)
pd.DataFrame(rows).to_csv('results/gatecounts.csv', index=False)
sim = AerSimulator.from_backend(backend); SH = 8192; out = []
n, k = 8, 3
for mi, t in enumerate(MONTHS[::12]):
    sp, c = inst(t, n, k); opt = sp.feas & (c <= 1e-12)
    for p in [1, 2]:
        for m in ['pen_tight', 'xy_dicke']:
            H = H_for(sp, c, m)
            pr, _, th = run(sp, 'xy_dicke' if m == 'xy_dicke' else 'pen', c, p, 5, SEED + mi, P=p_tight(sp, c), return_theta=True)
            qc = qaoa_circuit(H, n, th, p, m, k); qc.measure_all()
            tq = transpile(qc, backend, optimization_level=3, seed_transpiler=SEED)
            cnt = sim.run(tq, shots=SH, seed_simulator=SEED + mi).result().get_counts()
            q = np.zeros(sp.N)
            for b, v in cnt.items(): q[int(b.replace(' ', ''), 2)] += v / SH
            for label, dist in [('noiseless', pr), ('noisy', q)]:
                out.append(dict(month=t, p=p, method=m, sim=label, twoq=tq.count_ops().get('cz', 0),
                                p_feas=dist[sp.feas].sum(), p_opt=dist[opt].sum(), enhancement=dist[opt].sum() * sp.feas.sum(),
                                ratio_cond=1 - (dist[sp.feas] @ c[sp.feas]) / max(dist[sp.feas].sum(), 1e-12)))
    print('noisy month', mi, time.strftime('%H:%M:%S'), flush=True)
pd.DataFrame(out).to_csv('results/noisy.csv', index=False); print('done')
