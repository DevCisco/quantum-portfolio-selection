"""Scaling (point 1) and robustness (point 2) experiments. Writes results/scaling.csv, results/robustness.csv, results/penalty_grid.csv.
Run:  python experiments.py  (about 1.5 h on 2 CPUs)."""
import numpy as np, pandas as pd, sys, time
from multiprocessing import Pool
from qps_lib import *
SEED = 20261012
ORDER = ['ASML.AS','SAP.DE','SIE.DE','AIR.PA','SAN.MC','ALV.DE','TTE.PA','IBE.MC','MC.PA','OR.PA','SAN.PA','DTE.DE',
         'SU.PA','BBVA.MC','ENEL.MI','ITX.MC']
px = pd.read_csv('data/prices_monthly.csv', index_col=0, parse_dates=True)
L = np.log(px).diff().dropna()
MONTHS = [d for d in L.index if (L.index < d).sum() >= 24 and d >= pd.Timestamp('2016-01-01')]  # 24 so all windows fit
SPACES = {}
def space(n, k):
    if (n, k) not in SPACES: SPACES[(n, k)] = Space(n, k)
    return SPACES[(n, k)]
def inst(t, n, k, q, win):
    h = L.loc[L.index < t, ORDER[:n]].iloc[-win:]
    sp = space(n, k); mu = h.mean().values * 12; S = h.cov().values * 12
    c, sc = scaled(sp, objective(sp, mu, S, q))
    return sp, c, p_auto_qiskit(mu, S, q, k, sc)
def task(a):
    kind, t, n, k, p, q, win, method, P_rule, restarts = a
    sp, c, pa = inst(t, n, k, q, win)
    if method == 'xy_dicke': P = None
    elif P_rule == 'auto': P = pa
    elif P_rule == 'tight': P = p_tight(sp, c)
    else: P = float(P_rule) * p_tight(sp, c)
    seed = SEED + MONTHS.index(t) * 101 + n * 7 + p
    t0 = time.time(); pr, nf = run(sp, method, c, p, restarts, seed, P=P)
    return dict(kind=kind, month=t, n=n, k=k, p=p, q=q, win=win, method=method if method == 'xy_dicke' else f'pen_{P_rule}',
                P=P, nfev=nf, sec=time.time() - t0, **metrics(sp, pr, c))
if __name__ == '__main__':
    tasks = []
    sm = MONTHS[::len(MONTHS) // 12][:12]
    for n in [8, 10, 12, 14, 16]:
        k = round(n / 3)
        for p in [1, 2, 3, 4]:
            for t in sm:
                for m, r in [('xy_dicke', None), ('pen', 'tight'), ('pen', 'auto')]:
                    tasks.append(('scaling', t, n, k, p, 2.0, 12, m, r, 3))
    rm = MONTHS[::3]
    for q, win in [(2.0, 12), (0.5, 12), (8.0, 12), (2.0, 6), (2.0, 24)]:
        for t in rm:
            for m, r in [('xy_dicke', None), ('pen', 'tight'), ('pen', 'auto')]:
                tasks.append(('robust', t, 12, 4, 2, q, win, m, r, 5))
    for f in ['0.25', '0.5', '2', '4', '8', '16']:
        for t in rm:
            tasks.append(('pgrid', t, 12, 4, 2, 2.0, 12, 'pen', f, 5))
    tasks.sort(key=lambda a: -a[2] * a[4])  # big jobs first
    print(len(tasks), 'tasks', flush=True)
    out = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(task, tasks)):
            out.append(r)
            if i % 100 == 0:
                print(i, time.strftime('%H:%M:%S'), flush=True); pd.DataFrame(out).to_csv('results/experiments_partial.csv', index=False)
    d = pd.DataFrame(out)
    for kind in ['scaling', 'robust', 'pgrid']:
        d[d.kind == kind].to_csv(f'results/{kind}.csv', index=False)
    print('done')
