# Quantum portfolio selection (QAOA, Euro Stoxx 50, 2015–2026)

Penalty QAOA vs constraint-native XY-mixer QAOA for choosing k=4 of 12 stocks, on 139 real monthly instances, plus sector constraints and a mixed stocks+commodities universe (Part 2). Exact statevector simulation, exact optimum by enumeration, bootstrap CIs and paired Wilcoxon tests, parameter transfer across months, descriptive out-of-sample backtest.

**Data are not included** (Yahoo Finance / Curvo terms). To reproduce: `python download_prices.py`, `python download_commodities.py`, put the outputs and the Curvo Euro Stoxx 50 CSV (`benchmark_curvo.csv`) in `data/`, then run the notebook (~20 min on 2 CPUs).

Limits: survivorship bias (current index members), no transaction costs, no hardware, no quantum advantage claim. Not investment advice.

Author: Francesco Pisaturo (ORCID 0009-0002-5963-8036). Independent work by a student; not an official publication of, nor endorsed by, any university.

## Scaling and robustness (`scaling_robustness.ipynb`)
`experiments.py` (≈2.5 h on 2 CPUs) runs n = 8…16 qubits, p = 1…4, five (risk aversion, window) settings and a penalty scan; P_auto is computed by qiskit-optimization itself. The advantage of XY + Dicke over penalty QAOA grows with n (median ratio of enhancements ≈ 8× at n = 8, ≈ 56× at n = 16) and holds in every setting tested. Library: `qps_lib.py`.

## Hardware cost, noise and literature (`hardware_literature.ipynb`)
`hardware_sim.py` compiles both ansätze for an IBM Heron device model (FakeMarrakesh) and runs a noisy simulation at n = 8. The XY + Dicke ansatz needs ≈2–3.6× the 2-qubit gates, mostly for Dicke-state preparation. Under the device noise model its advantage survives only with postselection (Σx = k) at p = 1. The notebook also places the work in the literature (Hadfield 2019, Wang 2020, Bärtschi–Eidenbenz 2019, Hodson 2019, Brandhofer 2023, He 2023, Egger 2020, Herman 2023).
