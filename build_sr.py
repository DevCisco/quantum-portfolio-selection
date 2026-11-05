import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# Scaling and robustness of the penalty-vs-XY comparison
Companion to `quantum_portfolio_selection.ipynb`. Two questions a reviewer asks first:
1. **Scaling**: does the advantage of constraint-native QAOA (ring XY mixer + Dicke start) over penalty QAOA hold, grow or shrink with problem size n = 8…16 and depth p = 1…4?
2. **Robustness**: does it depend on choices I made — risk aversion q, estimation window, penalty value?

Changes vs the main notebook: P_auto is now the value **actually computed by qiskit-optimization's `LinearEqualityToPenalty`** (not an approximation), converted to the units of the rescaled cost; months start in 2016 so that a 24-month window fits.

Main metric: **enhancement = p(optimum) × |feasible set|**, i.e. how many times better than guessing uniformly among valid portfolios (1 = no better than random). p(optimum) itself shrinks with n simply because there are more portfolios, so it cannot be compared across n.

Results are produced by `experiments.py` (≈1.5 h on 2 CPUs, exact statevector, L-BFGS-B, seeded) and loaded here.""")
code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from math import comb
SEED=20261012
sc=pd.read_csv('results/scaling.csv',parse_dates=['month']); rb=pd.read_csv('results/robust.csv',parse_dates=['month']); pg=pd.read_csv('results/pgrid.csv',parse_dates=['month'])
def boot(x,B=5000):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; m=np.random.default_rng(SEED).choice(x,(B,len(x))).mean(1)
    return x.mean(),*np.percentile(m,[2.5,97.5])
f=lambda x: '{:.2f} [{:.2f}, {:.2f}]'.format(*boot(x))
print(sc.groupby('n').month.nunique().to_dict(), '| feasible:', {n:comb(n,round(n/3)) for n in sorted(sc.n.unique())})""")
md("## 1. Scaling with n and p (12 months spread over 2016–2026, 3 restarts)")
code("""t=sc.groupby(['n','p','method']).agg(enh=('enhancement',f),p_feas=('p_feas','mean'),ratio=('ratio_cond','mean'),nfev=('nfev','mean')).round(3)
display(t.unstack('method')['enh'])
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
for m,mk in [('xy_dicke','o-'),('pen_tight','s--'),('pen_auto','^:')]:
    for p,col in zip([1,2,3,4],['C0','C1','C2','C3']):
        d=sc[(sc.method==m)&(sc.p==p)].groupby('n')
        ax[0].plot(d.enhancement.mean(),mk,color=col,label=f'{m} p={p}' if m=='xy_dicke' or p==1 else None)
    d=sc[sc.method==m].groupby('n')
    ax[1].plot(d.ratio_cond.mean(),mk,label=m); ax[2].plot(d.p_feas.mean(),mk,label=m)
ax[0].set_yscale('log'); ax[0].set_title('enhancement over random valid guess'); ax[0].set_xlabel('n')
ax[1].set_title('approx. ratio | feasible (all p)'); ax[2].set_title('p(feasible) (all p)')
for a in ax: a.set_xlabel('n')
ax[0].legend(fontsize=6); ax[2].legend(fontsize=7); plt.tight_layout(); plt.savefig('results/scaling.png',dpi=130)""")
code("""rows=[]
for (n,p),g in sc.groupby(['n','p']):
    pv=g.pivot(index='month',columns='method',values='enhancement')
    for other in ['pen_tight','pen_auto']:
        r=pv.xy_dicke/pv[other]
        rows.append(dict(n=n,p=p,vs=other,median_ratio=np.median(r),xy_better=(r>1).mean(),wilcoxon_p=wilcoxon(pv.xy_dicke-pv[other]).pvalue))
paired=pd.DataFrame(rows); display(paired.pivot_table(index=['n','p'],columns='vs',values=['median_ratio','xy_better']).round(2))
g=paired.groupby('n').median_ratio.median(); slope=np.polyfit(g.index,np.log(g.values),1)[0]
print(f'median (XY / best penalty) enhancement ratio vs n: {g.round(2).to_dict()}; log-slope {slope:+.3f} per qubit')""")
md("""Only 12 months per (n, p) cell, so the per-cell Wilcoxon tests have low power (minimum attainable p ≈ 5·10⁻⁴). Read the trend across n, not single cells.

## 2. Robustness at n = 12, p = 2 (46 months, every third month 2016–2026, 5 restarts)""")
code("""rb['config']=rb.apply(lambda r:f"q={r.q:g}, win={int(r.win)}",axis=1)
t=rb.groupby(['config','method']).agg(enh=('enhancement',f),p_feas=('p_feas','mean')).unstack('method')
display(t)
rows=[]
for cfg,g in rb.groupby('config'):
    pv=g.pivot(index='month',columns='method',values='enhancement')
    for o in ['pen_tight','pen_auto']:
        d=pv.xy_dicke-pv[o]; rows.append(dict(config=cfg,vs=o,xy_better=(d>0).mean(),wilcoxon_p=wilcoxon(d).pvalue))
display(pd.DataFrame(rows).pivot(index='config',columns='vs'))""")
md("### Penalty scan: is there any P for which penalty QAOA catches up?")
code("""base=rb[(rb.q==2)&(rb.win==12)]
tight=base[base.method=='pen_tight'].assign(method='pen_1')
pgx=pd.concat([pg,tight]); pgx['factor']=pgx.method.str.replace('pen_','').astype(float)
s=pgx.groupby('factor').agg(enh=('enhancement','mean'),p_feas=('p_feas','mean'),ratio=('ratio_cond','mean'))
s.loc['qiskit auto (median factor %.1f)'%(base[base.method=='pen_auto'].P.median()/tight.P.median())]=base[base.method=='pen_auto'][['enhancement','p_feas','ratio_cond']].mean().values
display(s.round(3)); xy=base[base.method=='xy_dicke'].enhancement.mean()
print(f'xy_dicke enhancement on the same months: {xy:.2f}; best penalty factor reaches {s.enh.max():.2f}')
fig,ax=plt.subplots(figsize=(5,3.2)); ss=s.iloc[:-1]
ax.semilogx(ss.index.astype(float),ss.enh,'s-',label='penalty (factor × P_tight)'); ax.axhline(xy,color='C2',label='XY + Dicke'); ax.axhline(1,color='gray',ls=':',label='random valid guess')
ax.set_xlabel('penalty factor'); ax.set_ylabel('enhancement'); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig('results/penalty_scan.png',dpi=130)""")
md("## 3. Conclusions (printed from the numbers)")
code("""e=sc.groupby(['n','method']).enhancement.mean().unstack()
for n in e.index: print(f"n={n:2d}: enhancement XY {e.loc[n,'xy_dicke']:.1f} | pen_tight {e.loc[n,'pen_tight']:.1f} | pen_auto {e.loc[n,'pen_auto']:.1f}")
w=pd.DataFrame(rows)
print(f"Robustness: XY beats both penalties in {w.xy_better.min():.0%}–{w.xy_better.max():.0%} of months across the 5 (q, window) settings; largest Wilcoxon p = {w.wilcoxon_p.max():.1e}.")
print('Limits: exact simulation only up to 16 qubits; 12 months per scaling cell; p ≤ 4; L-BFGS-B with 3–5 restarts may under-train deep circuits equally for all methods.')""")
nb=nbf.v4.new_notebook(); nb.cells=C; nbf.write(nb,'scaling_robustness.ipynb')
