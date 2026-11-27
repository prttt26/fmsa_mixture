"""Medians, IQR, cluster bootstrap and paired differences for the HCTY (Yu & Jin 2008) pool.

Reads numerical_notes/results/benchmark_rdf/literature/hcty_metrics.csv.  Write-up:
FMSA_dp_output/md/response/hcty_full_comparison_and_table2.md.
"""
import pandas as pd, numpy as np
from scipy import stats
rng=np.random.default_rng(20261117); B=5000
d=pd.read_csv('numerical_notes/results/benchmark_rdf/literature/hcty_metrics.csv')
M={'OZ+MSA':'OZ-MSA','OZ+HNCB':'OZ-HNCB','FMSA-DP':'FMSA','MSA-exact':'MSA-exact'}
print('states',d.row.nunique(),d.groupby('model').converged.sum().to_dict(),'cases',d.groupby('case').row.nunique().to_dict())
d['model']=d.model.map(M)
for p in ['11','22','12']:
    d['e_g'+p]=(d['g'+p]-d['g'+p+'_ref']).abs()/d['g'+p+'_ref'].abs()
d['e_U']=(d.U-d.U_ref).abs()/d.U_ref.abs(); d['e_Z']=(d.Z-d.Z_ref).abs()/d.Z_ref.abs()
d['s_g11']=(d.g11-d.g11_ref)/d.g11_ref
def wide(col): return d.pivot(index='row',columns='model',values=col)
# contact per state: mean over three pairs
d['e_contact']=d[['e_g11','e_g22','e_g12']].mean(axis=1)
cl=d.drop_duplicates('row').set_index('row')[['table','case']].apply(lambda r:f'{r.table}-{r.case}',axis=1)
def bci(v,c):
    gs=[g.values for _,g in pd.Series(np.asarray(v)).groupby(np.asarray(c))]
    o=[np.median(np.concatenate([gs[i] for i in rng.integers(0,len(gs),len(gs))])) for _ in range(B)]
    return np.percentile(o,[2.5,97.5])
print('clusters (table)',cl.nunique())
tab=['OZ-MSA','OZ-HNCB','FMSA','MSA-exact']
for col,nm in [('e_contact','contact g(sigma), mean of 3 pairs per state'),('e_g11','g11'),('e_g12','g12'),('e_g22','g22'),('e_U','U/NkT'),('e_Z','Z')]:
    W=wide(col); print('\n==',nm,'n',len(W))
    for m in tab:
        v=W[m].dropna()
        if len(v)==0: print('  ',m,'n/a'); continue
        print(f'  {m:10s} median {100*v.median():6.2f}%  CI {np.round(100*bci(v.values,cl.loc[v.index].values),2)} IQR {np.round(100*np.percentile(v,[25,75]),2)} mean {100*v.mean():6.2f}%')
    for a,b in [('FMSA','OZ-MSA'),('FMSA','OZ-HNCB'),('OZ-MSA','OZ-HNCB'),('FMSA','MSA-exact'),('OZ-MSA','MSA-exact')]:
        dd=(W[a]-W[b]).dropna()
        if len(dd)==0: continue
        pos=(dd>0).sum()
        print(f'    {a}-{b}: median diff {100*dd.median():+.2f} pp CI {np.round(100*bci(dd.values,cl.loc[dd.index].values),2)} Wilcoxon p={stats.wilcoxon(dd.values).pvalue:.2g}; {a} worse in {pos}/{len(dd)}')
print('\n== by case (contact mean of 3 pairs) ==')
W=wide('e_contact'); c=d.drop_duplicates('row').set_index('row').case
for k in sorted(c.unique()):
    idx=c.index[c==k]; print(k,len(idx),{m:round(100*W.loc[idx,m].median(),2) for m in tab})
print('\n== by case signed g11 ==')
S=wide('s_g11')
for k in sorted(c.unique()):
    idx=c.index[c==k]; print(k,{m:round(100*S.loc[idx,m].median(),2) for m in tab})
