"""Per-dataset elution composition phi_e = organic fraction delivered by the pump at t_R - t0."""
import numpy as np, pandas as pd, os, pickle, warnings
warnings.filterwarnings('ignore')
H=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(H,'..','data','RepoRT','processed_data')
ORG=['acn','meoh','iproh','acetone','hex','chcl3','ch2cl2','hept']
num=lambda x: pd.to_numeric(x,errors='coerce')
recs=[]; phi={}
for d in sorted(os.listdir(P)):
    try:
        md=pd.read_csv(f'{P}/{d}/{d}_metadata.tsv',sep='\t',dtype=str).iloc[0]
        info=pd.read_csv(f'{P}/{d}/{d}_info.tsv',sep='\t',dtype=str).iloc[0]
        g=pd.read_csv(f'{P}/{d}/{d}_gradient.tsv',sep='\t')
        r=pd.read_csv(f'{P}/{d}/{d}_rtdata_canonical_success.tsv',sep='\t').dropna(subset=['rt','inchikey.std'])
    except Exception: continue
    if info.get('method.type')!='RP' or len(g)<2: continue
    org={E: sum(np.nan_to_num(num(md.get(f'eluent.{E}.{s}'))) for s in ORG) for E in 'ABCD'}
    cols={E:f'{E} [%]' for E in 'ABCD'}
    tt=g['t [min]'].values.astype(float)
    ph=sum((g[cols[E]].fillna(0).values if cols[E] in g else 0)/100.0*org[E]/100.0 for E in 'ABCD')
    t0=num(md.get('column.t0')); L=num(md.get('column.length')); ID=num(md.get('column.id')); F=num(md.get('column.flowrate'))
    t0_est=False
    if not (t0>0) and L>0 and ID>0 and F>0:
        t0=0.6*np.pi*(ID/2)**2*L/1000.0/F; t0_est=True          # mL / (mL/min), porosity 0.6 assumed
    if not (t0>0): continue
    rtv=r.groupby('inchikey.std').rt.median()
    tin=rtv.values-t0
    # keep compounds eluting while the program is ramping up (strictly increasing organic content)
    seg=np.searchsorted(tt,tin,side='right')-1
    ok=(seg>=0)&(seg<len(tt)-1)
    slope=np.full(len(tin),np.nan)
    slope[ok]=(ph[seg[ok]+1]-ph[seg[ok]])/(tt[seg[ok]+1]-tt[seg[ok]])
    keep=ok&(slope>1e-6)
    pe=np.interp(tin,tt,ph)
    s=pd.Series(pe[keep],index=rtv.index[keep])
    if len(s)<10: continue
    phi[d]=s
    bmeoh=num(md.get('eluent.B.meoh')) or 0; bacn=num(md.get('eluent.B.acn')) or 0
    modifier='acn' if bacn>0 and not bmeoh>0 else ('meoh' if bmeoh>0 and not bacn>0 else 'mixed')
    adds=sorted(c.split('.')[2] for c in md.index if c.startswith('eluent.A.') and c.split('.')[2] not in ORG+['h2o','pH'] and not c.endswith('unit') and num(md[c])>0)
    recs.append(dict(id=d,col=md.get('column.name'),modifier=modifier,additive='+'.join(adds) or 'none',pH=num(md.get('eluent.A.pH')),
        T=num(md.get('column.temperature')),L=L,ID=ID,dp=num(md.get('column.particle.size')),F=F,t0=t0,t0_est=t0_est,
        steep=float(np.median(slope[keep]))*t0,n=len(s),name=info.get('name'),src=info.get('url')))
M=pd.DataFrame(recs).set_index('id')
pickle.dump((M,phi),open(os.path.join(H,'v2_phi.pkl'),'wb'))
print(len(M),'RP datasets with >=10 compounds eluting on a ramp;',int(M.t0_est.sum()),'with estimated t0')
print(M.modifier.value_counts().to_dict())
