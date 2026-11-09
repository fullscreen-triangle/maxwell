import numpy as np, pandas as pd, os, pickle, itertools, json
from scipy.stats import spearmanr
H=os.path.dirname(os.path.abspath(__file__))
M,phi=pickle.load(open(os.path.join(H,'v2_phi.pkl'),'rb'))
import pickle as pk
_,rt=pk.load(open(os.path.join(H,'repo.pkl'),'rb'))
rows=[]
ids=list(M.index)
for a,b in itertools.combinations(ids,2):
    c=phi[a].index.intersection(phi[b].index)
    if len(c)<20: continue
    pa,pb=phi[a].loc[c].values,phi[b].loc[c].values
    A,B=M.loc[a],M.loc[b]
    ra,rb=rt[a].loc[c].values,rt[b].loc[c].values
    same_col=A.col==B.col; same_mod=A.modifier==B.modifier; same_add=A.additive==B.additive
    dT=abs(A['T']-B['T']) if np.isfinite(A['T']) and np.isfinite(B['T']) else np.nan
    carrier_same=all(np.isclose(A[k],B[k],equal_nan=True) for k in ['L','ID','dp','F']) and np.isclose(A.steep,B.steep,rtol=0.02)
    rows.append(dict(a=a,b=b,n=len(c),mae_phi=float(np.mean(np.abs(pa-pb))),rho_phi=spearmanr(pa,pb)[0],
        mae_rt=float(np.mean(np.abs(ra-rb))),rho_rt=spearmanr(ra,rb)[0],
        same_col=same_col,same_mod=same_mod,same_add=same_add,dT=dT,carrier_same=carrier_same,
        dlogL=abs(np.log(A.L/B.L)),dlogID=abs(np.log(A.ID/B.ID)),dlogdp=abs(np.log(A.dp/B.dp)),dlogF=abs(np.log(A.F/B.F)),
        dlogsteep=abs(np.log(A.steep/B.steep)),signed_dlogsteep=np.log(A.steep/B.steep),mean_dphi=float(np.mean(pa-pb)),
        same_src=(str(A.src)==str(B.src)) and str(A.src)!='nan'))
D=pd.DataFrame(rows)
D.to_pickle(os.path.join(H,'v2_pairs.pkl'))
print(len(D),'pairs with >=20 shared compounds eluting on a ramp')
same_contact=D.same_col&D.same_mod&D.same_add&((D.dT<=5)|D.dT.isna())
groups={
 'replicate (same contacts, same carrier)':same_contact&D.carrier_same,
 'same contacts, different carrier':same_contact&~D.carrier_same,
 'different column, same mobile phase':(~D.same_col)&D.same_mod&D.same_add&((D.dT<=5)|D.dT.isna()),
 'different organic modifier':~D.same_mod,
}
out={}
for k,m in groups.items():
    s=D[m]
    out[k]=dict(pairs=int(m.sum()),median_mae_phi=float(s.mae_phi.median()),median_rho_phi=float(s.rho_phi.median()),
               median_mae_rt=float(s.mae_rt.median()),median_rho_rt=float(s.rho_rt.median()))
print(json.dumps(out,indent=1))
# within same-contact pairs: does carrier geometry predict disagreement once steepness accounted?
s=D[same_contact&~D.carrier_same].copy()
X=np.column_stack([np.ones(len(s)),s.dlogsteep,s.dlogL,s.dlogID,s.dlogdp,s.dlogF])
beta,*_=np.linalg.lstsq(X,s.mae_phi.values,rcond=None)
print('same-contact OLS mae_phi ~ 1 + |dlog steep| + |dlog L| + |dlog ID| + |dlog dp| + |dlog F|:',np.round(beta,4))
# ladder sign prediction: steeper gradient per column volume -> later (higher) elution composition
r=spearmanr(s.signed_dlogsteep,s.mean_dphi); print('sign test: corr(signed dlog steep, mean dphi) =',round(r[0],3),'p=',r[1])
json.dump(dict(groups=out,ols_beta=dict(zip(['const','steep','L','ID','dp','F'],map(float,beta))),sign_rho=float(r[0]),sign_p=float(r[1]),n_pairs=len(D)),open(os.path.join(H,'v2_results.json'),'w'),indent=1)
