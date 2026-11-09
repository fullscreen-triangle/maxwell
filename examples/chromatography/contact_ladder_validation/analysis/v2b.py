import numpy as np, pandas as pd, os, pickle, json
from scipy.stats import spearmanr
H=os.path.dirname(os.path.abspath(__file__))
M,phi=pickle.load(open(os.path.join(H,'v2_phi.pkl'),'rb')); D=pd.read_pickle(os.path.join(H,'v2_pairs.pkl'))
rng=np.random.default_rng(0)
same_contact=D.same_col&D.same_mod&D.same_add&((D.dT<=5)|D.dT.isna())
S=D[same_contact&~D.carrier_same].copy()
# --- steepness correction, cross-fitted on random halves of the pairs (slope c ~ 1/S of a typical analyte)
res=[]
for rep in range(200):
    m=rng.random(len(S))<0.5
    c=np.polyfit(S.signed_dlogsteep[m],S.mean_dphi[m],1)
    corr=np.abs(S.mean_dphi[~m]-np.polyval(c,S.signed_dlogsteep[~m]))
    res.append((c[0],np.median(np.abs(S.mean_dphi[~m])),np.median(corr)))
res=np.array(res)
out={'steepness_slope_median':float(np.median(res[:,0])),'implied_S':float(1/np.median(res[:,0])),
     'median_abs_mean_shift_raw':float(np.median(res[:,1])),'median_abs_mean_shift_corrected':float(np.median(res[:,2]))}
# per-pair residual MAE after removing the mean shift (what remains is order / spacing disagreement)
S['mae_after_shift']=np.nan
for i,r in S.iterrows():
    c=phi[r.a].index.intersection(phi[r.b].index); d=phi[r.a].loc[c].values-phi[r.b].loc[c].values
    S.at[i,'mae_after_shift']=np.mean(np.abs(d-d.mean()))
R=D[same_contact&D.carrier_same]
Rm=[]
for i,r in R.iterrows():
    c=phi[r.a].index.intersection(phi[r.b].index); d=phi[r.a].loc[c].values-phi[r.b].loc[c].values; Rm.append(np.mean(np.abs(d-d.mean())))
out['replicate_mae_after_shift_median']=float(np.median(Rm)); out['samecontact_mae_after_shift_median']=float(S.mae_after_shift.median())
# geometry terms on the shift-removed disagreement, cluster bootstrap over datasets
cols=['dlogsteep','dlogL','dlogID','dlogdp','dlogF']
def ols(df): 
    X=np.column_stack([np.ones(len(df))]+[df[c].values for c in cols]); return np.linalg.lstsq(X,df.mae_after_shift.values,rcond=None)[0]
b0=ols(S); ds=np.array(sorted(set(S.a)|set(S.b))); boots=[]
for rep in range(500):
    pick=set(rng.choice(ds,len(ds),replace=True)); sub=S[S.a.isin(pick)&S.b.isin(pick)]
    if len(sub)>20: boots.append(ols(sub))
boots=np.array(boots)
out['geometry_ols']={k:dict(beta=float(b0[i]),ci95=[float(np.percentile(boots[:,i],2.5)),float(np.percentile(boots[:,i],97.5))]) for i,k in enumerate(['const']+cols)}
# --- contact distance: HSM distance between different columns vs disagreement (same modifier & additive)
hsm=pd.read_csv(os.path.join(H,'..','data','RepoRT','resources','hsm_database','hsm_database.tsv'),sep='\t').drop_duplicates('name_normalized').set_index('name_normalized')
P=['H','S*','A','B','C (pH 2.8)']
Dc=D[(~D.same_col)&D.same_mod&D.same_add&((D.dT<=5)|D.dT.isna())].copy()
ca=M.loc[Dc.a,'col'].values; cb=M.loc[Dc.b,'col'].values
ok=np.array([isinstance(x,str) and isinstance(y,str) and x in hsm.index and y in hsm.index for x,y in zip(ca,cb)])
Dc=Dc[ok]; ca,cb=ca[ok],cb[ok]
Z=hsm[P].astype(float); Z=(Z-Z.mean())/Z.std()
Dc['hsm_dist']=[np.linalg.norm(Z.loc[x].values-Z.loc[y].values) for x,y in zip(ca,cb)]
for metric in ['rho_phi']:
    r=spearmanr(Dc.hsm_dist,1-Dc[metric])
# permutation null: shuffle HSM vectors among columns
cols_all=sorted(set(ca)|set(cb)); obs=spearmanr(Dc.hsm_dist,1-Dc.rho_phi)[0]; null=[]
for rep in range(1000):
    perm=dict(zip(cols_all,rng.permutation(cols_all)))
    dd=[np.linalg.norm(Z.loc[perm[x]].values-Z.loc[perm[y]].values) for x,y in zip(ca,cb)]
    null.append(spearmanr(dd,1-Dc.rho_phi)[0])
null=np.array(null)
out['hsm_distance']=dict(pairs=int(len(Dc)),columns=len(cols_all),spearman=float(obs),perm_p=float((np.sum(null>=obs)+1)/(len(null)+1)),null_mean=float(null.mean()))
print(json.dumps(out,indent=1)); json.dump(out,open(os.path.join(H,'v2b_results.json'),'w'),indent=1)
