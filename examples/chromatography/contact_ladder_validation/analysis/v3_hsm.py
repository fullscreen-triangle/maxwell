"""Contacts <-> retention across columns using HSM vectors (leave-one-column-out)."""
import numpy as np, pandas as pd, os, pickle, json
from scipy.stats import spearmanr
H=os.path.dirname(os.path.abspath(__file__))
M,phi=pickle.load(open(os.path.join(H,'v2_phi.pkl'),'rb'))
hsm=pd.read_csv(os.path.join(H,'..','data','RepoRT','resources','hsm_database','hsm_database.tsv'),sep='\t')
hsm=hsm[hsm.name_normalized.apply(lambda x:isinstance(x,str))].drop_duplicates('name_normalized').set_index('name_normalized')
PAR=['H','S*','A','B','C (pH 2.8)']
rng=np.random.default_rng(1)
def als(Yi,Yd,Yv,nI,nD,r,lam=0.05,it=60,fixU=None):
    U=rng.normal(0,0.1,(nI,r)) if fixU is None else fixU.copy(); V=rng.normal(0,0.1,(nD,r)); mu=np.zeros(nD)
    for _ in range(it):
        for d in range(nD):                                   # dataset step: mu_d, v_d
            m=Yd==d
            if m.sum()<r+2: continue
            A=np.column_stack([np.ones(m.sum()),U[Yi[m]]]); R=np.eye(r+1)*lam; R[0,0]=0
            s=np.linalg.solve(A.T@A+R,A.T@Yv[m]); mu[d]=s[0]; V[d]=s[1:]
        if fixU is None:
            for i in range(nI):                               # compound step: u_i
                m=Yi==i
                if not m.any(): continue
                A=V[Yd[m]]; U[i]=np.linalg.solve(A.T@A+lam*np.eye(r),A.T@(Yv[m]-mu[Yd[m]]))
    return U,V,mu
out={}
for mod in ['acn','meoh']:
    ds=[d for d in M.index if M.loc[d,'modifier']==mod and isinstance(M.loc[d,'col'],str) and M.loc[d,'col'] in hsm.index]
    cnt=pd.Series([k for d in ds for k in phi[d].index]).value_counts(); keepC=set(cnt[cnt>=5].index)
    ds=[d for d in ds if len(set(phi[d].index)&keepC)>=20]
    cols=sorted(set(M.loc[ds,'col']))
    comp=sorted(set().union(*[set(phi[d].index)&keepC for d in ds])); ci={c:i for i,c in enumerate(comp)}; di={d:j for j,d in enumerate(ds)}
    Yi,Yd,Yv=[],[],[]
    for d in ds:
        s=phi[d][phi[d].index.isin(keepC)]; Yi+= [ci[c] for c in s.index]; Yd+=[di[d]]*len(s); Yv+=list(s.values)
    Yi,Yd,Yv=map(np.array,(Yi,Yd,Yv))
    o=dict(datasets=len(ds),columns=len(cols),compounds=len(comp),entries=len(Yv))
    # rank selection by held-out entries
    test=rng.random(len(Yv))<0.1; cv={}
    for r in [1,2,3,4]:
        U,V,mu=als(Yi[~test],Yd[~test],Yv[~test],len(comp),len(ds),r)
        pred=mu[Yd[test]]+np.sum(U[Yi[test]]*V[Yd[test]],1); cv[r]=float(np.sqrt(np.mean((pred-Yv[test])**2)))
    r=min(cv,key=cv.get); o['cv_rmse_by_rank']=cv; o['rank']=r
    Z=hsm.loc[cols,PAR].astype(float)
    # leave-one-column-out
    fw=[];inv_pred={};
    for c in cols:
        tr=np.array([M.loc[ds[j],'col']!=c for j in Yd]); held=[d for d in ds if M.loc[d,'col']==c]
        U,V,mu=als(Yi[tr],Yd[tr],Yv[tr],len(comp),len(ds),r)
        trd=[j for j,d in enumerate(ds) if M.loc[d,'col']!=c]
        # forward map HSM -> v (ridge)
        Xh=np.column_stack([np.ones(len(trd)),Z.loc[[M.loc[ds[j],'col'] for j in trd]].values]); lamb=np.eye(Xh.shape[1])*1.0; lamb[0,0]=0
        Wf=np.linalg.solve(Xh.T@Xh+lamb,Xh.T@V[trd])
        vbar=V[trd].mean(0)
        # oracle: v of held-out datasets with U fixed
        Uo,Vo,muo=als(Yi[~tr],Yd[~tr],Yv[~tr],len(comp),len(ds),r,fixU=U)
        for d in held:
            j=di[d]; m=(Yd==j)&np.isin(Yi,np.unique(Yi[tr]))
            if m.sum()<15: continue
            y=Yv[m]; Ui=U[Yi[m]]
            p_hsm=Ui@(np.r_[1,Z.loc[c].values]@Wf); p_bar=Ui@vbar; p_or=Ui@Vo[j]
            spec_obs=y-np.polyval(np.polyfit(p_bar,y,1),p_bar)          # column-specific part of the order
            fw.append(dict(col=c,ds=d,n=int(m.sum()),rho_hsm=spearmanr(p_hsm,y)[0],rho_bar=spearmanr(p_bar,y)[0],rho_oracle=spearmanr(p_or,y)[0],
                           rho_specific=spearmanr(p_hsm-p_bar,spec_obs)[0]))
        # inverse: v (column mean of its datasets, from oracle fit) -> HSM, map learned on training columns
        Vcol=pd.DataFrame(V[trd],index=[M.loc[ds[j],'col'] for j in trd]).groupby(level=0).mean()
        Xv=np.column_stack([np.ones(len(Vcol)),Vcol.values]); li=np.eye(Xv.shape[1])*0.1; li[0,0]=0
        Wi=np.linalg.solve(Xv.T@Xv+li,Xv.T@Z.loc[Vcol.index].values)
        vh=np.mean([Vo[di[d]] for d in held],0); inv_pred[c]=np.r_[1,vh]@Wi
    F=pd.DataFrame(fw)
    o['forward']=dict(n_datasets=len(F),mean_rho_hsm=float(F.rho_hsm.mean()),mean_rho_universal=float(F.rho_bar.mean()),mean_rho_oracle=float(F.rho_oracle.mean()),
        frac_hsm_beats_universal=float((F.rho_hsm>F.rho_bar).mean()),mean_rho_specific=float(F.rho_specific.mean()),
        frac_specific_positive=float((F.rho_specific>0).mean()))
    IP=pd.DataFrame(inv_pred,index=PAR).T
    o['inverse']={p:dict(spearman=float(spearmanr(IP[p],Z.loc[IP.index,p])[0]),r2=float(1-np.sum((IP[p]-Z.loc[IP.index,p])**2)/np.sum((Z.loc[IP.index,p]-Z.loc[IP.index,p].mean())**2))) for p in PAR}
    # permutation null for the inverse: shuffle HSM rows among columns, same pipeline is expensive -> null on final correlations
    null={p:[] for p in PAR}
    for rep in range(2000):
        perm=rng.permutation(IP.index)
        for p in PAR: null[p].append(spearmanr(IP[p].values,Z.loc[perm,p].values)[0])
    for p in PAR: o['inverse'][p]['perm_p']=float((np.sum(np.array(null[p])>=o['inverse'][p]['spearman'])+1)/2001)
    out[mod]=o; F.to_csv(os.path.join(H,f'v3_forward_{mod}.csv'),index=False); IP.to_csv(os.path.join(H,f'v3_inverse_{mod}.csv'))
    print(mod,json.dumps(o,indent=1),flush=True)
json.dump(out,open(os.path.join(H,'v3_results.json'),'w'),indent=1)
