import numpy as np, pandas as pd, os, json, time
from ladder_gradient import make_program, retention
H=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(H,'..','data','RepoRT','processed_data')
pos={0.2:'0189',0.25:'0191',0.3:'0187',0.35:'0193',0.4:'0195'}
neg={0.2:'0190',0.25:'0192',0.3:'0188',0.35:'0194',0.4:'0196'}
def load(d):
    r=pd.read_csv(f'{P}/{d}/{d}_rtdata_canonical_success.tsv',sep='\t').dropna(subset=['rt','inchikey.std'])
    return r.groupby('inchikey.std').rt.median()
R={F:load(d) for F,d in pos.items()}; Rn={F:load(d) for F,d in neg.items()}
common=sorted(set.intersection(*[set(s.index) for s in R.values()]))
X=pd.DataFrame({F:R[F].loc[common] for F in R}); nC=len(common)
g=pd.read_csv(f'{P}/0187/0187_gradient.tsv',sep='\t')
prog=make_program(g['t [min]'].values, g['B [%]'].values/100.0)
V0=0.735*0.3
Fs=np.array(sorted(R)); VDs=np.round(np.arange(0,0.61,0.05),3)
a1=np.arange(-8,10.001,0.25); S1=np.arange(0,40.001,1.0)
T=time.time()
VDm,Fm,am,Sm=np.meshgrid(VDs,Fs,a1,S1,indexing='ij')
cf=os.path.join(H,'v1_coarse.npy')
if os.path.exists(cf): coarse=np.load(cf)
else:
    coarse=retention(prog,V0/Fm,VDm/Fm,am,Sm); np.save(cf,coarse)   # [VD,F,a,S]
print('coarse grid',round(time.time()-T),'s, compounds',nC,flush=True)
fi={F:i for i,F in enumerate(Fs)}
def coarse_fit(train,iv):
    sse=sum((np.nan_to_num(coarse[iv,fi[F]],nan=1e3)[None]-X[F].values[:,None,None])**2 for F in train)
    k=sse.reshape(nC,-1).argmin(1); return k, sse.reshape(nC,-1)[np.arange(nC),k]
def refine(train,iv,k):
    ia,iS=np.unravel_index(k,(a1.size,S1.size))
    da=np.linspace(-0.25,0.25,11); dS=np.linspace(-1,1,11)
    A=a1[ia][:,None,None]+da[None,:,None]; S=np.clip(S1[iS][:,None,None]+dS[None,None,:],0,None)
    A,S=[x.copy() for x in np.broadcast_arrays(A,S)]
    VD=VDs[iv]; sse=0
    for F in train: sse=sse+(np.nan_to_num(retention(prog,V0/F,VD/F,A,S),nan=1e3)-X[F].values[:,None,None])**2
    j=sse.reshape(nC,-1).argmin(1)
    return A.reshape(nC,-1)[np.arange(nC),j], S.reshape(nC,-1)[np.arange(nC),j]
designs={'interpolation':([0.2,0.3,0.4],[0.25,0.35]),'extrapolation':([0.2,0.25,0.3],[0.35,0.4]),
         'far_extrapolation':([0.2,0.25],[0.3,0.35,0.4])}
noise=[]
for F in R:
    c=R[F].index.intersection(Rn[F].index); noise+=list((R[F].loc[c]-Rn[F].loc[c]).abs().values)
noise=np.array(noise)*60
out={'n_compounds':nC,'replicate_pos_vs_neg_s':dict(n=len(noise),median=float(np.median(noise)),mean=float(noise.mean()))}
fits={}
for name,(tr,te) in designs.items():
    iv=int(np.argmin([coarse_fit(tr,iv)[1].sum() for iv in range(len(VDs))]))
    k,_=coarse_fit(tr,iv); a,S=refine(tr,iv,k); fits[name]=(a,S)
    res={'VD_mL':float(VDs[iv])}; Ftr=np.array(tr)
    for F in te:
        y=X[F].values*60
        lad=retention(prog,V0/F,VDs[iv]/F,a,S)*60
        b1=X[tr].mean(1).values*60
        b2=(X[tr]*Ftr).mean(1).values/F*60
        if len(tr)>=2:
            Q=np.vstack([np.ones_like(Ftr),Ftr]).T; c3=np.linalg.lstsq(Q,X[tr].values.T,rcond=None)[0]; b3=(c3[0]+c3[1]*F)*60
            Q=np.vstack([np.ones_like(Ftr),1/Ftr]).T; c4=np.linalg.lstsq(Q,X[tr].values.T,rcond=None)[0]; b4=(c4[0]+c4[1]/F)*60
        m=lambda p: dict(mae=float(np.nanmean(np.abs(p-y))), median=float(np.nanmedian(np.abs(p-y))), n=int(np.isfinite(p).sum()))
        res[str(F)]=dict(ladder=m(lad),const_time=m(b1),const_volume=m(b2),linear_F=m(b3),linear_invF=m(b4))
    out[name]=res
out['contact_numbers_interp']=dict(a_median=float(np.median(fits['interpolation'][0])),S_median=float(np.median(fits['interpolation'][1])))
print(json.dumps(out,indent=1))
json.dump(out,open(os.path.join(H,'v1_results.json'),'w'),indent=1)
np.savez(os.path.join(H,'v1_fits.npz'),common=np.array(common),**{f'{k}_a':v[0] for k,v in fits.items()},**{f'{k}_S':v[1] for k,v in fits.items()},X=X.values,Fs=Fs)
