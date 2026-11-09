import numpy as np, pandas as pd, os, json
from ladder_gradient import make_program, retention
H=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(H,'..','data','RepoRT','processed_data')
f=np.load(os.path.join(H,'v1_fits.npz')); X=f['X']; Fs=list(f['Fs'])
g=pd.read_csv(f'{P}/0187/0187_gradient.tsv',sep='\t'); prog=make_program(g['t [min]'].values,g['B [%]'].values/100)
V0=0.735*0.3; R=json.load(open(os.path.join(H,'v1_results.json')))
designs={'interpolation':([0.2,0.3,0.4],[0.25,0.35]),'extrapolation':([0.2,0.25,0.3],[0.35,0.4]),'far_extrapolation':([0.2,0.25],[0.3,0.35,0.4])}
out={}
for name,(tr,te) in designs.items():
    a,S=f[name+'_a'],f[name+'_S']; VD=R[name]['VD_mL']
    lnk0=a-S*(0.05-0.5); retained=lnk0>np.log(1.0)          # k > 1 at the starting composition
    Ftr=np.array(tr); o={'retained':int(retained.sum()),'unretained':int((~retained).sum())}
    for F in te:
        y=X[:,Fs.index(F)]*60; lad=retention(prog,V0/F,VD/F,a,S)*60
        Q=np.vstack([np.ones_like(Ftr),1/Ftr]).T; c4=np.linalg.lstsq(Q,X[:,[Fs.index(x) for x in tr]].T,rcond=None)[0]; b4=(c4[0]+c4[1]/F)*60
        o[str(F)]={grp:dict(ladder=float(np.mean(np.abs(lad-y)[m])),linear_invF=float(np.mean(np.abs(b4-y)[m]))) for grp,m in [('retained',retained),('unretained',~retained)]}
    out[name]=o
print(json.dumps(out,indent=1)); json.dump(out,open(os.path.join(H,'v1_split.json'),'w'),indent=1)
