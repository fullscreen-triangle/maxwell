"""Sensitivity of the two-flow (far extrapolation) design to the dwell volume, which two flows cannot identify."""
import numpy as np, pandas as pd, os, json
from ladder_gradient import make_program, retention
H=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(H,'..','data','RepoRT','processed_data')
exec(open(os.path.join(H,'v1_flow.py')).read().split("designs=")[0].replace("print('coarse grid'","pass #"))
out={}
for iv,VD in enumerate(VDs[:5]):
    tr,te=[0.2,0.25],[0.3,0.35,0.4]
    k,_=coarse_fit(tr,iv); a,S=refine(tr,iv,k)
    out[str(VD)]={str(F):float(np.nanmean(np.abs(retention(prog,V0/F,VD/F,a,S)-X[F].values))*60) for F in te}
print(json.dumps(out,indent=1)); json.dump(out,open(os.path.join(H,'v1_vd.json'),'w'),indent=1)
