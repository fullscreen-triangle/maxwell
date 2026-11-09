import pandas as pd, numpy as np, os, itertools, collections, pickle
B=os.path.join(os.path.dirname(__file__),'..','data','RepoRT'); P=os.path.join(B,'processed_data')
num=lambda x: pd.to_numeric(x,errors='coerce')
recs=[]; rt={}
for d in sorted(os.listdir(P)):
    f=lambda s: os.path.join(P,d,f'{d}_{s}')
    try:
        md=pd.read_csv(f('metadata.tsv'),sep='\t',dtype=str).iloc[0]; info=pd.read_csv(f('info.tsv'),sep='\t',dtype=str).iloc[0]
        r=pd.read_csv(f('rtdata_canonical_success.tsv'),sep='\t',dtype=str); gr=pd.read_csv(f('gradient.tsv'),sep='\t')
    except Exception: continue
    r['rt']=num(r.rt); r=r.dropna(subset=['rt','inchikey.std'])
    r=r.groupby('inchikey.std').rt.median()          # duplicates -> median
    rt[d]=r
    org='acn' if num(md.get('eluent.B.acn'))>0 else ('meoh' if num(md.get('eluent.B.meoh'))>0 else 'other')
    recs.append(dict(id=d,type=info.get('method.type'),col=md.get('column.name'),L=num(md.get('column.length')),ID=num(md.get('column.id')),
       dp=num(md.get('column.particle.size')),T=num(md.get('column.temperature')),F=num(md.get('column.flowrate')),t0=num(md.get('column.t0')),
       org=org,pHA=num(md.get('eluent.A.pH')),formic=num(md.get('eluent.A.formic')),nh4ac=num(md.get('eluent.A.nh4ac')),
       n=len(r),gr=gr.to_json()))
M=pd.DataFrame(recs).set_index('id')
pickle.dump((M,rt),open(os.path.join(os.path.dirname(__file__),'repo.pkl'),'wb'))
rp=M[M.type=='RP'].index
# overlap
ov=[]
for a,b in itertools.combinations(rp,2):
    k=len(rt[a].index.intersection(rt[b].index))
    if k>=10: ov.append((a,b,k))
print('RP datasets',len(rp),' pairs with >=10 shared compounds:',len(ov),' >=30:',sum(1 for x in ov if x[2]>=30))
