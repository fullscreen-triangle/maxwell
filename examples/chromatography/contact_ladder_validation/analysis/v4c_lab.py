"""Is the eluent signal a contact or a laboratory? Model B (direct contact response) scored
leave-one-source-out: all datasets of one source (publication or laboratory) are held out,
so eluent effects must be learned from other laboratories. Lambda by inner 3-fold over sources."""
import numpy as np, pandas as pd, os, json
from scipy.stats import spearmanr, wilcoxon
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, 'v4_contacts.py'), encoding='utf-8').read()
g = {'__file__': os.path.join(H, 'v4_contacts.py')}; exec(src[:src.index('out = {}')], g)
M, phi, hsm, tan, contacts, direct_fit, score, standardise = (g[k] for k in ['M', 'phi', 'hsm', 'tan', 'contacts', 'direct_fit', 'score', 'standardise'])
rng = np.random.default_rng(11)
LAMS = [1, 3, 10, 30, 100, 300]
out = {}
for mod in ['acn', 'meoh']:
    ds = [d for d in M.index if M.loc[d, 'modifier'] == mod and isinstance(M.loc[d, 'col'], str)
          and M.loc[d, 'col'] in hsm.index and M.loc[d, 'col'] in tan.index]
    cnt = pd.Series([k for d in ds for k in phi[d].index]).value_counts(); keepC = set(cnt[cnt >= 5].index)
    ds = [d for d in ds if len(set(phi[d].index) & keepC) >= 20]
    Z, SETS = contacts(ds)
    SETS = dict(hsm5=SETS['hsm5'], eluent=SETS['eluent'], pH=['pH', 'pH_na'],
                additive=['formic', 'tfa', 'acetic', 'ammonium', 'basic'], full=SETS['full'])
    srcof = np.array([str(M.loc[d, 'src']) if isinstance(M.loc[d, 'src'], str) else 'nosrc_' + d for d in ds])
    comp = sorted(set().union(*[set(phi[d].index) & keepC for d in ds])); ci = {c: i for i, c in enumerate(comp)}
    Yi, Yd, Yv = [], [], []
    for j, d in enumerate(ds):
        s = phi[d][phi[d].index.isin(keepC)]; Yi += [ci[c] for c in s.index]; Yd += [j] * len(s); Yv += list(s.values)
    Yi, Yd, Yv = map(np.array, (Yi, Yd, Yv))
    sources = sorted(set(srcof)); rows = []
    for sname in sources:
        ted = np.where(srcof == sname)[0]; trd = np.where(srcof != sname)[0]; tr = np.isin(Yd, trd)
        for name, feats in SETS.items():
            Ztr, Zte = standardise(Z.iloc[trd][feats].values, Z.iloc[ted][feats].values)
            Zall = np.zeros((len(ds), len(feats))); Zall[trd] = Ztr; Zall[ted] = Zte
            inner = sorted(set(srcof[trd])); folds = np.array_split(rng.permutation(inner), 3); err = {}
            for lam in LAMS:
                e = []
                for f in folds:
                    k = np.isin(srcof[Yd], f) & tr; kk = tr & ~k
                    B_, W_ = direct_fit(Yi[kk], Yd[kk], Yv[kk], Zall, len(comp), lam); seenk = np.unique(Yi[kk])
                    for jd in np.unique(Yd[k]):
                        m = (Yd == jd) & np.isin(Yi, seenk)
                        if m.sum() < 15: continue
                        e.append(score(B_[Yi[m]] + W_[Yi[m]] @ Zall[jd], B_[Yi[m]], Yv[m])[1])
                err[lam] = -np.nanmean(e) if e else 0
            lam = min(err, key=err.get); B_, W_ = direct_fit(Yi[tr], Yd[tr], Yv[tr], Zall, len(comp), lam); seen = np.unique(Yi[tr])
            for jd in ted:
                m = (Yd == jd) & np.isin(Yi, seen)
                if m.sum() < 15: continue
                pb = B_[Yi[m]]; rr, sp = score(pb + W_[Yi[m]] @ Zall[jd], pb, Yv[m])
                rows.append(dict(src=sname, ds=ds[jd], set=name, lam=lam, rho=rr, rho_universal=spearmanr(pb, Yv[m])[0], spec=sp))
        print(mod, len(rows), flush=True)
    F = pd.DataFrame(rows); o = dict(datasets=len(ds), sources=len(sources))
    for name in SETS:
        g_ = F[F.set == name]; bysrc = g_.groupby('src').spec.mean().dropna()
        o[name] = dict(n=len(g_), n_sources=len(bysrc), rho=float(g_.rho.mean()), rho_universal=float(g_.rho_universal.mean()),
                       spec_mean=float(g_.spec.mean()), spec_frac_pos=float((g_.spec > 0).mean()),
                       spec_by_source_mean=float(bysrc.mean()), spec_by_source_frac_pos=float((bysrc > 0).mean()),
                       spec_by_source_wilcoxon_p=float(wilcoxon(bysrc, alternative='greater').pvalue))
    out[mod] = o; F.to_csv(os.path.join(H, f'v4c_lab_{mod}.csv'), index=False); print(mod, json.dumps(o, indent=1), flush=True)
json.dump(out, open(os.path.join(H, 'v4c_results.json'), 'w'), indent=1)
