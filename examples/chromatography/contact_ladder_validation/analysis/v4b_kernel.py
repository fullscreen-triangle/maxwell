"""P4 as similarity, not coordinates.

1. Nearest-contact test with a column-permutation null: shuffle which contact vector
   belongs to which column, recompute the advantage of the contact-nearest training
   dataset over an average training dataset of another column.
2. Kernel transfer: predict a held-out column's elution order as a contact-weighted
   average of the (rank-normalised) orders measured on training columns,
   weights exp(-d^2 / 2h^2); h chosen by inner leave-one-column-out; h = inf is the
   universal (column-agnostic) order. Scored on the column-specific part as in v3/v4.
"""
import numpy as np, pandas as pd, os, pickle, json
from scipy.stats import spearmanr, wilcoxon, rankdata

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, 'v4_contacts.py'), encoding='utf-8').read()
head = src[:src.index('out = {}')]                       # reuse loaders and contact builder only
g = {'__file__': os.path.join(H, 'v4_contacts.py')}; exec(head, g)
M, phi, hsm, tan, contacts = g['M'], g['phi'], g['hsm'], g['tan'], g['contacts']
rng = np.random.default_rng(7)
HS = [0.5, 1, 2, 4, 8, np.inf]
out = {}
for mod in ['acn', 'meoh']:
    ds = [d for d in M.index if M.loc[d, 'modifier'] == mod and isinstance(M.loc[d, 'col'], str)
          and M.loc[d, 'col'] in hsm.index and M.loc[d, 'col'] in tan.index]
    cnt = pd.Series([k for d in ds for k in phi[d].index]).value_counts(); keepC = set(cnt[cnt >= 5].index)
    ds = [d for d in ds if len(set(phi[d].index) & keepC) >= 20]
    Z, SETS = contacts(ds); colof = np.array([M.loc[d, 'col'] for d in ds]); cols = sorted(set(colof)); n = len(ds)
    # rank-normalised orders
    CI = pd.Index(sorted(set().union(*[set(phi[d].index) for d in ds])))
    rk = {d: pd.Series((rankdata(phi[d].values) - 0.5) / len(phi[d]), index=phi[d].index) for d in ds}
    P0 = np.zeros((n, len(CI))); MK = np.zeros((n, len(CI)))
    for q, d in enumerate(ds):
        ix = CI.get_indexer(rk[d].index); P0[q, ix] = rk[d].values; MK[q, ix] = 1
    RHO = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            if colof[a] == colof[b]: continue
            sh = phi[ds[a]].index.intersection(phi[ds[b]].index)
            if len(sh) >= 15: RHO[a, b] = RHO[b, a] = spearmanr(phi[ds[a]][sh], phi[ds[b]][sh])[0]
    o = {}
    for name in ['hsm5', 'column', 'full']:
        feats = SETS[name]
        Cv = pd.DataFrame(Z[feats].values, index=colof).groupby(level=0).first()   # one contact vector per column (column sets)
        Ccol = pd.DataFrame(Z[SETS['column']].values, index=colof).groupby(level=0).first()
        def advantage(vecs):
            """mean over datasets of rho(nearest other-column dataset) - mean rho(other-column datasets)."""
            X = np.vstack([vecs[c] for c in colof]) if name != 'full' else vecs
            X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1)
            D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); adv = []
            for a in range(n):
                ok = ~np.isnan(RHO[a])
                if ok.sum() < 3: continue
                idx = np.where(ok)[0]; adv.append(RHO[a, idx[np.argmin(D[a, idx])]] - RHO[a, idx].mean())
            return np.mean(adv), len(adv)
        if name == 'full':
            obs, nn_n = advantage(Z[feats].values)
            null = []
            for _ in range(1000):                     # permute column part across columns, keep eluent part
                perm = dict(zip(cols, rng.permutation(cols)))
                Xc = np.vstack([Ccol.loc[perm[c]].values for c in colof])
                null.append(advantage(np.column_stack([Xc, Z[SETS['eluent']].values]))[0])
        else:
            vec = {c: Cv.loc[c].values for c in cols}; obs, nn_n = advantage(vec); null = []
            for _ in range(1000):
                perm = dict(zip(cols, rng.permutation(cols))); null.append(advantage({c: vec[perm[c]] for c in cols})[0])
        null = np.array(null)
        o[name] = dict(n=nn_n, advantage=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                       perm_p=float((np.sum(null >= obs) + 1) / (len(null) + 1)))
    # kernel transfer
    rows = []
    feats = SETS['hsm5']
    for c in cols:
        trd = np.where(colof != c)[0]; ted = np.where(colof == c)[0]
        Xtr = Z.iloc[trd][feats].values; m_, s_ = Xtr.mean(0), Xtr.std(0); s_[s_ < 1e-9] = 1
        X = (Z[feats].values - m_) / s_
        def predict(j, pool, h):
            d2 = ((X[pool] - X[j]) ** 2).sum(1); w = np.ones(len(pool)) if np.isinf(h) else np.exp(-d2 / (2 * h * h))
            num = w @ P0[pool]; den = w @ MK[pool]; ok = den > 1e-12
            return pd.Series(num[ok] / den[ok], index=CI[ok])
        # inner choice of h: leave-one-training-column-out on the specific part
        inner = sorted(set(colof[trd])); sc = {h: [] for h in HS}
        for cc in inner:
            pool = np.array([q for q in trd if colof[q] != cc])
            for j in [q for q in trd if colof[q] == cc]:
                pu = predict(j, pool, np.inf); y = phi[ds[j]]
                for h in HS[:-1]:
                    p = predict(j, pool, h); sh = y.index.intersection(p.index).intersection(pu.index)
                    if len(sh) < 15: continue
                    spec_obs = y[sh] - np.polyval(np.polyfit(pu[sh], y[sh], 1), pu[sh])
                    sc[h].append(spearmanr(p[sh] - pu[sh], spec_obs)[0])
        hbest = max(HS[:-1], key=lambda h: np.nanmean(sc[h]) if sc[h] else -1)
        for j in ted:
            y = phi[ds[j]]; pu = predict(j, trd, np.inf); p = predict(j, trd, hbest)
            sh = y.index.intersection(p.index)
            if len(sh) < 15: continue
            spec_obs = y[sh] - np.polyval(np.polyfit(pu[sh], y[sh], 1), pu[sh])
            rows.append(dict(col=c, ds=ds[j], n=len(sh), h=hbest, rho_kernel=spearmanr(p[sh], y[sh])[0],
                             rho_universal=spearmanr(pu[sh], y[sh])[0], spec=spearmanr(p[sh] - pu[sh], spec_obs)[0]))
        print(mod, c, hbest, flush=True)
    K = pd.DataFrame(rows); cm = K.groupby('col').spec.mean().dropna()
    o['kernel_hsm5'] = dict(n_datasets=len(K), rho_kernel=float(K.rho_kernel.mean()), rho_universal=float(K.rho_universal.mean()),
                            frac_kernel_better=float((K.rho_kernel > K.rho_universal).mean()),
                            wilcoxon_rho_p=float(wilcoxon(K.rho_kernel - K.rho_universal, alternative='greater').pvalue),
                            spec_mean=float(K.spec.mean()), spec_by_column_mean=float(cm.mean()),
                            spec_by_column_wilcoxon_p=float(wilcoxon(cm, alternative='greater').pvalue),
                            h_counts=K.groupby('col').h.first().value_counts().to_dict())
    K.to_csv(os.path.join(H, f'v4b_kernel_{mod}.csv'), index=False)
    out[mod] = o; print(mod, json.dumps(o, indent=1, default=str), flush=True)
json.dump(out, open(os.path.join(H, 'v4b_results.json'), 'w'), indent=1, default=str)
