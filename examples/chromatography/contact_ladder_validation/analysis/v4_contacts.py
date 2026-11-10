"""P4 revisited: does a richer contact description carry retention to an unseen column?

Contact descriptions of a dataset (one column + one eluent), nested:
  hsm5     H, S*, A, B, C(2.8)                                   (the original P4 test)
  column   hsm5 + C(7.0), EB retention, Tanaka kPB, aCH2, aT/O, aC/P,
           aB/P(7.6), aB/P(2.7), pore size                         (14 numbers)
  eluent   pH, temperature, additive classes (formic, TFA, acetic,
           ammonium, basic), missing-value flags                   (eluent only, 9)
  full     column + eluent                                         (23)
Two models, both scored leave-one-column-out on the datasets whose column has
both HSM and Tanaka parameters:
  A  latent: rank-r ALS on training datasets, ridge map contacts -> dataset vector
  B  direct: every compound responds linearly to the standardised contact numbers,
     phi_{i,d} = mu_d + b_i + w_i . z_d  (ridge per compound)
Ridge strength is chosen by an inner leave-one-column-out on the training columns.
Third test: is the training dataset nearest in contact space a better predictor
than an average training dataset of another column?
"""
import numpy as np, pandas as pd, os, pickle, json
from scipy.stats import spearmanr, wilcoxon
H = os.path.dirname(os.path.abspath(__file__))
M, phi = pickle.load(open(os.path.join(H, 'v2_phi.pkl'), 'rb'))
R = os.path.join(H, '..', 'data', 'RepoRT', 'resources')
hsm = pd.read_csv(os.path.join(R, 'hsm_database', 'hsm_database.tsv'), sep='\t')
hsm = hsm[hsm.name_normalized.apply(lambda x: isinstance(x, str))].drop_duplicates('name_normalized').set_index('name_normalized')
tan = pd.read_csv(os.path.join(R, 'tanaka_database', 'tanaka_database.tsv'), sep='\t', encoding='utf-8')
tan = tan[tan.name_normalized.apply(lambda x: isinstance(x, str))].drop_duplicates('name_normalized').set_index('name_normalized')
tan.columns = ['usp', 'phase', 'kPB', 'aCH2', 'aTO', 'aCP', 'aBP76', 'aBP27', 'dp_t', 'pore', 'n']
HSM5 = ['H', 'S*', 'A', 'B', 'C (pH 2.8)']
HSMX = HSM5 + ['C (pH 7.0)', 'EB retention factor']
TAN = ['kPB', 'aCH2', 'aTO', 'aCP', 'aBP76', 'aBP27', 'pore']
rng = np.random.default_rng(4)


def eluent(d):
    a = M.loc[d, 'additive'] if isinstance(M.loc[d, 'additive'], str) else ''
    pH, T = M.loc[d, 'pH'], M.loc[d, 'T']
    return dict(pH=pH, pH_na=float(pd.isna(pH)), T=T, T_na=float(pd.isna(T)),
                formic=float('formic' in a), tfa=float('trifluoro' in a), acetic=float('acetic' in a),
                ammonium=float('nh4' in a), basic=float(('nh4oh' in a) or ('bicarb' in a)))


def contacts(ds):
    Z = pd.DataFrame([dict(**hsm.loc[M.loc[d, 'col'], HSMX].astype(float).to_dict(),
                           **tan.loc[M.loc[d, 'col'], TAN].astype(float).to_dict(), **eluent(d)) for d in ds], index=ds)
    for c in ['pH', 'T']:
        Z[c] = Z[c].fillna(Z[c].median())
    ELU = ['pH', 'pH_na', 'T', 'T_na', 'formic', 'tfa', 'acetic', 'ammonium', 'basic']
    return Z, dict(hsm5=HSM5, column=HSMX + TAN, eluent=ELU, full=HSMX + TAN + ELU)


def als(Yi, Yd, Yv, nI, nD, r, lam=0.05, it=60, fixU=None):
    U = rng.normal(0, 0.1, (nI, r)) if fixU is None else fixU.copy(); V = rng.normal(0, 0.1, (nD, r)); mu = np.zeros(nD)
    for _ in range(it):
        for d in range(nD):
            m = Yd == d
            if m.sum() < r + 2: continue
            A = np.column_stack([np.ones(m.sum()), U[Yi[m]]]); Rg = np.eye(r + 1) * lam; Rg[0, 0] = 0
            s = np.linalg.solve(A.T @ A + Rg, A.T @ Yv[m]); mu[d] = s[0]; V[d] = s[1:]
        if fixU is None:
            for i in range(nI):
                m = Yi == i
                if not m.any(): continue
                A = V[Yd[m]]; U[i] = np.linalg.solve(A.T @ A + lam * np.eye(r), A.T @ (Yv[m] - mu[Yd[m]]))
    return U, V, mu


def ridge(X, Y, lam):
    X1 = np.column_stack([np.ones(len(X)), X]); L = np.eye(X1.shape[1]) * lam; L[0, 0] = 0
    return np.linalg.solve(X1.T @ X1 + L, X1.T @ Y)


def standardise(Ztr, Zte):
    m, s = Ztr.mean(0), Ztr.std(0); s[s < 1e-9] = 1
    return (Ztr - m) / s, (Zte - m) / s


LAMS_A = [0.3, 1, 3, 10, 30, 100]
LAMS_B = [1, 3, 10, 30, 100, 300]


def direct_fit(Yi, Yd, Yv, Zs, nI, lam):
    """phi = mu_d + b_i + w_i.z_d ; returns (b, W). Batched per-compound ridge, mu_d by alternation."""
    p = Zs.shape[1]; mu = np.zeros(Zs.shape[0])
    X = np.column_stack([np.ones(len(Yi)), Zs[Yd]])
    XtX = np.zeros((nI, p + 1, p + 1)); np.add.at(XtX, Yi, X[:, :, None] * X[:, None, :])
    L = np.eye(p + 1) * lam; L[0, 0] = 1e-6; XtX += L
    for _ in range(4):
        r = Yv - mu[Yd]
        Xty = np.zeros((nI, p + 1)); np.add.at(Xty, Yi, X * r[:, None])
        S_ = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]; B = S_[:, 0]; W = S_[:, 1:]
        res = Yv - B[Yi] - np.sum(W[Yi] * Zs[Yd], 1)
        mu = np.bincount(Yd, res, minlength=len(mu)) / np.maximum(np.bincount(Yd, minlength=len(mu)), 1)
    return B, W


def score(p, pbar, y):
    spec = y - np.polyval(np.polyfit(pbar, y, 1), pbar)
    return spearmanr(p, y)[0], spearmanr(p - pbar, spec)[0]


out = {}
for mod in ['acn', 'meoh']:
    ds = [d for d in M.index if M.loc[d, 'modifier'] == mod and isinstance(M.loc[d, 'col'], str)
          and M.loc[d, 'col'] in hsm.index and M.loc[d, 'col'] in tan.index]
    cnt = pd.Series([k for d in ds for k in phi[d].index]).value_counts(); keepC = set(cnt[cnt >= 5].index)
    ds = [d for d in ds if len(set(phi[d].index) & keepC) >= 20]
    Z, SETS = contacts(ds)
    colof = np.array([M.loc[d, 'col'] for d in ds]); cols = sorted(set(colof))
    comp = sorted(set().union(*[set(phi[d].index) & keepC for d in ds])); ci = {c: i for i, c in enumerate(comp)}
    Yi, Yd, Yv = [], [], []
    for j, d in enumerate(ds):
        s = phi[d][phi[d].index.isin(keepC)]; Yi += [ci[c] for c in s.index]; Yd += [j] * len(s); Yv += list(s.values)
    Yi, Yd, Yv = map(np.array, (Yi, Yd, Yv))
    o = dict(datasets=len(ds), columns=len(cols), compounds=len(comp), entries=int(len(Yv)),
             n_contacts={k: len(v) for k, v in SETS.items()})
    test = rng.random(len(Yv)) < 0.1; cv = {}
    for r in [1, 2, 3, 4, 6]:
        U, V, mu = als(Yi[~test], Yd[~test], Yv[~test], len(comp), len(ds), r)
        pred = mu[Yd[test]] + np.sum(U[Yi[test]] * V[Yd[test]], 1); cv[r] = float(np.sqrt(np.mean((pred - Yv[test]) ** 2)))
    r = min(cv, key=cv.get); o['cv_rmse_by_rank'] = cv; o['rank'] = r
    rows = []; nnrows = []
    for c in cols:
        trd = np.where(colof != c)[0]; ted = np.where(colof == c)[0]
        tr = np.isin(Yd, trd); seen = np.unique(Yi[tr])
        U, V, mu = als(Yi[tr], Yd[tr], Yv[tr], len(comp), len(ds), r)
        Uo, Vo, _ = als(Yi[~tr], Yd[~tr], Yv[~tr], len(comp), len(ds), r, fixU=U)
        vbar = V[trd].mean(0)
        # per-compound mean over training (universal order for model B)
        bsum = np.bincount(Yi[tr], Yv[tr] - mu[Yd[tr]], minlength=len(comp)); bn = np.bincount(Yi[tr], minlength=len(comp))
        bbar = bsum / np.maximum(bn, 1)
        predA, predB = {}, {}
        for name, feats in SETS.items():
            Ztr, Zte = standardise(Z.iloc[trd][feats].values, Z.iloc[ted][feats].values)
            gtr = colof[trd]; inner = sorted(set(gtr))
            # --- model A: choose lambda by inner LOCO on training columns (dataset vectors V)
            errA = {}
            for lam in LAMS_A:
                e = 0
                for cc in inner:
                    k = gtr == cc
                    Wf = ridge(Ztr[~k], V[trd][~k], lam); e += np.sum((np.column_stack([np.ones(k.sum()), Ztr[k]]) @ Wf - V[trd][k]) ** 2)
                errA[lam] = e
            lamA = min(errA, key=errA.get); Wf = ridge(Ztr, V[trd], lamA)
            predA[name] = (np.column_stack([np.ones(len(ted)), Zte]) @ Wf, lamA)
            # --- model B: direct per-compound contact response; lambda by inner LOCO (3 inner folds of columns for cost)
            Zall = np.zeros((len(ds), len(feats))); Zall[trd] = Ztr; Zall[ted] = Zte
            folds = np.array_split(rng.permutation(inner), 3); errB = {}
            for lam in LAMS_B:
                e = []
                for f in folds:
                    k = np.isin(colof[Yd], f) & tr; kk = tr & ~k
                    B_, W_ = direct_fit(Yi[kk], Yd[kk], Yv[kk], Zall, len(comp), lam)
                    for jd in np.unique(Yd[k]):
                        m = (Yd == jd) & np.isin(Yi, np.unique(Yi[kk]))
                        if m.sum() < 15: continue
                        p = B_[Yi[m]] + W_[Yi[m]] @ Zall[jd]; e.append(score(p, B_[Yi[m]], Yv[m])[1])
                errB[lam] = -np.nanmean(e)
            lamB = min(errB, key=errB.get); B_, W_ = direct_fit(Yi[tr], Yd[tr], Yv[tr], Zall, len(comp), lamB)
            predB[name] = (B_, W_, Zall, lamB)
        for jd in ted:
            m = (Yd == jd) & np.isin(Yi, seen)
            if m.sum() < 15: continue
            y = Yv[m]; Ui = U[Yi[m]]; pbar = Ui @ vbar
            row = dict(col=c, ds=ds[jd], n=int(m.sum()), rho_universal=spearmanr(pbar, y)[0], rho_oracle=spearmanr(Ui @ Vo[jd], y)[0])
            k = list(ted).index(jd)
            for name in SETS:
                vh, lamA = predA[name]; rr, sp = score(Ui @ vh[k], pbar, y)
                row[f'A_{name}_rho'] = rr; row[f'A_{name}_spec'] = sp; row[f'A_{name}_lam'] = lamA
                B_, W_, Zall, lamB = predB[name]; pb = B_[Yi[m]]; rr, sp = score(pb + W_[Yi[m]] @ Zall[jd], pb, y)
                row[f'B_{name}_rho'] = rr; row[f'B_{name}_spec'] = sp; row[f'B_{name}_lam'] = lamB
                row[f'B_universal_rho'] = spearmanr(pb, y)[0]
            rows.append(row)
            # nearest training dataset in contact space vs average training dataset of another column
            for name, feats in SETS.items():
                Ztr, Zte = standardise(Z.iloc[trd][feats].values, Z.iloc[[jd]][feats].values)
                dist = np.sqrt(((Ztr - Zte) ** 2).sum(1)); rh = []
                for q, jt in enumerate(trd):
                    a = phi[ds[jd]]; b = phi[ds[jt]]; sh = a.index.intersection(b.index)
                    rh.append((dist[q], spearmanr(a[sh], b[sh])[0] if len(sh) >= 15 else np.nan))
                rh = np.array(rh); ok = ~np.isnan(rh[:, 1])
                if ok.sum() < 3: continue
                rh = rh[ok]; nn = rh[np.argmin(rh[:, 0]), 1]
                nnrows.append(dict(ds=ds[jd], col=c, set=name, rho_nn=nn, rho_mean=rh[:, 1].mean(),
                                   dist_rho=spearmanr(rh[:, 0], rh[:, 1])[0]))
        print(mod, c, len(rows), flush=True)
    F = pd.DataFrame(rows); N = pd.DataFrame(nnrows)
    summ = dict(n_datasets=len(F), rho_universal=float(F.rho_universal.mean()), rho_oracle=float(F.rho_oracle.mean()))
    for mdl in 'AB':
        for name in SETS:
            sp = F[f'{mdl}_{name}_spec'].dropna(); colmean = F.groupby('col')[f'{mdl}_{name}_spec'].mean().dropna()
            summ[f'{mdl}_{name}'] = dict(rho=float(F[f'{mdl}_{name}_rho'].mean()), spec_mean=float(sp.mean()),
                                         spec_frac_pos=float((sp > 0).mean()),
                                         spec_by_column_mean=float(colmean.mean()),
                                         spec_by_column_wilcoxon_p=float(wilcoxon(colmean, alternative='greater').pvalue) if len(colmean) > 5 else None,
                                         lam_median=float(F[f'{mdl}_{name}_lam'].median()))
    summ['B_universal_rho'] = float(F.B_universal_rho.mean())
    nn = {}
    for name in SETS:
        g = N[N.set == name]
        nn[name] = dict(n=len(g), rho_nn=float(g.rho_nn.mean()), rho_mean=float(g.rho_mean.mean()),
                        frac_nn_better=float((g.rho_nn > g.rho_mean).mean()),
                        wilcoxon_p=float(wilcoxon(g.rho_nn - g.rho_mean, alternative='greater').pvalue),
                        mean_dist_rho=float(g.dist_rho.mean()))
    o['forward'] = summ; o['nearest_contact'] = nn; out[mod] = o
    F.to_csv(os.path.join(H, f'v4_forward_{mod}.csv'), index=False); N.to_csv(os.path.join(H, f'v4_nearest_{mod}.csv'), index=False)
    print(mod, json.dumps(o, indent=1), flush=True)
json.dump(out, open(os.path.join(H, 'v4_results.json'), 'w'), indent=1)
