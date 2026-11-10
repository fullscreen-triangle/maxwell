import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, '..', '..', '..', '..', 'docs', 'thermodynamics', 'chromatography-contact-ladder', 'figures')
if not os.path.isdir(OUT):
    OUT = r'C:\Users\kunda\Documents\physics\maxwell\docs\thermodynamics\chromatography-contact-ladder\figures'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'acn': '#7f7f7f', 'meoh': '#1f4e79'}; LAB = {'acn': 'acetonitrile', 'meoh': 'methanol'}
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[4, 5]))
sets = [('hsm5', 'HSM\n(5)'), ('column', 'column\n(14)'), ('eluent', 'eluent\n(9)'), ('full', 'all\n(23)')]
for i, m in enumerate(['acn', 'meoh']):
    F = pd.read_csv(os.path.join(H, f'v4_forward_{m}.csv'))
    for k, (s, _) in enumerate(sets):
        v = F.groupby('col')[f'B_{s}_spec'].mean().dropna().values; x = k + (i - 0.5) * 0.36
        ax[0].scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=9, color=C[m], alpha=0.7, label=LAB[m] if k == 0 else None)
        ax[0].plot([x - 0.14, x + 0.14], [v.mean()] * 2, color='k', lw=1.2)
ax[0].axhline(0, lw=0.6, color='k'); ax[0].set_xticks(range(4)); ax[0].set_xticklabels([l for _, l in sets], fontsize=7)
ax[0].set_ylabel('specific-order prediction\n(Spearman, mean per unit)'); ax[0].set_title('held-out column', fontsize=9)
ax[0].legend(fontsize=7, frameon=False, loc='lower left')
sets2 = [('hsm5', 'HSM\n(5)'), ('pH', 'pH\n(2)'), ('additive', 'additive\n(5)'), ('eluent', 'eluent\n(9)'), ('full', 'all\n(23)')]
for i, m in enumerate(['acn', 'meoh']):
    F = pd.read_csv(os.path.join(H, f'v4c_lab_{m}.csv'))
    for k, (s, _) in enumerate(sets2):
        v = F[F.set == s].groupby('src').spec.mean().dropna().values; x = k + (i - 0.5) * 0.36
        ax[1].scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=9, color=C[m], alpha=0.7)
        ax[1].plot([x - 0.14, x + 0.14], [v.mean()] * 2, color='k', lw=1.2)
ax[1].axhline(0, lw=0.6, color='k'); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([l for _, l in sets2], fontsize=7)
ax[1].set_title('held-out laboratory', fontsize=9)
for a in ax: a.set_ylim(-1.05, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_contacts.pdf')); fig.savefig(os.path.join(OUT, 'fig4_contacts.png'), dpi=200); print('ok', OUT)
