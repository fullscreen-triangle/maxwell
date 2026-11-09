import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(H,'..','..','..','..','docs','thermodynamics','chromatography-contact-ladder','figures')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
# ---- Figure 1: flow-rate transfer
R=json.load(open(os.path.join(H,'v1_results.json'))); Sp=json.load(open(os.path.join(H,'v1_split.json')))
meth=[('ladder','contact ladder','#1f4e79'),('linear_invF','$t_R$ affine in $1/F$','#c55a11'),('linear_F','$t_R$ linear in $F$','#7f7f7f'),('const_volume','constant volume','#a5a5a5'),('const_time','constant time','#d9d9d9')]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),sharey=False)
for k,(name,title) in enumerate([('interpolation','fit 0.2, 0.3, 0.4'),('extrapolation','fit 0.2, 0.25, 0.3'),('far_extrapolation','fit 0.2, 0.25')]):
    Fs=[f for f in R[name] if f!='VD_mL']; x=np.arange(len(Fs)); w=0.16
    for j,(m,lab,c) in enumerate(meth):
        ax[k].bar(x+(j-2)*w,[R[name][f][m]['mae'] for f in Fs],w,color=c,label=lab)
    ax[k].axhline(R['replicate_pos_vs_neg_s']['mean'],ls='--',lw=0.8,color='k')
    ax[k].set_xticks(x); ax[k].set_xticklabels([f'{f} mL/min' for f in Fs],fontsize=7); ax[k].set_title(title,fontsize=8); ax[k].set_yscale('log')
ax[0].set_ylabel('mean absolute error (s)'); ax[0].legend(fontsize=6,frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig1_flow.pdf')); fig.savefig(os.path.join(OUT,'fig1_flow.png'),dpi=200)
# ---- Figure 2: contacts versus carrier
D=pd.read_pickle(os.path.join(H,'v2_pairs.pkl'))
sc=D.same_col&D.same_mod&D.same_add&((D.dT<=5)|D.dT.isna())
groups=[('replicate',sc&D.carrier_same),('same contacts,\ndifferent carrier',sc&~D.carrier_same),('different column,\nsame mobile phase',(~D.same_col)&D.same_mod&D.same_add&((D.dT<=5)|D.dT.isna())),('different\norganic modifier',~D.same_mod)]
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
ax[0].boxplot([D.loc[m,'rho_phi'].dropna() for _,m in groups],showfliers=False,widths=0.6)
ax[0].set_xticks(range(1,5)); ax[0].set_xticklabels([g for g,_ in groups],fontsize=7); ax[0].set_ylabel('Spearman correlation of elution order'); ax[0].set_ylim(0.6,1.005)
s=D[sc&~D.carrier_same]
ax[1].scatter(s.signed_dlogsteep,s.mean_dphi,s=5,alpha=0.5,color='#1f4e79')
ax[1].axhline(0,lw=0.5,color='k'); ax[1].axvline(0,lw=0.5,color='k')
ax[1].set_xlabel(r'$\ln(G_a/G_b)$, gradient steepness per column volume'); ax[1].set_ylabel(r'mean $\varphi_e(a)-\varphi_e(b)$')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig2_contacts_carrier.pdf')); fig.savefig(os.path.join(OUT,'fig2_contacts_carrier.png'),dpi=200)
print('ok')
