import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(H,'..','..','..','..','docs','thermodynamics','chromatography-contact-ladder','figures')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
lab={'acn':'acetonitrile','meoh':'methanol'}
for i,m in enumerate(['acn','meoh']):
    F=pd.read_csv(os.path.join(H,f'v3_forward_{m}.csv'))
    vals=[F.rho_bar.mean(),F.rho_hsm.mean(),F.rho_oracle.mean()]
    ax[0].bar(np.arange(3)+i*0.38,vals,0.36,label=lab[m],color=['#7f7f7f','#1f4e79'][i])
    C=F.groupby('col').rho_specific.mean()
    ax[1].scatter(np.full(len(C),i)+np.random.default_rng(0).uniform(-0.12,0.12,len(C)),C,s=12,color=['#7f7f7f','#1f4e79'][i])
ax[0].set_xticks(np.arange(3)+0.19); ax[0].set_xticklabels(['universal order','from HSM vector','column fitted\nfrom own data'],fontsize=7)
ax[0].set_ylim(0.7,1.0); ax[0].set_ylabel('mean Spearman correlation'); ax[0].legend(fontsize=7,frameon=False)
ax[1].axhline(0,lw=0.6,color='k'); ax[1].set_xticks([0,1]); ax[1].set_xticklabels(['acetonitrile','methanol'])
ax[1].set_ylabel('column-specific prediction\n(Spearman, per column)')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig3_hsm.pdf')); fig.savefig(os.path.join(OUT,'fig3_hsm.png'),dpi=200); print('ok')
