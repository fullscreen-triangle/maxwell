#!/usr/bin/env python3
"""
Generate Panel L-5: Partition Lag Dynamics
Shows the undetermined residue and minimum partition time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gamma
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Partition Lag Time Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Gamma distribution for lag times (realistic)
    tau = np.linspace(0, 20, 500)
    shape = 3
    scale = 2
    dist = gamma.pdf(tau, shape, scale=scale)
    
    ax1.fill_between(tau, 0, dist, alpha=0.3, color='blue')
    ax1.plot(tau, dist, 'b-', linewidth=2, label='P(τ_lag)')
    
    # Minimum lag time
    tau_min = 0.5
    ax1.axvline(x=tau_min, color='red', linestyle='--', linewidth=2, label=r'τ_min = ℏ/ΔE')
    ax1.fill_between([0, tau_min], [0, 0], [max(dist)*1.1, max(dist)*1.1], 
                     color='red', alpha=0.2, label='Forbidden region')
    
    # Mean lag
    mean_tau = shape * scale
    ax1.axvline(x=mean_tau, color='green', linestyle=':', linewidth=2, label=r'⟨τ_lag⟩')
    
    ax1.set_xlabel('Partition lag τ_lag (arbitrary units)', fontsize=11)
    ax1.set_ylabel('Probability density', fontsize=11)
    ax1.set_title('(A) Partition Lag Distribution: Non-Zero Minimum', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 20)
    
    ax1.text(10, max(dist)*0.7, r'$\tau_{lag} \geq \frac{\hbar}{\Delta E}$' + '\n(Uncertainty principle)',
             fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel B: Undetermined Residue Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    
    t = np.linspace(0, 10, 100)
    
    # Multiple molecules with different lag times
    np.random.seed(42)
    for i in range(5):
        tau_i = np.random.uniform(0.5, 3)
        determination = 1 - np.exp(-t / tau_i)
        ax2.plot(t, determination, linewidth=1.5, alpha=0.6, label=f'Molecule {i+1}')
    
    # Average determination
    avg_tau = 1.5
    avg_determination = 1 - np.exp(-t / avg_tau)
    ax2.plot(t, avg_determination, 'k-', linewidth=3, label='Average')
    
    # Residue region
    ax2.fill_between(t, avg_determination, 1, alpha=0.2, color='red', label='Undetermined residue')
    
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.set_ylabel('Determination fraction', fontsize=11)
    ax2.set_title('(B) Undetermined Residue: Partition Takes Finite Time', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax2.text(5, 0.5, 'Residue = Information\nnot yet determined\n→ contributes to entropy',
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    # Panel C: Entropy Production Rate
    ax3 = fig.add_subplot(gs[1, 0])
    
    t = np.linspace(0.1, 10, 100)
    
    # Entropy production rate from partition lag
    tau_lag = 1.5
    k_B = 1.0
    
    # dS/dt proportional to residue
    residue = np.exp(-t / tau_lag)
    dS_dt = k_B * residue / tau_lag
    
    # Cumulative entropy
    S = k_B * (1 - np.exp(-t / tau_lag))
    
    ax3.plot(t, dS_dt, 'r-', linewidth=2.5, label=r'dS/dt (production rate)')
    ax3.plot(t, S, 'b-', linewidth=2.5, label=r'S(t) (cumulative)')
    
    ax3.set_xlabel('Time t', fontsize=11)
    ax3.set_ylabel('Entropy (units of k_B)', fontsize=11)
    ax3.set_title('(C) Entropy Production: Continuous During Partition Lag', fontweight='bold')
    ax3.legend(loc='right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.text(5, 0.6, r'$\frac{dS}{dt} = k_B \frac{\text{Residue}}{\tau_{lag}}$',
             fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Minimum Lag Scaling with Energy
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Energy gap
    Delta_E = np.logspace(-3, 0, 100)  # eV
    hbar = 6.582e-16  # eV·s
    
    tau_min = hbar / Delta_E
    
    ax4.loglog(Delta_E, tau_min, 'b-', linewidth=2.5)
    
    # Reference points
    ref_energies = [0.001, 0.01, 0.1, 1.0]
    ref_labels = ['Phonon\n(~meV)', 'Vibrational\n(~10 meV)', 'Electronic\n(~100 meV)', 'Core\n(~eV)']
    
    for E, label in zip(ref_energies, ref_labels):
        tau_ref = hbar / E
        ax4.scatter([E], [tau_ref], s=100, zorder=5, edgecolors='black')
        ax4.annotate(label, xy=(E, tau_ref), xytext=(E*1.5, tau_ref*0.3),
                    fontsize=8, ha='left')
    
    ax4.set_xlabel('Energy gap ΔE (eV)', fontsize=11)
    ax4.set_ylabel('Minimum partition lag τ_min (s)', fontsize=11)
    ax4.set_title('(D) Minimum Lag Scales with Energy Gap', fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    ax4.text(0.003, 1e-12, r'$\tau_{min} = \frac{\hbar}{\Delta E}$' + '\nFundamental limit\n(Heisenberg)',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Panel L-5: Partition Lag — The Finite Time of Categorical Determination', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_partition_lag.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_partition_lag.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_partition_lag.png'}")
    print(f"Saved: {output_dir / 'panel_partition_lag.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

