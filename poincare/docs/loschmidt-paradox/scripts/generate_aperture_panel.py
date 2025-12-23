#!/usr/bin/env python3
"""
Generate Panel L-4: Aperture Selectivity and Categorical Potential
Visualizes how partition boundaries function as selective apertures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Selectivity Function
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 3)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    
    # Aperture (partition boundary)
    aperture_x = 1.5
    aperture_width = 0.1
    aperture_gap_y = (0.8, 1.2)  # Gap in the partition
    
    # Draw partition with aperture
    ax1.fill_between([aperture_x - aperture_width/2, aperture_x + aperture_width/2], 
                     [0, 0], [aperture_gap_y[0], aperture_gap_y[0]], color='gray', alpha=0.8)
    ax1.fill_between([aperture_x - aperture_width/2, aperture_x + aperture_width/2], 
                     [aperture_gap_y[1], aperture_gap_y[1]], [1.8, 1.8], color='gray', alpha=0.8)
    
    # Aperture opening
    ax1.plot([aperture_x - aperture_width/2, aperture_x - aperture_width/2], 
             [aperture_gap_y[0], aperture_gap_y[1]], 'g-', linewidth=3)
    ax1.plot([aperture_x + aperture_width/2, aperture_x + aperture_width/2], 
             [aperture_gap_y[0], aperture_gap_y[1]], 'g-', linewidth=3)
    
    # Particles that pass (green arrows)
    for y in [0.9, 1.0, 1.1]:
        ax1.annotate('', xy=(2.2, y), xytext=(0.8, y),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Particles that are blocked (red X)
    blocked_y = [0.3, 0.5, 1.5, 1.7]
    for y in blocked_y:
        ax1.annotate('', xy=(1.3, y), xytext=(0.8, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, ls='--'))
        ax1.plot([1.25, 1.35], [y-0.05, y+0.05], 'r-', linewidth=2)
        ax1.plot([1.25, 1.35], [y+0.05, y-0.05], 'r-', linewidth=2)
    
    # Labels
    ax1.text(0.5, 1.85, 'Incoming\nconfigurations', ha='center', fontsize=9)
    ax1.text(2.5, 1.0, 'σ(ω) = 1\n(pass)', ha='center', fontsize=10, color='green', fontweight='bold')
    ax1.text(0.2, 0.4, 'σ(ω) = 0\n(blocked)', ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax1.text(aperture_x, -0.3, 'Aperture\n(partition boundary)', ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_title('(A) Selection Function σ(ω): Pass or Block', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Categorical Potential vs Selectivity
    ax2 = fig.add_subplot(gs[0, 1])
    
    s = np.linspace(0.01, 1.0, 100)
    kT = 1.0  # Normalized
    Phi = -kT * np.log(s)
    
    ax2.plot(s, Phi, 'b-', linewidth=2.5)
    ax2.fill_between(s, 0, Phi, alpha=0.2, color='blue')
    
    # Annotate key points
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    ax2.scatter([1], [0], s=100, c='green', zorder=5, edgecolors='black')
    ax2.text(1.02, 0.3, 's = 1\nNo selectivity\nΦ = 0', fontsize=9, color='green')
    
    ax2.scatter([0.1], [-kT * np.log(0.1)], s=100, c='red', zorder=5, edgecolors='black')
    ax2.text(0.15, -kT * np.log(0.1) - 0.3, 's → 0\nHigh selectivity\nΦ → ∞', fontsize=9, color='red')
    
    ax2.set_xlabel('Selectivity s = Ω_pass / Ω_total', fontsize=11)
    ax2.set_ylabel('Categorical Potential Φ = -kT ln(s)', fontsize=11)
    ax2.set_title('(B) Aperture Potential: Φ = -kT ln(s)', fontweight='bold')
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(-0.5, 5)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Entropy from Selectivity
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Different selectivities
    selectivities = [0.9, 0.5, 0.1, 0.01]
    labels = ['s=0.9\n(weak)', 's=0.5\n(medium)', 's=0.1\n(strong)', 's=0.01\n(very strong)']
    entropies = [-np.log(s) for s in selectivities]  # Normalized
    colors = ['lightgreen', 'yellow', 'orange', 'red']
    
    bars = ax3.bar(range(len(selectivities)), entropies, color=colors, edgecolor='black', linewidth=1.5)
    
    ax3.set_xticks(range(len(selectivities)))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('Entropy production ΔS/k = ln(1/s)', fontsize=11)
    ax3.set_title('(C) Entropy from Selectivity: Higher Selectivity → More Entropy', fontweight='bold')
    
    # Add formula
    ax3.text(1.5, max(entropies) * 0.8, r'$\Delta S = k_B \ln\left(\frac{1}{s}\right) = \frac{\Phi}{T}$',
             fontsize=12, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Aperture Barrier Visualization
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    
    # Energy landscape with aperture barrier
    x = np.linspace(0, 3, 100)
    
    # Barrier profile
    barrier_center = 1.5
    barrier_width = 0.3
    barrier_height = 1.5
    
    y_barrier = barrier_height * np.exp(-((x - barrier_center)**2) / (2 * barrier_width**2))
    
    ax4.fill_between(x, 0, y_barrier, color='lightcoral', alpha=0.5)
    ax4.plot(x, y_barrier, 'r-', linewidth=2)
    
    # Particle approaching barrier
    ax4.annotate('', xy=(1.0, 0.3), xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.scatter([0.5], [0.3], s=150, c='blue', zorder=5, edgecolors='black')
    ax4.text(0.5, 0.5, 'Particle', ha='center', fontsize=9)
    
    # Barrier annotation
    ax4.annotate('', xy=(1.5, 1.5), xytext=(1.5, 0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax4.text(1.7, 0.9, r'Φ = -kT ln(s)', fontsize=10, fontweight='bold')
    
    # Transmitted vs reflected
    ax4.annotate('', xy=(2.8, 0.3), xytext=(2.0, 0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax4.text(2.5, 0.5, 'Pass (prob. s)', fontsize=9, color='green')
    
    ax4.annotate('', xy=(0.5, 0.8), xytext=(1.2, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, ls='--'))
    ax4.text(0.5, 1.0, 'Block (prob. 1-s)', fontsize=9, color='red')
    
    # Key insight
    ax4.text(1.5, 2.2, 'Aperture = Categorical Barrier\nBlocked states → Non-actualisations',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.set_xlabel('Reaction coordinate', fontsize=11)
    ax4.set_ylabel('Potential energy', fontsize=11)
    ax4.set_title('(D) Aperture as Energy Barrier: Selectivity Creates Entropy', fontweight='bold')
    
    plt.suptitle('Panel L-4: Partition Boundaries as Categorical Apertures', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_aperture_selectivity.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_aperture_selectivity.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_aperture_selectivity.png'}")
    print(f"Saved: {output_dir / 'panel_aperture_selectivity.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

