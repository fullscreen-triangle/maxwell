#!/usr/bin/env python3
"""
Generate Panel L-1: Mixing-Separation Entropy Cycle
Demonstrates that entropy increases in a full mixing-separation cycle.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def generate_molecules(n, xlim, ylim, seed=42):
    """Generate random molecule positions."""
    np.random.seed(seed)
    x = np.random.uniform(xlim[0] + 0.1, xlim[1] - 0.1, n)
    y = np.random.uniform(ylim[0] + 0.1, ylim[1] - 0.1, n)
    return x, y

def draw_container(ax, xlim, ylim, color='black', linewidth=2):
    """Draw container boundary."""
    rect = Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                     fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)

def draw_partition(ax, x, ylim, color='gray', linewidth=3):
    """Draw partition line."""
    ax.plot([x, x], ylim, color=color, linewidth=linewidth, linestyle='-')

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel A: Initial State (Separated)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('(A) Initial State: Separated Gases', fontweight='bold')
    
    # Left container - Gas A (blue)
    draw_container(ax1, (0, 0), (1, 1))
    x_a, y_a = generate_molecules(20, (0, 1), (0, 1), seed=42)
    ax1.scatter(x_a, y_a, c='royalblue', s=60, alpha=0.8, edgecolors='darkblue', linewidth=0.5, label='Gas A')
    
    # Right container - Gas B (red)
    draw_container(ax1, (1.5, 0), (2.5, 1))
    x_b, y_b = generate_molecules(20, (1.5, 2.5), (0, 1), seed=43)
    ax1.scatter(x_b, y_b, c='tomato', s=60, alpha=0.8, edgecolors='darkred', linewidth=0.5, label='Gas B')
    
    # Partition
    draw_partition(ax1, 1.25, (0, 1))
    
    ax1.text(0.5, -0.3, r'$S_A^{(0)}$', ha='center', fontsize=12)
    ax1.text(2.0, -0.3, r'$S_B^{(0)}$', ha='center', fontsize=12)
    ax1.text(1.25, 1.15, r'$S_{initial} = S_A^{(0)} + S_B^{(0)}$', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')
    
    # Panel B: Mixed State
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('(B) Mixed State: Partition Removed', fontweight='bold')
    
    # Combined container
    draw_container(ax2, (0, 0), (2.5, 1))
    
    # Mixed molecules
    x_mixed_a, y_mixed_a = generate_molecules(20, (0, 2.5), (0, 1), seed=44)
    x_mixed_b, y_mixed_b = generate_molecules(20, (0, 2.5), (0, 1), seed=45)
    ax2.scatter(x_mixed_a, y_mixed_a, c='royalblue', s=60, alpha=0.8, edgecolors='darkblue', linewidth=0.5)
    ax2.scatter(x_mixed_b, y_mixed_b, c='tomato', s=60, alpha=0.8, edgecolors='darkred', linewidth=0.5)
    
    ax2.text(1.25, -0.3, r'$S_{mixed} = S_{initial} + \Delta S_{mix}$', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.text(1.25, 1.15, 'Phase-lock network connected', ha='center', fontsize=10, style='italic')
    ax2.axis('off')
    
    # Panel C: Re-separated State
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_title('(C) Re-separated: Partition Restored', fontweight='bold')
    
    # Left container - mixed gases
    draw_container(ax3, (0, 0), (1, 1))
    x_resep_l_a, y_resep_l_a = generate_molecules(10, (0, 1), (0, 1), seed=46)
    x_resep_l_b, y_resep_l_b = generate_molecules(10, (0, 1), (0, 1), seed=47)
    ax3.scatter(x_resep_l_a, y_resep_l_a, c='royalblue', s=60, alpha=0.8, edgecolors='darkblue', linewidth=0.5)
    ax3.scatter(x_resep_l_b, y_resep_l_b, c='tomato', s=60, alpha=0.8, edgecolors='darkred', linewidth=0.5)
    
    # Right container - mixed gases
    draw_container(ax3, (1.5, 0), (2.5, 1))
    x_resep_r_a, y_resep_r_a = generate_molecules(10, (1.5, 2.5), (0, 1), seed=48)
    x_resep_r_b, y_resep_r_b = generate_molecules(10, (1.5, 2.5), (0, 1), seed=49)
    ax3.scatter(x_resep_r_a, y_resep_r_a, c='royalblue', s=60, alpha=0.8, edgecolors='darkblue', linewidth=0.5)
    ax3.scatter(x_resep_r_b, y_resep_r_b, c='tomato', s=60, alpha=0.8, edgecolors='darkred', linewidth=0.5)
    
    # Partition
    draw_partition(ax3, 1.25, (0, 1))
    
    ax3.text(1.25, -0.3, r'$S_{final} = S_{initial} + \Delta S_{residual}$', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.text(1.25, 1.15, 'Residual phase correlations persist!', ha='center', fontsize=10, 
             style='italic', color='darkred')
    ax3.axis('off')
    
    # Panel D: Entropy Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Time points
    t = np.array([0, 1, 2, 3, 4, 5])
    t_labels = ['Initial', 'Mixing\nstarts', 'Fully\nmixed', 'Re-sep\nstarts', 'Re-sep\ncomplete', 'Final\nstate']
    
    # Entropy values (normalized)
    S = np.array([1.0, 1.0, 1.8, 1.8, 1.5, 1.5])
    
    # Classical prediction (reversible)
    S_classical = np.array([1.0, 1.0, 1.8, 1.8, 1.0, 1.0])
    
    ax4.plot(t, S, 'o-', color='darkgreen', linewidth=2.5, markersize=10, 
             label='Categorical prediction')
    ax4.plot(t, S_classical, 's--', color='gray', linewidth=2, markersize=8, 
             label='Classical (reversible)', alpha=0.7)
    
    # Highlight final difference
    ax4.annotate('', xy=(5, 1.5), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax4.text(5.15, 1.25, r'$\Delta S_{irrev} > 0$', fontsize=11, color='red', fontweight='bold')
    
    ax4.set_xticks(t)
    ax4.set_xticklabels(t_labels, fontsize=9)
    ax4.set_ylabel('Entropy S (normalized)', fontsize=11)
    ax4.set_title('(D) Entropy Evolution: Irreversibility Demonstrated', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_ylim(0.8, 2.0)
    ax4.grid(True, alpha=0.3)
    
    # Add key result box
    ax4.text(2.5, 0.95, r'$S_{final} > S_{initial}$' + '\nEntropy increases despite\nidentical spatial config!',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9, edgecolor='darkred'))
    
    plt.suptitle('Panel L-1: Mixing-Separation Cycle Demonstrates Irreversibility', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_mixing_separation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_mixing_separation.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_mixing_separation.png'}")
    print(f"Saved: {output_dir / 'panel_mixing_separation.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

