#!/usr/bin/env python3
"""
Generate Panel L-6: Termination and Irreversibility
Shows why entropy is only defined at termination and why reversal fails.
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
    
    # Panel A: Reality Stream vs Terminated State
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    
    # Reality stream (in-progress)
    stream_x = np.linspace(0, 2.5, 50)
    stream_y = 1.5 + 0.1 * np.sin(10 * stream_x)
    ax1.plot(stream_x, stream_y, 'b-', linewidth=3, alpha=0.7)
    ax1.scatter([0], [1.5], s=100, c='blue', zorder=5, edgecolors='black')
    
    # Wavy arrow for ongoing process
    ax1.annotate('', xy=(2.7, 1.5), xytext=(2.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(1.25, 1.8, 'Reality Stream (in progress)', ha='center', fontsize=10, 
             fontweight='bold', color='blue')
    ax1.text(1.25, 1.2, 'S = undefined\n(superposition of possibilities)', 
             ha='center', fontsize=9, style='italic')
    
    # Terminated state (fixed)
    ax1.fill([3.0, 3.5, 3.5, 3.0], [1.3, 1.3, 1.7, 1.7], color='green', alpha=0.5)
    ax1.plot([3.0, 3.5, 3.5, 3.0, 3.0], [1.3, 1.3, 1.7, 1.7, 1.3], 'g-', linewidth=2)
    ax1.scatter([3.25], [1.5], s=100, c='green', zorder=5, edgecolors='black')
    ax1.text(3.25, 1.85, 'Terminated State', ha='center', fontsize=10, 
             fontweight='bold', color='green')
    ax1.text(3.25, 1.1, 'S = well-defined\n(categorical fact)', 
             ha='center', fontsize=9, style='italic')
    
    # Arrow from stream to terminated
    ax1.annotate('', xy=(2.95, 1.5), xytext=(2.7, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax1.text(2.8, 1.35, 'Collapse', fontsize=8, rotation=0)
    
    # Key insight
    ax1.text(1.75, 0.3, 'Entropy requires termination:\nNo termination → No entropy',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax1.set_title('(A) Reality Stream vs Terminated: Entropy Requires Collapse', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Categorical Completion = Geometric Partitioning
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.5, 3)
    ax2.set_ylim(-0.5, 2)
    ax2.set_aspect('equal')
    
    # Before completion (undivided)
    ax2.add_patch(Circle((0.5, 1.5), 0.4, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax2.text(0.5, 1.5, '?', ha='center', va='center', fontsize=20, fontweight='bold')
    ax2.text(0.5, 0.95, 'Before\n(undetermined)', ha='center', fontsize=9)
    
    # Arrow
    ax2.annotate('', xy=(1.3, 1.5), xytext=(1.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(1.15, 1.7, 'Partition', fontsize=9, ha='center')
    
    # After completion (divided)
    # Top half
    ax2.add_patch(plt.matplotlib.patches.Wedge((2.0, 1.5), 0.4, 0, 180, 
                 facecolor='lightgreen', edgecolor='black', linewidth=2))
    # Bottom half
    ax2.add_patch(plt.matplotlib.patches.Wedge((2.0, 1.5), 0.4, 180, 360, 
                 facecolor='lightyellow', edgecolor='black', linewidth=2))
    
    ax2.text(2.0, 1.7, 'A', ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(2.0, 1.3, 'B', ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(2.0, 0.95, 'After\n(determined)', ha='center', fontsize=9)
    
    # Identity statement
    ax2.text(2.7, 1.5, '≡', fontsize=24, ha='center', va='center')
    
    # Geometric representation
    ax2.plot([0, 0.6, 0.6, 0, 0], [0, 0, 0.6, 0.6, 0], 'k-', linewidth=2,
            transform=ax2.transData)
    rect = Rectangle((2.8, 1.2), 0.6, 0.6, facecolor='none', edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.plot([2.8, 3.4], [1.5, 1.5], 'r-', linewidth=2)  # Partition line
    
    ax2.text(3.55, 1.5, 'Boundary\ncreated', fontsize=8, ha='left')
    
    ax2.text(1.25, 0.3, 'Categorical Completion ≡ Geometric Partitioning\nBoth create boundaries → Both create entropy',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    ax2.set_title('(B) Identity: Completion = Partitioning = Entropy Creation', fontweight='bold')
    ax2.axis('off')
    
    # Panel C: Why Reversal Fails
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(-0.5, 3.5)
    ax3.set_ylim(-0.5, 2.2)
    ax3.set_aspect('equal')
    
    # Forward direction
    ax3.text(0.2, 1.9, 'FORWARD', fontsize=11, fontweight='bold', color='green')
    
    # State A
    ax3.add_patch(Circle((0.5, 1.3), 0.25, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax3.text(0.5, 1.3, 'A', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow forward
    ax3.annotate('', xy=(1.5, 1.3), xytext=(0.85, 1.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Creates non-actualisations
    for i, y in enumerate([1.7, 1.5, 1.3, 1.1, 0.9]):
        ax3.scatter([1.3], [y], s=30, c='red', alpha=0.5)
    ax3.text(1.4, 1.0, '+ ∞ non-acts', fontsize=8, color='red', rotation=90)
    
    # State B
    ax3.add_patch(Circle((2.0, 1.3), 0.25, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax3.text(2.0, 1.3, 'B', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Backward direction (blocked)
    ax3.text(2.5, 1.9, 'BACKWARD', fontsize=11, fontweight='bold', color='red')
    
    # Arrow backward (dashed)
    ax3.annotate('', xy=(2.35, 1.3), xytext=(3.2, 1.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
    
    # X mark
    ax3.plot([2.65, 2.95], [1.15, 1.45], 'r-', linewidth=4)
    ax3.plot([2.65, 2.95], [1.45, 1.15], 'r-', linewidth=4)
    
    # Requires un-creating
    ax3.text(2.8, 0.7, 'Would require\nun-creating ∞\nnon-actualisations', 
             ha='center', fontsize=9, color='red',
             bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    # Key insight
    ax3.text(1.25, 0.2, 'Reversal is not about energy or velocity reversal\nIt requires un-creating categorical facts → IMPOSSIBLE',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax3.set_title('(C) Why Reversal Fails: Cannot Un-Create Categorical Facts', fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Forward/Backward Asymmetry Ratio
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Number of processes
    n_processes = np.arange(1, 11)
    
    # Asymmetry grows exponentially
    non_actualisations_per_process = 1000  # Symbolic
    asymmetry_ratio = non_actualisations_per_process ** n_processes
    
    ax4.semilogy(n_processes, asymmetry_ratio, 'ro-', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Number of categorical completions', fontsize=11)
    ax4.set_ylabel('Forward/Backward asymmetry ratio', fontsize=11)
    ax4.set_title('(D) Asymmetry Grows Exponentially with Each Completion', fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    # Annotation
    ax4.text(5.5, 1e15, r'$\frac{P_{forward}}{P_{backward}} = \prod_{i=1}^{N} \Omega_i \rightarrow \infty$',
             fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Reversible (ratio = 1)')
    ax4.legend(loc='lower right', fontsize=9)
    
    plt.suptitle('Panel L-6: Termination, Completion, and the Impossibility of Reversal', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_termination_irreversibility.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_termination_irreversibility.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_termination_irreversibility.png'}")
    print(f"Saved: {output_dir / 'panel_termination_irreversibility.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

