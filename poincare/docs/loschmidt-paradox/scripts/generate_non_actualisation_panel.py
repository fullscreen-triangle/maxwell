#!/usr/bin/env python3
"""
Generate Panel L-3: Non-Actualisation Accumulation
The cup example and asymmetry between actualisation and non-actualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def draw_cup_on_table(ax, x, y, width=0.3, height=0.25, color='saddlebrown'):
    """Draw a simple cup."""
    # Cup body
    cup = FancyBboxPatch((x - width/2, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(cup)
    
    # Handle
    handle = Wedge((x + width/2, y + height/2), 0.08, -90, 90, width=0.03,
                   facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(handle)

def draw_broken_cup(ax, x, y, color='saddlebrown'):
    """Draw broken cup pieces."""
    np.random.seed(42)
    for i in range(5):
        dx = np.random.uniform(-0.2, 0.2)
        dy = np.random.uniform(-0.05, 0.05)
        size = np.random.uniform(0.05, 0.12)
        
        piece = FancyBboxPatch((x + dx - size/2, y + dy - size/2), size, size*0.6,
                               boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(piece)

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Cup Example Visualization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 1.8)
    ax1.set_aspect('equal')
    
    # Table surface
    ax1.fill_between([-0.3, 1.0], [-0.1, -0.1], [0, 0], color='peru', alpha=0.5)
    ax1.plot([-0.3, 1.0], [0, 0], 'k-', linewidth=2)
    
    # Cup on table
    draw_cup_on_table(ax1, 0.35, 0.02)
    ax1.text(0.35, 0.4, 'Cup on table', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow showing fall
    ax1.annotate('', xy=(1.5, 0.15), xytext=(0.7, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, 
                               connectionstyle='arc3,rad=-0.3'))
    ax1.text(1.1, 0.5, 'FALLS', fontsize=11, color='red', fontweight='bold')
    
    # Floor
    ax1.fill_between([1.2, 2.3], [-0.1, -0.1], [0, 0], color='gray', alpha=0.3)
    ax1.plot([1.2, 2.3], [0, 0], 'k-', linewidth=2)
    
    # Broken cup
    draw_broken_cup(ax1, 1.75, 0.02)
    ax1.text(1.75, 0.25, 'Broken cup', ha='center', fontsize=10, fontweight='bold')
    
    # Non-actualisations (things that didn't happen)
    non_acts = [
        'Not turning to gold',
        'Not becoming sentient',
        'Not teleporting',
        'Not melting',
        'Not reassembling'
    ]
    
    for i, na in enumerate(non_acts):
        y_pos = 1.6 - i * 0.25
        ax1.text(2.0, y_pos, f'✗ {na}', fontsize=9, color='darkred', alpha=0.8)
    
    ax1.text(1.75, 1.75, 'Non-actualisations:', fontsize=10, fontweight='bold', 
             color='darkred', ha='center')
    
    ax1.set_title('(A) The Cup Example: Non-Actualisations Created', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Actualisation Tree
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-0.5, 2)
    ax2.set_aspect('equal')
    
    # Root node (event)
    root = Circle((1, 1.7), 0.15, facecolor='green', edgecolor='black', linewidth=2)
    ax2.add_patch(root)
    ax2.text(1, 1.7, '1', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax2.text(1, 1.95, 'Actualisation\n(finite)', ha='center', fontsize=9, fontweight='bold', color='green')
    
    # Branch to non-actualisations
    n_branches = 8
    angles = np.linspace(-60, 60, n_branches)
    
    for i, angle in enumerate(angles):
        rad = np.radians(angle - 90)
        x_end = 1 + 0.8 * np.sin(rad)
        y_end = 1.7 - 0.8 * np.cos(rad)
        
        ax2.annotate('', xy=(x_end, y_end), xytext=(1, 1.55),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.6))
        
        node = Circle((x_end, y_end - 0.12), 0.08, facecolor='red', edgecolor='black', linewidth=1, alpha=0.7)
        ax2.add_patch(node)
    
    ax2.text(1, 0.6, '∞ Non-actualisations\n(infinite)', ha='center', fontsize=10, 
             fontweight='bold', color='darkred')
    
    # Asymmetry annotation
    ax2.annotate('', xy=(2.2, 1.7), xytext=(2.2, 0.7),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax2.text(2.4, 1.2, 'ASYMMETRY\n1 : ∞', fontsize=10, color='purple', fontweight='bold')
    
    ax2.set_title('(B) Branching Asymmetry: 1 Actualisation → ∞ Non-actualisations', fontweight='bold')
    ax2.axis('off')
    
    # Panel C: Non-actualisation Count Over Time
    ax3 = fig.add_subplot(gs[1, 0])
    
    t = np.linspace(0, 10, 100)
    n_events = np.floor(t * 2)  # Number of events
    n_non_act = n_events * 1000  # Each event creates many non-actualisations
    
    ax3.fill_between(t, 0, n_non_act, alpha=0.3, color='red', label='Non-actualisations')
    ax3.plot(t, n_non_act, 'r-', linewidth=2)
    ax3.plot(t, n_events * 100, 'g-', linewidth=2, label='Actualisations (scaled ×100)')
    
    ax3.set_xlabel('Time (categorical units)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('(C) Accumulation: Non-actualisations Grow Without Bound', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_ylim(0, max(n_non_act) * 1.1)
    
    ax3.text(5, max(n_non_act) * 0.7, 
             'Non-actualisations\naccumulate monotonically\n→ Cannot be un-created',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    # Panel D: Forward vs Backward Asymmetry
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-1, 3)
    ax4.set_ylim(-0.5, 2)
    ax4.set_aspect('equal')
    
    # Forward direction
    ax4.annotate('', xy=(2, 1.5), xytext=(0.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax4.text(1.25, 1.65, 'FORWARD', ha='center', fontsize=11, fontweight='bold', color='green')
    ax4.text(1.25, 1.3, '1 actual → ∞ non-actual', ha='center', fontsize=10)
    ax4.text(1.25, 1.1, 'Always possible', ha='center', fontsize=9, style='italic', color='green')
    
    # Backward direction (blocked)
    ax4.annotate('', xy=(0.5, 0.5), xytext=(2, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=3, ls='--'))
    ax4.text(1.25, 0.65, 'BACKWARD', ha='center', fontsize=11, fontweight='bold', color='red')
    ax4.text(1.25, 0.3, 'Requires un-creating non-actualisations', ha='center', fontsize=10)
    ax4.text(1.25, 0.1, 'IMPOSSIBLE', ha='center', fontsize=11, fontweight='bold', color='red')
    
    # Big X
    ax4.plot([0.3, 0.7], [0.3, 0.7], 'r-', linewidth=4)
    ax4.plot([0.3, 0.7], [0.7, 0.3], 'r-', linewidth=4)
    
    # Ratio box
    ax4.text(1.25, -0.3, r'$\frac{B_{forward}}{B_{backward}} \rightarrow \infty$', 
             ha='center', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))
    
    ax4.set_title('(D) Irreversibility: Forward/Backward Asymmetry Ratio → ∞', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Panel L-3: Non-Actualisation Asymmetry — The Deepest Reason for Irreversibility', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_non_actualisation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_non_actualisation.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_non_actualisation.png'}")
    print(f"Saved: {output_dir / 'panel_non_actualisation.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

