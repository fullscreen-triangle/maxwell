"""
Panel 1: Triple Equivalence - Oscillation ≡ Category ≡ Partition

Demonstrates using virtual gas molecules (each vibrational mode as a pendulum):
- Oscillatory perspective: molecules vibrate with characteristic frequencies
- Categorical perspective: each position is a distinguishable state
- Partition perspective: the oscillation period is divided into intervals

All three views describe the same physical reality.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc, Wedge
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Set up style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'

def create_triple_equivalence_panel():
    """Create a 2x3 panel showing the triple equivalence."""
    
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    
    # Color scheme
    osc_color = '#E74C3C'  # Red for oscillation
    cat_color = '#27AE60'  # Green for category
    part_color = '#3498DB'  # Blue for partition
    
    # =========================================================================
    # Panel A: Virtual Gas Molecules as Pendulums
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(A) Virtual Gas Molecules as Pendulums', fontsize=12, fontweight='bold', pad=10)
    
    # Draw a "container" box
    container = FancyBboxPatch((-1.3, -1.3), 2.6, 2.3, boxstyle="round,pad=0.05",
                                facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=2)
    ax1.add_patch(container)
    
    # Draw multiple pendulum molecules
    pendulum_positions = [(-0.8, 0.5), (0, 0.5), (0.8, 0.5), (-0.4, -0.3), (0.4, -0.3)]
    pendulum_angles = [0.3, -0.2, 0.4, -0.3, 0.25]
    pendulum_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#27AE60', '#3498DB']
    
    for (px, py), angle, color in zip(pendulum_positions, pendulum_angles, pendulum_colors):
        # Pivot point
        ax1.plot(px, py + 0.3, 'ko', markersize=4)
        
        # Rod
        rod_length = 0.35
        end_x = px + rod_length * np.sin(angle)
        end_y = py + 0.3 - rod_length * np.cos(angle)
        ax1.plot([px, end_x], [py + 0.3, end_y], 'k-', linewidth=1.5)
        
        # Mass (molecule)
        molecule = Circle((end_x, end_y), 0.08, facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(molecule)
        
        # Oscillation arc (faint)
        arc = Arc((px, py + 0.3), 0.5, 0.5, angle=0, theta1=250, theta2=290,
                  color=color, linewidth=1, linestyle='--', alpha=0.5)
        ax1.add_patch(arc)
    
    ax1.text(0, -1.1, 'Each vibrational mode = One pendulum', ha='center', fontsize=9,
             style='italic', color='#2C3E50')
    ax1.text(0, 0.95, 'Virtual Gas Container', ha='center', fontsize=10, fontweight='bold')
    
    # =========================================================================
    # Panel B: Oscillatory Perspective
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 4*np.pi)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title('(B) Oscillatory Perspective', fontsize=12, fontweight='bold', pad=10, color=osc_color)
    
    t = np.linspace(0, 4*np.pi, 500)
    theta = np.sin(t)
    ax2.plot(t, theta, color=osc_color, linewidth=2.5, label=r'$\theta(t) = \theta_0 \sin(\omega t)$')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    
    # Mark period
    ax2.annotate('', xy=(2*np.pi, -1.3), xytext=(0, -1.3),
                arrowprops=dict(arrowstyle='<->', color='#2C3E50', lw=2))
    ax2.text(np.pi, -1.45, r'Period $T = 2\pi/\omega$', ha='center', fontsize=10)
    
    # Mark amplitude
    ax2.annotate('', xy=(np.pi/2, 1), xytext=(np.pi/2, 0),
                arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=1.5))
    ax2.text(np.pi/2 + 0.3, 0.5, r'$\theta_0$', fontsize=10, color='#8E44AD')
    
    # Quantum levels (right side)
    for n in range(5):
        y_level = -0.8 + n * 0.4
        ax2.plot([3.8*np.pi, 4*np.pi], [y_level, y_level], 'k-', linewidth=1.5)
        ax2.text(4.05*np.pi, y_level, f'$n={n}$', fontsize=8, va='center')
    
    ax2.text(3.9*np.pi, 1.2, 'Quantum\nStates', ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel(r'Angle $\theta$', fontsize=10)
    ax2.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax2.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
    
    # =========================================================================
    # Panel C: Categorical Perspective
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('(C) Categorical Perspective', fontsize=12, fontweight='bold', pad=10, color=cat_color)
    
    # Draw pendulum with categorical states marked
    pivot = (0, 1.2)
    ax3.plot(*pivot, 'ko', markersize=8)
    
    # Draw arc showing range
    arc_angles = np.linspace(-0.5, 0.5, 100)
    arc_x = 0.8 * np.sin(arc_angles)
    arc_y = 1.2 - 0.8 * np.cos(arc_angles)
    ax3.plot(arc_x, arc_y, 'k--', linewidth=1, alpha=0.5)
    
    # Draw categorical divisions (n=8 states)
    n_cats = 8
    cat_angles = np.linspace(-0.5, 0.5, n_cats + 1)
    cat_colors_list = plt.cm.Greens(np.linspace(0.3, 0.9, n_cats))
    
    for i in range(n_cats):
        angle = (cat_angles[i] + cat_angles[i+1]) / 2
        x = 0.8 * np.sin(angle)
        y = 1.2 - 0.8 * np.cos(angle)
        
        # Draw category marker
        cat_circle = Circle((x, y), 0.08, facecolor=cat_colors_list[i], 
                            edgecolor='black', linewidth=1)
        ax3.add_patch(cat_circle)
        ax3.text(x, y - 0.15, f'$C_{{{i+1}}}$', ha='center', fontsize=7)
    
    # Current position indicator
    current_angle = 0.15
    cx = 0.8 * np.sin(current_angle)
    cy = 1.2 - 0.8 * np.cos(current_angle)
    ax3.plot([0, cx], [1.2, cy], 'k-', linewidth=2)
    mass = Circle((cx, cy), 0.1, facecolor=cat_color, edgecolor='black', linewidth=2)
    ax3.add_patch(mass)
    
    ax3.text(0, -0.3, f'$n = {n_cats}$ distinguishable positions', ha='center', fontsize=9,
             style='italic', color='#2C3E50')
    ax3.text(0, -0.45, r'Each position $\theta_i$ is a categorical state $C_i$', 
             ha='center', fontsize=8, color='gray')
    
    # =========================================================================
    # Panel D: Partition Perspective
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(-0.5, 4.5)
    ax4.set_ylim(-0.5, 2.5)
    ax4.axis('off')
    ax4.set_title('(D) Partition Perspective', fontsize=12, fontweight='bold', pad=10, color=part_color)
    
    # Draw partition tree
    # Level 0 (root)
    root = Circle((2, 2.2), 0.15, facecolor=part_color, edgecolor='black', linewidth=2)
    ax4.add_patch(root)
    ax4.text(2, 2.2, 'T', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Level 1 (n=4 branches)
    level1_x = [0.5, 1.5, 2.5, 3.5]
    for i, x in enumerate(level1_x):
        # Connection line
        ax4.plot([2, x], [2.05, 1.45], 'k-', linewidth=1.5)
        # Node
        node = Circle((x, 1.3), 0.12, facecolor=plt.cm.Blues(0.3 + i*0.15), 
                       edgecolor='black', linewidth=1.5)
        ax4.add_patch(node)
        ax4.text(x, 1.3, f'{i+1}', ha='center', va='center', fontsize=8, color='white')
    
    # Level 2 (each branches into 2)
    level2_x = [0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8]
    for i, x in enumerate(level2_x):
        parent_x = level1_x[i // 2]
        ax4.plot([parent_x, x], [1.18, 0.62], 'k-', linewidth=1)
        node = Circle((x, 0.5), 0.08, facecolor=plt.cm.Blues(0.2 + i*0.08), 
                       edgecolor='black', linewidth=1)
        ax4.add_patch(node)
    
    ax4.text(2, -0.2, r'Partition tree: depth $M$, branching $n$', ha='center', fontsize=9)
    ax4.text(2, -0.4, r'Leaves = $n^M$ terminal states', ha='center', fontsize=9, style='italic')
    
    # Labels
    ax4.text(-0.3, 2.2, 'Level 0', fontsize=8, va='center')
    ax4.text(-0.3, 1.3, 'Level 1', fontsize=8, va='center')
    ax4.text(-0.3, 0.5, 'Level 2', fontsize=8, va='center')
    
    # =========================================================================
    # Panel E: Equivalence Diagram
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_aspect('equal')
    ax5.axis('off')
    ax5.set_title('(E) The Fundamental Equivalence', fontsize=12, fontweight='bold', pad=10)
    
    # Three circles for three frameworks
    radius = 0.55
    centers = [(0, 0.7), (-0.7, -0.4), (0.7, -0.4)]
    colors = [osc_color, cat_color, part_color]
    labels = ['Oscillation', 'Category', 'Partition']
    symbols = [r'$\omega, n$', r'$M, n$', r'$M, n$']
    
    for center, color, label, symbol in zip(centers, colors, labels, symbols):
        circle = Circle(center, radius, facecolor=color, edgecolor='black', 
                        linewidth=2, alpha=0.7)
        ax5.add_patch(circle)
        ax5.text(center[0], center[1] + 0.1, label, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        ax5.text(center[0], center[1] - 0.15, symbol, ha='center', va='center', 
                fontsize=9, color='white')
    
    # Equivalence arrows
    arrow_props = dict(arrowstyle='<->', color='#2C3E50', lw=2,
                       connectionstyle='arc3,rad=0')
    
    ax5.annotate('', xy=(0.35, 0.4), xytext=(-0.35, 0.4), arrowprops=arrow_props)
    ax5.annotate('', xy=(-0.35, -0.1), xytext=(0.35, -0.1), arrowprops=arrow_props)
    ax5.annotate('', xy=(-0.55, 0.3), xytext=(-0.55, -0.15), arrowprops=arrow_props)
    ax5.annotate('', xy=(0.55, 0.3), xytext=(0.55, -0.15), arrowprops=arrow_props)
    
    # Central equivalence
    ax5.text(0, -0.05, r'$\equiv$', ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Bottom formula
    # Formula box
    formula_bg = FancyBboxPatch((-1.2, -1.45), 2.4, 0.4, boxstyle="round,pad=0.05",
                                 facecolor='#F8F9FA', edgecolor='#2C3E50', linewidth=2)
    ax5.add_patch(formula_bg)
    ax5.text(0, -1.25, r'$S = k_B M \ln n$', ha='center', fontsize=14, 
             fontweight='bold', color='#2C3E50')
    ax5.text(0, -1.5, 'Same entropy from all three perspectives', ha='center', 
             fontsize=9, style='italic')
    
    # =========================================================================
    # Panel F: Unified Interpretation Table
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    ax6.set_title('(F) Parameter Correspondence', fontsize=12, fontweight='bold', pad=10)
    
    # Create table
    table_data = [
        ['Concept', 'Oscillatory', 'Categorical', 'Partition'],
        ['DOF ($M$)', 'Modes', 'Dimensions', 'Levels'],
        ['States ($n$)', 'Quantum #', 'Levels', 'Branches'],
        ['Total', r'$n^M$ states', r'$|\mathcal{C}|$', 'Leaves'],
        ['Entropy', r'$k_B \ln W$', r'$k_B \ln|\mathcal{C}|$', r'$k_B M \ln n$'],
    ]
    
    cell_height = 0.8
    cell_widths = [2.0, 2.2, 2.2, 2.2]
    start_y = 8.5
    start_x = 0.5
    
    header_colors = ['#2C3E50', osc_color, cat_color, part_color]
    
    for row_idx, row in enumerate(table_data):
        y = start_y - row_idx * cell_height
        x = start_x
        for col_idx, (cell, width) in enumerate(zip(row, cell_widths)):
            if row_idx == 0:
                facecolor = header_colors[col_idx]
                textcolor = 'white'
                fontweight = 'bold'
            else:
                facecolor = '#ECF0F1' if row_idx % 2 == 0 else 'white'
                textcolor = 'black'
                fontweight = 'normal'
            
            rect = FancyBboxPatch((x, y - cell_height + 0.1), width - 0.1, cell_height - 0.1,
                                   boxstyle="round,pad=0.02", facecolor=facecolor,
                                   edgecolor='#BDC3C7', linewidth=1)
            ax6.add_patch(rect)
            ax6.text(x + width/2 - 0.05, y - cell_height/2 + 0.05, cell, 
                    ha='center', va='center', fontsize=8, color=textcolor,
                    fontweight=fontweight)
            x += width
    
    # Bottom note
    ax6.text(5, 3.5, 'The pendulum demonstrates all three:', ha='center', fontsize=10, 
             fontweight='bold')
    ax6.text(5, 3.0, r'Oscillation: $\theta(t) = \theta_0 \cos(\omega t)$', ha='center', 
             fontsize=9, color=osc_color)
    ax6.text(5, 2.5, r'Category: $n$ distinguishable positions $\{C_1, ..., C_n\}$', 
             ha='center', fontsize=9, color=cat_color)
    ax6.text(5, 2.0, r'Partition: Period $T$ divided into $n$ intervals', ha='center', 
             fontsize=9, color=part_color)
    
    # Final formula with box
    final_box = FancyBboxPatch((2.5, 0.8), 5, 0.7, boxstyle="round,pad=0.05",
                                facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=2)
    ax6.add_patch(final_box)
    ax6.text(5, 1.15, r'All yield: $S = k_B \ln n$', ha='center', fontsize=11, 
             fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import os
    
    # Get the script directory and construct output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fig = create_triple_equivalence_panel()
    
    # Save in multiple formats
    fig.savefig(os.path.join(output_dir, 'panel1_triple_equivalence.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'panel1_triple_equivalence.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    print("Panel 1 (Triple Equivalence) saved successfully!")
    plt.close()

