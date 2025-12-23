"""
Panel 3: Categorical Enthalpy Derivation

Shows:
- Apertures and selectivity concept
- Categorical potential Φ = -k_B T ln s
- Categorical enthalpy H = U + Σn_a Φ_a
- Recovery of classical enthalpy H = U + PV in the non-selective limit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge, Polygon
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.gridspec as gridspec

# Set up style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'

def create_categorical_enthalpy_panel():
    """Create a 2x3 panel showing categorical enthalpy derivation."""
    
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    
    # Color scheme
    aperture_color = '#9B59B6'  # Purple for apertures
    pass_color = '#27AE60'      # Green for passing molecules
    block_color = '#E74C3C'    # Red for blocked molecules
    enthalpy_color = '#E67E22' # Orange for enthalpy
    
    # =========================================================================
    # Panel A: Aperture Concept - Selective Passage
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-2, 8)
    ax1.set_ylim(-1, 5)
    ax1.axis('off')
    ax1.set_title('(A) Aperture: Selective Molecular Passage', fontsize=12,
                  fontweight='bold', pad=10)
    
    # Draw membrane with aperture
    membrane_left = Rectangle((-0.5, 0), 1, 4.5, facecolor='#7F8C8D', 
                               edgecolor='black', linewidth=2)
    membrane_right = Rectangle((0.5, 0), 1, 4.5, facecolor='#7F8C8D',
                                 edgecolor='black', linewidth=2)
    ax1.add_patch(membrane_left)
    ax1.add_patch(membrane_right)
    
    # Aperture (gap in membrane)
    aperture_height = 1.2
    aperture_y = 2.0
    ax1.fill_between([0, 0.5], [aperture_y, aperture_y], 
                     [aperture_y + aperture_height, aperture_y + aperture_height],
                     color='white')
    
    # Aperture label
    ax1.annotate('', xy=(0.8, aperture_y + aperture_height/2), 
                xytext=(1.8, aperture_y + aperture_height/2),
                arrowprops=dict(arrowstyle='->', color=aperture_color, lw=2))
    ax1.text(2.0, aperture_y + aperture_height/2, 'Aperture $a$', fontsize=10,
             va='center', color=aperture_color, fontweight='bold')
    
    # Passing molecules (small, green)
    pass_positions = [(-1.2, 2.5), (-0.8, 2.8), (1.8, 2.5), (2.2, 2.7)]
    for x, y in pass_positions[:2]:
        circle = Circle((x, y), 0.15, facecolor=pass_color, edgecolor='black', linewidth=1)
        ax1.add_patch(circle)
        ax1.annotate('', xy=(x + 0.3, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=pass_color, lw=1.5))
    for x, y in pass_positions[2:]:
        circle = Circle((x, y), 0.15, facecolor=pass_color, edgecolor='black', linewidth=1)
        ax1.add_patch(circle)
    
    # Blocked molecules (large, red) 
    block_positions = [(-1.3, 3.5), (-1.0, 1.2)]
    for x, y in block_positions:
        circle = Circle((x, y), 0.3, facecolor=block_color, edgecolor='black', linewidth=1)
        ax1.add_patch(circle)
        ax1.plot([x + 0.35, x + 0.5], [y, y], 'k-', linewidth=2)
        ax1.plot([x + 0.5, x + 0.4], [y + 0.1, y], 'k-', linewidth=2)
        ax1.plot([x + 0.5, x + 0.4], [y - 0.1, y], 'k-', linewidth=2)
    
    # Legend
    ax1.text(-1.5, 4.7, 'Small molecules: PASS', fontsize=9, color=pass_color)
    ax1.text(-1.5, 4.3, 'Large molecules: BLOCKED', fontsize=9, color=block_color)
    
    # Selectivity formula
    ax1.text(3.5, 3.8, r'Selectivity:', fontsize=10, fontweight='bold')
    ax1.text(3.5, 3.2, r'$s_a = \frac{\Omega_{pass}}{\Omega_{total}}$', fontsize=12)
    ax1.text(3.5, 2.4, r'$0 \leq s_a \leq 1$', fontsize=10)
    
    ax1.text(3.5, 1.2, r'$s = 1$: All pass', fontsize=9)
    ax1.text(3.5, 0.7, r'$s = 0$: None pass', fontsize=9)
    ax1.text(3.5, 0.2, r'$0 < s < 1$: Selective', fontsize=9)
    
    # =========================================================================
    # Panel B: Categorical Potential
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(B) Categorical Potential vs Selectivity', fontsize=12,
                  fontweight='bold', pad=10)
    
    # Plot Φ = -k_B T ln(s)
    s_vals = np.linspace(0.01, 1.0, 100)
    phi_vals = -np.log(s_vals)  # In units of k_B T
    
    ax2.plot(s_vals, phi_vals, color=aperture_color, linewidth=3)
    ax2.fill_between(s_vals, 0, phi_vals, color=aperture_color, alpha=0.2)
    
    ax2.set_xlabel(r'Selectivity $s = \Omega_{pass}/\Omega_{total}$', fontsize=11)
    ax2.set_ylabel(r'Potential $\Phi_a / k_B T$', fontsize=11)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 5)
    ax2.grid(True, alpha=0.3)
    
    # Mark key points
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    
    # s = 1: Φ = 0
    ax2.plot(1, 0, 'go', markersize=12)
    ax2.text(0.85, 0.3, r'$s=1$: $\Phi=0$' + '\n(no barrier)', fontsize=9, color=pass_color)
    
    # s = 0.5
    ax2.plot(0.5, -np.log(0.5), 'yo', markersize=10)
    ax2.text(0.35, -np.log(0.5) + 0.3, r'$s=0.5$: $\Phi=0.69 k_B T$', fontsize=8)
    
    # s → 0: Φ → ∞
    ax2.annotate(r'$s \to 0$: $\Phi \to \infty$' + '\n(impermeable)', 
                xy=(0.05, 4.5), fontsize=9, color=block_color, ha='left')
    
    # Formula box
    formula_box = FancyBboxPatch((0.55, 3.5), 0.4, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='white', edgecolor=aperture_color, linewidth=2,
                                  transform=ax2.transAxes)
    ax2.add_patch(formula_box)
    ax2.text(0.75, 0.92, r'$\Phi_a = -k_B T \ln s_a$', fontsize=11, fontweight='bold',
             ha='center', transform=ax2.transAxes, color=aperture_color)
    
    # =========================================================================
    # Panel C: Categorical Enthalpy Definition
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-1, 11)
    ax3.set_ylim(-1, 6)
    ax3.axis('off')
    ax3.set_title('(C) Categorical Enthalpy Definition', fontsize=12,
                  fontweight='bold', pad=10)
    
    # System with multiple apertures
    # Container
    container = FancyBboxPatch((0.5, 1.5), 4, 3, boxstyle="round,pad=0.1",
                                facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=2)
    ax3.add_patch(container)
    ax3.text(2.5, 4.8, 'System', ha='center', fontsize=10, fontweight='bold')
    
    # Internal energy U (molecules inside)
    molecule_positions = [(1.3, 2.5), (2.0, 3.5), (2.8, 2.8), (3.5, 3.2)]
    for x, y in molecule_positions:
        circle = Circle((x, y), 0.2, facecolor='#3498DB', edgecolor='black', linewidth=1)
        ax3.add_patch(circle)
    ax3.text(2.5, 2.0, r'Internal Energy $U$', ha='center', fontsize=9)
    
    # Apertures on boundary
    aperture_positions = [(0.5, 2.5), (0.5, 3.5), (4.5, 2.8), (2.5, 4.5)]
    for i, (x, y) in enumerate(aperture_positions):
        ap = Circle((x, y), 0.15, facecolor=aperture_color, edgecolor='black', linewidth=1.5)
        ax3.add_patch(ap)
        ax3.text(x, y - 0.35, f'$a_{i+1}$', ha='center', fontsize=7, color=aperture_color)
    
    # Formula on right
    ax3.text(7.5, 4.5, 'Categorical Enthalpy:', fontsize=11, fontweight='bold', ha='center')
    
    formula_box = FancyBboxPatch((5.5, 2.8), 4, 1.4, boxstyle="round,pad=0.1",
                                  facecolor='white', edgecolor=enthalpy_color, linewidth=2)
    ax3.add_patch(formula_box)
    ax3.text(7.5, 3.5, r'$\mathcal{H} = U + \sum_a n_a \Phi_a$', 
             ha='center', fontsize=14, fontweight='bold', color=enthalpy_color)
    
    # Component labels
    ax3.text(6.0, 2.2, r'$U$ = internal energy', fontsize=9)
    ax3.text(6.0, 1.7, r'$n_a$ = number of type-$a$ apertures', fontsize=9)
    ax3.text(6.0, 1.2, r'$\Phi_a$ = categorical potential of $a$', fontsize=9)
    ax3.text(6.0, 0.7, r'$\sum_a n_a \Phi_a$ = aperture energy', fontsize=9, fontweight='bold')
    
    # =========================================================================
    # Panel D: Classical Limit - Non-selective Apertures
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(-1, 11)
    ax4.set_ylim(-1, 6)
    ax4.axis('off')
    ax4.set_title('(D) Classical Limit: Non-Selective Apertures', fontsize=12,
                  fontweight='bold', pad=10)
    
    # Left: Many selective apertures
    ax4.text(2.5, 5.5, 'Selective Apertures', ha='center', fontsize=10, fontweight='bold')
    
    # Draw membrane with few apertures (selective)
    for i in range(3):
        y = 3.5 + i * 0.7
        rect = Rectangle((0.5, y - 0.1), 4, 0.2, facecolor='#7F8C8D', edgecolor='black')
        ax4.add_patch(rect)
        # Single small aperture
        circle = Circle((2.5, y), 0.15, facecolor=aperture_color, edgecolor='black')
        ax4.add_patch(circle)
    
    ax4.text(2.5, 2.8, r'$s_a < 1$ (selective)', ha='center', fontsize=9, color=aperture_color)
    ax4.text(2.5, 2.3, r'$\Phi_a > 0$ (barrier)', ha='center', fontsize=9)
    
    # Arrow to right
    ax4.annotate('', xy=(6.5, 4), xytext=(4.8, 4),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=3))
    ax4.text(5.65, 4.5, r'$s_a \to 1$', fontsize=10, ha='center')
    ax4.text(5.65, 3.5, r'$n_a \to \infty$', fontsize=10, ha='center')
    
    # Right: Infinitely many non-selective apertures
    ax4.text(8.5, 5.5, 'Non-Selective Limit', ha='center', fontsize=10, fontweight='bold')
    
    # Draw membrane with many apertures (non-selective = open)
    for i in range(3):
        y = 3.5 + i * 0.7
        # Dashed membrane (mostly apertures)
        for j in range(10):
            x = 6.5 + j * 0.4
            rect = Rectangle((x, y - 0.05), 0.15, 0.1, facecolor='#BDC3C7', edgecolor='none')
            ax4.add_patch(rect)
    
    ax4.text(8.5, 2.8, r'$s_a = 1$ (no selectivity)', ha='center', fontsize=9, color=pass_color)
    ax4.text(8.5, 2.3, r'$\Phi_a = 0$ (no barrier)', ha='center', fontsize=9)
    
    # Formula transformation at bottom
    ax4.text(5.5, 1.5, 'In the classical limit:', ha='center', fontsize=10, fontweight='bold')
    ax4.text(5.5, 0.8, r'$\sum_a n_a \Phi_a \to \int_{\partial\Omega} P \, dA = PV$', 
             ha='center', fontsize=12)
    # Boxed formula
    result_box = FancyBboxPatch((2, -0.3), 7, 0.7, boxstyle="round,pad=0.05",
                                 facecolor='#FEF9E7', edgecolor=enthalpy_color, linewidth=2)
    ax4.add_patch(result_box)
    ax4.text(5.5, 0.05, r'$\mathcal{H} \to U + PV = H_{classical}$', 
             ha='center', fontsize=13, fontweight='bold', color=enthalpy_color)
    
    # =========================================================================
    # Panel E: Pressure as Emergent Quantity
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(-1, 11)
    ax5.set_ylim(-1, 6)
    ax5.axis('off')
    ax5.set_title('(E) Pressure: Emergent from Aperture Statistics', fontsize=12,
                  fontweight='bold', pad=10)
    
    # Derivation chain
    steps = [
        (5, 5.2, r'Aperture density: $\rho_a$ per unit area', 10),
        (5, 4.4, r'Each aperture has potential: $\Phi_a = -k_B T \ln s_a$', 10),
        (5, 3.6, r'Total aperture contribution: $\rho_a \cdot A \cdot \Phi_a$', 10),
        (5, 2.6, r'Taking limits: $s_a \to 1$, $\rho_a \to \infty$', 10),
        (5, 1.8, r'Define: $P = \lim_{s_a \to 1} \rho_a \cdot (-k_B T \ln s_a)$', 11),
    ]
    
    for x, y, text, size in steps:
        ax5.text(x, y, text, ha='center', fontsize=size)
    
    # Arrow down
    ax5.annotate('', xy=(5, 0.9), xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Final result box
    result_box = FancyBboxPatch((1.5, -0.2), 7, 1.0, boxstyle="round,pad=0.1",
                                 facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=2)
    ax5.add_patch(result_box)
    ax5.text(5, 0.3, r'Pressure $P$ = coarse-grained aperture potential density', 
             ha='center', fontsize=11, fontweight='bold', color='#16A085')
    
    # Side note
    ax5.text(10, 3, 'Key insight:', fontsize=9, fontweight='bold', color='gray', rotation=90, va='center')
    
    # =========================================================================
    # Panel F: Enthalpy Formula Comparison
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(-1, 11)
    ax6.set_ylim(-1, 6)
    ax6.axis('off')
    ax6.set_title('(F) Enthalpy: From Categorical to Classical', fontsize=12,
                  fontweight='bold', pad=10)
    
    # Categorical formula (top)
    cat_box = FancyBboxPatch((0.5, 4), 9, 1.3, boxstyle="round,pad=0.1",
                              facecolor='#F5EEF8', edgecolor=aperture_color, linewidth=2)
    ax6.add_patch(cat_box)
    ax6.text(5, 5.0, 'CATEGORICAL (fundamental)', ha='center', fontsize=9, 
             fontweight='bold', color=aperture_color)
    ax6.text(5, 4.4, r'$\mathcal{H} = U + \int_{\partial\Omega} \sigma(x) \cdot \phi(x) \, dA$',
             ha='center', fontsize=12, color=aperture_color)
    
    # Arrow
    ax6.annotate('', xy=(5, 3.2), xytext=(5, 3.9),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    ax6.text(6, 3.55, r'$\sigma(x) \to 1$', fontsize=9)
    ax6.text(6, 3.25, r'$\phi(x) \to P$', fontsize=9)
    
    # Classical formula (bottom)
    class_box = FancyBboxPatch((0.5, 1.5), 9, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FEF9E7', edgecolor=enthalpy_color, linewidth=2)
    ax6.add_patch(class_box)
    ax6.text(5, 2.7, 'CLASSICAL (coarse-grained limit)', ha='center', fontsize=9, 
             fontweight='bold', color=enthalpy_color)
    ax6.text(5, 2.0, r'$H = U + PV$', ha='center', fontsize=14, fontweight='bold',
             color=enthalpy_color)
    
    # Key insight at bottom
    ax6.text(5, 0.6, 'Classical thermodynamics emerges as', ha='center', fontsize=10)
    ax6.text(5, 0.1, 'the coarse-grained limit of categorical aperture dynamics', 
             ha='center', fontsize=10, style='italic', color='#7F8C8D')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import os
    
    # Get the script directory and construct output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fig = create_categorical_enthalpy_panel()
    
    # Save in multiple formats
    fig.savefig(os.path.join(output_dir, 'panel3_categorical_enthalpy.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'panel3_categorical_enthalpy.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    print("Panel 3 (Categorical Enthalpy) saved successfully!")
    plt.close()

