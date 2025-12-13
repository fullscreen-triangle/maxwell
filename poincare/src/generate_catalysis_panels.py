"""
Catalysis Paper Panel Charts
Generates publication-quality panel charts for:
1. Three Paradoxes of Temporal Catalysis
2. Aperture Model (vs Maxwell's Demon)
3. Carbonic Anhydrase
4. Haber Process
5. Rubisco Multi-Aperture
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from pathlib import Path
import json

# Style configuration
COLORS = {
    'reactant': '#3498DB',      # Blue
    'product': '#E67E22',       # Orange
    'enzyme': '#9B59B6',        # Purple
    'catalyst': '#1ABC9C',      # Teal
    'barrier': '#E74C3C',       # Red
    'lowered': '#27AE60',       # Green
    'iron': '#7F8C8D',          # Gray
    'nitrogen': '#3498DB',      # Blue
    'hydrogen': '#F1C40F',      # Yellow
    'ammonia': '#27AE60',       # Green
    'zinc': '#9B59B6',          # Purple
    'co2': '#95A5A6',           # Light gray
    'rubisco': '#E74C3C',       # Red
    'oxygen': '#E74C3C',        # Red
    'aperture': '#1ABC9C',      # Teal
    'demon': '#E74C3C',         # Red
    'temporal': '#E74C3C',      # Red
    'categorical': '#27AE60',   # Green
    'primary': '#2C3E50',
    'background': '#FAFAFA'
}


def setup_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.linewidth': 1.0,
    })


def generate_three_paradoxes_panel(output_dir: str):
    """Generate panel chart for the three paradoxes of temporal catalysis."""
    setup_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Instantaneous Concentration Paradox - Theory
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # If temporal acceleration, v should go to infinity
    S = np.linspace(0.1, 100, 100)
    v_temporal = S * 10  # Linear increase (temporal model)
    v_actual = 100 * S / (10 + S)  # Michaelis-Menten
    
    ax1.plot(S, v_temporal, '--', color=COLORS['temporal'], lw=2, 
             label='Temporal model: v → ∞')
    ax1.plot(S, v_actual, '-', color=COLORS['categorical'], lw=2.5,
             label='Observed: v → Vmax')
    ax1.axhline(y=100, color=COLORS['categorical'], linestyle=':', lw=1.5)
    ax1.text(80, 105, 'Vmax', fontsize=9, color=COLORS['categorical'])
    
    ax1.set_xlabel('[S] (substrate concentration)')
    ax1.set_ylabel('Reaction velocity v')
    ax1.legend(loc='center right')
    ax1.set_title('A. Instantaneous Concentration Paradox\n(Temporal model predicts v → ∞)',
                 fontweight='bold')
    
    # =========================================================================
    # Panel B: Instantaneous Concentration - Categorical Resolution
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Show categorical steps
    steps = [(2, 5), (4, 5), (6, 5), (8, 5)]
    labels = ['S', 'ES', 'EP', 'P']
    colors_steps = [COLORS['reactant'], COLORS['enzyme'], COLORS['enzyme'], COLORS['product']]
    
    for i, ((x, y), label, color) in enumerate(zip(steps, labels, colors_steps)):
        ax2.add_patch(Circle((x, y), 0.5, color=color))
        ax2.text(x, y, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        if i < len(steps) - 1:
            ax2.annotate('', xy=(steps[i+1][0] - 0.6, steps[i+1][1]),
                        xytext=(x + 0.6, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    ax2.text(5, 8, 'dcat = 3 (fixed)', fontsize=11, ha='center', 
            fontweight='bold', color=COLORS['categorical'])
    ax2.text(5, 7, 'Cannot be reduced by [S]', fontsize=9, ha='center')
    
    ax2.text(5, 2, 'Vmax = [E]total / (dcat · τstep)', fontsize=10, 
            ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax2.set_title('B. Categorical Resolution\n(dcat is fixed, not reducible)', fontweight='bold')
    ax2.axis('off')
    
    # =========================================================================
    # Panel C: Reversible Reaction Paradox
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Reactants and Products
    ax3.add_patch(Circle((2, 5), 0.8, color=COLORS['reactant']))
    ax3.text(2, 5, 'A', ha='center', va='center', fontsize=14, 
            fontweight='bold', color='white')
    
    ax3.add_patch(Circle((8, 5), 0.8, color=COLORS['product']))
    ax3.text(8, 5, 'B', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')
    
    # Forward arrow (top)
    ax3.annotate('', xy=(7, 6.5), xytext=(3, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['categorical'],
                               connectionstyle='arc3,rad=0.2'))
    ax3.text(5, 7.8, 'kf (faster)', fontsize=9, ha='center', color=COLORS['categorical'])
    
    # Reverse arrow (bottom)
    ax3.annotate('', xy=(3, 3.5), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['categorical'],
                               connectionstyle='arc3,rad=0.2'))
    ax3.text(5, 2.2, 'kr (also faster!)', fontsize=9, ha='center', color=COLORS['categorical'])
    
    # Keq unchanged
    ax3.text(5, 9, 'Keq = kf/kr = UNCHANGED', fontsize=10, ha='center',
            fontweight='bold', color=COLORS['primary'],
            bbox=dict(boxstyle='round', facecolor='#FEF9E7'))
    
    ax3.text(5, 0.8, 'Both directions use SAME pathway\n→ Keq automatically preserved',
            fontsize=8, ha='center', style='italic')
    
    ax3.set_title('C. Reversible Reaction Paradox\n(Time can\'t flow both ways)', fontweight='bold')
    ax3.axis('off')
    
    # =========================================================================
    # Panel D: Step-Exclusion Paradox - Same Steps Faster?
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # Uncatalyzed pathway
    ax4.text(1, 9, 'Uncatalyzed:', fontsize=9, fontweight='bold')
    uncat_steps = [(1, 7), (3, 7), (5, 7), (7, 7)]
    for i, (x, y) in enumerate(uncat_steps):
        ax4.add_patch(Circle((x, y), 0.4, color=COLORS['temporal'], alpha=0.7))
        ax4.text(x, y, chr(65+i), ha='center', va='center', fontsize=8, color='white')
        if i < len(uncat_steps) - 1:
            ax4.annotate('', xy=(uncat_steps[i+1][0] - 0.5, y),
                        xytext=(x + 0.5, y),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['temporal']))
    
    # "Faster" same steps - with question
    ax4.text(1, 5, '"Same steps faster":', fontsize=9, fontweight='bold')
    for i, (x, y) in enumerate([(1, 3), (3, 3), (5, 3), (7, 3)]):
        ax4.add_patch(Circle((x, y), 0.4, color=COLORS['temporal'], alpha=0.4))
        ax4.text(x, y, chr(65+i), ha='center', va='center', fontsize=8, color='white')
    
    ax4.text(8.5, 3, '?', fontsize=20, color=COLORS['temporal'], fontweight='bold')
    ax4.text(5, 1, 'Where does stabilization\nenergy come from?', 
            fontsize=9, ha='center', color=COLORS['temporal'],
            bbox=dict(boxstyle='round', facecolor='#FADBD8'))
    
    ax4.set_title('D. Step-Exclusion Paradox (Case 1)\n(Same steps faster requires energy)',
                 fontweight='bold')
    ax4.axis('off')
    
    # =========================================================================
    # Panel E: Step-Exclusion Paradox - Skip Steps?
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    
    # Uncatalyzed pathway
    ax5.text(1, 9, 'Uncatalyzed:', fontsize=9, fontweight='bold')
    for i, (x, y) in enumerate([(1, 7), (3, 7), (5, 7), (7, 7)]):
        ax5.add_patch(Circle((x, y), 0.4, color=COLORS['temporal'], alpha=0.7))
        ax5.text(x, y, chr(65+i), ha='center', va='center', fontsize=8, color='white')
        if i < 3:
            ax5.annotate('', xy=(x + 1.5, y), xytext=(x + 0.5, y),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['temporal']))
    
    # "Skip steps" pathway
    ax5.text(1, 5, '"Skip steps":', fontsize=9, fontweight='bold')
    ax5.add_patch(Circle((1, 3), 0.4, color=COLORS['temporal'], alpha=0.7))
    ax5.text(1, 3, 'A', ha='center', va='center', fontsize=8, color='white')
    ax5.add_patch(Circle((7, 3), 0.4, color=COLORS['temporal'], alpha=0.7))
    ax5.text(7, 3, 'D', ha='center', va='center', fontsize=8, color='white')
    ax5.annotate('', xy=(6.5, 3), xytext=(1.5, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['temporal']))
    
    # X through B and C
    ax5.plot([2.5, 5.5], [2.5, 3.5], '-', color=COLORS['temporal'], lw=2)
    ax5.plot([2.5, 5.5], [3.5, 2.5], '-', color=COLORS['temporal'], lw=2)
    ax5.text(4, 3, 'B, C', fontsize=8, ha='center', color=COLORS['temporal'], alpha=0.5)
    
    ax5.text(5, 1, 'If B, C can be skipped,\nwhy are they in uncatalyzed path?',
            fontsize=9, ha='center', color=COLORS['temporal'],
            bbox=dict(boxstyle='round', facecolor='#FADBD8'))
    
    ax5.set_title('E. Step-Exclusion Paradox (Case 2)\n(Skip steps → they were unnecessary?)',
                 fontweight='bold')
    ax5.axis('off')
    
    # =========================================================================
    # Panel F: Categorical Resolution - New Pathway
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    # Uncatalyzed (top, faded)
    ax6.text(1, 9, 'Uncatalyzed:', fontsize=8, alpha=0.5)
    for i, (x, y) in enumerate([(1, 7.5), (3, 7.5), (5, 7.5), (7, 7.5)]):
        ax6.add_patch(Circle((x, y), 0.3, color=COLORS['temporal'], alpha=0.3))
    
    # Catalyzed - NEW pathway
    ax6.text(1, 5.5, 'Catalyzed (NEW pathway):', fontsize=9, fontweight='bold',
            color=COLORS['categorical'])
    cat_steps = [(1, 4), (2.5, 4), (4, 4), (5.5, 4), (7, 4), (8.5, 4)]
    cat_labels = ['S', 'E·S', 'E·I₁', 'E·I₂', 'E·P', 'P']
    
    for i, ((x, y), label) in enumerate(zip(cat_steps, cat_labels)):
        color = COLORS['enzyme'] if 'E' in label else COLORS['categorical']
        ax6.add_patch(Circle((x, y), 0.35, color=color, alpha=0.8))
        ax6.text(x, y, label[:3], ha='center', va='center', fontsize=6, 
                color='white', fontweight='bold')
        if i < len(cat_steps) - 1:
            ax6.annotate('', xy=(cat_steps[i+1][0] - 0.4, y),
                        xytext=(x + 0.4, y),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['categorical']))
    
    ax6.text(5, 2, 'Enzyme creates NEW intermediate states\n(E·S, E·I₁, E·I₂, E·P)',
            fontsize=9, ha='center', color=COLORS['categorical'],
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax6.text(5, 0.8, 'Different categorical space, not faster same space',
            fontsize=8, ha='center', style='italic')
    
    ax6.set_title('F. Categorical Resolution\n(New pathway, not faster old pathway)',
                 fontweight='bold')
    ax6.axis('off')
    
    # Main title
    plt.suptitle('Three Paradoxes of Temporal Catalysis and Their Categorical Resolution',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "three_paradoxes_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_aperture_model_panel(output_dir: str):
    """Generate panel chart for the aperture model vs Maxwell's demon."""
    setup_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Maxwell's Demon (information processing)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Two chambers
    ax1.add_patch(FancyBboxPatch((0.5, 2), 4, 6, boxstyle="round",
                                  facecolor='#EBF5FB', edgecolor=COLORS['primary']))
    ax1.add_patch(FancyBboxPatch((5.5, 2), 4, 6, boxstyle="round",
                                  facecolor='#FDEDEC', edgecolor=COLORS['primary']))
    
    # Demon at door
    ax1.add_patch(Circle((5, 5), 0.6, color=COLORS['demon']))
    ax1.text(5, 5, 'D', ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
    ax1.text(5, 3.5, 'Demon', fontsize=8, ha='center', color=COLORS['demon'])
    
    # Molecules with velocities
    fast_pos = [(1.5, 6), (3, 4)]
    slow_pos = [(2, 7), (3.5, 5.5)]
    
    for x, y in fast_pos:
        ax1.add_patch(Circle((x, y), 0.25, color=COLORS['demon']))
        ax1.annotate('', xy=(x+0.5, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['demon'], lw=1.5))
    
    for x, y in slow_pos:
        ax1.add_patch(Circle((x, y), 0.25, color=COLORS['categorical']))
        ax1.annotate('', xy=(x+0.2, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['categorical'], lw=1))
    
    ax1.text(5, 9, 'Selects by VELOCITY', fontsize=10, ha='center',
            fontweight='bold', color=COLORS['demon'])
    ax1.text(5, 1, 'Requires: measurement, memory, erasure',
            fontsize=8, ha='center', style='italic')
    
    ax1.set_title('A. Maxwell\'s Demon\n(Information Processing)', fontweight='bold')
    ax1.axis('off')
    
    # =========================================================================
    # Panel B: Categorical Aperture (geometric selection)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Two chambers
    ax2.add_patch(FancyBboxPatch((0.5, 2), 4, 6, boxstyle="round",
                                  facecolor='#E8F8F5', edgecolor=COLORS['primary']))
    ax2.add_patch(FancyBboxPatch((5.5, 2), 4, 6, boxstyle="round",
                                  facecolor='#E8F8F5', edgecolor=COLORS['primary']))
    
    # Shaped aperture
    aperture_x = [4.5, 5.5, 5.5, 5.2, 5.2, 4.8, 4.8, 4.5]
    aperture_y = [3, 3, 7, 7, 5.5, 5.5, 7, 7]
    ax2.fill(aperture_x, aperture_y, color=COLORS['aperture'], alpha=0.8)
    ax2.text(5, 4.2, 'Aperture', fontsize=7, ha='center', color='white', rotation=90)
    
    # Molecules with shapes
    # Fitting shape (triangle)
    triangle = Polygon([(1.5, 5), (2, 6), (2.5, 5)], color=COLORS['categorical'])
    ax2.add_patch(triangle)
    ax2.annotate('', xy=(4.3, 5.3), xytext=(2.7, 5.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['categorical'], lw=2))
    
    # Non-fitting shape (large square)
    ax2.add_patch(FancyBboxPatch((1, 6.5), 1.5, 1.5, boxstyle="round",
                                  facecolor=COLORS['demon'], alpha=0.7))
    ax2.plot([1.75, 4.5], [7.25, 6], '--', color=COLORS['demon'], alpha=0.5)
    ax2.text(3.2, 6.8, '✗', fontsize=14, color=COLORS['demon'])
    
    ax2.text(5, 9, 'Selects by CONFIGURATION', fontsize=10, ha='center',
            fontweight='bold', color=COLORS['categorical'])
    ax2.text(5, 1, 'No measurement, no memory, no erasure',
            fontsize=8, ha='center', style='italic')
    
    ax2.set_title('B. Categorical Aperture\n(Geometric Selection)', fontweight='bold')
    ax2.axis('off')
    
    # =========================================================================
    # Panel C: Information Content Comparison
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    categories = ['Demon', 'Aperture']
    info_acquired = [1, 0]  # bits
    
    bars = ax3.bar(categories, info_acquired, 
                   color=[COLORS['demon'], COLORS['categorical']],
                   edgecolor='black', linewidth=1)
    
    ax3.set_ylabel('Shannon Information (bits)')
    ax3.set_ylim(0, 1.5)
    ax3.axhline(y=0, color='black', lw=0.5)
    
    # Annotations
    ax3.text(0, 1.1, 'I > 0\nMust erase', ha='center', fontsize=9, color=COLORS['demon'])
    ax3.text(1, 0.15, 'I = 0\nNo erasure', ha='center', fontsize=9, color=COLORS['categorical'])
    
    ax3.set_title('C. Information Acquired\n(Zero for apertures)', fontweight='bold')
    
    # =========================================================================
    # Panel D: Enzyme Active Site as Aperture
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # Enzyme pocket
    pocket = Wedge((5, 5), 3, 200, 340, width=1, color=COLORS['enzyme'], alpha=0.7)
    ax4.add_patch(pocket)
    
    # Active site features
    ax4.add_patch(Circle((3.5, 4), 0.3, color=COLORS['demon']))  # H-bond acceptor
    ax4.add_patch(Circle((6.5, 4), 0.3, color=COLORS['categorical']))  # H-bond donor
    ax4.add_patch(Circle((5, 3), 0.3, color=COLORS['zinc']))  # Catalytic residue
    
    ax4.text(3.5, 3.3, 'A', fontsize=7, ha='center')
    ax4.text(6.5, 3.3, 'D', fontsize=7, ha='center')
    ax4.text(5, 2.3, 'Cat', fontsize=7, ha='center')
    
    # Substrate (complementary shape)
    substrate = Polygon([(4, 5.5), (6, 5.5), (5.5, 7), (4.5, 7)], 
                        color=COLORS['reactant'], alpha=0.8)
    ax4.add_patch(substrate)
    ax4.text(5, 6.2, 'S', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')
    
    ax4.text(5, 9, 'Enzyme = Shaped Aperture', fontsize=10, ha='center',
            fontweight='bold')
    ax4.text(5, 1, 'Substrate fits or doesn\'t (geometric)',
            fontsize=9, ha='center', style='italic')
    
    ax4.set_title('D. Enzyme Active Site\n= Categorical Aperture', fontweight='bold')
    ax4.axis('off')
    
    # =========================================================================
    # Panel E: Topological Completion
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    
    # Before completion
    ax5.text(2.5, 9, 'Before', fontsize=9, ha='center', fontweight='bold')
    # Enzyme
    ax5.add_patch(Wedge((2.5, 6), 1.5, 180, 360, color=COLORS['enzyme'], alpha=0.7))
    # Substrate approaching
    ax5.add_patch(Circle((2.5, 8), 0.5, color=COLORS['reactant']))
    ax5.annotate('', xy=(2.5, 7.2), xytext=(2.5, 7.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
    
    # After completion
    ax5.text(7.5, 9, 'After', fontsize=9, ha='center', fontweight='bold')
    # Enzyme + substrate complex
    ax5.add_patch(Wedge((7.5, 6), 1.5, 180, 360, color=COLORS['enzyme'], alpha=0.7))
    ax5.add_patch(Circle((7.5, 6.8), 0.5, color=COLORS['reactant']))
    
    # Completion indicator
    ax5.plot([6, 9], [6.5, 6.5], '-', color=COLORS['categorical'], lw=2)
    ax5.text(7.5, 5, 'Topology\nCOMPLETED', ha='center', fontsize=9,
            color=COLORS['categorical'], fontweight='bold')
    
    ax5.text(5, 2, 'Substrate completes enzyme topology\n→ Reaction proceeds',
            ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax5.set_title('E. Topological Completion\n(Fit enables reaction)', fontweight='bold')
    ax5.axis('off')
    
    # =========================================================================
    # Panel F: Comparison Table
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    # Table data
    rows = [
        ['Property', 'Demon', 'Aperture'],
        ['Selection', 'Velocity', 'Config.'],
        ['Measurement', 'Yes', 'No'],
        ['Information', 'I > 0', 'I = 0'],
        ['Memory', 'Required', 'None'],
        ['Erasure cost', 'kT ln2', '0'],
        ['Paradox', 'Yes', 'No'],
    ]
    
    for i, row in enumerate(rows):
        y = 8.5 - i * 1.1
        for j, cell in enumerate(row):
            x = 1.5 + j * 2.8
            if i == 0:
                ax6.text(x, y, cell, ha='center', fontsize=9, fontweight='bold')
            elif j == 1 and i > 0:
                ax6.text(x, y, cell, ha='center', fontsize=8, color=COLORS['demon'])
            elif j == 2 and i > 0:
                ax6.text(x, y, cell, ha='center', fontsize=8, color=COLORS['categorical'])
            else:
                ax6.text(x, y, cell, ha='center', fontsize=8)
    
    # Table lines
    ax6.plot([0.5, 9.5], [7.8, 7.8], 'k-', lw=1)
    ax6.plot([3.5, 3.5], [1, 9], 'k-', lw=0.5)
    ax6.plot([6.3, 6.3], [1, 9], 'k-', lw=0.5)
    
    ax6.text(5, 0.8, 'Enzymes are apertures, NOT demons',
            ha='center', fontsize=10, fontweight='bold',
            color=COLORS['categorical'])
    
    ax6.set_title('F. Property Comparison', fontweight='bold')
    ax6.axis('off')
    
    # Main title
    plt.suptitle('Categorical Aperture Model: Geometric Selection Without Information Processing',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir) / "aperture_model_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_carbonic_anhydrase_panel(output_dir: str):
    """Generate panel chart for carbonic anhydrase."""
    setup_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Active Site Geometry
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Zn2+ center
    ax1.add_patch(Circle((5, 5), 0.6, color=COLORS['zinc']))
    ax1.text(5, 5, 'Zn²⁺', ha='center', va='center', fontsize=9, 
            fontweight='bold', color='white')
    
    # Three His ligands
    his_pos = [(3, 3.5), (7, 3.5), (5, 2)]
    for i, (x, y) in enumerate(his_pos):
        ax1.add_patch(Circle((x, y), 0.4, color=COLORS['enzyme']))
        ax1.text(x, y, 'His', ha='center', va='center', fontsize=7, color='white')
        ax1.plot([x, 5], [y + 0.4, 5 - 0.5], '-', color=COLORS['zinc'], lw=2)
    
    # OH- ligand
    ax1.add_patch(Circle((5, 7), 0.4, color=COLORS['categorical']))
    ax1.text(5, 7, 'OH⁻', ha='center', va='center', fontsize=7, 
            fontweight='bold', color='white')
    ax1.plot([5, 5], [6.6, 5.6], '-', color=COLORS['zinc'], lw=2)
    
    # Distances
    ax1.text(3.8, 4.2, '2.0Å', fontsize=7, color=COLORS['zinc'])
    ax1.text(5.3, 6, '1.9Å', fontsize=7, color=COLORS['zinc'])
    
    ax1.text(5, 9, 'Tetrahedral Coordination', fontsize=10, ha='center', fontweight='bold')
    
    ax1.set_title('A. Zn²⁺ Active Site Geometry', fontweight='bold')
    ax1.axis('off')
    
    # =========================================================================
    # Panel B: Phase-Lock Network
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Network nodes
    nodes = [
        (1.5, 5, 'CO₂', COLORS['co2']),
        (3.5, 5, 'Zn-OH⁻', COLORS['zinc']),
        (6, 5, 'His64', COLORS['enzyme']),
        (8.5, 5, 'H₂O', COLORS['categorical']),
    ]
    
    for x, y, label, color in nodes:
        ax2.add_patch(Circle((x, y), 0.6, color=color))
        ax2.text(x, y, label[:4], ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white' if color != COLORS['co2'] else 'black')
    
    # Phase-lock arrows
    arrows = [('attack', 2.3, 3.9), ('transfer', 4.7, 5.3), ('release', 6.8, 7.7)]
    for label, x1, x2 in arrows:
        ax2.annotate('', xy=(x2, 5), xytext=(x1, 5),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['aperture'], lw=2))
        ax2.text((x1 + x2) / 2, 6.2, label, ha='center', fontsize=8)
    
    # His64 distance
    ax2.text(4.7, 3.5, '~7 Å', fontsize=9, ha='center', fontweight='bold',
            color=COLORS['enzyme'])
    
    ax2.set_title('B. Phase-Lock Network', fontweight='bold')
    ax2.axis('off')
    
    # =========================================================================
    # Panel C: Rate Comparison
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    enzymes = ['Uncatalyzed', 'CA']
    rates = [0.03, 1e6]
    
    bars = ax3.bar(enzymes, rates, color=[COLORS['temporal'], COLORS['categorical']],
                   edgecolor='black', log=True)
    
    ax3.set_ylabel('Rate constant (s⁻¹)')
    ax3.set_ylim(0.01, 1e8)
    
    # Enhancement factor
    ax3.text(0.5, 1e4, '3×10⁷\nenhancement', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['categorical'])
    
    ax3.set_title('C. Rate Enhancement\n(From geometry, not time)', fontweight='bold')
    
    # =========================================================================
    # Panel D: Categorical Steps
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    steps = [
        (1.5, 7, 'H₂O→OH⁻', 'Water\nactivation'),
        (5, 7, 'CO₂+OH⁻', 'Nucleophilic\nattack'),
        (8.5, 7, 'HCO₃⁻', 'Product\nrelease'),
    ]
    
    for i, (x, y, label, desc) in enumerate(steps):
        ax4.add_patch(Circle((x, y), 0.7, color=COLORS['categorical']))
        ax4.text(x, y, f'C{i+1}', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        ax4.text(x, y - 1.5, label, ha='center', fontsize=8)
        ax4.text(x, y - 2.3, desc, ha='center', fontsize=7, style='italic')
        if i < len(steps) - 1:
            ax4.annotate('', xy=(steps[i+1][0] - 0.8, y),
                        xytext=(x + 0.8, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    ax4.text(5, 2, 'dcat = 3 (three categorical steps)', fontsize=10, ha='center',
            fontweight='bold', color=COLORS['categorical'],
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax4.set_title('D. Categorical Distance = 3', fontweight='bold')
    ax4.axis('off')
    
    # =========================================================================
    # Panel E: His64 Proton Shuttle
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    
    # Zn site
    ax5.add_patch(Circle((2, 5), 0.5, color=COLORS['zinc']))
    ax5.text(2, 5, 'Zn', ha='center', va='center', fontsize=9, color='white')
    
    # His64
    ax5.add_patch(Circle((7, 5), 0.5, color=COLORS['enzyme']))
    ax5.text(7, 5, 'H64', ha='center', va='center', fontsize=8, color='white')
    
    # 7 Å distance
    ax5.annotate('', xy=(6.4, 5), xytext=(2.6, 5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['aperture'], lw=2))
    ax5.text(4.5, 5.8, '7 Å', fontsize=11, ha='center', fontweight='bold')
    
    # Proton transfer
    ax5.annotate('', xy=(6.5, 4), xytext=(2.5, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['demon'], lw=1.5,
                               connectionstyle='arc3,rad=0.3'))
    ax5.text(4.5, 2.8, 'H⁺ transfer\n(rate-limiting)', ha='center', fontsize=9)
    
    # Bulk water
    ax5.add_patch(Circle((9, 5), 0.4, color=COLORS['categorical']))
    ax5.text(9, 5, 'H₂O', ha='center', va='center', fontsize=7, color='white')
    ax5.annotate('', xy=(8.6, 5), xytext=(7.5, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['categorical'], lw=1.5))
    
    ax5.text(5, 8.5, 'Optimal spacing: 7 Å', fontsize=10, ha='center', fontweight='bold')
    ax5.text(5, 1, 'Shorter: steric clash\nLonger: increased dcat',
            ha='center', fontsize=8, style='italic')
    
    ax5.set_title('E. His64 Proton Shuttle\n(Geometry determines speed)', fontweight='bold')
    ax5.axis('off')
    
    # =========================================================================
    # Panel F: Speed = Geometry
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    # Equation
    ax6.text(5, 8, 'kcat = 1 / (dcat · τstep)', fontsize=12, ha='center',
            fontfamily='monospace')
    
    ax6.text(5, 6.5, 'For CA:', fontsize=10, ha='center', fontweight='bold')
    ax6.text(5, 5.5, 'dcat = 3', fontsize=10, ha='center')
    ax6.text(5, 4.5, 'τstep ≈ 3×10⁻⁷ s', fontsize=10, ha='center')
    
    ax6.text(5, 3, 'kcat = 1/(3 × 3×10⁻⁷)', fontsize=10, ha='center',
            fontfamily='monospace')
    ax6.text(5, 2, '≈ 10⁶ s⁻¹', fontsize=14, ha='center', 
            fontweight='bold', color=COLORS['categorical'])
    
    ax6.add_patch(FancyBboxPatch((1.5, 0.5), 7, 1.5, boxstyle="round",
                                  facecolor='#E8F8F5', edgecolor=COLORS['categorical']))
    ax6.text(5, 1.2, 'Speed from GEOMETRY', fontsize=11, ha='center',
            fontweight='bold', color=COLORS['categorical'])
    
    ax6.set_title('F. Speed = Optimal Geometry', fontweight='bold')
    ax6.axis('off')
    
    # Main title
    plt.suptitle('Carbonic Anhydrase: 10⁶ s⁻¹ Through Geometric Optimization',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir) / "carbonic_anhydrase_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_haber_process_panel(output_dir: str):
    """Generate panel chart for the Haber process."""
    setup_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Uncatalyzed - No Pathway
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Reactants
    ax1.add_patch(Circle((2, 7), 0.5, color=COLORS['nitrogen']))
    ax1.text(2, 7, 'N₂', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')
    
    ax1.add_patch(Circle((2, 5), 0.4, color=COLORS['hydrogen']))
    ax1.text(2, 5, 'H₂', ha='center', va='center', fontsize=9, color='black')
    
    # Product
    ax1.add_patch(Circle((8, 6), 0.6, color=COLORS['ammonia']))
    ax1.text(8, 6, 'NH₃', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')
    
    # Huge barrier
    barrier_x = [3.5, 4, 5.5, 6, 6.5]
    barrier_y = [6, 9, 9, 9, 6]
    ax1.fill(barrier_x, barrier_y, color=COLORS['barrier'], alpha=0.5)
    ax1.text(5, 8, 'N≡N\n945 kJ/mol', ha='center', fontsize=8, 
            fontweight='bold', color='white')
    
    # X through arrow
    ax1.plot([3, 7], [5, 7], '--', color=COLORS['barrier'], lw=2, alpha=0.5)
    ax1.text(5, 5.5, '✗', fontsize=20, ha='center', color=COLORS['barrier'])
    
    ax1.text(5, 2, 'dcat = ∞\nNo pathway exists', fontsize=10, ha='center',
            color=COLORS['barrier'], fontweight='bold')
    
    ax1.set_title('A. Uncatalyzed: Infinite Distance', fontweight='bold')
    ax1.axis('off')
    
    # =========================================================================
    # Panel B: Iron Surface Apertures
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Iron surface
    for i in range(8):
        for j in range(3):
            ax2.add_patch(Circle((1 + i * 1.1, 2 + j * 1.1), 0.4, 
                                 color=COLORS['iron'], alpha=0.8))
    
    ax2.text(5, 0.5, 'Fe Surface', fontsize=10, ha='center', fontweight='bold')
    
    # N2 adsorbing
    ax2.add_patch(Circle((3.2, 5.5), 0.3, color=COLORS['nitrogen']))
    ax2.add_patch(Circle((3.8, 5.5), 0.3, color=COLORS['nitrogen']))
    ax2.plot([3.2, 3.8], [5.5, 5.5], '-', color=COLORS['nitrogen'], lw=3)
    ax2.annotate('', xy=(3.5, 4.5), xytext=(3.5, 5.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax2.text(3.5, 6.2, 'N₂ adsorbs', fontsize=8, ha='center')
    
    # H2 dissociated
    ax2.add_patch(Circle((6.5, 4.5), 0.2, color=COLORS['hydrogen']))
    ax2.add_patch(Circle((7.5, 4.5), 0.2, color=COLORS['hydrogen']))
    ax2.text(7, 5.2, 'H atoms', fontsize=8, ha='center')
    
    ax2.text(5, 8.5, 'Fe creates APERTURES\nfor adsorption & dissociation',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['iron'])
    
    ax2.set_title('B. Iron Surface: Aperture Creation', fontweight='bold')
    ax2.axis('off')
    
    # =========================================================================
    # Panel C: Categorical Pathway
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Pathway steps
    steps = [
        (1, 8, 'N₂(g)'),
        (3, 8, 'N₂*'),
        (5, 8, '2N*'),
        (1, 5, 'H₂*'),
        (3, 5, '2H*'),
        (5, 5, 'NH*'),
        (7, 5, 'NH₂*'),
        (9, 5, 'NH₃*'),
        (9, 2, 'NH₃(g)'),
    ]
    
    colors = [COLORS['nitrogen'], COLORS['nitrogen'], COLORS['nitrogen'],
              COLORS['hydrogen'], COLORS['hydrogen'],
              COLORS['ammonia'], COLORS['ammonia'], COLORS['ammonia'], COLORS['ammonia']]
    
    for (x, y, label), color in zip(steps, colors):
        ax3.add_patch(Circle((x, y), 0.4, color=color, alpha=0.8))
        ax3.text(x, y - 0.8, label, ha='center', fontsize=7)
    
    # Arrows
    arrow_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    for i, j in arrow_pairs:
        x1, y1, _ = steps[i]
        x2, y2, _ = steps[j]
        ax3.annotate('', xy=(x2 - 0.3, y2), xytext=(x1 + 0.3, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1))
    
    # N dissociation arrow to NH formation
    ax3.annotate('', xy=(5, 5.5), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1,
                               connectionstyle='arc3,rad=0.3'))
    
    ax3.text(5, 1, 'dcat ≈ 8 (finite!)', fontsize=11, ha='center',
            fontweight='bold', color=COLORS['categorical'])
    
    ax3.set_title('C. Catalyzed Pathway: dcat = 8', fontweight='bold')
    ax3.axis('off')
    
    # =========================================================================
    # Panel D: Rate-Limiting Step
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Energy diagram
    x = np.linspace(0, 10, 100)
    y = 2 + 4 * np.exp(-((x - 2.5)**2) / 0.8)  # N2 dissociation barrier
    
    ax4.plot(x, y, '-', color=COLORS['iron'], lw=2.5)
    ax4.fill_between(x, 0, y, alpha=0.2, color=COLORS['iron'])
    
    ax4.axhline(y=2, color=COLORS['categorical'], linestyle='--', lw=1)
    
    # Labels
    ax4.text(1, 1.5, 'N₂*', fontsize=10, fontweight='bold')
    ax4.text(4.5, 1.5, '2N*', fontsize=10, fontweight='bold')
    ax4.text(2.5, 6.5, 'N≡N\ndissociation', ha='center', fontsize=9)
    
    ax4.set_xlabel('Reaction coordinate')
    ax4.set_ylabel('Energy')
    ax4.set_ylim(0, 8)
    
    ax4.set_title('D. Rate-Limiting Step\n(N₂ dissociation on surface)', fontweight='bold')
    
    # =========================================================================
    # Panel E: Crystal Face Dependence
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    faces = ['Fe(111)', 'Fe(100)', 'Fe(110)']
    activities = [100, 25, 1]
    
    bars = ax5.bar(faces, activities, 
                   color=[COLORS['categorical'], COLORS['iron'], COLORS['temporal']],
                   edgecolor='black')
    
    ax5.set_ylabel('Relative Activity')
    ax5.set_ylim(0, 120)
    
    # Annotations
    ax5.text(0, 105, 'C7 sites\n(optimal)', ha='center', fontsize=8, color=COLORS['categorical'])
    ax5.text(1, 30, '4-fold\nsites', ha='center', fontsize=8)
    ax5.text(2, 6, 'Poor\ngeometry', ha='center', fontsize=8, color=COLORS['temporal'])
    
    ax5.set_title('E. Crystal Face Activity\n(Geometry determines activity)', fontweight='bold')
    
    # =========================================================================
    # Panel F: Summary
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    # Summary box
    summary_lines = [
        ('Without Fe:', 'dcat = ∞', COLORS['barrier']),
        ('With Fe:', 'dcat = 8', COLORS['categorical']),
        ('', '', 'black'),
        ('Iron creates:', '', COLORS['iron']),
        ('• Adsorption sites', '', COLORS['iron']),
        ('• Dissociation apertures', '', COLORS['iron']),
        ('• Reaction pathway', '', COLORS['iron']),
    ]
    
    y_pos = 8
    for left, right, color in summary_lines:
        if left:
            ax6.text(2, y_pos, left, fontsize=10, color=color, fontweight='bold')
        if right:
            ax6.text(6, y_pos, right, fontsize=10, color=color, fontweight='bold')
        y_pos -= 0.9
    
    ax6.add_patch(FancyBboxPatch((1, 0.5), 8, 2, boxstyle="round",
                                  facecolor='#E8F8F5', edgecolor=COLORS['categorical']))
    ax6.text(5, 1.5, 'Fe makes the reaction EXIST', fontsize=12, ha='center',
            fontweight='bold', color=COLORS['categorical'])
    
    ax6.set_title('F. Iron Creates Categorical Space', fontweight='bold')
    ax6.axis('off')
    
    # Main title
    plt.suptitle('Haber Process: Iron Creates Pathway Where None Existed (dcat: ∞ → 8)',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir) / "haber_process_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_rubisco_panel(output_dir: str):
    """Generate panel chart for Rubisco."""
    setup_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: The "Inefficiency" Claim
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    enzymes = ['Catalase', 'CA', 'Chymo.', 'Rubisco']
    kcat_values = [4e7, 1e6, 1e2, 10]
    colors = [COLORS['categorical'], COLORS['enzyme'], COLORS['iron'], COLORS['rubisco']]
    
    bars = ax1.bar(enzymes, kcat_values, color=colors, edgecolor='black', log=True)
    
    ax1.set_ylabel('kcat (s⁻¹)')
    ax1.set_ylim(1, 1e9)
    
    # "Inefficient" label with X
    ax1.text(3, 100, '"Inefficient"?', fontsize=10, ha='center', 
            color=COLORS['rubisco'], fontweight='bold')
    ax1.text(3, 20, '✗', fontsize=16, ha='center', color=COLORS['rubisco'])
    
    ax1.set_title('A. Traditional "Efficiency" Ranking\n(Misleading comparison)', fontweight='bold')
    
    # =========================================================================
    # Panel B: Categorical Distance Comparison
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    enzymes = ['Catalase', 'CA', 'Chymo.', 'Rubisco']
    dcat_values = [1.5, 3, 4, 12]
    
    bars = ax2.bar(enzymes, dcat_values, color=colors, edgecolor='black')
    
    ax2.set_ylabel('Categorical Distance (dcat)')
    ax2.set_ylim(0, 15)
    
    # Highlight Rubisco's complexity
    ax2.text(3, 13, 'Complex\nreaction!', fontsize=9, ha='center',
            color=COLORS['rubisco'], fontweight='bold')
    
    ax2.set_title('B. Categorical Distance\n(Reaction complexity)', fontweight='bold')
    
    # =========================================================================
    # Panel C: kcat ∝ 1/dcat
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    dcat = np.array([1.5, 3, 4, 12])
    kcat = np.array([4e7, 1e6, 1e2, 10])
    
    ax3.scatter(dcat, kcat, s=100, c=colors, edgecolors='black', zorder=5)
    
    # Fit line
    x_fit = np.linspace(1, 15, 100)
    y_fit = 5e7 / x_fit  # kcat ∝ 1/dcat
    ax3.plot(x_fit, y_fit, '--', color=COLORS['primary'], lw=1.5, alpha=0.7)
    
    ax3.set_xlabel('Categorical Distance (dcat)')
    ax3.set_ylabel('kcat (s⁻¹)')
    ax3.set_yscale('log')
    ax3.set_xlim(0, 14)
    ax3.set_ylim(1, 1e9)
    
    # Labels
    for i, (e, d, k) in enumerate(zip(enzymes, dcat, kcat)):
        ax3.annotate(e, (d, k), textcoords="offset points", 
                    xytext=(5, 5), fontsize=8)
    
    ax3.set_title('C. kcat ∝ 1/dcat\n(Expected relationship)', fontweight='bold')
    
    # =========================================================================
    # Panel D: CO2/O2 Discrimination
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # CO2 and O2 molecules
    ax4.add_patch(Circle((3, 7), 0.5, color=COLORS['co2']))
    ax4.text(3, 7, 'CO₂', ha='center', va='center', fontsize=9, fontweight='bold')
    ax4.text(3, 6, '0.04%', ha='center', fontsize=8)
    
    ax4.add_patch(Circle((7, 7), 0.5, color=COLORS['oxygen']))
    ax4.text(7, 7, 'O₂', ha='center', va='center', fontsize=9, 
            fontweight='bold', color='white')
    ax4.text(7, 6, '21%', ha='center', fontsize=8)
    
    # Similarity
    ax4.annotate('', xy=(6.4, 7), xytext=(3.6, 7),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=2))
    ax4.text(5, 7.8, 'dcat ≈ 2-3', fontsize=9, ha='center')
    ax4.text(5, 8.5, 'Categorically SIMILAR', fontsize=10, ha='center', fontweight='bold')
    
    # Specificity
    ax4.text(5, 4, 'Specificity: 80-100:1', fontsize=11, ha='center', 
            fontweight='bold', color=COLORS['rubisco'])
    ax4.text(5, 3, 'Despite 500:1 O₂ excess!', fontsize=9, ha='center')
    ax4.text(5, 1.5, 'Effective discrimination:\n80 × 500 = 40,000:1', 
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax4.set_title('D. CO₂/O₂ Discrimination\n(Remarkable, not poor)', fontweight='bold')
    ax4.axis('off')
    
    # =========================================================================
    # Panel E: Speed-Specificity Trade-off
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Pareto frontier
    speed = np.linspace(1, 20, 50)
    specificity = 1000 / speed
    
    ax5.plot(speed, specificity, '-', color=COLORS['primary'], lw=2)
    ax5.fill_between(speed, 0, specificity, alpha=0.1, color=COLORS['primary'])
    
    # Rubisco position (on frontier)
    ax5.scatter([10], [100], s=150, color=COLORS['rubisco'], 
               edgecolors='black', zorder=5)
    ax5.annotate('Rubisco', (10, 100), textcoords="offset points",
                xytext=(10, 10), fontsize=10, fontweight='bold',
                color=COLORS['rubisco'])
    
    # Impossible region
    ax5.text(15, 150, 'Impossible\nregion', fontsize=9, ha='center',
            color=COLORS['temporal'], style='italic')
    
    ax5.set_xlabel('Speed (kcat)')
    ax5.set_ylabel('Specificity (CO₂/O₂)')
    ax5.set_xlim(0, 20)
    ax5.set_ylim(0, 200)
    
    ax5.set_title('E. Speed-Specificity Trade-off\n(Rubisco is Pareto optimal)', fontweight='bold')
    
    # =========================================================================
    # Panel F: Conclusion
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    conclusions = [
        'Rubisco is NOT inefficient',
        '',
        '• dcat ≈ 12 (enormous complexity)',
        '• CO₂ is chemically inert',
        '• CO₂/O₂ are categorically similar',
        '• Pareto optimal for trade-off',
        '• Most abundant protein on Earth',
        '',
        'Feeds the entire biosphere'
    ]
    
    y_pos = 8.5
    for line in conclusions:
        if line.startswith('Rubisco'):
            ax6.text(5, y_pos, line, fontsize=12, ha='center', 
                    fontweight='bold', color=COLORS['rubisco'])
        elif line.startswith('Feeds'):
            ax6.text(5, y_pos, line, fontsize=11, ha='center',
                    fontweight='bold', color=COLORS['categorical'],
                    bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
        elif line:
            ax6.text(2, y_pos, line, fontsize=9)
        y_pos -= 0.85
    
    ax6.set_title('F. Rubisco: Categorical Sophistication', fontweight='bold')
    ax6.axis('off')
    
    # Main title
    plt.suptitle('Rubisco: Categorical Complexity, Not Evolutionary Failure',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir) / "rubisco_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all catalysis panel charts."""
    output_dir = "../docs/catalysis/figures"
    
    print("=" * 60)
    print("GENERATING CATALYSIS PANEL CHARTS")
    print("=" * 60)
    
    generate_three_paradoxes_panel(output_dir)
    generate_aperture_model_panel(output_dir)
    generate_carbonic_anhydrase_panel(output_dir)
    generate_haber_process_panel(output_dir)
    generate_rubisco_panel(output_dir)
    
    print("\n" + "=" * 60)
    print("ALL PANELS GENERATED")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  • three_paradoxes_panel.png")
    print("  • aperture_model_panel.png")
    print("  • carbonic_anhydrase_panel.png")
    print("  • haber_process_panel.png")
    print("  • rubisco_panel.png")


if __name__ == "__main__":
    main()

