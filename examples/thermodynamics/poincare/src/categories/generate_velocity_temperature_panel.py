#!/usr/bin/env python3
"""
Generate panel chart for Velocity-Temperature Non-Correspondence.

This visualization demonstrates:
1. Maxwell-Boltzmann distributions overlap
2. Same velocity = different "temperature meaning"
3. Temperature is ensemble property, not molecular
4. Sorting paradox when moving molecules
5. Category change on transfer
6. Why sorting by velocity doesn't sort by temperature
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
from scipy import stats
from pathlib import Path
import json

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'figure.facecolor': '#0a0a12',
    'axes.facecolor': '#12121a',
    'axes.edgecolor': '#3a3a4a',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'grid.color': '#2a2a3a',
    'grid.alpha': 0.3
})

# Color scheme
COLORS = {
    'cold': '#4ecdc4',         # Teal for cold container
    'hot': '#ff6b6b',          # Red for hot container
    'overlap': '#a78bfa',      # Purple for overlap region
    'molecule': '#ffd93d',     # Yellow for molecule
    'fast': '#ff9f43',         # Orange for "fast"
    'slow': '#45b7d1',         # Blue for "slow"
    'arrow': '#f8b4d9',        # Pink for arrows
    'background': '#1a1a2e',
    'text': '#e0e0e0',
    'equilibrium': '#95e1a3',  # Green
}


def maxwell_boltzmann(v, T, m=28):
    """Maxwell-Boltzmann speed distribution (arbitrary units for visualization)."""
    kB = 1.0  # Normalized
    prefactor = 4 * np.pi * (m / (2 * np.pi * kB * T))**(3/2)
    return prefactor * v**2 * np.exp(-m * v**2 / (2 * kB * T))


def panel_distributions(ax):
    """Panel A: Two overlapping Maxwell-Boltzmann distributions."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    
    # Generate distributions
    v = np.linspace(0.01, 10, 200)
    T_cold = 300  # Arbitrary units
    T_hot = 340
    
    f_cold = maxwell_boltzmann(v, T_cold)
    f_hot = maxwell_boltzmann(v, T_hot)
    
    # Normalize for visualization
    f_cold = f_cold / f_cold.max() * 0.9
    f_hot = f_hot / f_hot.max() * 0.9
    
    # Plot distributions
    ax.fill_between(v, 0, f_cold, color=COLORS['cold'], alpha=0.4, label='Cold (300K)')
    ax.fill_between(v, 0, f_hot, color=COLORS['hot'], alpha=0.4, label='Hot (340K)')
    ax.plot(v, f_cold, color=COLORS['cold'], lw=2)
    ax.plot(v, f_hot, color=COLORS['hot'], lw=2)
    
    # Mark overlap region
    overlap_mask = (f_cold > 0.05) & (f_hot > 0.05)
    ax.fill_between(v, 0, np.minimum(f_cold, f_hot), where=overlap_mask,
                    color=COLORS['overlap'], alpha=0.5, label='Overlap')
    
    # Mark means
    v_mean_cold = np.sqrt(8 * 300 / (np.pi * 28)) * 1.8  # Scaled for visualization
    v_mean_hot = np.sqrt(8 * 340 / (np.pi * 28)) * 1.8
    
    ax.axvline(v_mean_cold, color=COLORS['cold'], linestyle='--', lw=1.5, alpha=0.8)
    ax.axvline(v_mean_hot, color=COLORS['hot'], linestyle='--', lw=1.5, alpha=0.8)
    
    ax.text(v_mean_cold - 0.3, 1.05, '<v>_cold', fontsize=8, color=COLORS['cold'])
    ax.text(v_mean_hot + 0.1, 1.05, '<v>_hot', fontsize=8, color=COLORS['hot'])
    
    ax.set_xlabel('Velocity', fontsize=9)
    ax.set_ylabel('Probability', fontsize=9)
    ax.set_title('A. Overlapping Distributions', fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_yticks([])


def panel_same_velocity(ax):
    """Panel B: Same velocity, different meaning."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B. Same Velocity, Different Meaning', fontsize=11, fontweight='bold', pad=10)
    
    # Central molecule
    mol = Circle((5, 5), 0.5, facecolor=COLORS['molecule'], edgecolor='white', 
                 linewidth=2, zorder=5)
    ax.add_patch(mol)
    ax.text(5, 5, 'v*', ha='center', va='center', fontsize=12, 
            color='black', fontweight='bold')
    ax.text(5, 4, '500 m/s', ha='center', fontsize=9, color=COLORS['text'])
    
    # In cold container (left)
    ax.add_patch(FancyBboxPatch((0.5, 6.5), 3.5, 2.5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['cold'], alpha=0.2,
                 edgecolor=COLORS['cold'], linewidth=2))
    ax.text(2.25, 8.5, 'In COLD (300K)', fontsize=9, ha='center', 
            color=COLORS['cold'], fontweight='bold')
    ax.text(2.25, 7.5, '"FAST"', fontsize=14, ha='center', 
            color=COLORS['fast'], fontweight='bold')
    ax.text(2.25, 6.8, '(above average)', fontsize=8, ha='center', color=COLORS['text'])
    
    # In hot container (right)
    ax.add_patch(FancyBboxPatch((6, 6.5), 3.5, 2.5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['hot'], alpha=0.2,
                 edgecolor=COLORS['hot'], linewidth=2))
    ax.text(7.75, 8.5, 'In HOT (310K)', fontsize=9, ha='center', 
            color=COLORS['hot'], fontweight='bold')
    ax.text(7.75, 7.5, '"SLOW"', fontsize=14, ha='center', 
            color=COLORS['slow'], fontweight='bold')
    ax.text(7.75, 6.8, '(below average)', fontsize=8, ha='center', color=COLORS['text'])
    
    # Arrows from molecule to boxes
    ax.annotate('', xy=(2.25, 6.5), xytext=(4.5, 5.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['cold'], lw=2))
    ax.annotate('', xy=(7.75, 6.5), xytext=(5.5, 5.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['hot'], lw=2))
    
    # Key insight
    ax.text(5, 1.5, 'Same velocity, opposite categorical meaning!', fontsize=10, 
            ha='center', color='white', fontweight='bold')


def panel_percentile(ax):
    """Panel C: Percentile visualization."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('C. Velocity Percentile by Context', fontsize=11, fontweight='bold', pad=10)
    
    # Two horizontal bars showing percentile
    # Cold container
    ax.add_patch(Rectangle((1, 7), 8, 1, facecolor=COLORS['cold'], alpha=0.3,
                           edgecolor=COLORS['cold'], linewidth=2))
    ax.text(5, 8.5, 'Cold Container (300K)', fontsize=9, ha='center', 
            color=COLORS['cold'], fontweight='bold')
    
    # Marker at 53% (above average)
    marker_cold = 1 + 0.53 * 8
    ax.plot([marker_cold, marker_cold], [7, 8], color=COLORS['molecule'], lw=4)
    ax.text(marker_cold, 6.5, '53%', fontsize=9, ha='center', color=COLORS['fast'])
    ax.text(marker_cold, 6, '"Fast"', fontsize=8, ha='center', color=COLORS['fast'])
    
    # Mean marker
    ax.plot([5, 5], [7, 8], color='white', lw=2, linestyle='--')
    ax.text(5, 8.2, '50%', fontsize=7, ha='center', color='white')
    
    # Hot container
    ax.add_patch(Rectangle((1, 3.5), 8, 1, facecolor=COLORS['hot'], alpha=0.3,
                           edgecolor=COLORS['hot'], linewidth=2))
    ax.text(5, 5, 'Hot Container (310K)', fontsize=9, ha='center', 
            color=COLORS['hot'], fontweight='bold')
    
    # Marker at 47% (below average)
    marker_hot = 1 + 0.47 * 8
    ax.plot([marker_hot, marker_hot], [3.5, 4.5], color=COLORS['molecule'], lw=4)
    ax.text(marker_hot, 3, '47%', fontsize=9, ha='center', color=COLORS['slow'])
    ax.text(marker_hot, 2.5, '"Slow"', fontsize=8, ha='center', color=COLORS['slow'])
    
    # Mean marker
    ax.plot([5, 5], [3.5, 4.5], color='white', lw=2, linestyle='--')
    ax.text(5, 4.7, '50%', fontsize=7, ha='center', color='white')
    
    # Arrow showing same velocity
    ax.annotate('', xy=(marker_hot, 4.5), xytext=(marker_cold, 7),
               arrowprops=dict(arrowstyle='<->', color=COLORS['molecule'], lw=2,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(8.5, 5.5, 'Same v!', fontsize=9, color=COLORS['molecule'], fontweight='bold')


def panel_sorting_paradox(ax):
    """Panel D: The sorting paradox."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Sorting Paradox', fontsize=11, fontweight='bold', pad=10)
    
    # Step 1: Demon sees "fast" molecule
    ax.add_patch(FancyBboxPatch((0.3, 6.5), 4.2, 3,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['cold'], alpha=0.2,
                 edgecolor=COLORS['cold'], linewidth=2))
    ax.text(2.4, 9, 'COLD', fontsize=9, ha='center', color=COLORS['cold'], fontweight='bold')
    
    mol1 = Circle((2.4, 7.5), 0.3, facecolor=COLORS['molecule'], edgecolor='white', lw=2)
    ax.add_patch(mol1)
    ax.text(2.4, 7.5, 'v*', ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    ax.text(2.4, 6.8, '"FAST"', fontsize=10, ha='center', color=COLORS['fast'], fontweight='bold')
    
    # Arrow showing demon's decision
    ax.annotate('', xy=(6.5, 7.5), xytext=(4.8, 7.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=3))
    ax.text(5.6, 8, 'Demon:', fontsize=8, ha='center', color=COLORS['arrow'])
    ax.text(5.6, 7, '"Move to hot!"', fontsize=8, ha='center', color=COLORS['arrow'])
    
    # Step 2: Arrives in hot as "slow"
    ax.add_patch(FancyBboxPatch((5.5, 6.5), 4.2, 3,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['hot'], alpha=0.2,
                 edgecolor=COLORS['hot'], linewidth=2))
    ax.text(7.6, 9, 'HOT', fontsize=9, ha='center', color=COLORS['hot'], fontweight='bold')
    
    mol2 = Circle((7.6, 7.5), 0.3, facecolor=COLORS['molecule'], edgecolor='white', lw=2)
    ax.add_patch(mol2)
    ax.text(7.6, 7.5, 'v*', ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    ax.text(7.6, 6.8, '"SLOW"', fontsize=10, ha='center', color=COLORS['slow'], fontweight='bold')
    
    # Result box
    ax.add_patch(FancyBboxPatch((1, 1), 8, 4,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.8,
                 edgecolor=COLORS['hot'], linewidth=2))
    
    ax.text(5, 4.3, 'PARADOX:', fontsize=11, ha='center', color=COLORS['hot'], fontweight='bold')
    ax.text(5, 3.3, 'Intended: Add "fast" to make hotter', fontsize=9, 
            ha='center', color=COLORS['text'])
    ax.text(5, 2.5, 'Result: Added "slow" - makes COLDER!', fontsize=9, 
            ha='center', color=COLORS['slow'], fontweight='bold')
    ax.text(5, 1.5, 'Demon achieved the OPPOSITE', fontsize=9, 
            ha='center', color='white', style='italic')


def panel_no_molecular_temp(ax):
    """Panel E: Molecules don't have temperature."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('E. No Molecular Temperature', fontsize=11, fontweight='bold', pad=10)
    
    # Wrong view
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['hot'], linewidth=2, linestyle='--'))
    ax.text(2.5, 8.5, 'WRONG', fontsize=10, ha='center', 
            color=COLORS['hot'], fontweight='bold')
    ax.text(2.5, 7.5, 'Molecule has v', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 6.8, '->', fontsize=12, ha='center', color='white')
    ax.text(2.5, 6.1, 'Molecule has T', fontsize=9, ha='center', color=COLORS['text'])
    
    # Right view
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.6,
                 edgecolor=COLORS['equilibrium'], linewidth=2))
    ax.text(7.5, 8.5, 'CORRECT', fontsize=10, ha='center', 
            color=COLORS['equilibrium'], fontweight='bold')
    ax.text(7.5, 7.5, 'Ensemble has {v_i}', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.5, 6.8, '->', fontsize=12, ha='center', color='white')
    ax.text(7.5, 6.1, 'Ensemble has T', fontsize=9, ha='center', color=COLORS['text'])
    
    # Key equation
    ax.text(5, 4, 'T = T[{v_1, v_2, ..., v_N}]', fontsize=11, 
            ha='center', color='white', fontweight='bold')
    ax.text(5, 3.2, 'Temperature is a FUNCTIONAL of distribution', fontsize=9, 
            ha='center', color=COLORS['text'])
    ax.text(5, 2.4, 'NOT a function of individual velocity', fontsize=9, 
            ha='center', color=COLORS['text'])
    
    # Bottom insight
    ax.text(5, 1, 'Only ENSEMBLES have temperature', fontsize=11, 
            ha='center', color=COLORS['equilibrium'], fontweight='bold')


def panel_category_change(ax):
    """Panel F: Category changes on transfer."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('F. Category Changes on Transfer', fontsize=11, fontweight='bold', pad=10)
    
    # Before transfer
    ax.text(2.5, 9, 'Before', fontsize=10, ha='center', color=COLORS['text'])
    ax.add_patch(FancyBboxPatch((0.5, 6), 4, 2.5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['cold'], alpha=0.15,
                 edgecolor=COLORS['cold'], linewidth=2))
    
    mol1 = Circle((2.5, 7.25), 0.25, facecolor=COLORS['fast'], edgecolor='white', lw=2)
    ax.add_patch(mol1)
    ax.text(2.5, 6.5, 'Category: HOT', fontsize=8, ha='center', color=COLORS['fast'])
    
    # After transfer
    ax.text(7.5, 9, 'After', fontsize=10, ha='center', color=COLORS['text'])
    ax.add_patch(FancyBboxPatch((5.5, 6), 4, 2.5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['hot'], alpha=0.15,
                 edgecolor=COLORS['hot'], linewidth=2))
    
    mol2 = Circle((7.5, 7.25), 0.25, facecolor=COLORS['slow'], edgecolor='white', lw=2)
    ax.add_patch(mol2)
    ax.text(7.5, 6.5, 'Category: COLD', fontsize=8, ha='center', color=COLORS['slow'])
    
    # Transfer arrow
    ax.annotate('', xy=(5.5, 7.25), xytext=(4.5, 7.25),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(5, 7.7, 'Transfer', fontsize=8, ha='center', color='white')
    
    # Velocity unchanged
    ax.add_patch(FancyBboxPatch((2, 3), 6, 2,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.8,
                 edgecolor=COLORS['molecule'], linewidth=2))
    ax.text(5, 4.3, 'Velocity: UNCHANGED', fontsize=10, ha='center', 
            color=COLORS['molecule'], fontweight='bold')
    ax.text(5, 3.5, 'Category: INVERTED', fontsize=10, ha='center', 
            color=COLORS['arrow'], fontweight='bold')
    
    ax.text(5, 1.5, 'Context determines meaning', fontsize=10, 
            ha='center', color='white', style='italic')


def panel_demon_failure(ax):
    """Panel G: Why demon fails."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('G. Why Demon Cannot Sort by T', fontsize=11, fontweight='bold', pad=10)
    
    reasons = [
        ('1.', 'Temperature is not', 'a molecular property'),
        ('2.', 'Velocity does not', 'determine T contribution'),
        ('3.', 'T contribution is', 'ensemble-relative'),
        ('4.', 'Transfer changes', 'categorical meaning'),
    ]
    
    for i, (num, line1, line2) in enumerate(reasons):
        y = 8 - i * 1.8
        ax.text(1, y, num, fontsize=11, color=COLORS['hot'], fontweight='bold')
        ax.text(1.8, y, line1, fontsize=9, color='white')
        ax.text(1.8, y - 0.5, line2, fontsize=9, color=COLORS['text'])
    
    # Conclusion
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 9, 1.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.8,
                 edgecolor=COLORS['hot'], linewidth=2))
    ax.text(5, 1.25, 'Sorting by velocity =/= sorting by temperature', fontsize=10, 
            ha='center', color='white', fontweight='bold')


def panel_overlap_problem(ax):
    """Panel H: The overlap region problem."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('H. Overlap Region Problem', fontsize=11, fontweight='bold', pad=10)
    
    # Venn-like diagram
    cold_circle = Circle((3.5, 5.5), 2.5, facecolor=COLORS['cold'], alpha=0.3,
                         edgecolor=COLORS['cold'], linewidth=2)
    hot_circle = Circle((6.5, 5.5), 2.5, facecolor=COLORS['hot'], alpha=0.3,
                        edgecolor=COLORS['hot'], linewidth=2)
    ax.add_patch(cold_circle)
    ax.add_patch(hot_circle)
    
    ax.text(2, 5.5, 'Cold\nonly', fontsize=9, ha='center', va='center', 
            color=COLORS['cold'], fontweight='bold')
    ax.text(8, 5.5, 'Hot\nonly', fontsize=9, ha='center', va='center', 
            color=COLORS['hot'], fontweight='bold')
    ax.text(5, 5.5, 'BOTH', fontsize=11, ha='center', va='center', 
            color=COLORS['overlap'], fontweight='bold')
    
    ax.text(5, 3.5, 'Overlap = ALL velocities!', fontsize=10, ha='center', 
            color=COLORS['overlap'])
    
    # Explanation
    ax.text(5, 2, 'Every velocity exists in both distributions', fontsize=9, 
            ha='center', color=COLORS['text'])
    ax.text(5, 1.2, 'The overlap is COMPLETE', fontsize=10, ha='center', 
            color='white', fontweight='bold')


def panel_summary(ax):
    """Panel I: Summary."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('I. The Insight', fontsize=11, fontweight='bold', pad=10)
    
    # Main box
    ax.add_patch(FancyBboxPatch((0.5, 3), 9, 6,
                 boxstyle="round,pad=0.1,rounding_size=0.3",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['overlap'], linewidth=2))
    
    ax.text(5, 8.2, 'Velocity =/= Temperature', fontsize=14, ha='center', 
            color='white', fontweight='bold')
    
    ax.text(5, 7, 'Same velocity v:', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(5, 6.2, 'In cold: "FAST" (contributes to hotness)', fontsize=9, 
            ha='center', color=COLORS['fast'])
    ax.text(5, 5.4, 'In hot: "SLOW" (contributes to coldness)', fontsize=9, 
            ha='center', color=COLORS['slow'])
    
    ax.axhline(y=4.5, xmin=0.1, xmax=0.9, color='white', alpha=0.3)
    
    ax.text(5, 3.8, 'The demon sorts by velocity', fontsize=10, 
            ha='center', color=COLORS['text'])
    ax.text(5, 3.2, 'but CANNOT sort by temperature', fontsize=10, 
            ha='center', color=COLORS['hot'], fontweight='bold')
    
    ax.text(5, 1.5, 'Temperature is contextual, not intrinsic', fontsize=11, 
            ha='center', color=COLORS['equilibrium'], fontweight='bold')


def main():
    """Generate the 9-panel velocity-temperature visualization."""
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a12')
    
    # Create 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Row 2
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Row 3
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Generate panels
    panel_distributions(ax1)
    panel_same_velocity(ax2)
    panel_percentile(ax3)
    panel_sorting_paradox(ax4)
    panel_no_molecular_temp(ax5)
    panel_category_change(ax6)
    panel_demon_failure(ax7)
    panel_overlap_problem(ax8)
    panel_summary(ax9)
    
    # Main title
    fig.suptitle('Velocity-Temperature Non-Correspondence:\nSame Velocity, Different "Temperature Meaning"',
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'resolution' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'velocity_temperature_panel.png'
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    # Save results summary
    results = {
        'panel_type': 'velocity_temperature_non_correspondence',
        'title': 'Velocity-Temperature Non-Correspondence',
        'key_insights': [
            'Maxwell-Boltzmann distributions overlap completely',
            'Same velocity has different meaning in different ensembles',
            'A 500 m/s molecule is "fast" in cold, "slow" in hot',
            'Temperature is not a molecular property',
            'Only ensembles have temperature',
            'Sorting by velocity does not sort by temperature',
            'Category changes on transfer (fast -> slow)',
            'Demon achieves opposite of intention for overlap molecules',
            'The overlap region includes ALL velocities',
            'Temperature contribution depends on destination, not source'
        ],
        'sorting_paradox': {
            'demon_sees': 'fast molecule in cold container',
            'demon_action': 'move to hot container',
            'intended_result': 'add fast molecule to make hot hotter',
            'actual_result': 'add slow molecule, makes hot COLDER'
        },
        'fundamental_problem': 'velocity does not determine temperature contribution'
    }
    
    results_path = output_dir / 'velocity_temperature_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    
    plt.close()


if __name__ == '__main__':
    main()

