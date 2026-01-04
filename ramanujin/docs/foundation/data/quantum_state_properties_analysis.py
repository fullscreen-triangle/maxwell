"""
O₂ Quantum State Properties Analysis
2-panel visualization of oscillatory properties and symmetries
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

if __name__ == "__main__":
    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("Loading o2_state_catalog_sample.json...")
    with open('o2_state_catalog_sample.json', 'r') as f:
        data = json.load(f)

    states = data['states']

    # Extract data
    J_values = np.array([s['quantum_numbers']['J'] for s in states])
    frequencies = np.array([s['frequency'] for s in states])
    dampings = np.array([s['damping'] for s in states])
    symmetries = np.array([s['symmetry'] for s in states])
    boltzmann_weights = np.array([s['boltzmann_weight'] for s in states])

    print(f"Loaded {len(states)} quantum states")
    print(f"Frequency range: {frequencies.min():.2e} - {frequencies.max():.2e} Hz")
    print(f"Damping range: {dampings.min():.3f} - {dampings.max():.3f}")
    print(f"Symmetry range: {symmetries.min():.3f} - {symmetries.max():.3f}")

    # ============================================================================
    # CREATE FIGURE
    # ============================================================================
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # ============================================================================
    # PANEL A: Frequency vs Damping (colored by population)
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    # Filter out zero frequencies for log scale
    nonzero_mask = frequencies > 0
    freq_plot = frequencies[nonzero_mask]
    damp_plot = dampings[nonzero_mask]
    weight_plot = boltzmann_weights[nonzero_mask]

    # Convert to THz
    freq_THz = freq_plot / 1e12

    # Create scatter plot
    scatter = ax_a.scatter(freq_THz, damp_plot, 
                        c=weight_plot, s=80, alpha=0.7,
                        cmap='viridis', edgecolors='black', linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax_a, pad=0.02)
    cbar.set_label('Boltzmann Weight', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    # Styling
    ax_a.set_xlabel('Frequency (THz)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Damping Factor', fontsize=11, fontweight='bold')
    ax_a.set_title('A. Oscillatory Properties', fontsize=12, fontweight='bold', pad=10)
    ax_a.grid(True, alpha=0.3, linestyle='--')
    ax_a.set_xscale('log')

    # Info box
    info_text = f'States with ν > 0: {nonzero_mask.sum()}\nT = {data["temperature"]} K'
    ax_a.text(0.03, 0.97, info_text, 
            transform=ax_a.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black'))

    # ============================================================================
    # PANEL B: Symmetry Distribution by J level
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Group by J
    J_unique = np.unique(J_values)

    # Create violin plot data
    symmetry_by_J = [symmetries[J_values == J] for J in J_unique]

    # Violin plot
    parts = ax_b.violinplot(symmetry_by_J, positions=J_unique, 
                            widths=0.6, showmeans=True, showmedians=True)

    # Color the violins
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.2)

    # Style the other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.5)

    # Overlay scatter points
    for J in J_unique:
        mask = J_values == J
        sym_vals = symmetries[mask]
        weights = boltzmann_weights[mask]
        
        # Jitter x positions slightly
        x_jitter = J + np.random.normal(0, 0.05, len(sym_vals))
        
        ax_b.scatter(x_jitter, sym_vals, 
                    c=weights, s=30, alpha=0.6,
                    cmap='plasma', edgecolors='black', linewidth=0.3)

    # Styling
    ax_b.set_xlabel('Rotational Quantum Number (J)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Symmetry Factor', fontsize=11, fontweight='bold')
    ax_b.set_title('B. Symmetry Distribution', fontsize=12, fontweight='bold', pad=10)
    ax_b.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_b.set_xlim(-0.5, J_unique.max() + 0.5)
    ax_b.set_ylim(-0.05, 1.05)

    # Info box
    mean_sym = symmetries.mean()
    std_sym = symmetries.std()
    info_text2 = f'Mean: {mean_sym:.3f}\nStd: {std_sym:.3f}'
    ax_b.text(0.97, 0.97, info_text2, 
            transform=ax_b.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8, edgecolor='black'))

    # ============================================================================
    # OVERALL TITLE AND SAVE
    # ============================================================================
    fig.suptitle(f'O₂ Quantum State Properties: Oscillatory Signatures', 
                fontsize=13, fontweight='bold', y=0.98)

    plt.savefig('quantum_state_properties_analysis.pdf', bbox_inches='tight')
    plt.savefig('quantum_state_properties_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: quantum_state_properties_analysis.pdf/.png")

    plt.show()

    print("\n" + "="*80)
    print("PROPERTY STATISTICS")
    print("="*80)
    print(f"Frequency range: {frequencies[frequencies>0].min()/1e9:.2f} - {frequencies.max()/1e12:.2f} THz")
    print(f"Mean damping: {dampings.mean():.3f} ± {dampings.std():.3f}")
    print(f"Mean symmetry: {symmetries.mean():.3f} ± {symmetries.std():.3f}")
    print(f"States with oscillations (ν > 0): {(frequencies > 0).sum()}/{len(frequencies)}")
    print("="*80)
