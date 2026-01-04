"""
O₂ Quantum State Catalog Analysis
Clean 2-panel visualization with minimal overlapping text
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy

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

    print(f"Timestamp: {data['timestamp']}")
    print(f"N_states: {data['n_states']}")
    print(f"Cycle frequency: {data['cycle_frequency']:.2e} Hz")
    print(f"Temperature: {data['temperature']} K")
    print(f"Information capacity: {data['information_capacity_bits']:.2f} bits")

    # ============================================================================
    # EXTRACT STATE DATA
    # ============================================================================
    states = data['states']

    J_values = np.array([s['quantum_numbers']['J'] for s in states])
    M_values = np.array([s['quantum_numbers']['M'] for s in states])
    v_values = np.array([s['quantum_numbers']['v'] for s in states])
    spin_values = np.array([s['quantum_numbers']['spin'] for s in states])

    energies = np.array([s['energy'] for s in states])
    frequencies = np.array([s['frequency'] for s in states])
    amplitudes = np.array([s['amplitude'] for s in states])
    phases = np.array([s['phase'] for s in states])
    dampings = np.array([s['damping'] for s in states])
    symmetries = np.array([s['symmetry'] for s in states])
    degeneracies = np.array([s['degeneracy'] for s in states])
    boltzmann_weights = np.array([s['boltzmann_weight'] for s in states])

    print(f"\nQuantum number ranges:")
    print(f"  J: {J_values.min()} - {J_values.max()}")
    print(f"  M: {M_values.min()} - {M_values.max()}")
    print(f"  v: {v_values.min()} - {v_values.max()}")
    print(f"  spin: {spin_values.min()} - {spin_values.max()}")
    print(f"Energy range: {energies.min():.4e} - {energies.max():.4e} J")
    print(f"Boltzmann weight range: {boltzmann_weights.min():.4e} - {boltzmann_weights.max():.4e}")

    # Calculate entropy
    weights_nonzero = boltzmann_weights[boltzmann_weights > 0]
    state_entropy = entropy(weights_nonzero, base=2)
    print(f"State entropy: {state_entropy:.2f} bits")

    # ============================================================================
    # CREATE FIGURE
    # ============================================================================
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # ============================================================================
    # PANEL A: Rotational State Distribution (J-level populations)
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    # Group by J quantum number
    J_unique = np.unique(J_values)
    J_weights = np.array([boltzmann_weights[J_values == J].sum() for J in J_unique])

    # Normalize to percentage
    J_weights_pct = J_weights / J_weights.sum() * 100

    # Create bar plot
    bars = ax_a.bar(J_unique, J_weights_pct, 
                color='#3498db', alpha=0.8,
                edgecolor='black', linewidth=1.2, width=0.7)

    # Highlight most populated
    max_idx = np.argmax(J_weights_pct)
    bars[max_idx].set_color('#e74c3c')
    bars[max_idx].set_linewidth(2)

    # Styling
    ax_a.set_xlabel('Rotational Quantum Number (J)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Population (%)', fontsize=11, fontweight='bold')
    ax_a.set_title('A. Rotational State Distribution', fontsize=12, fontweight='bold', pad=10)
    ax_a.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_a.set_xlim(-0.5, J_unique.max() + 0.5)

    # Add single info box (top right, non-overlapping)
    info_text = f'T = {data["temperature"]} K\nMax: J = {J_unique[max_idx]} ({J_weights_pct[max_idx]:.1f}%)'
    ax_a.text(0.97, 0.97, info_text, 
            transform=ax_a.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black'))

    # ============================================================================
    # PANEL B: Energy Level Diagram with Populations
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Convert energy to eV for better readability
    energies_eV = energies / 1.602e-19

    # Group by J and calculate average energy and total weight
    J_energies = []
    J_weights_for_plot = []

    for J in J_unique:
        mask = J_values == J
        avg_energy = energies_eV[mask].mean()
        total_weight = boltzmann_weights[mask].sum()
        J_energies.append(avg_energy)
        J_weights_for_plot.append(total_weight)

    J_energies = np.array(J_energies)
    J_weights_for_plot = np.array(J_weights_for_plot)

    # Normalize weights for marker size
    marker_sizes = (J_weights_for_plot / J_weights_for_plot.max()) * 1000 + 100

    # Plot energy levels
    for i, J in enumerate(J_unique):
        color = '#e74c3c' if i == max_idx else '#3498db'
        ax_b.scatter(J, J_energies[i], s=marker_sizes[i], 
                    c=color, alpha=0.7, edgecolors='black', linewidth=1.5,
                    zorder=10)
        
        # Draw horizontal line for energy level
        ax_b.hlines(J_energies[i], J - 0.3, J + 0.3, 
                colors=color, linewidth=2, alpha=0.6)

    # Connect levels
    ax_b.plot(J_unique, J_energies, 'k--', alpha=0.3, linewidth=1, zorder=1)

    # Styling
    ax_b.set_xlabel('Rotational Quantum Number (J)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Energy (eV)', fontsize=11, fontweight='bold')
    ax_b.set_title('B. Energy Level Diagram', fontsize=12, fontweight='bold', pad=10)
    ax_b.grid(True, alpha=0.3, linestyle='--')
    ax_b.set_xlim(-0.5, J_unique.max() + 0.5)

    # Add legend for marker size
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=8, label='Low population', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=14, label='High population', markeredgecolor='black'),
    ]
    ax_b.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    # Add single info box (bottom right)
    info_text2 = f'Info: {data["information_capacity_bits"]:.2f} bits\nEntropy: {state_entropy:.2f} bits'
    ax_b.text(0.97, 0.03, info_text2, 
            transform=ax_b.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='black'))

    # ============================================================================
    # OVERALL TITLE AND SAVE
    # ============================================================================
    fig.suptitle(f'O₂ Quantum State Catalog: {data["n_states"]} states at {data["cycle_frequency"]/1e12:.0f} THz', 
                fontsize=13, fontweight='bold', y=0.98)

    plt.savefig('quantum_state_catalog_analysis.pdf', bbox_inches='tight')
    plt.savefig('quantum_state_catalog_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: quantum_state_catalog_analysis.pdf/.png")

    plt.show()

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total states: {len(states)}")
    print(f"J range: {J_values.min()} - {J_values.max()}")
    print(f"Most populated J level: {J_unique[max_idx]} ({J_weights_pct[max_idx]:.1f}%)")
    print(f"Energy range: {energies_eV.min():.6f} - {energies_eV.max():.6f} eV")
    print(f"Information capacity: {data['information_capacity_bits']:.2f} bits")
    print(f"State entropy: {state_entropy:.2f} bits")
    print(f"Temperature: {data['temperature']} K")
    print(f"Cycle frequency: {data['cycle_frequency']/1e12:.0f} THz")
    print("="*80)
