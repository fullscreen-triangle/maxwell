"""
Unified Virtual Gas Ensemble Panel
==================================

Extended visualization showing the gas ensemble with three conjugate views:
1. Virtual Gas (molecules)
2. Categorical Memory (addresses)
3. Categorical Processor (computational states)
4. Semantic Processor (meaning encoding)

Each row demonstrates how the same measurement creates equivalent structures
in all three domains.
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge, Arc, Rectangle
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np

from virtual_molecule import VirtualMolecule, SCoordinate
from virtual_chamber import VirtualChamber


# Color scheme
COLORS = {
    'observed': '#2E86AB',      # Blue - currently observed
    'related1': '#A23B72',      # Magenta - related molecule 1
    'related2': '#F18F01',      # Orange - related molecule 2
    'memory': '#6A0DAD',        # Purple - memory
    'processor': '#228B22',     # Forest green - processor
    'semantic': '#DC143C',      # Crimson - semantic
    'vdw': '#6C757D',           # Gray - Van der Waals
    'dipole': '#3A7D44',        # Green - dipole
    'vibration': '#C73E1D',     # Red - vibrational
    'bg': '#F8F9FA',
    'dark': '#212529',
    'grid': '#DEE2E6',
}

HARDWARE_PARAMS = ['CPU\nCycle', 'Memory\nBus', 'Power\nSupply', 
                   'I/O\nLatency', 'Cache\nTiming', 'Network\nJitter']


def create_molecule_radar(ax, molecule: VirtualMolecule, color: str, 
                         is_observed: bool = False, alpha: float = 1.0,
                         label: str = None):
    """Create radar chart for molecule."""
    n_params = len(HARDWARE_PARAMS)
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]
    
    base_values = [
        molecule.s_coord.S_k,
        molecule.s_coord.S_t,
        molecule.s_coord.S_e,
        molecule.phase / (2 * np.pi) if molecule.phase else 0.5,
        molecule.amplitude if molecule.amplitude else 0.5,
        (molecule.frequency % 1e9) / 1e9 if molecule.frequency else 0.5,
    ]
    
    values = [max(0.1, min(1.0, v)) for v in base_values]
    values += values[:1]
    
    if is_observed:
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.plot(angles, values, 'o-', color=color, linewidth=2, markersize=5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(HARDWARE_PARAMS, fontsize=6)
    else:
        ax.plot(angles, values, '-', color=color, linewidth=1.5, alpha=alpha)
        ax.fill(angles, values, color=color, alpha=0.1 * alpha)
    
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=5)
    ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    
    if label:
        ax.set_title(label, fontsize=8, fontweight='bold', color=color, pad=8)


def draw_interaction_line(ax, pos1, pos2, itype, strength):
    """Draw interaction line between molecules."""
    colors = {'vdw': COLORS['vdw'], 'dipole': COLORS['dipole'], 'vibration': COLORS['vibration']}
    styles = {'vdw': '--', 'dipole': '-.', 'vibration': ':'}
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
           linestyle=styles.get(itype, '--'), 
           color=colors.get(itype, COLORS['vdw']), 
           linewidth=1.5 * strength, alpha=0.6)


def draw_molecule_circle(ax, pos, color, size=0.12, is_observed=False, label=None):
    """Draw molecule as circle."""
    if is_observed:
        for i in range(3):
            glow = Circle(pos, size * (1.3 - i * 0.1), 
                         facecolor=color, alpha=0.1 * (3 - i), edgecolor='none')
            ax.add_patch(glow)
    
    circle = Circle(pos, size, facecolor=color, edgecolor='white', 
                   linewidth=2 if is_observed else 1, alpha=0.9)
    ax.add_patch(circle)
    
    if label:
        ax.text(pos[0], pos[1], label, ha='center', va='center',
               fontsize=9 if is_observed else 7, fontweight='bold', color='white')


def create_ensemble_view(ax, molecules, observed_idx, time_label):
    """Create ensemble snapshot view."""
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.2, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    positions = [(1.0, 1.5), (0.3, 0.4), (1.7, 0.4)]
    colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
    labels = ['α', 'β', 'γ']
    
    # Draw interactions
    for i, j, itype, strength in [(0, 1, 'vdw', 0.7), (0, 2, 'dipole', 0.5), (1, 2, 'vibration', 0.6)]:
        draw_interaction_line(ax, positions[i], positions[j], itype, strength)
    
    # Draw molecules
    for i, (mol, pos, color, label) in enumerate(zip(molecules, positions, colors, labels)):
        is_obs = (i == observed_idx)
        draw_color = COLORS['observed'] if is_obs else color
        draw_molecule_circle(ax, pos, draw_color, size=0.18 if is_obs else 0.12,
                           is_observed=is_obs, label=label)
    
    ax.text(1.0, 1.95, time_label, ha='center', fontsize=8, fontweight='bold')


def create_memory_view(ax, molecule: VirtualMolecule, is_active: bool = True):
    """
    Create the categorical memory equivalence view.
    Shows S-coordinates as memory addresses in a hierarchical structure.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    color = COLORS['memory'] if is_active else COLORS['grid']
    alpha = 1.0 if is_active else 0.3
    
    # Title
    ax.text(5, 9.5, 'CATEGORICAL MEMORY', ha='center', fontsize=8, 
           fontweight='bold', color=color)
    
    # Draw 3^k hierarchy (simplified)
    # Root
    root = FancyBboxPatch((4, 7), 2, 1, boxstyle="round,pad=0.1",
                         facecolor=color, alpha=0.3 * alpha, edgecolor=color)
    ax.add_patch(root)
    ax.text(5, 7.5, 'Root', ha='center', va='center', fontsize=7, color=color)
    
    # Three branches
    branch_x = [1.5, 5, 8.5]
    branch_labels = ['$S_k$', '$S_t$', '$S_e$']
    
    for i, (bx, bl) in enumerate(zip(branch_x, branch_labels)):
        # Connection line
        ax.plot([5, bx], [7, 5.5], '-', color=color, alpha=0.5 * alpha, linewidth=1)
        
        # Branch box
        highlight = (i == 0 and molecule.s_coord.S_k > 0.5) or \
                   (i == 1 and molecule.s_coord.S_t > 0.5) or \
                   (i == 2 and molecule.s_coord.S_e > 0.5)
        
        box_alpha = 0.7 if highlight and is_active else 0.2
        box = FancyBboxPatch((bx - 1, 4.5), 2, 1, boxstyle="round,pad=0.1",
                            facecolor=color, alpha=box_alpha * alpha, edgecolor=color)
        ax.add_patch(box)
        ax.text(bx, 5, bl, ha='center', va='center', fontsize=7, color=color)
    
    # Address display
    addr_box = FancyBboxPatch((1.5, 1.5), 7, 2, boxstyle="round,pad=0.1",
                             facecolor='white', alpha=0.9, edgecolor=color, linewidth=2)
    ax.add_patch(addr_box)
    
    addr_text = f"Address: ({molecule.s_coord.S_k:.3f}, {molecule.s_coord.S_t:.3f}, {molecule.s_coord.S_e:.3f})"
    ax.text(5, 2.5, addr_text, ha='center', va='center', fontsize=7, 
           fontweight='bold' if is_active else 'normal', color=color)
    
    ax.text(5, 0.5, 'S-coordinate = Address', ha='center', fontsize=6, 
           style='italic', color=color, alpha=alpha)


def create_processor_view(ax, molecule: VirtualMolecule, is_active: bool = True):
    """
    Create the categorical processor equivalence view.
    Shows molecule as computational state with oscillator representation.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    color = COLORS['processor'] if is_active else COLORS['grid']
    alpha = 1.0 if is_active else 0.3
    
    # Title
    ax.text(5, 9.5, 'CATEGORICAL PROCESSOR', ha='center', fontsize=8, 
           fontweight='bold', color=color)
    
    # Oscillator wave
    t = np.linspace(0, 4 * np.pi, 100)
    freq = molecule.frequency / 1e12 if molecule.frequency else 1.0
    phase = molecule.phase if molecule.phase else 0
    amp = molecule.amplitude if molecule.amplitude else 0.5
    
    y = amp * np.sin(freq * t + phase)
    y_scaled = 5.5 + y * 1.5  # Scale to fit
    t_scaled = 1 + t / (4 * np.pi) * 8  # Scale to 1-9
    
    ax.plot(t_scaled, y_scaled, '-', color=color, linewidth=2, alpha=alpha)
    ax.axhline(y=5.5, xmin=0.1, xmax=0.9, color=color, linestyle='--', 
              alpha=0.3 * alpha, linewidth=1)
    
    # Frequency label
    ax.text(5, 7.8, f'ω = {molecule.frequency:.2e} Hz' if molecule.frequency else 'ω',
           ha='center', fontsize=7, color=color, alpha=alpha)
    
    # Phase-lock indicator
    phase_box = FancyBboxPatch((3, 2), 4, 2, boxstyle="round,pad=0.1",
                              facecolor=color, alpha=0.2 * alpha, edgecolor=color)
    ax.add_patch(phase_box)
    
    ax.text(5, 3.2, 'Phase-Lock State', ha='center', fontsize=7, color=color, alpha=alpha)
    ax.text(5, 2.5, f'φ = {phase:.2f} rad', ha='center', fontsize=6, color=color, alpha=alpha)
    
    # Processing rate
    rate = molecule.frequency / (2 * np.pi) if molecule.frequency else 0
    ax.text(5, 0.5, f'R = ω/2π = {rate:.2e} ops/s', ha='center', fontsize=6,
           style='italic', color=color, alpha=alpha)


def create_semantic_view(ax, molecule: VirtualMolecule, is_active: bool = True):
    """
    Create the semantic processor equivalence view.
    Shows molecule as word encoding with harmonic structure.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    color = COLORS['semantic'] if is_active else COLORS['grid']
    alpha = 1.0 if is_active else 0.3
    
    # Title
    ax.text(5, 9.5, 'SEMANTIC PROCESSOR', ha='center', fontsize=8, 
           fontweight='bold', color=color)
    
    # Word encoding representation
    word_box = FancyBboxPatch((2, 6.5), 6, 2, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.2 * alpha, edgecolor=color)
    ax.add_patch(word_box)
    
    # Show as word → molecule mapping
    ax.text(5, 7.8, '"word" → Molecule', ha='center', fontsize=7, color=color, alpha=alpha)
    ax.text(5, 6.9, f'f₀ = {molecule.frequency:.2e}' if molecule.frequency else 'f₀',
           ha='center', fontsize=6, color=color, alpha=alpha)
    
    # Harmonic spectrum (simplified)
    harmonics_x = [2, 4, 6, 8]
    harmonics_h = [0.9, 0.6, 0.4, 0.2]  # Decreasing harmonics
    
    for i, (hx, hh) in enumerate(zip(harmonics_x, harmonics_h)):
        bar_height = hh * 2.5
        rect = Rectangle((hx - 0.3, 3), 0.6, bar_height, 
                         facecolor=color, alpha=(0.8 - i * 0.15) * alpha)
        ax.add_patch(rect)
        ax.text(hx, 2.6, f'n={i+1}', ha='center', fontsize=5, color=color, alpha=alpha)
    
    ax.text(5, 1.8, 'Harmonic Overtones', ha='center', fontsize=6, color=color, alpha=alpha)
    
    # Meaning label
    ax.text(5, 0.5, 'Frequency = Meaning', ha='center', fontsize=6,
           style='italic', color=color, alpha=alpha)


def create_unified_view(ax, molecules: List[VirtualMolecule]):
    """
    Create the unified view showing all three conjugate representations together.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'UNIFIED: Gas = Memory = Processor = Semantics', 
           ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])
    
    # Central molecule
    center = Circle((5, 5.5), 1.2, facecolor=COLORS['observed'], 
                   edgecolor='white', linewidth=3, alpha=0.9)
    ax.add_patch(center)
    ax.text(5, 5.5, 'M', ha='center', va='center', fontsize=14, 
           fontweight='bold', color='white')
    
    # Three emanating arrows to conjugate views
    views = [
        ('Memory\n$\\mathbf{S}$ = Address', COLORS['memory'], (1.5, 2.5), 150),
        ('Processor\nω = Rate', COLORS['processor'], (5, 1.5), 270),
        ('Semantics\nf = Meaning', COLORS['semantic'], (8.5, 2.5), 30),
    ]
    
    for label, color, pos, angle in views:
        # Arrow from center
        dx = pos[0] - 5
        dy = pos[1] - 5.5
        length = np.sqrt(dx**2 + dy**2)
        
        # Start point (edge of central circle)
        start_x = 5 + (dx / length) * 1.2
        start_y = 5.5 + (dy / length) * 1.2
        
        ax.annotate('', xy=pos, xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Label box
        box = FancyBboxPatch((pos[0] - 1.2, pos[1] - 0.8), 2.4, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3, edgecolor=color)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, ha='center', va='center', 
               fontsize=6, color=color, fontweight='bold')
    
    # Bottom text
    ax.text(5, 0.3, 'One measurement → Three conjugate views of the same categorical state',
           ha='center', fontsize=6, style='italic', color=COLORS['dark'])


def generate_unified_ensemble_panel():
    """Generate the complete unified panel."""
    
    # Generate molecules
    chamber = VirtualChamber()
    molecules = []
    for _ in range(3):
        mol = chamber.sample()
        molecules.append(mol)
        time.sleep(0.008)
    
    # Create figure with 4 rows, 3 columns
    fig = plt.figure(figsize=(16, 18))
    
    # Grid: 4 rows x 3 columns
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                          height_ratios=[1, 1, 1, 1.2])
    
    # Row configurations
    # Each row: (time_label, observed_idx, active_view)
    rows = [
        ('Window $t_1$: Molecule α', 0, 'memory'),
        ('Window $t_2$: Molecule β', 1, 'processor'),
        ('Window $t_3$: Molecule γ', 2, 'semantic'),
    ]
    
    # Generate first three rows
    for row_idx, (time_label, obs_idx, active_view) in enumerate(rows):
        # Column 1: Ensemble view
        ax_ensemble = fig.add_subplot(gs[row_idx, 0])
        create_ensemble_view(ax_ensemble, molecules, obs_idx, time_label)
        
        # Column 2: Radar chart
        ax_radar = fig.add_subplot(gs[row_idx, 1], projection='polar')
        colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
        labels = ['Molecule α', 'Molecule β', 'Molecule γ']
        create_molecule_radar(ax_radar, molecules[obs_idx], colors[obs_idx],
                            is_observed=True, label=labels[obs_idx])
        
        # Column 3: Conjugate view (cycles through Memory → Processor → Semantic)
        ax_conjugate = fig.add_subplot(gs[row_idx, 2])
        
        if active_view == 'memory':
            create_memory_view(ax_conjugate, molecules[obs_idx], is_active=True)
        elif active_view == 'processor':
            create_processor_view(ax_conjugate, molecules[obs_idx], is_active=True)
        else:
            create_semantic_view(ax_conjugate, molecules[obs_idx], is_active=True)
    
    # Row 4: Unified view spanning all columns
    ax_unified_left = fig.add_subplot(gs[3, 0])
    ax_unified_left.set_xlim(-0.3, 2.3)
    ax_unified_left.set_ylim(-0.2, 2.0)
    ax_unified_left.set_aspect('equal')
    ax_unified_left.axis('off')
    
    # Draw all three molecules with equal emphasis
    positions = [(1.0, 1.5), (0.3, 0.4), (1.7, 0.4)]
    colors = [COLORS['observed'], COLORS['related1'], COLORS['related2']]
    labels = ['α', 'β', 'γ']
    
    for i, j, itype, strength in [(0, 1, 'vdw', 0.7), (0, 2, 'dipole', 0.5), (1, 2, 'vibration', 0.6)]:
        draw_interaction_line(ax_unified_left, positions[i], positions[j], itype, strength)
    
    for i, (pos, color, label) in enumerate(zip(positions, colors, labels)):
        draw_molecule_circle(ax_unified_left, pos, color, size=0.15, is_observed=True, label=label)
    
    ax_unified_left.text(1.0, 1.95, 'Complete Ensemble', ha='center', fontsize=8, fontweight='bold')
    
    # Middle: overlaid radar
    ax_unified_radar = fig.add_subplot(gs[3, 1], projection='polar')
    for i, mol in enumerate(molecules):
        create_molecule_radar(ax_unified_radar, mol, colors[i], 
                            is_observed=(i == 0), alpha=0.8 if i == 0 else 0.5)
    ax_unified_radar.set_title('All Three Overlaid', fontsize=8, fontweight='bold', pad=10)
    
    # Right: unified schematic
    ax_unified_right = fig.add_subplot(gs[3, 2])
    create_unified_view(ax_unified_right, molecules)
    
    # Main title
    fig.suptitle('Virtual Gas Ensemble: Unified Categorical Framework\n'
                'Molecule = Address = Oscillator = Meaning',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Bottom explanation
    explanation = (
        "Each row shows the same categorical state viewed through different lenses:\n"
        "Row 1: Memory view (S-coordinates as hierarchical addresses)\n"
        "Row 2: Processor view (oscillator frequency as processing rate)\n"
        "Row 3: Semantic view (vibrational modes as meaning encoding)\n"
        "Row 4: Unified view (all three are the same underlying structure)"
    )
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=8,
            color=COLORS['dark'])
    
    return fig, molecules


def save_panel(output_dir: Path):
    """Generate and save the unified panel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(" Generating Unified Ensemble Panel")
    print("=" * 70)
    
    print("\n  Generating panel...")
    fig, molecules = generate_unified_ensemble_panel()
    
    output_file = output_dir / 'panel_unified_ensemble.png'
    fig.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    
    print(f"    ✓ Saved to: {output_file}")
    
    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'file': str(output_file),
        'molecules': [
            {
                'S_k': mol.s_coord.S_k,
                'S_t': mol.s_coord.S_t,
                'S_e': mol.s_coord.S_e,
                'frequency': mol.frequency,
                'phase': mol.phase,
            }
            for mol in molecules
        ]
    }
    
    results_file = output_dir / 'unified_ensemble_summary.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f" Panel saved to: {output_dir}")
    print(f"{'='*70}")
    
    return results


def main():
    output_dir = Path(__file__).parent.parent / 'results' / 'panels'
    return save_panel(output_dir)


if __name__ == "__main__":
    main()

