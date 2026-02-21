#!/usr/bin/env python3
"""
Generate panel chart for Velocity-Entropy Independence.

This visualization demonstrates:
1. Entropy counts arrangements, not velocities
2. Snapshots are velocity-blind
3. Elastic collisions: temperature without entropy
4. Velocity is kinetic, entropy is configurational
5. The demon's category error
6. Velocity-sorting is orthogonal to entropy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
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
    'arrangement': '#4ecdc4',    # Teal for arrangements
    'velocity': '#ff6b6b',       # Red for velocity
    'entropy': '#95e1a3',        # Green for entropy
    'temperature': '#ff9f43',    # Orange for temperature
    'molecule': '#ffd93d',       # Yellow for molecules
    'kinetic': '#f8b4d9',        # Pink for kinetic
    'config': '#a78bfa',         # Purple for configurational
    'background': '#1a1a2e',
    'text': '#e0e0e0',
}


def draw_molecule(ax, x, y, color=None, size=0.15, velocity=None):
    """Draw a molecule with optional velocity arrow."""
    if color is None:
        color = COLORS['molecule']
    mol = Circle((x, y), size, facecolor=color, edgecolor='white', 
                 linewidth=1.5, zorder=5)
    ax.add_patch(mol)
    
    if velocity is not None:
        vx, vy = velocity
        ax.annotate('', xy=(x + vx, y + vy), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['velocity'], lw=2))


def panel_entropy_counts(ax):
    """Panel A: Entropy counts arrangements."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Entropy Counts Arrangements', fontsize=11, fontweight='bold', pad=10)
    
    # Equation
    ax.text(5, 9, r'S = k$_B$ ln($\Omega$)', fontsize=16, ha='center', 
            color='white', fontweight='bold')
    ax.text(5, 8, r'$\Omega$ = number of spatial arrangements', fontsize=10, 
            ha='center', color=COLORS['text'])
    
    # Show different arrangements
    arrangements = [
        [(1.5, 5), (2.5, 5), (3, 6)],
        [(5, 5), (5.5, 6), (6, 5)],
        [(8, 5), (8.5, 5), (9, 6)],
    ]
    
    for i, arr in enumerate(arrangements):
        for x, y in arr:
            draw_molecule(ax, x, y, size=0.2)
        ax.text(sum(p[0] for p in arr)/3, 3.8, f'Arrangement {i+1}', 
                fontsize=8, ha='center', color=COLORS['text'])
    
    ax.text(5, 2.5, 'Different positions = different arrangements', fontsize=10, 
            ha='center', color=COLORS['arrangement'])
    ax.text(5, 1.5, 'Velocity NOT in the count!', fontsize=11, ha='center', 
            color=COLORS['velocity'], fontweight='bold')


def panel_snapshot(ax):
    """Panel B: Snapshot is velocity-blind."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B. Snapshot = Positions Only', fontsize=11, fontweight='bold', pad=10)
    
    # Same positions, different velocities
    positions = [(2, 7), (4, 6.5), (3, 5.5), (5, 7.5)]
    
    # Snapshot box
    ax.add_patch(FancyBboxPatch((0.5, 4.5), 9, 5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['arrangement'], alpha=0.1,
                 edgecolor=COLORS['arrangement'], linewidth=2))
    ax.text(5, 9, 'SAME SNAPSHOT', fontsize=11, ha='center', 
            color=COLORS['arrangement'], fontweight='bold')
    
    # Molecules at same positions
    for x, y in positions:
        draw_molecule(ax, x, y, size=0.25)
    
    # Different velocity arrows (showing it doesn't matter)
    velocities = [(0.5, 0.3), (-0.3, 0.4), (0.4, -0.3), (-0.5, -0.2)]
    for (x, y), (vx, vy) in zip(positions, velocities):
        ax.annotate('', xy=(x + vx, y + vy), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['velocity'], 
                                  lw=1.5, alpha=0.5))
    
    # Label
    ax.text(7, 6, 'Velocities\n(ignored)', fontsize=8, ha='center', 
            color=COLORS['velocity'], alpha=0.7)
    
    # Key point
    ax.text(5, 3.5, 'Snapshot records POSITIONS', fontsize=10, ha='center', 
            color='white')
    ax.text(5, 2.5, 'Not velocities, not temperatures', fontsize=9, ha='center', 
            color=COLORS['text'])
    ax.text(5, 1.5, 'Same snapshot at 100K or 1000K!', fontsize=10, ha='center', 
            color=COLORS['temperature'])


def panel_elastic_collision(ax):
    """Panel C: Elastic collision - T changes, S doesn't."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('C. Elastic Collision', fontsize=11, fontweight='bold', pad=10)
    
    # Before
    ax.text(2.5, 9.2, 'Before', fontsize=10, ha='center', color=COLORS['text'])
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.6,
                 edgecolor='white', linewidth=1))
    
    draw_molecule(ax, 1.5, 7, size=0.25, velocity=(0.6, 0))
    draw_molecule(ax, 3.5, 7, size=0.25, velocity=(-0.3, 0))
    ax.text(2.5, 6, 'v1, v2', fontsize=9, ha='center', color=COLORS['velocity'])
    
    # After
    ax.text(7.5, 9.2, 'After', fontsize=10, ha='center', color=COLORS['text'])
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.6,
                 edgecolor='white', linewidth=1))
    
    draw_molecule(ax, 6.5, 7, size=0.25, velocity=(0.2, 0.3))
    draw_molecule(ax, 8.5, 7, size=0.25, velocity=(0.5, -0.2))
    ax.text(7.5, 6, "v1', v2' (changed)", fontsize=9, ha='center', color=COLORS['velocity'])
    
    # Arrow
    ax.annotate('', xy=(5.3, 7), xytext=(4.7, 7),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Results
    ax.add_patch(FancyBboxPatch((1, 1), 8, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.8,
                 edgecolor=COLORS['entropy'], linewidth=2))
    
    ax.text(5, 3.8, 'Positions: SAME (collision point)', fontsize=9, 
            ha='center', color=COLORS['arrangement'])
    ax.text(5, 3, 'Velocities: CHANGED', fontsize=9, ha='center', 
            color=COLORS['velocity'])
    ax.text(5, 2.2, 'Temperature: CAN CHANGE', fontsize=9, ha='center', 
            color=COLORS['temperature'])
    ax.text(5, 1.4, 'Entropy: UNCHANGED!', fontsize=10, ha='center', 
            color=COLORS['entropy'], fontweight='bold')


def panel_orthogonality(ax):
    """Panel D: Velocity perpendicular to entropy."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Orthogonality', fontsize=11, fontweight='bold', pad=10)
    
    # Coordinate system
    ax.annotate('', xy=(9, 5), xytext=(1, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['velocity'], lw=3))
    ax.annotate('', xy=(5, 9), xytext=(5, 1),
               arrowprops=dict(arrowstyle='->', color=COLORS['entropy'], lw=3))
    
    ax.text(9, 4.3, 'Velocity', fontsize=10, ha='center', color=COLORS['velocity'])
    ax.text(5.7, 9, 'Entropy', fontsize=10, ha='left', color=COLORS['entropy'])
    
    # Perpendicular symbol
    ax.plot([5.3, 5.3, 5], [5, 5.3, 5.3], color='white', lw=1.5)
    
    # Equation
    ax.text(7, 7.5, r'$\partial S / \partial v = 0$', fontsize=14, 
            ha='center', color='white', fontweight='bold')
    
    # Labels
    ax.text(7, 3, 'Kinetic axis', fontsize=9, ha='center', color=COLORS['velocity'])
    ax.text(3, 7, 'Config axis', fontsize=9, ha='center', color=COLORS['entropy'])


def panel_category_error(ax):
    """Panel E: The demon's category error."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('E. Demon\'s Category Error', fontsize=11, fontweight='bold', pad=10)
    
    # Kinetic category
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['kinetic'], alpha=0.2,
                 edgecolor=COLORS['kinetic'], linewidth=2))
    ax.text(2.5, 8.5, 'KINETIC', fontsize=11, ha='center', 
            color=COLORS['kinetic'], fontweight='bold')
    ax.text(2.5, 7.5, 'Velocity', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 6.8, 'Momentum', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 6.1, 'Kinetic energy', fontsize=9, ha='center', color=COLORS['text'])
    
    # Configurational category
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor=COLORS['config'], alpha=0.2,
                 edgecolor=COLORS['config'], linewidth=2))
    ax.text(7.5, 8.5, 'CONFIGURATIONAL', fontsize=11, ha='center', 
            color=COLORS['config'], fontweight='bold')
    ax.text(7.5, 7.5, 'Position', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.5, 6.8, 'Arrangement', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.5, 6.1, 'Entropy', fontsize=9, ha='center', color=COLORS['text'])
    
    # Demon's error
    ax.annotate('', xy=(5.5, 6.5), xytext=(4.5, 6.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['velocity'], lw=3))
    ax.text(5, 7.2, 'Demon\ntries', fontsize=8, ha='center', color=COLORS['velocity'])
    
    # X mark
    ax.text(5, 5, 'X', fontsize=40, ha='center', va='center', 
            color=COLORS['velocity'], fontweight='bold', alpha=0.7)
    
    # Bottom text
    ax.text(5, 3.5, 'Different categories!', fontsize=11, ha='center', 
            color='white', fontweight='bold')
    ax.text(5, 2.5, 'Manipulating kinetic properties', fontsize=9, ha='center', 
            color=COLORS['kinetic'])
    ax.text(5, 1.8, 'cannot affect configurational properties', fontsize=9, 
            ha='center', color=COLORS['config'])


def panel_what_changes_entropy(ax):
    """Panel F: What actually changes entropy."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('F. What Changes Entropy', fontsize=11, fontweight='bold', pad=10)
    
    # Does change
    ax.add_patch(FancyBboxPatch((0.5, 5), 4.2, 4.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.6,
                 edgecolor=COLORS['entropy'], linewidth=2))
    ax.text(2.6, 9, 'CHANGES S', fontsize=10, ha='center', 
            color=COLORS['entropy'], fontweight='bold')
    
    changes = ['Mixing', 'Expansion', 'Chemical rxn', 'Phase change']
    for i, c in enumerate(changes):
        ax.text(2.6, 8 - i*0.8, c, fontsize=9, ha='center', color='white')
    
    # Doesn't change
    ax.add_patch(FancyBboxPatch((5.3, 5), 4.2, 4.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['velocity'], linewidth=2))
    ax.text(7.4, 9, 'NO CHANGE', fontsize=10, ha='center', 
            color=COLORS['velocity'], fontweight='bold')
    
    no_changes = ['Elastic collision', 'Velocity sorting', 'Adiabatic T change']
    for i, c in enumerate(no_changes):
        ax.text(7.4, 8 - i*0.8, c, fontsize=9, ha='center', color='white')
    
    # Key
    ax.text(5, 3.5, 'Spatial rearrangement = entropy change', fontsize=10, 
            ha='center', color=COLORS['entropy'])
    ax.text(5, 2.5, 'Velocity redistribution = NO entropy change', fontsize=10, 
            ha='center', color=COLORS['velocity'])
    
    ax.text(5, 1.2, 'Demon does velocity sorting!', fontsize=11, ha='center', 
            color='white', fontweight='bold')


def panel_demon_strategy(ax):
    """Panel G: Demon's failed strategy chain."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('G. Demon\'s Broken Chain', fontsize=11, fontweight='bold', pad=10)
    
    # Strategy chain
    steps = [
        ('Measure v', 2, 8, COLORS['velocity']),
        ('Sort by v', 5, 8, COLORS['velocity']),
        ('Change T', 8, 8, COLORS['temperature']),
        ('Change S?', 5, 4, COLORS['entropy']),
    ]
    
    for label, x, y, color in steps:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2,
                     boxstyle="round,pad=0.1",
                     facecolor='#1a1a2e', alpha=0.8,
                     edgecolor=color, linewidth=2))
        ax.text(x, y, label, fontsize=9, ha='center', va='center', 
                color=color, fontweight='bold')
    
    # Arrows with X marks
    ax.annotate('', xy=(3.8, 8), xytext=(3.2, 8),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(6.8, 8), xytext=(6.2, 8),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.annotate('', xy=(5, 5.4), xytext=(5, 7.4),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # X marks on arrows
    ax.text(3.5, 8.5, 'X', fontsize=14, ha='center', color=COLORS['velocity'])
    ax.text(6.5, 8.5, 'X', fontsize=14, ha='center', color=COLORS['velocity'])
    ax.text(5.4, 6.4, 'X', fontsize=14, ha='center', color=COLORS['velocity'])
    
    # Problems
    ax.text(3.5, 7.2, 'v doesnt tell T', fontsize=7, ha='center', color=COLORS['text'])
    ax.text(6.5, 7.2, 'T can change', fontsize=7, ha='center', color=COLORS['text'])
    ax.text(6, 5.8, 'without S!', fontsize=7, ha='center', color=COLORS['text'])
    
    # Final result
    ax.add_patch(FancyBboxPatch((1.5, 1), 7, 1.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.8,
                 edgecolor=COLORS['velocity'], linewidth=2))
    ax.text(5, 1.75, 'EVERY STEP BROKEN', fontsize=11, ha='center', 
            color='white', fontweight='bold')


def panel_partial_derivative(ax):
    """Panel H: The zero partial derivative."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('H. The Mathematical Proof', fontsize=11, fontweight='bold', pad=10)
    
    # Main equation box
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 9, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['config'], linewidth=2))
    
    ax.text(5, 8.2, r'$\Omega$ = f(positions only)', fontsize=11, 
            ha='center', color=COLORS['arrangement'])
    ax.text(5, 7.2, r'$\partial\Omega/\partial v_i = 0$', fontsize=14, 
            ha='center', color='white', fontweight='bold')
    ax.text(5, 6.2, r'$\Rightarrow \partial S/\partial v_i = 0$', fontsize=14, 
            ha='center', color=COLORS['entropy'], fontweight='bold')
    
    # Explanation
    ax.text(5, 4.5, 'Arrangement count is velocity-independent', fontsize=10, 
            ha='center', color=COLORS['text'])
    ax.text(5, 3.5, 'Therefore entropy is velocity-independent', fontsize=10, 
            ha='center', color=COLORS['text'])
    
    # Consequence
    ax.text(5, 2, 'Velocity sorting has ZERO effect on entropy', fontsize=11, 
            ha='center', color=COLORS['entropy'], fontweight='bold')


def panel_summary(ax):
    """Panel I: Summary."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('I. The Final Defeat', fontsize=11, fontweight='bold', pad=10)
    
    # Main box
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 9, 6.5,
                 boxstyle="round,pad=0.1,rounding_size=0.3",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['config'], linewidth=2))
    
    ax.text(5, 8.3, 'VELOCITY and ENTROPY', fontsize=12, ha='center', 
            color='white', fontweight='bold')
    ax.text(5, 7.5, 'are ORTHOGONAL', fontsize=14, ha='center', 
            color=COLORS['entropy'], fontweight='bold')
    
    insights = [
        ('Velocity:', 'rate of position change (kinetic)', COLORS['velocity']),
        ('Entropy:', 'count of arrangements (config)', COLORS['entropy']),
        ('Demon sorts:', 'velocities', COLORS['velocity']),
        ('2nd Law protects:', 'entropy', COLORS['entropy']),
    ]
    
    for i, (label, desc, color) in enumerate(insights):
        y = 6.2 - i * 0.9
        ax.text(2.5, y, label, fontsize=9, ha='right', color=color, fontweight='bold')
        ax.text(2.7, y, desc, fontsize=9, ha='left', color=COLORS['text'])
    
    # Final statement
    ax.text(5, 3, 'The demon operates in the WRONG CATEGORY', fontsize=10, 
            ha='center', color='white', fontweight='bold')
    
    ax.text(5, 1.2, 'Category error = fundamental defeat', fontsize=11, 
            ha='center', color=COLORS['config'], fontweight='bold')


def main():
    """Generate the 9-panel velocity-entropy independence visualization."""
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
    panel_entropy_counts(ax1)
    panel_snapshot(ax2)
    panel_elastic_collision(ax3)
    panel_orthogonality(ax4)
    panel_category_error(ax5)
    panel_what_changes_entropy(ax6)
    panel_demon_strategy(ax7)
    panel_partial_derivative(ax8)
    panel_summary(ax9)
    
    # Main title
    fig.suptitle('Velocity-Entropy Independence: The Demon\'s Category Error',
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'resolution' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'velocity_entropy_panel.png'
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    # Save results summary
    results = {
        'panel_type': 'velocity_entropy_independence',
        'title': 'Velocity-Entropy Independence',
        'key_insights': [
            'Entropy counts spatial arrangements (S = kB ln Omega)',
            'Arrangement count is velocity-independent (dOmega/dv = 0)',
            'A snapshot is velocity-blind - same positions at any temperature',
            'Elastic collisions change velocity/temperature without changing entropy',
            'Velocity is kinetic (rate of position change)',
            'Entropy is configurational (count of arrangements)',
            'These are ORTHOGONAL quantities',
            'The demon commits a category error',
            'Manipulating kinetic properties cannot affect configurational properties',
            'Velocity sorting has ZERO effect on entropy'
        ],
        'category_distinction': {
            'kinetic': ['velocity', 'momentum', 'kinetic energy'],
            'configurational': ['position', 'arrangement', 'entropy']
        },
        'demon_error': 'treating kinetic properties as if they determined configurational properties'
    }
    
    results_path = output_dir / 'velocity_entropy_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    
    plt.close()


if __name__ == '__main__':
    main()

