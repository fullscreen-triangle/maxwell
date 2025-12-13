#!/usr/bin/env python3
"""
Generate panel chart for Heat-Entropy Decoupling.

This visualization demonstrates:
1. Heat can flow either direction in individual collisions
2. Entropy ALWAYS increases regardless of heat direction
3. The three collision cases at the demon's door
4. Heat is statistical, entropy is categorical
5. Why Maxwell conflated heat and entropy
6. The fundamental decoupling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch, Polygon, Arc
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
    'hot': '#ff6b6b',          # Red for hot/fast
    'cold': '#4ecdc4',         # Teal for cold/slow
    'entropy': '#95e1a3',      # Green for entropy
    'heat': '#ff9f43',         # Orange for heat
    'partition': '#8b8b8b',    # Gray for partition
    'aperture': '#ffd93d',     # Yellow for aperture
    'collision': '#f8b4d9',    # Pink for collision
    'arrow': '#a78bfa',        # Purple for arrows
    'background': '#1a1a2e',
    'text': '#e0e0e0'
}


def draw_molecule(ax, x, y, color, size=0.15, velocity=None, label=None, speed='medium'):
    """Draw a molecule with optional velocity arrow."""
    mol = Circle((x, y), size, facecolor=color, edgecolor='white', 
                 linewidth=2, zorder=5)
    ax.add_patch(mol)
    
    if velocity is not None:
        vx, vy = velocity
        ax.annotate('', xy=(x + vx, y + vy), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=8, 
                color='white', fontweight='bold')


def panel_collision_setup(ax):
    """Panel A: The collision scenario setup."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Door Collision Scenario', fontsize=11, fontweight='bold', pad=10)
    
    # Two containers
    ax.add_patch(FancyBboxPatch((0.5, 2), 3.5, 6,
                 boxstyle="round,pad=0.05",
                 facecolor=COLORS['hot'], alpha=0.15,
                 edgecolor=COLORS['hot'], linewidth=2))
    ax.add_patch(FancyBboxPatch((6, 2), 3.5, 6,
                 boxstyle="round,pad=0.05",
                 facecolor=COLORS['cold'], alpha=0.15,
                 edgecolor=COLORS['cold'], linewidth=2))
    
    ax.text(2.25, 8.5, 'HOT (A)', fontsize=10, ha='center', 
            color=COLORS['hot'], fontweight='bold')
    ax.text(7.75, 8.5, 'COLD (B)', fontsize=10, ha='center', 
            color=COLORS['cold'], fontweight='bold')
    
    # Partition with aperture
    ax.add_patch(Rectangle((4.4, 2), 0.2, 6, facecolor=COLORS['partition']))
    ax.add_patch(Circle((4.5, 5), 0.35, facecolor=COLORS['aperture'], 
                        edgecolor='white', linewidth=2))
    ax.text(4.5, 1.5, 'Door', fontsize=8, ha='center', color=COLORS['aperture'])
    
    # Fast molecule approaching
    draw_molecule(ax, 3, 5, COLORS['hot'], size=0.2, velocity=(0.8, 0))
    ax.text(3, 5.6, 'Fast', fontsize=8, ha='center', color=COLORS['hot'])
    ax.text(3, 4.3, '$m_A$', fontsize=9, ha='center', color='white')
    
    # Cold molecule near door
    draw_molecule(ax, 5.5, 5, COLORS['cold'], size=0.2)
    ax.text(5.5, 5.6, 'Slow', fontsize=8, ha='center', color=COLORS['cold'])
    ax.text(5.5, 4.3, '$m_B$', fontsize=9, ha='center', color='white')
    
    # Collision indicator
    ax.add_patch(Circle((4.8, 5), 0.8, facecolor=COLORS['collision'], 
                        alpha=0.3, edgecolor=COLORS['collision'], 
                        linewidth=2, linestyle='--'))
    ax.text(4.8, 6.2, 'Collision\nzone', fontsize=7, ha='center', 
            color=COLORS['collision'])


def panel_case1(ax):
    """Panel B: Case 1 - Bounce back."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B. Case 1: Bounce Back', fontsize=11, fontweight='bold', pad=10)
    
    # Before
    ax.text(2.5, 9, 'Before', fontsize=9, ha='center', color=COLORS['text'])
    draw_molecule(ax, 1.5, 7.5, COLORS['hot'], size=0.2, velocity=(0.6, 0))
    draw_molecule(ax, 3.5, 7.5, COLORS['cold'], size=0.2)
    ax.add_patch(Rectangle((2.4, 6.8), 0.15, 1.4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((2.475, 7.5), 0.2, facecolor=COLORS['aperture']))
    
    # After
    ax.text(7.5, 9, 'After', fontsize=9, ha='center', color=COLORS['text'])
    draw_molecule(ax, 6, 7.5, COLORS['hot'], size=0.2, velocity=(-0.4, 0))
    ax.text(6, 6.8, 'slower', fontsize=7, ha='center', color=COLORS['hot'])
    draw_molecule(ax, 8.5, 7.5, COLORS['cold'], size=0.2, velocity=(0.5, 0))
    ax.text(8.5, 6.8, 'faster', fontsize=7, ha='center', color=COLORS['cold'])
    ax.add_patch(Rectangle((7.4, 6.8), 0.15, 1.4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((7.475, 7.5), 0.2, facecolor=COLORS['aperture']))
    
    # Arrow
    ax.annotate('', xy=(5.5, 7.5), xytext=(4.2, 7.5),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Results box
    ax.add_patch(FancyBboxPatch((1, 2), 8, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.8,
                 edgecolor=COLORS['collision'], linewidth=2))
    
    ax.text(5, 4.8, 'Molecule returns to source', fontsize=9, 
            ha='center', color='white')
    ax.text(5, 3.8, 'Heat: HOT -> COLD', fontsize=10, ha='center', 
            color=COLORS['heat'], fontweight='bold')
    ax.text(5, 2.8, 'Entropy: INCREASES', fontsize=10, ha='center', 
            color=COLORS['entropy'], fontweight='bold')


def panel_case2(ax):
    """Panel C: Case 2 - Cold accelerates."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('C. Case 2: Cold Accelerates', fontsize=11, fontweight='bold', pad=10)
    
    # Before
    ax.text(2.5, 9, 'Before', fontsize=9, ha='center', color=COLORS['text'])
    draw_molecule(ax, 1.5, 7.5, COLORS['hot'], size=0.2, velocity=(0.6, 0))
    draw_molecule(ax, 3.5, 7.5, COLORS['cold'], size=0.2)
    ax.add_patch(Rectangle((2.4, 6.8), 0.15, 1.4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((2.475, 7.5), 0.2, facecolor=COLORS['aperture']))
    
    # After
    ax.text(7.5, 9, 'After', fontsize=9, ha='center', color=COLORS['text'])
    draw_molecule(ax, 6, 7.5, COLORS['hot'], size=0.2, velocity=(0.2, 0))
    ax.text(6, 6.8, 'slower', fontsize=7, ha='center', color=COLORS['hot'])
    draw_molecule(ax, 8.5, 7.5, COLORS['cold'], size=0.2, velocity=(0.8, 0))
    ax.text(8.5, 6.8, 'FASTER', fontsize=7, ha='center', color=COLORS['cold'], fontweight='bold')
    ax.add_patch(Rectangle((7.4, 6.8), 0.15, 1.4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((7.475, 7.5), 0.2, facecolor=COLORS['aperture']))
    
    # Arrow
    ax.annotate('', xy=(5.5, 7.5), xytext=(4.2, 7.5),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Results box
    ax.add_patch(FancyBboxPatch((1, 2), 8, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.8,
                 edgecolor=COLORS['entropy'], linewidth=2))
    
    ax.text(5, 4.8, 'Standard energy transfer', fontsize=9, 
            ha='center', color='white')
    ax.text(5, 3.8, 'Heat: HOT -> COLD', fontsize=10, ha='center', 
            color=COLORS['heat'], fontweight='bold')
    ax.text(5, 2.8, 'Entropy: INCREASES', fontsize=10, ha='center', 
            color=COLORS['entropy'], fontweight='bold')


def panel_case3(ax):
    """Panel D: Case 3 - Cold decelerates (reverse heat!)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Case 3: Cold Decelerates (!)', fontsize=11, fontweight='bold', pad=10)
    
    # Before
    ax.text(2.5, 9, 'Before', fontsize=9, ha='center', color=COLORS['text'])
    draw_molecule(ax, 1.5, 7.5, COLORS['hot'], size=0.2, velocity=(0.6, 0))
    draw_molecule(ax, 3.5, 7.5, COLORS['cold'], size=0.2, velocity=(-0.3, 0))
    ax.add_patch(Rectangle((2.4, 6.8), 0.15, 1.4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((2.475, 7.5), 0.2, facecolor=COLORS['aperture']))
    
    # After
    ax.text(7.5, 9, 'After', fontsize=9, ha='center', color=COLORS['text'])
    draw_molecule(ax, 6, 7.5, COLORS['hot'], size=0.2, velocity=(-0.9, 0))
    ax.text(6, 6.8, 'FASTER!', fontsize=7, ha='center', color=COLORS['hot'], fontweight='bold')
    draw_molecule(ax, 8.5, 7.5, COLORS['cold'], size=0.2)
    ax.text(8.5, 6.8, 'slower', fontsize=7, ha='center', color=COLORS['cold'])
    ax.add_patch(Rectangle((7.4, 6.8), 0.15, 1.4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((7.475, 7.5), 0.2, facecolor=COLORS['aperture']))
    
    # Arrow
    ax.annotate('', xy=(5.5, 7.5), xytext=(4.2, 7.5),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Results box - highlighted differently
    ax.add_patch(FancyBboxPatch((1, 2), 8, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.8,
                 edgecolor=COLORS['hot'], linewidth=3))
    
    ax.text(5, 4.8, 'Hot returns with MORE energy!', fontsize=9, 
            ha='center', color='white')
    ax.text(5, 3.8, 'Heat: COLD -> HOT', fontsize=10, ha='center', 
            color=COLORS['hot'], fontweight='bold')
    ax.text(5, 2.8, 'Entropy: STILL INCREASES', fontsize=10, ha='center', 
            color=COLORS['entropy'], fontweight='bold')


def panel_decoupling(ax):
    """Panel E: The fundamental decoupling."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('E. Fundamental Decoupling', fontsize=11, fontweight='bold', pad=10)
    
    # Heat box
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['heat'], linewidth=2))
    ax.text(2.5, 8.5, 'HEAT', fontsize=12, ha='center', 
            color=COLORS['heat'], fontweight='bold')
    ax.text(2.5, 7.5, 'Energy flow', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 6.7, 'Can go EITHER way', fontsize=9, ha='center', color='white')
    ax.text(2.5, 6, 'in collisions', fontsize=9, ha='center', color=COLORS['text'])
    
    # Entropy box
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.6,
                 edgecolor=COLORS['entropy'], linewidth=2))
    ax.text(7.5, 8.5, 'ENTROPY', fontsize=12, ha='center', 
            color=COLORS['entropy'], fontweight='bold')
    ax.text(7.5, 7.5, 'Categorical completion', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.5, 6.7, 'ALWAYS increases', fontsize=9, ha='center', color='white')
    ax.text(7.5, 6, 'in collisions', fontsize=9, ha='center', color=COLORS['text'])
    
    # NOT EQUAL sign
    ax.text(5, 7.25, '=/=', fontsize=20, ha='center', va='center', 
            color='white', fontweight='bold')
    
    # Summary
    ax.add_patch(FancyBboxPatch((1, 1), 8, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['arrow'], linewidth=2))
    ax.text(5, 3.8, 'The Second Law constrains', fontsize=10, 
            ha='center', color='white')
    ax.text(5, 2.8, 'ENTROPY (categorical)', fontsize=11, ha='center', 
            color=COLORS['entropy'], fontweight='bold')
    ax.text(5, 1.8, 'not HEAT (statistical)', fontsize=10, ha='center', 
            color=COLORS['heat'])


def panel_maxwell_error(ax):
    """Panel F: Maxwell's conflation."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('F. Maxwell\'s Conflation', fontsize=11, fontweight='bold', pad=10)
    
    # Wrong assumption
    ax.add_patch(FancyBboxPatch((0.5, 6), 9, 3,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['hot'], linewidth=2, linestyle='--'))
    ax.text(5, 8.3, 'Maxwell assumed:', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(5, 7.2, r'$\Delta Q > 0 \Leftrightarrow \Delta S > 0$', fontsize=14, 
            ha='center', color='white')
    ax.text(5, 6.3, '(heat flow = entropy change)', fontsize=9, 
            ha='center', color=COLORS['text'], style='italic')
    
    # X mark
    ax.text(8.5, 7.5, 'X', fontsize=30, ha='center', va='center', 
            color=COLORS['hot'], fontweight='bold')
    
    # Reality
    ax.add_patch(FancyBboxPatch((0.5, 1.5), 9, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.6,
                 edgecolor=COLORS['entropy'], linewidth=2))
    ax.text(5, 4.3, 'Reality:', fontsize=10, ha='center', color=COLORS['text'])
    ax.text(5, 3.3, r'$\Delta Q$ fluctuates, $\Delta S \geq 0$ always', fontsize=12, 
            ha='center', color='white')
    ax.text(5, 2.2, 'Microscopic: heat direction random', fontsize=9, 
            ha='center', color=COLORS['text'])
    ax.text(5, 1.7, 'Macroscopic: entropy always increases', fontsize=9, 
            ha='center', color=COLORS['entropy'])


def panel_statistical_vs_categorical(ax):
    """Panel G: Statistical vs Categorical."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('G. Statistical vs Categorical', fontsize=11, fontweight='bold', pad=10)
    
    # Heat: statistical average
    ax.text(5, 9, 'Heat = Statistical Average', fontsize=10, ha='center', 
            color=COLORS['heat'], fontweight='bold')
    
    # Show fluctuating line
    x = np.linspace(0.5, 9.5, 100)
    np.random.seed(42)
    y_heat = 6.5 + np.cumsum(np.random.randn(100) * 0.15)
    y_heat = y_heat - y_heat.mean() + 6.5
    ax.plot(x, y_heat, color=COLORS['heat'], lw=2, alpha=0.8)
    ax.axhline(y=6.5, color=COLORS['heat'], linestyle='--', alpha=0.5)
    ax.text(9.7, 6.5, 'avg', fontsize=7, color=COLORS['heat'])
    
    # Entropy: monotonic increase
    ax.text(5, 5, 'Entropy = Categorical (monotonic)', fontsize=10, ha='center', 
            color=COLORS['entropy'], fontweight='bold')
    
    y_entropy = 2 + 1.5 * (1 - np.exp(-x/3))
    ax.plot(x, y_entropy, color=COLORS['entropy'], lw=3)
    ax.fill_between(x, 1.5, y_entropy, color=COLORS['entropy'], alpha=0.2)
    
    # Arrow showing monotonic
    ax.annotate('', xy=(8, 3.3), xytext=(2, 2.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['entropy'], lw=2))
    ax.text(5, 1.2, 'Never decreases', fontsize=9, ha='center', 
            color=COLORS['entropy'], style='italic')


def panel_demon_misdirection(ax):
    """Panel H: Demon attacks wrong quantity."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('H. Demon\'s Misdirection', fontsize=11, fontweight='bold', pad=10)
    
    # Demon attacks heat
    ax.add_patch(FancyBboxPatch((0.5, 5), 4, 4,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['heat'], linewidth=2))
    ax.text(2.5, 8.3, 'Demon attacks:', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 7.3, 'HEAT', fontsize=14, ha='center', 
            color=COLORS['heat'], fontweight='bold')
    ax.text(2.5, 6.3, '(statistical)', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(2.5, 5.5, 'fluctuating', fontsize=9, ha='center', color=COLORS['heat'])
    
    # Second Law protects entropy
    ax.add_patch(FancyBboxPatch((5.5, 5), 4, 4,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.6,
                 edgecolor=COLORS['entropy'], linewidth=2))
    ax.text(7.5, 8.3, '2nd Law protects:', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.5, 7.3, 'ENTROPY', fontsize=14, ha='center', 
            color=COLORS['entropy'], fontweight='bold')
    ax.text(7.5, 6.3, '(categorical)', fontsize=9, ha='center', color=COLORS['text'])
    ax.text(7.5, 5.5, 'invariant', fontsize=9, ha='center', color=COLORS['entropy'])
    
    # Shield
    shield_x = [7.5, 8.3, 8.3, 7.5, 6.7, 6.7]
    shield_y = [4.2, 4.5, 3.5, 3, 3.5, 4.5]
    ax.fill(shield_x, shield_y, color=COLORS['entropy'], alpha=0.5)
    ax.plot(shield_x + [shield_x[0]], shield_y + [shield_y[0]], 
            color='white', lw=2)
    
    # Result
    ax.text(5, 1.5, 'Wrong target = inevitable failure', fontsize=11, 
            ha='center', color='white', fontweight='bold')


def panel_summary(ax):
    """Panel I: Summary insight."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('I. The Insight', fontsize=11, fontweight='bold', pad=10)
    
    # Main box
    ax.add_patch(FancyBboxPatch((0.5, 3), 9, 6,
                 boxstyle="round,pad=0.1,rounding_size=0.3",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['arrow'], linewidth=2))
    
    # Three cases summary
    cases = [
        ('Case 1:', 'Heat -> cold', 'Entropy UP', COLORS['heat']),
        ('Case 2:', 'Heat -> cold', 'Entropy UP', COLORS['heat']),
        ('Case 3:', 'Heat -> HOT', 'Entropy UP', COLORS['hot']),
    ]
    
    for i, (case, heat, entropy, color) in enumerate(cases):
        y = 7.5 - i * 1.2
        ax.text(1.5, y, case, fontsize=9, ha='left', color='white')
        ax.text(4, y, heat, fontsize=9, ha='center', color=color)
        ax.text(7, y, entropy, fontsize=9, ha='center', 
                color=COLORS['entropy'], fontweight='bold')
    
    # Divider
    ax.axhline(y=4.5, xmin=0.1, xmax=0.9, color='white', alpha=0.3)
    
    # Conclusion
    ax.text(5, 3.8, 'Heat direction: variable', fontsize=10, 
            ha='center', color=COLORS['heat'])
    ax.text(5, 3.2, 'Entropy direction: ALWAYS UP', fontsize=11, 
            ha='center', color=COLORS['entropy'], fontweight='bold')
    
    # Bottom text
    ax.text(5, 1.5, 'The demon manipulates the wrong quantity', fontsize=10, 
            ha='center', color='white', style='italic')
    ax.text(5, 0.8, 'HEAT is not ENTROPY', fontsize=12, 
            ha='center', color=COLORS['arrow'], fontweight='bold')


def main():
    """Generate the 9-panel heat-entropy decoupling visualization."""
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
    panel_collision_setup(ax1)
    panel_case1(ax2)
    panel_case2(ax3)
    panel_case3(ax4)
    panel_decoupling(ax5)
    panel_maxwell_error(ax6)
    panel_statistical_vs_categorical(ax7)
    panel_demon_misdirection(ax8)
    panel_summary(ax9)
    
    # Main title
    fig.suptitle('Heat-Entropy Decoupling: Why the Demon Attacks the Wrong Quantity',
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'resolution' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'heat_entropy_decoupling_panel.png'
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    # Save results summary
    results = {
        'panel_type': 'heat_entropy_decoupling',
        'title': 'Heat-Entropy Decoupling',
        'key_insights': [
            'Heat can flow in either direction in individual collisions',
            'Entropy ALWAYS increases regardless of heat direction',
            'Case 1 (bounce back): Heat hot->cold, entropy up',
            'Case 2 (cold accelerates): Heat hot->cold, entropy up',
            'Case 3 (cold decelerates): Heat COLD->HOT, entropy STILL up',
            'Heat is a statistical emergent property (fluctuating)',
            'Entropy is a categorical fundamental property (monotonic)',
            'Second Law constrains entropy, not heat direction',
            'Maxwell conflated heat and entropy',
            'The demon attacks the wrong quantity'
        ],
        'fundamental_decoupling': {
            'heat': 'statistical average, can fluctuate either direction',
            'entropy': 'categorical completion, always non-decreasing',
            'second_law': 'constrains entropy, not heat'
        },
        'collision_cases': [
            {'name': 'bounce_back', 'heat_direction': 'hot->cold', 'entropy': 'increase'},
            {'name': 'cold_accelerates', 'heat_direction': 'hot->cold', 'entropy': 'increase'},
            {'name': 'cold_decelerates', 'heat_direction': 'cold->hot', 'entropy': 'increase'}
        ]
    }
    
    results_path = output_dir / 'heat_entropy_decoupling_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    
    plt.close()


if __name__ == '__main__':
    main()

