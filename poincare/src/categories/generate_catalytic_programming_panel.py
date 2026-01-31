#!/usr/bin/env python3
"""
Generate panel chart for Catalytic Programming paradigm.

This visualization demonstrates:
1. Programs as catalytic structures (apertures, not instructions)
2. Execution as gas dynamics
3. Solutions as equilibrium states
4. Comparison with traditional programming
5. Velocity independence
6. Conservation and termination
7. Autocatalytic acceleration
8. Le Chatelier and problem perturbation
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
    'instruction': '#ff6b6b',     # Red for instruction-based
    'catalytic': '#4ecdc4',       # Teal for catalytic
    'aperture': '#ffd93d',        # Yellow for apertures
    'partition': '#8b8b8b',       # Gray for partitions
    'gas': '#a78bfa',             # Purple for gas molecules
    'equilibrium': '#95e1a3',     # Green for equilibrium
    'arrow': '#ff9f43',           # Orange for arrows
    'accent': '#f8b4d9',          # Pink accent
    'background': '#1a1a2e',
    'text': '#e0e0e0'
}


def draw_molecule(ax, x, y, color, size=0.1, velocity=None, alpha=1.0):
    """Draw a gas molecule with optional velocity arrow."""
    mol = Circle((x, y), size, facecolor=color, edgecolor='white', 
                 linewidth=1, alpha=alpha, zorder=5)
    ax.add_patch(mol)
    
    if velocity is not None:
        vx, vy = velocity
        ax.annotate('', xy=(x + vx, y + vy), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1, alpha=0.7))


def panel_paradigm_comparison(ax):
    """Panel A: Compare instruction-based vs catalytic programming."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Two Programming Paradigms', fontsize=11, fontweight='bold', pad=10)
    
    # Instruction-based (left)
    ax.add_patch(FancyBboxPatch((0.3, 5), 4.2, 4.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['instruction'], linewidth=2))
    ax.text(2.4, 9, 'INSTRUCTION-BASED', fontsize=9, ha='center', 
            color=COLORS['instruction'], fontweight='bold')
    
    # Instruction boxes
    for i, txt in enumerate(['fetch', 'decode', 'execute', 'store']):
        y = 8 - i * 0.8
        ax.add_patch(Rectangle((1, y-0.25), 2.8, 0.5, facecolor=COLORS['instruction'], 
                               alpha=0.3, edgecolor='white'))
        ax.text(2.4, y, txt, fontsize=8, ha='center', va='center', color='white')
        if i < 3:
            ax.annotate('', xy=(2.4, y-0.35), xytext=(2.4, y-0.65),
                       arrowprops=dict(arrowstyle='->', color='white', lw=1))
    
    # Catalytic (right)
    ax.add_patch(FancyBboxPatch((5.5, 5), 4.2, 4.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a2a', alpha=0.6,
                 edgecolor=COLORS['catalytic'], linewidth=2))
    ax.text(7.6, 9, 'CATALYTIC', fontsize=9, ha='center', 
            color=COLORS['catalytic'], fontweight='bold')
    
    # Partition with apertures
    ax.add_patch(Rectangle((7.3, 5.5), 0.15, 2.5, facecolor=COLORS['partition']))
    for y in [6, 7, 7.8]:
        ax.add_patch(Circle((7.375, y), 0.12, facecolor=COLORS['aperture']))
    
    # Gas molecules on both sides
    np.random.seed(42)
    for _ in range(5):
        x = np.random.uniform(5.8, 7)
        y = np.random.uniform(5.8, 8)
        draw_molecule(ax, x, y, COLORS['gas'], size=0.08)
    for _ in range(3):
        x = np.random.uniform(7.7, 9.2)
        y = np.random.uniform(5.8, 8)
        draw_molecule(ax, x, y, COLORS['gas'], size=0.08)
    
    # Labels
    ax.text(6.4, 5.3, 'gas', fontsize=7, color=COLORS['gas'])
    ax.text(8.3, 5.3, 'dynamics', fontsize=7, color=COLORS['gas'])
    ax.text(7.375, 8.5, 'apertures', fontsize=7, ha='center', color=COLORS['aperture'])
    
    # Bottom comparison
    ax.text(2.4, 4.3, 'Sequential', fontsize=9, ha='center', color=COLORS['instruction'])
    ax.text(7.6, 4.3, 'Equilibrium', fontsize=9, ha='center', color=COLORS['catalytic'])


def panel_program_as_catalyst(ax):
    """Panel B: Program = Catalytic Structure."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B. Program = Aperture Geometry', fontsize=11, fontweight='bold', pad=10)
    
    # Main equation
    ax.text(5, 8.5, 'Program(P) = {Partitions, Apertures}', fontsize=11, 
            ha='center', color='white', fontweight='bold')
    
    # Visual: Complex partition structure
    # Multiple partitions
    partitions = [
        [(2, 3), (2, 7)],  # Vertical left
        [(5, 4), (5, 6)],  # Vertical middle
        [(8, 3), (8, 7)],  # Vertical right
    ]
    
    for (x1, y1), (x2, y2) in partitions:
        ax.plot([x1, x2], [y1, y2], color=COLORS['partition'], lw=4)
    
    # Apertures (holes in partitions)
    apertures = [
        (2, 4.5, 0.2), (2, 5.5, 0.15),
        (5, 5, 0.25),
        (8, 4, 0.18), (8, 6, 0.2),
    ]
    for x, y, r in apertures:
        ax.add_patch(Circle((x, y), r, facecolor=COLORS['aperture'], 
                           edgecolor='white', linewidth=1, zorder=10))
    
    # Gas molecules flowing through
    np.random.seed(123)
    for _ in range(8):
        x = np.random.uniform(1, 9)
        y = np.random.uniform(3.5, 6.5)
        draw_molecule(ax, x, y, COLORS['gas'], size=0.1, alpha=0.7)
    
    # Arrows showing flow
    ax.annotate('', xy=(3.8, 5), xytext=(2.5, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2))
    ax.annotate('', xy=(6.8, 5), xytext=(5.5, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2))
    
    # Labels
    ax.text(5, 2.3, 'Geometry defines constraints', fontsize=9, 
            ha='center', color=COLORS['text'], style='italic')
    ax.text(5, 1.5, 'Gas finds equilibrium', fontsize=9, 
            ha='center', color=COLORS['equilibrium'])


def panel_velocity_independence(ax):
    """Panel C: Aperture traversal is velocity-independent."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('C. Velocity Independence', fontsize=11, fontweight='bold', pad=10)
    
    # Two scenarios: fast and slow, both traverse
    scenarios = [
        ('Fast', 8, 2.5, COLORS['instruction']),
        ('Slow', 5.5, 1.0, COLORS['catalytic']),
    ]
    
    for i, (label, y, speed, color) in enumerate(scenarios):
        # Partition
        ax.add_patch(Rectangle((4.9, y-0.8), 0.2, 1.6, facecolor=COLORS['partition']))
        # Aperture
        ax.add_patch(Circle((5, y), 0.25, facecolor=COLORS['aperture'], 
                           edgecolor='white', linewidth=2))
        
        # Molecule approaching
        mol_x = 3 if i == 0 else 3.5
        draw_molecule(ax, mol_x, y, color, size=0.18)
        
        # Velocity arrow (different lengths)
        arrow_len = 1.5 if speed > 1.5 else 0.8
        ax.annotate('', xy=(mol_x + arrow_len, y), xytext=(mol_x + 0.25, y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=3))
        
        # Result: both pass
        draw_molecule(ax, 7, y, color, size=0.18, alpha=0.6)
        ax.text(7, y + 0.5, 'PASS', fontsize=9, ha='center', 
                color=COLORS['equilibrium'], fontweight='bold')
        
        ax.text(1.5, y, label, fontsize=10, ha='center', va='center', 
                color=color, fontweight='bold')
    
    # Key insight
    ax.add_patch(FancyBboxPatch((1, 4), 8, 1.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['accent'], linewidth=2))
    ax.text(5, 4.75, 'Fits through = PASS (regardless of speed)', fontsize=10, 
            ha='center', color='white')


def panel_equilibrium_solution(ax):
    """Panel D: Solution = Equilibrium State."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Solution = Equilibrium', fontsize=11, fontweight='bold', pad=10)
    
    # Two containers with partition
    ax.add_patch(FancyBboxPatch((0.5, 3), 3.5, 4,
                 boxstyle="round,pad=0.05",
                 facecolor=COLORS['gas'], alpha=0.1,
                 edgecolor=COLORS['gas'], linewidth=2))
    ax.add_patch(FancyBboxPatch((6, 3), 3.5, 4,
                 boxstyle="round,pad=0.05",
                 facecolor=COLORS['gas'], alpha=0.1,
                 edgecolor=COLORS['gas'], linewidth=2))
    
    # Partition with aperture
    ax.add_patch(Rectangle((4.4, 3), 0.2, 4, facecolor=COLORS['partition']))
    ax.add_patch(Circle((4.5, 5), 0.3, facecolor=COLORS['aperture'], 
                        edgecolor='white', linewidth=2))
    
    # Molecules at equilibrium (roughly equal distribution)
    np.random.seed(77)
    for _ in range(4):
        x = np.random.uniform(1, 3.5)
        y = np.random.uniform(3.5, 6.5)
        draw_molecule(ax, x, y, COLORS['gas'], size=0.15)
    for _ in range(4):
        x = np.random.uniform(6.5, 9)
        y = np.random.uniform(3.5, 6.5)
        draw_molecule(ax, x, y, COLORS['gas'], size=0.15)
    
    # Bidirectional arrows (equal rates)
    ax.annotate('', xy=(5.5, 5.3), xytext=(3.5, 5.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['catalytic'], lw=2))
    ax.annotate('', xy=(3.5, 4.7), xytext=(5.5, 4.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['catalytic'], lw=2))
    
    ax.text(4.5, 7.5, 'Rate_fwd = Rate_rev', fontsize=10, ha='center', 
            color=COLORS['equilibrium'], fontweight='bold')
    
    # Labels
    ax.text(2.25, 2.5, 'Problem', fontsize=10, ha='center', color=COLORS['gas'])
    ax.text(7.75, 2.5, 'Solution', fontsize=10, ha='center', color=COLORS['equilibrium'])
    
    ax.text(5, 1.5, 'Equilibrium IS the answer', fontsize=10, 
            ha='center', color='white', style='italic')


def panel_catalyst_ignorance(ax):
    """Panel E: Catalysts don't know the answer."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('E. Catalyst Ignorance', fontsize=11, fontweight='bold', pad=10)
    
    # Big aperture with question mark
    ax.add_patch(Circle((5, 5.5), 1.5, facecolor=COLORS['aperture'], 
                        edgecolor='white', linewidth=3, alpha=0.8))
    ax.text(5, 5.5, '?', fontsize=50, ha='center', va='center', 
            color='#333', fontweight='bold')
    
    # Text explanations
    ax.text(5, 8.5, 'Aperture defines: WHERE you can go', fontsize=10, 
            ha='center', color=COLORS['aperture'], fontweight='bold')
    ax.text(5, 7.8, 'NOT: WHAT the answer is', fontsize=10, 
            ha='center', color=COLORS['text'])
    
    # Molecules approaching from different angles
    angles = [30, 150, 210, 330]
    for angle in angles:
        rad = np.radians(angle)
        x = 5 + 3 * np.cos(rad)
        y = 5.5 + 2.5 * np.sin(rad)
        draw_molecule(ax, x, y, COLORS['gas'], size=0.15)
        # Arrow toward center
        ax.annotate('', xy=(5 + 1.8*np.cos(rad), 5.5 + 1.8*np.sin(rad)), 
                   xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1, alpha=0.5))
    
    ax.text(5, 2.2, 'Solution emerges from dynamics', fontsize=10, 
            ha='center', color=COLORS['equilibrium'], style='italic')
    ax.text(5, 1.5, 'not from catalyst "knowing"', fontsize=9, 
            ha='center', color=COLORS['text'])


def panel_conservation_termination(ax):
    """Panel F: Conservation causes termination."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('F. Conservation -> Termination', fontsize=11, fontweight='bold', pad=10)
    
    # Progression showing why equilibrium is reached
    states = [
        ('Start', 4, 2),
        ('A scores', 2, 4),
        ('Equilibrium', 3, 3),
    ]
    
    for i, (label, na, nb) in enumerate(states):
        y = 7.5 - i * 2.3
        
        # Mini containers
        ax.add_patch(Rectangle((1, y-0.5), 2.5, 1, facecolor=COLORS['gas'], 
                               alpha=0.15, edgecolor=COLORS['gas']))
        ax.add_patch(Rectangle((5.5, y-0.5), 2.5, 1, facecolor=COLORS['gas'], 
                               alpha=0.15, edgecolor=COLORS['gas']))
        
        # Partition + aperture
        ax.add_patch(Rectangle((3.9, y-0.5), 0.2, 1, facecolor=COLORS['partition']))
        ax.add_patch(Circle((4, y), 0.15, facecolor=COLORS['aperture']))
        
        # Molecules
        np.random.seed(i*100)
        for j in range(na):
            ax.add_patch(Circle((1.3 + j*0.5, y), 0.12, facecolor=COLORS['gas']))
        for j in range(nb):
            ax.add_patch(Circle((5.8 + j*0.5, y), 0.12, facecolor=COLORS['gas']))
        
        ax.text(0.5, y, label, fontsize=8, va='center', color='white')
        ax.text(9, y, f'{na}:{nb}', fontsize=9, va='center', color=COLORS['text'])
    
    # Arrows
    ax.annotate('', xy=(4, 6.2), xytext=(4, 6.8),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.annotate('', xy=(4, 3.9), xytext=(4, 4.5),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    # Conservation equation
    ax.text(5, 1.3, 'n_A + n_B = N (constant)', fontsize=10, 
            ha='center', color=COLORS['accent'])
    ax.text(5, 0.6, 'Cannot empty one side -> must reach equilibrium', fontsize=9, 
            ha='center', color=COLORS['text'])


def panel_autocatalytic(ax):
    """Panel G: Autocatalytic acceleration."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('G. Autocatalytic Feedback', fontsize=11, fontweight='bold', pad=10)
    
    # Resistance curve decreasing
    x = np.linspace(0.5, 9, 50)
    R = 5 * np.exp(-0.4 * x) + 1
    
    ax.plot(x, R + 2, color=COLORS['instruction'], lw=3, label='Resistance')
    ax.fill_between(x, 2, R + 2, color=COLORS['instruction'], alpha=0.2)
    
    # Labels
    ax.text(1, 8, 'R (resistance)', fontsize=9, color=COLORS['instruction'])
    ax.text(8.5, 3, 'decreases', fontsize=9, color=COLORS['instruction'])
    
    # Arrow showing each transit reduces R
    ax.annotate('Each transit', xy=(3, 5), xytext=(1.5, 6.5),
               fontsize=8, color=COLORS['text'],
               arrowprops=dict(arrowstyle='->', color=COLORS['text']))
    ax.annotate('reduces R', xy=(6, 3.5), xytext=(4.5, 5),
               fontsize=8, color=COLORS['text'],
               arrowprops=dict(arrowstyle='->', color=COLORS['text']))
    
    # Equation
    ax.text(5, 1.2, 'dR/dt < 0 : positive feedback', fontsize=10, 
            ha='center', color=COLORS['equilibrium'], fontweight='bold')
    
    ax.axhline(y=2, color='white', linestyle=':', alpha=0.3)
    ax.text(9.3, 2, '0', fontsize=8, color=COLORS['text'])


def panel_le_chatelier(ax):
    """Panel H: Le Chatelier for problem modification."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('H. Problem Perturbation', fontsize=11, fontweight='bold', pad=10)
    
    # Original equilibrium
    ax.add_patch(FancyBboxPatch((0.5, 6), 9, 2.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a1a2e', alpha=0.6,
                 edgecolor=COLORS['equilibrium'], linewidth=1))
    ax.text(5, 8, 'Original Equilibrium', fontsize=9, ha='center', 
            color=COLORS['equilibrium'])
    
    # Show perturbation
    ax.text(5, 7, '+ Add constraints  -> shift right', fontsize=9, 
            ha='center', color=COLORS['text'])
    ax.text(5, 6.3, '- Remove constraints -> shift left', fontsize=9, 
            ha='center', color=COLORS['text'])
    
    # Arrow to new equilibrium
    ax.annotate('', xy=(7, 4), xytext=(5, 5.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2))
    ax.text(7.5, 4.3, 'New', fontsize=9, color=COLORS['arrow'])
    ax.text(7.5, 3.7, 'Equilibrium', fontsize=9, color=COLORS['arrow'])
    
    # Key insight
    ax.add_patch(FancyBboxPatch((1, 1), 8, 1.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.8,
                 edgecolor=COLORS['equilibrium'], linewidth=2))
    ax.text(5, 1.8, 'No restart: system adjusts incrementally', fontsize=10, 
            ha='center', color='white')


def panel_summary_table(ax):
    """Panel I: Summary comparison table."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('I. Paradigm Summary', fontsize=11, fontweight='bold', pad=10)
    
    # Table data
    rows = [
        ('Program', 'Instructions', 'Apertures'),
        ('Execute', 'Step-by-step', 'Gas dynamics'),
        ('Terminate', 'Halt', 'Equilibrium'),
        ('Speed', 'FLOPS', 'Irrelevant'),
        ('Solution', 'Computed', 'Emergent'),
    ]
    
    # Header
    ax.text(5.5, 8.5, 'Traditional', fontsize=9, ha='center', 
            color=COLORS['instruction'], fontweight='bold')
    ax.text(8, 8.5, 'Catalytic', fontsize=9, ha='center', 
            color=COLORS['catalytic'], fontweight='bold')
    
    ax.axhline(y=8, xmin=0.1, xmax=0.9, color='white', alpha=0.3)
    
    for i, (aspect, trad, cat) in enumerate(rows):
        y = 7.2 - i * 1.2
        ax.text(2, y, aspect, fontsize=9, ha='left', color='white', fontweight='bold')
        ax.text(5.5, y, trad, fontsize=9, ha='center', color=COLORS['instruction'])
        ax.text(8, y, cat, fontsize=9, ha='center', color=COLORS['catalytic'])
        
        if i < len(rows) - 1:
            ax.axhline(y=y-0.5, xmin=0.1, xmax=0.9, color='white', alpha=0.1)
    
    # Bottom text
    ax.text(5, 1, 'Declarative | Geometric | Equilibrium-based', fontsize=10, 
            ha='center', color=COLORS['equilibrium'], fontweight='bold')


def main():
    """Generate the 9-panel catalytic programming visualization."""
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
    panel_paradigm_comparison(ax1)
    panel_program_as_catalyst(ax2)
    panel_velocity_independence(ax3)
    panel_equilibrium_solution(ax4)
    panel_catalyst_ignorance(ax5)
    panel_conservation_termination(ax6)
    panel_autocatalytic(ax7)
    panel_le_chatelier(ax8)
    panel_summary_table(ax9)
    
    # Main title
    fig.suptitle('Catalytic Programming: Apertures as Programs, Equilibrium as Solutions',
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'docs' / 'poincare-computing' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'catalytic_programming_panel.png'
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    # Save results summary
    results = {
        'panel_type': 'catalytic_programming',
        'title': 'Catalytic Programming Paradigm',
        'key_insights': [
            'Programs are catalytic structures (partitions + apertures), not instructions',
            'Execution is gas dynamics, not step-by-step operation',
            'Solutions are equilibrium states, not computed outputs',
            'Aperture traversal is velocity-independent (geometry only)',
            'Catalysts are ignorant of solutions - they define constraints, not answers',
            'Conservation prevents complete emptying -> forces equilibrium termination',
            'Autocatalytic feedback: each transit reduces resistance',
            'Le Chatelier governs problem modification: incremental adjustment, no restart'
        ],
        'paradigm_properties': {
            'type': 'declarative',
            'control': 'geometric',
            'termination': 'equilibrium-based',
            'speed_measure': 'irrelevant',
            'solution_type': 'emergent'
        },
        'theorems': [
            'Program-Catalyst Correspondence',
            'Equilibrium-Penultimate Equivalence',
            'No Velocity Dependence',
            'Computational Termination from Conservation',
            'Autocatalytic Acceleration',
            'Catalyst Ignorance',
            'Problem Perturbation via Le Chatelier'
        ]
    }
    
    results_path = output_dir / 'catalytic_programming_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    
    plt.close()


if __name__ == '__main__':
    main()

