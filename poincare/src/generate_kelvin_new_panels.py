"""
Generate panel charts for the new sections of the Kelvin Paradox paper:
1. Asymmetric Branching
2. Dark Matter Termination
3. Emergent Time
4. Heat Death Self-Refutation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, Wedge, FancyArrowPatch
from matplotlib.patches import ConnectionPatch, Polygon
import matplotlib.patches as mpatches
from pathlib import Path
import os

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "kelvin-paradox" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.facecolor': '#0a0a12',
    'axes.facecolor': '#12121f',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'axes.edgecolor': '#3a3a5a',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
})

# Color palette
COLORS = {
    'primary': '#7eb8da',
    'secondary': '#da7e7e',
    'tertiary': '#7eda9c',
    'quaternary': '#dac27e',
    'accent': '#b87eda',
    'highlight': '#ffffff',
    'dim': '#5a5a7a',
    'dark': '#2a2a3a',
    'forward': '#4ade80',
    'backward': '#f87171',
    'terminated': '#60a5fa',
    'non_terminated': '#a855f7',
    'kinetic': '#f59e0b',
    'categorical': '#10b981',
}


def generate_asymmetric_branching_panel():
    """Generate 6-panel chart for asymmetric branching section."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')
    
    # Panel A: Event actualisation resolving infinite non-possibilities
    ax = axes[0, 0]
    ax.set_title('A. Actualisation Resolves Non-Possibilities', fontweight='bold', color=COLORS['primary'])
    
    # Draw the "can happen" possibilities (finite)
    can_happen = ['Fall', 'Stay', 'Pushed']
    cannot_happen = ['Gold', 'Sentient', 'Fly', '...∞']
    
    # Central event
    ax.scatter([0.5], [0.5], s=400, c=COLORS['quaternary'], zorder=10, marker='o')
    ax.text(0.5, 0.5, 'Event', ha='center', va='center', fontsize=9, fontweight='bold', color='black')
    
    # Can happen - green arrows pointing out (finite)
    for i, event in enumerate(can_happen):
        angle = np.pi/2 + i * np.pi/4
        x_end = 0.5 + 0.35 * np.cos(angle)
        y_end = 0.5 + 0.35 * np.sin(angle)
        ax.annotate('', xy=(x_end, y_end), xytext=(0.5, 0.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))
        ax.text(x_end + 0.08*np.cos(angle), y_end + 0.08*np.sin(angle), event, 
               ha='center', va='center', fontsize=8, color=COLORS['forward'])
    
    # Cannot happen - red arrows pointing out (infinite)
    for i, event in enumerate(cannot_happen):
        angle = -np.pi/2 - i * np.pi/6
        x_end = 0.5 + 0.35 * np.cos(angle)
        y_end = 0.5 + 0.35 * np.sin(angle)
        ax.annotate('', xy=(x_end, y_end), xytext=(0.5, 0.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=1.5, ls='--'))
        ax.text(x_end + 0.1*np.cos(angle), y_end + 0.1*np.sin(angle), event, 
               ha='center', va='center', fontsize=8, color=COLORS['backward'])
    
    ax.text(0.5, 0.02, '|Can| = finite,  |Cannot| = ∞', ha='center', fontsize=9, 
           color=COLORS['highlight'], style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel B: Forward vs backward branching ratio
    ax = axes[0, 1]
    ax.set_title('B. Branching Ratio: Forward/Backward → ∞', fontweight='bold', color=COLORS['primary'])
    
    # Draw tree structure - forward explosion
    levels = 4
    for level in range(levels):
        n_nodes = 3 ** level if level < 3 else 20  # Cap visual nodes
        y = 0.85 - level * 0.2
        x_positions = np.linspace(0.1, 0.9, min(n_nodes, 20))
        
        for x in x_positions:
            size = 80 / (level + 1)
            ax.scatter([x], [y], s=size, c=COLORS['forward'], alpha=0.7)
        
        # Connect to parent (simplified)
        if level > 0:
            parent_y = 0.85 - (level-1) * 0.2
            parent_x = np.linspace(0.1, 0.9, min(3**(level-1), 7))
            for px in parent_x[:min(len(parent_x), 5)]:
                ax.plot([px, px-0.05, px+0.05], [parent_y-0.02, y+0.02, y+0.02], 
                       'g-', alpha=0.3, lw=0.5)
    
    # Backward - single thin line
    ax.annotate('', xy=(0.15, 0.85), xytext=(0.15, 0.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2))
    ax.text(0.08, 0.55, 'Back:\nO(1)', ha='center', fontsize=8, color=COLORS['backward'])
    ax.text(0.75, 0.55, 'Forward:\n∞ + O(n)', ha='center', fontsize=8, color=COLORS['forward'])
    
    ax.text(0.5, 0.02, 'Ratio → ∞: Irreversibility is categorical', ha='center', 
           fontsize=9, color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel C: Category self-division C/C ≠ 1
    ax = axes[0, 2]
    ax.set_title('C. Category Self-Division: C/C ≠ 1', fontweight='bold', color=COLORS['primary'])
    
    # Show C_0 -> traverse -> C_0' ≠ C_0
    ax.add_patch(Circle((0.2, 0.6), 0.1, fill=True, color=COLORS['quaternary'], alpha=0.8))
    ax.text(0.2, 0.6, 'C₀', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.annotate('', xy=(0.45, 0.6), xytext=(0.32, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))
    ax.text(0.38, 0.68, 'traverse', ha='center', fontsize=8, color=COLORS['dim'])
    
    ax.add_patch(Circle((0.55, 0.6), 0.08, fill=True, color=COLORS['accent'], alpha=0.8))
    ax.text(0.55, 0.6, '...', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.annotate('', xy=(0.75, 0.6), xytext=(0.65, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['backward'], lw=2, ls='--'))
    ax.text(0.70, 0.68, '"return"', ha='center', fontsize=8, color=COLORS['dim'])
    
    ax.add_patch(Circle((0.85, 0.6), 0.1, fill=True, color=COLORS['secondary'], alpha=0.8))
    ax.text(0.85, 0.6, "C₀'", ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Inequality
    ax.text(0.5, 0.35, "C₀/C₀ = C₀' ≠ C₀", ha='center', fontsize=14, 
           fontweight='bold', color=COLORS['highlight'])
    ax.text(0.5, 0.2, 'Residue: record of non-actualisations', ha='center', fontsize=9, 
           color=COLORS['tertiary'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel D: Information content - broken cup > intact cup
    ax = axes[1, 0]
    ax.set_title('D. Information: Broken Cup > Intact Cup', fontweight='bold', color=COLORS['primary'])
    
    # Intact cup - simple
    ax.add_patch(FancyBboxPatch((0.1, 0.5), 0.15, 0.2, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['tertiary'], edgecolor='white', lw=2))
    ax.text(0.175, 0.6, 'CUP', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.175, 0.35, 'Intact\nI = I₀', ha='center', fontsize=9, color=COLORS['tertiary'])
    
    # Arrow
    ax.annotate('', xy=(0.45, 0.6), xytext=(0.3, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['dim'], lw=2))
    ax.text(0.375, 0.7, 'falls', ha='center', fontsize=9, color=COLORS['dim'])
    
    # Broken cup - complex (multiple fragments)
    fragments_x = [0.55, 0.65, 0.75, 0.60, 0.70, 0.80]
    fragments_y = [0.55, 0.65, 0.50, 0.45, 0.55, 0.60]
    for fx, fy in zip(fragments_x, fragments_y):
        ax.scatter([fx], [fy], s=100, marker='s', c=COLORS['secondary'], alpha=0.8)
    
    ax.text(0.675, 0.35, 'Broken\nI = I₀ + |didn\'t|', ha='center', fontsize=9, color=COLORS['secondary'])
    
    # Didn't happen facts
    didnt = ["X gold", "X sentient", "X fly", "X ...∞"]
    for i, d in enumerate(didnt):
        ax.text(0.95, 0.8 - i*0.12, d, ha='right', fontsize=8, color=COLORS['backward'], alpha=0.7)
    
    ax.text(0.5, 0.08, 'Determined facts > Undetermined possibilities', ha='center', 
           fontsize=9, color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel E: Accumulation of "didn't happen"
    ax = axes[1, 1]
    ax.set_title("E. Entropy = Accumulated 'Didn't Happen'", fontweight='bold', color=COLORS['primary'])
    
    # Time axis with growing shadow
    time_points = np.linspace(0.1, 0.9, 6)
    events = np.cumsum(np.ones(6))
    non_events = events * 100  # Each event resolves ~100x non-events
    
    # Plot events (small)
    ax.bar(time_points - 0.02, events/max(events) * 0.3, width=0.03, 
          color=COLORS['forward'], alpha=0.8, label='Actualised')
    
    # Plot non-events (large shadow)
    ax.bar(time_points + 0.02, non_events/max(non_events) * 0.8, width=0.03, 
          color=COLORS['backward'], alpha=0.5, label='Non-actualised')
    
    ax.axhline(y=0, color=COLORS['dim'], lw=1)
    ax.set_xlabel('Cosmic Time →', fontsize=9)
    ax.set_ylabel('Categorical Information', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.text(0.5, -0.15, 'S = |resolved non-actualisations|', ha='center', 
           fontsize=9, color=COLORS['highlight'], transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Panel F: Why reversal is impossible
    ax = axes[1, 2]
    ax.set_title('F. Why Reversal Is Impossible', fontweight='bold', color=COLORS['primary'])
    
    # Show that un-resolving is impossible
    ax.text(0.5, 0.85, 'To reverse:', ha='center', fontsize=10, color=COLORS['highlight'])
    
    steps = [
        ('1. Return C\' → C', COLORS['forward'], 0.7),
        ('2. Un-resolve "didn\'t gold"', COLORS['backward'], 0.55),
        ('3. Un-resolve "didn\'t fly"', COLORS['backward'], 0.45),
        ('4. Un-resolve ∞ more...', COLORS['backward'], 0.35),
    ]
    
    for text, color, y in steps:
        if 'Un-resolve' in text:
            ax.text(0.5, y, text + ' [X]', ha='center', fontsize=9, color=color)
        else:
            ax.text(0.5, y, text + ' ?', ha='center', fontsize=9, color=color)
    
    ax.text(0.5, 0.15, '"Did not happen" cannot become\n"undetermined non-possibility"', 
           ha='center', fontsize=9, color=COLORS['highlight'], style='italic')
    
    ax.text(0.5, 0.02, 'Determined facts are irreducible', ha='center', 
           fontsize=9, color=COLORS['tertiary'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'asymmetric_branching_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'asymmetric_branching_panel.png'}")


def generate_dark_matter_termination_panel():
    """Generate 6-panel chart for dark matter termination section."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')
    
    # Panel A: Terminated vs non-terminated oscillation trajectories
    ax = axes[0, 0]
    ax.set_title('A. Terminated vs Non-Terminated Oscillations', fontweight='bold', color=COLORS['primary'])
    
    t = np.linspace(0, 4*np.pi, 200)
    
    # Terminated oscillation - damped, reaches endpoint
    y_term = np.exp(-t/8) * np.sin(t) * 0.4
    ax.plot(t/(4*np.pi), y_term + 0.7, color=COLORS['terminated'], lw=2, label='Terminated')
    ax.scatter([1], [0.7], s=100, c=COLORS['terminated'], marker='*', zorder=10)
    ax.text(1.02, 0.72, 'endpoint', fontsize=8, color=COLORS['terminated'])
    
    # Non-terminated oscillation - continues forever
    y_non = np.sin(t*1.5) * 0.3
    ax.plot(t/(4*np.pi), y_non + 0.3, color=COLORS['non_terminated'], lw=2, label='Non-terminated', alpha=0.7)
    ax.text(0.85, 0.1, '→ ∞', fontsize=12, color=COLORS['non_terminated'])
    
    ax.axhline(y=0.7, color=COLORS['dim'], ls=':', lw=1, alpha=0.5)
    ax.axhline(y=0.3, color=COLORS['dim'], ls=':', lw=1, alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Process Evolution →', fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Panel B: Observer can only "see" terminated endpoints
    ax = axes[0, 1]
    ax.set_title('B. Observers See Only Terminated States', fontweight='bold', color=COLORS['primary'])
    
    # Observer eye
    ax.scatter([0.15], [0.5], s=300, marker='o', c=COLORS['quaternary'])
    ax.text(0.15, 0.5, 'O', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.15, 0.35, 'Observer', ha='center', fontsize=8, color=COLORS['quaternary'])
    
    # Sight lines to terminated
    for i, y in enumerate([0.75, 0.5, 0.25]):
        ax.plot([0.25, 0.6], [0.5, y], color=COLORS['terminated'], lw=1, alpha=0.5)
        ax.scatter([0.6], [y], s=80, c=COLORS['terminated'], marker='*')
    
    ax.text(0.7, 0.5, 'Terminated\n(visible)', ha='center', fontsize=9, color=COLORS['terminated'])
    
    # Barrier
    ax.axvline(x=0.75, color=COLORS['dim'], lw=3, ls='--')
    ax.text(0.75, 0.9, 'Termination\nBoundary', ha='center', fontsize=8, color=COLORS['dim'])
    
    # Non-terminated (behind barrier)
    ax.scatter([0.85, 0.9, 0.88], [0.6, 0.4, 0.5], s=60, c=COLORS['non_terminated'], marker='o', alpha=0.4)
    ax.text(0.88, 0.25, 'Non-terminated\n(invisible)', ha='center', fontsize=9, color=COLORS['non_terminated'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel C: Dark matter properties from non-termination
    ax = axes[0, 2]
    ax.set_title('C. Dark Matter = Non-Terminated Processes', fontweight='bold', color=COLORS['primary'])
    
    properties = [
        ('Has gravity', 'Mass-energy exists', COLORS['forward']),
        ('No light', 'No terminated state', COLORS['backward']),
        ('Not detected', 'No endpoint to measure', COLORS['backward']),
        ('~5.4x ordinary', 'Termination ratio', COLORS['quaternary']),
    ]
    
    for i, (prop, reason, color) in enumerate(properties):
        y = 0.85 - i * 0.2
        ax.text(0.05, y, '●', fontsize=14, color=color)
        ax.text(0.15, y, prop, fontsize=10, fontweight='bold', color=color)
        ax.text(0.15, y - 0.06, f'← {reason}', fontsize=8, color=COLORS['dim'])
    
    ax.text(0.5, 0.05, 'Dark matter IS without BEING', ha='center', fontsize=10, 
           color=COLORS['highlight'], style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel D: Resolved non-actualisations as dark matter
    ax = axes[1, 0]
    ax.set_title('D. Dark Matter = "What Didn\'t Happen"', fontweight='bold', color=COLORS['primary'])
    
    # Pie chart showing ratio
    sizes = [5.4, 1]
    colors_pie = [COLORS['non_terminated'], COLORS['terminated']]
    labels = ['Dark\n(non-actualised)', 'Ordinary\n(actualised)']
    
    wedges, texts = ax.pie(sizes, colors=colors_pie, startangle=90,
                           wedgeprops=dict(width=0.5, edgecolor='white'))
    
    ax.text(0, 0, '5.4 : 1', ha='center', va='center', fontsize=14, 
           fontweight='bold', color=COLORS['highlight'])
    
    # Legend
    ax.text(-0.3, -0.7, 'Dark: resolved absences', fontsize=9, color=COLORS['non_terminated'])
    ax.text(-0.3, -0.85, 'Ordinary: actualised presence', fontsize=9, color=COLORS['terminated'])
    
    # Panel E: The 5.4 ratio from termination statistics
    ax = axes[1, 1]
    ax.set_title('E. Ratio from Termination Statistics', fontweight='bold', color=COLORS['primary'])
    
    # Show branching creating more non-terminated
    levels = ['Event', 'Level 1', 'Level 2', 'Level 3']
    terminated = [1, 1, 1, 1]
    non_terminated = [0, 2, 6, 18]
    
    x = np.arange(len(levels))
    width = 0.35
    
    ax.bar(x - width/2, terminated, width, label='Terminated', color=COLORS['terminated'], alpha=0.8)
    ax.bar(x + width/2, non_terminated, width, label='Non-terminated', color=COLORS['non_terminated'], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8)
    
    # Ratio annotation
    ax.text(3, 15, f'Ratio:\n{18/1:.1f}', fontsize=10, color=COLORS['highlight'], ha='center')
    ax.text(0.5, -0.15, 'Each termination creates multiple non-terminations', 
           fontsize=9, color=COLORS['dim'], ha='center', transform=ax.transAxes)
    
    # Panel F: Why detection fails
    ax = axes[1, 2]
    ax.set_title('F. Why Dark Matter Cannot Be Detected', fontweight='bold', color=COLORS['primary'])
    
    # Detection requirements (all fail for non-terminated)
    requirements = [
        ('Terminated state', 'X continuously evolving'),
        ('Definite value', 'X undetermined'),
        ('Info transfer', 'X no endpoint'),
    ]
    
    ax.text(0.5, 0.9, 'Detection requires:', ha='center', fontsize=10, 
           fontweight='bold', color=COLORS['highlight'])
    
    for i, (req, fail) in enumerate(requirements):
        y = 0.7 - i * 0.2
        ax.text(0.15, y, f'{i+1}. {req}', fontsize=10, color=COLORS['forward'])
        ax.text(0.55, y, fail, fontsize=10, color=COLORS['backward'])
    
    ax.text(0.5, 0.15, 'We see dark matter\'s shadow\n(gravitational effects on ordinary matter)\nnot dark matter itself', 
           ha='center', fontsize=9, color=COLORS['dim'], style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dark_matter_termination_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'dark_matter_termination_panel.png'}")


def generate_emergent_time_panel():
    """Generate 6-panel chart for emergent time section."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')
    
    # Panel A: Time emerging from category counting
    ax = axes[0, 0]
    ax.set_title('A. Time = Categorical Completion Count', fontweight='bold', color=COLORS['primary'])
    
    # Categories being filled, time emerging
    n_categories = 10
    x = np.arange(n_categories)
    heights = np.ones(n_categories)
    
    colors_bar = [COLORS['categorical'] if i < 6 else COLORS['dim'] for i in range(n_categories)]
    ax.bar(x, heights, color=colors_bar, edgecolor='white', lw=1)
    
    # Time arrow below
    ax.annotate('', xy=(6, -0.3), xytext=(0, -0.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['quaternary'], lw=3))
    ax.text(3, -0.5, 'τ = |completed categories|', ha='center', fontsize=10, 
           color=COLORS['quaternary'])
    
    # Labels
    ax.text(3, 1.3, 'Filled', ha='center', fontsize=9, color=COLORS['categorical'])
    ax.text(8, 1.3, 'Unfilled', ha='center', fontsize=9, color=COLORS['dim'])
    
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.7, 1.5)
    ax.set_ylabel('Category', fontsize=9)
    ax.set_xticks([])
    
    # Panel B: Constant branching ratio giving uniform time flow
    ax = axes[0, 1]
    ax.set_title('B. Constant Branching Ratio → Uniform Flow', fontweight='bold', color=COLORS['primary'])
    
    # Show 3^k growth but constant ratio
    k_values = np.arange(0, 6)
    categories = 3 ** k_values
    ratios = np.ones(len(k_values) - 1) * 3  # Ratio is always 3
    
    ax.semilogy(k_values, categories, 'o-', color=COLORS['forward'], lw=2, markersize=8, label='|C|')
    ax.set_xlabel('Level k', fontsize=9)
    ax.set_ylabel('Category Count (log)', fontsize=9, color=COLORS['forward'])
    
    ax2 = ax.twinx()
    ax2.plot(k_values[1:], ratios, 's--', color=COLORS['quaternary'], lw=2, markersize=8, label='Ratio')
    ax2.set_ylabel('C(k+1)/C(k)', fontsize=9, color=COLORS['quaternary'])
    ax2.set_ylim(0, 5)
    
    ax.text(0.5, 0.05, 'Ratio = 3 = constant → uniform time', ha='center', fontsize=9, 
           color=COLORS['highlight'], transform=ax.transAxes)
    
    # Panel C: Self-similar structure
    ax = axes[0, 2]
    ax.set_title('C. Self-Similar: Each Level Looks the Same', fontweight='bold', color=COLORS['primary'])
    
    # Draw self-similar fractal-like pattern
    def draw_triangle(ax, x, y, size, depth, max_depth):
        if depth > max_depth:
            return
        # Draw current triangle
        alpha = 1 - depth * 0.2
        ax.scatter([x], [y], s=size*50, c=COLORS['categorical'], alpha=alpha)
        
        # Recurse
        if depth < max_depth:
            offset = size * 0.4
            draw_triangle(ax, x - offset, y - offset*0.8, size*0.5, depth+1, max_depth)
            draw_triangle(ax, x + offset, y - offset*0.8, size*0.5, depth+1, max_depth)
            draw_triangle(ax, x, y + offset*0.5, size*0.5, depth+1, max_depth)
    
    draw_triangle(ax, 0.5, 0.5, 1, 0, 3)
    
    ax.text(0.5, 0.02, 'Same structure at every scale', ha='center', fontsize=9, 
           color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel D: Singularity - no categories, no time
    ax = axes[1, 0]
    ax.set_title('D. At Singularity: Time is Undefined', fontweight='bold', color=COLORS['primary'])
    
    # Timeline with singularity gap
    ax.axhline(y=0.5, xmin=0.1, xmax=0.4, color=COLORS['categorical'], lw=3)
    ax.axhline(y=0.5, xmin=0.6, xmax=0.9, color=COLORS['categorical'], lw=3)
    
    # Singularity point
    ax.scatter([0.5], [0.5], s=200, c=COLORS['accent'], marker='*', zorder=10)
    ax.text(0.5, 0.65, 'Singularity\n|C| = 1', ha='center', fontsize=9, color=COLORS['accent'])
    
    # Time undefined
    ax.text(0.5, 0.3, 'ρ_C = 0 → τ undefined', ha='center', fontsize=10, 
           fontweight='bold', color=COLORS['highlight'])
    
    ax.text(0.15, 0.35, '"Before"?', ha='center', fontsize=9, color=COLORS['dim'])
    ax.text(0.85, 0.35, '"After"?', ha='center', fontsize=9, color=COLORS['dim'])
    
    ax.text(0.5, 0.1, 'The question "what was before?" is malformed', ha='center', 
           fontsize=9, color=COLORS['secondary'], style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel E: Categories begetting categories
    ax = axes[1, 1]
    ax.set_title('E. Categories Beget Categories', fontweight='bold', color=COLORS['primary'])
    
    # Show C -> {C1, C2, C3}
    ax.add_patch(Circle((0.2, 0.5), 0.1, fill=True, color=COLORS['quaternary'], alpha=0.8))
    ax.text(0.2, 0.5, 'C', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows to children
    children = [(0.6, 0.75), (0.6, 0.5), (0.6, 0.25)]
    for i, (cx, cy) in enumerate(children):
        ax.annotate('', xy=(cx-0.08, cy), xytext=(0.3, 0.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))
        ax.add_patch(Circle((cx, cy), 0.07, fill=True, color=COLORS['categorical'], alpha=0.8))
        ax.text(cx, cy, f'C{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Second generation (smaller)
    for cx, cy in children[:2]:
        for j in range(2):
            gcx = cx + 0.2
            gcy = cy + 0.08 - j*0.16
            ax.plot([cx+0.07, gcx], [cy, gcy], color=COLORS['forward'], lw=1, alpha=0.5)
            ax.scatter([gcx], [gcy], s=30, c=COLORS['categorical'], alpha=0.6)
    
    ax.text(0.5, 0.02, 'Self-generating ensures time never stops', ha='center', fontsize=9, 
           color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel F: Arrow of time = direction of completion
    ax = axes[1, 2]
    ax.set_title('F. Arrow of Time = Completion Direction', fontweight='bold', color=COLORS['primary'])
    
    # Past (completed) -> Future (potential)
    ax.add_patch(FancyBboxPatch((0.05, 0.35), 0.35, 0.3, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['categorical'], edgecolor='white', alpha=0.5))
    ax.text(0.225, 0.5, 'PAST\nCompleted\nCategories', ha='center', va='center', fontsize=9, 
           fontweight='bold', color='white')
    
    ax.add_patch(FancyBboxPatch((0.6, 0.35), 0.35, 0.3, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['dim'], edgecolor='white', alpha=0.5))
    ax.text(0.775, 0.5, 'FUTURE\nPotential\nCategories', ha='center', va='center', fontsize=9, 
           fontweight='bold', color='white')
    
    # Arrow of time
    ax.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['quaternary'], lw=4))
    ax.text(0.5, 0.72, 'Arrow of Categorical Completion', ha='center', fontsize=10, 
           fontweight='bold', color=COLORS['quaternary'])
    
    ax.text(0.5, 0.15, 'Completion is irreversible →\nTime is irreversible', ha='center', 
           fontsize=9, color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'emergent_time_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'emergent_time_panel.png'}")


def generate_heat_death_refutation_panel():
    """Generate 6-panel chart for heat death self-refutation section."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')
    
    # Panel A: Requirements for true heat death (all impossible)
    ax = axes[0, 0]
    ax.set_title('A. Requirements for True Heat Death', fontweight='bold', color=COLORS['primary'])
    
    requirements = [
        ('T = 0 K exactly', 'X Third Law'),
        ('No quantum fluctuations', 'X Uncertainty'),
        ('No process of any kind', 'X Requires T=0'),
        ('Permanent persistence', 'X Unstable'),
    ]
    
    ax.text(0.5, 0.92, 'True Stasis Requires:', ha='center', fontsize=10, 
           fontweight='bold', color=COLORS['highlight'])
    
    for i, (req, reason) in enumerate(requirements):
        y = 0.75 - i * 0.18
        ax.text(0.1, y, f'{i+1}. {req}', fontsize=10, color=COLORS['forward'])
        ax.text(0.65, y, reason, fontsize=10, color=COLORS['backward'], fontweight='bold')
    
    ax.text(0.5, 0.05, 'ALL requirements thermodynamically forbidden', ha='center', 
           fontsize=10, color=COLORS['secondary'], style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel B: Actual heat death state T_min > 0
    ax = axes[0, 1]
    ax.set_title('B. Actual "Heat Death" State', fontweight='bold', color=COLORS['primary'])
    
    # Temperature approaching but never reaching 0
    t = np.linspace(0, 10, 100)
    T = 0.5 * np.exp(-t/3) + 0.1  # Asymptotic to T_min > 0
    
    ax.plot(t/10, T, color=COLORS['quaternary'], lw=3)
    ax.axhline(y=0.1, color=COLORS['forward'], ls='--', lw=2, label='T_min > 0')
    ax.axhline(y=0, color=COLORS['backward'], ls=':', lw=2, label='T = 0 (unreachable)')
    
    ax.fill_between(t/10, 0, 0.1, color=COLORS['backward'], alpha=0.2)
    ax.text(0.5, 0.05, 'Forbidden\nZone', ha='center', fontsize=9, color=COLORS['backward'])
    
    ax.set_xlabel('Cosmic Time →', fontsize=9)
    ax.set_ylabel('Temperature', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.7)
    
    # Panel C: Apertures functioning at heat death
    ax = axes[0, 2]
    ax.set_title('C. Apertures Still Function at Heat Death', fontweight='bold', color=COLORS['primary'])
    
    # Draw partition with apertures
    ax.axvline(x=0.5, ymin=0.1, ymax=0.35, color=COLORS['dim'], lw=6)
    ax.axvline(x=0.5, ymin=0.45, ymax=0.55, color=COLORS['dim'], lw=6)
    ax.axvline(x=0.5, ymin=0.65, ymax=0.9, color=COLORS['dim'], lw=6)
    
    # Apertures (gaps)
    ax.scatter([0.5], [0.4], s=200, c=COLORS['categorical'], marker='s', alpha=0.3)
    ax.scatter([0.5], [0.6], s=200, c=COLORS['categorical'], marker='s', alpha=0.3)
    ax.text(0.6, 0.4, 'Aperture', fontsize=8, color=COLORS['categorical'])
    
    # Molecules with vibrations
    for x, y in [(0.25, 0.5), (0.3, 0.7), (0.2, 0.3)]:
        ax.scatter([x], [y], s=100, c=COLORS['terminated'], marker='o')
        # Vibration indicator
        ax.plot([x-0.03, x+0.03], [y, y], color=COLORS['terminated'], lw=1)
        ax.plot([x, x], [y-0.03, y+0.03], color=COLORS['terminated'], lw=1)
    
    # Some passing through
    ax.annotate('', xy=(0.65, 0.4), xytext=(0.35, 0.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))
    
    ax.text(0.5, 0.02, 'Geometric selection continues\nwhen T > 0', ha='center', fontsize=9, 
           color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel D: Self-refutation logical structure
    ax = axes[1, 0]
    ax.set_title('D. The Self-Refutation Logic', fontweight='bold', color=COLORS['primary'])
    
    steps = [
        '1. Heat death defined as "no process"',
        '2. "No process" requires T = 0',
        '3. T = 0 is impossible (Third Law)',
        '4. Therefore T > 0 at "heat death"',
        '5. T > 0 → processes continue',
        '6. Therefore "no process" is FALSE',
    ]
    
    colors_logic = [COLORS['dim'], COLORS['dim'], COLORS['backward'], 
                   COLORS['forward'], COLORS['forward'], COLORS['secondary']]
    
    for i, (step, color) in enumerate(zip(steps, colors_logic)):
        y = 0.9 - i * 0.13
        ax.text(0.1, y, step, fontsize=9, color=color)
    
    ax.text(0.5, 0.08, 'Heat death refutes itself!', ha='center', fontsize=11, 
           fontweight='bold', color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel E: Kinetic death vs categorical death timeline
    ax = axes[1, 1]
    ax.set_title('E. Kinetic Death << Categorical Death', fontweight='bold', color=COLORS['primary'])
    
    # Timeline
    ax.axhline(y=0.5, color=COLORS['dim'], lw=2)
    
    # Markers
    ax.scatter([0.15], [0.5], s=200, c=COLORS['kinetic'], marker='v', zorder=10)
    ax.text(0.15, 0.35, 'Kinetic\nDeath', ha='center', fontsize=9, color=COLORS['kinetic'])
    ax.text(0.15, 0.62, '~10¹⁰⁰ yr', ha='center', fontsize=8, color=COLORS['dim'])
    
    ax.scatter([0.85], [0.5], s=200, c=COLORS['categorical'], marker='v', zorder=10)
    ax.text(0.85, 0.35, 'Categorical\nDeath', ha='center', fontsize=9, color=COLORS['categorical'])
    ax.text(0.85, 0.62, 'N_max\ncategories', ha='center', fontsize=8, color=COLORS['dim'])
    
    # Long era in between
    ax.annotate('', xy=(0.75, 0.5), xytext=(0.25, 0.5),
               arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=3))
    ax.text(0.5, 0.72, 'Long Categorical Era\n(invisible to kinetic measurement)', 
           ha='center', fontsize=9, color=COLORS['accent'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel F: Kelvin's category error
    ax = axes[1, 2]
    ax.set_title("F. Kelvin's Error: Wrong Entropy", fontweight='bold', color=COLORS['primary'])
    
    # Two entropy curves
    t = np.linspace(0, 1, 100)
    S_kin = 1 - np.exp(-5*t)  # Saturates
    S_cat = t ** 0.3  # Continues growing
    
    ax.plot(t, S_kin, color=COLORS['kinetic'], lw=3, label='S_kinetic (saturates)')
    ax.plot(t, S_cat, color=COLORS['categorical'], lw=3, label='S_categorical (grows)')
    
    # Heat death marker
    ax.axvline(x=0.3, color=COLORS['dim'], ls='--', lw=2)
    ax.text(0.32, 0.5, '"Heat\nDeath"', fontsize=8, color=COLORS['dim'])
    
    ax.set_xlabel('Cosmic Time →', fontsize=9)
    ax.set_ylabel('Entropy', fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    
    ax.text(0.5, -0.2, 'Kelvin measured kinetic entropy (saturates)\nMissed categorical entropy (continues)', 
           ha='center', fontsize=9, color=COLORS['highlight'], transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heat_death_refutation_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'heat_death_refutation_panel.png'}")


if __name__ == "__main__":
    print("Generating Kelvin Paradox New Section Panels...")
    print("=" * 60)
    
    generate_asymmetric_branching_panel()
    generate_dark_matter_termination_panel()
    generate_emergent_time_panel()
    generate_heat_death_refutation_panel()
    
    print("=" * 60)
    print("All panels generated successfully!")

