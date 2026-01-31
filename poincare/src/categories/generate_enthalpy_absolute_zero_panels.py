"""
Generate panel charts for the new sections:
1. Categorical Enthalpy Through Partition Dynamics
2. Absolute Zero as the Boundary of Time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Rectangle, Polygon
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from pathlib import Path

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
    'aperture': '#4ade80',
    'partition': '#64748b',
    'energy': '#f59e0b',
    'classical': '#60a5fa',
    'categorical': '#10b981',
    'boundary': '#f43f5e',
    'time': '#8b5cf6',
}


def generate_categorical_enthalpy_panel():
    """Generate 6-panel chart for categorical enthalpy section."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')

    # Panel A: Standard enthalpy - uniform PV work
    ax = axes[0, 0]
    ax.set_title('A. Standard Enthalpy: Uniform PV Work', fontweight='bold', color=COLORS['primary'])

    # System box
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 0.4, 0.6, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['dark'], edgecolor=COLORS['classical'], lw=3))
    ax.text(0.4, 0.5, 'System\nU', ha='center', va='center', fontsize=12,
           fontweight='bold', color=COLORS['classical'])

    # Uniform pressure arrows
    for y in [0.3, 0.5, 0.7]:
        ax.annotate('', xy=(0.18, y), xytext=(0.05, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['energy'], lw=2))
        ax.annotate('', xy=(0.75, y), xytext=(0.62, y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['energy'], lw=2))

    ax.text(0.05, 0.85, 'P', fontsize=14, fontweight='bold', color=COLORS['energy'])
    ax.text(0.8, 0.85, 'P', fontsize=14, fontweight='bold', color=COLORS['energy'])

    ax.text(0.4, 0.08, 'H = U + PV', ha='center', fontsize=12,
           fontweight='bold', color=COLORS['highlight'])
    ax.text(0.4, 0.02, 'Uniform resistance', ha='center', fontsize=9, color=COLORS['dim'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel B: Categorical enthalpy - aperture work
    ax = axes[0, 1]
    ax.set_title('B. Categorical Enthalpy: Aperture Work', fontweight='bold', color=COLORS['primary'])

    # System with partitions and apertures
    ax.add_patch(Rectangle((0.1, 0.2), 0.8, 0.6, fill=False,
                           edgecolor=COLORS['partition'], lw=2))

    # Partitions
    ax.plot([0.4, 0.4], [0.2, 0.5], color=COLORS['partition'], lw=4)
    ax.plot([0.4, 0.4], [0.6, 0.8], color=COLORS['partition'], lw=4)
    ax.plot([0.6, 0.6], [0.2, 0.35], color=COLORS['partition'], lw=4)
    ax.plot([0.6, 0.6], [0.45, 0.65], color=COLORS['partition'], lw=4)
    ax.plot([0.6, 0.6], [0.75, 0.8], color=COLORS['partition'], lw=4)

    # Apertures (gaps) - highlighted
    ax.scatter([0.4], [0.55], s=200, c=COLORS['aperture'], marker='s', alpha=0.5)
    ax.scatter([0.6], [0.4], s=150, c=COLORS['aperture'], marker='s', alpha=0.5)
    ax.scatter([0.6], [0.7], s=100, c=COLORS['aperture'], marker='s', alpha=0.5)

    # Molecules passing through
    ax.annotate('', xy=(0.48, 0.55), xytext=(0.32, 0.55),
               arrowprops=dict(arrowstyle='->', color=COLORS['tertiary'], lw=2))

    ax.text(0.5, 0.08, 'H = U + Sum(n_a * Phi_a)', ha='center', fontsize=11,
           fontweight='bold', color=COLORS['highlight'])
    ax.text(0.5, 0.02, 'Selective apertures', ha='center', fontsize=9, color=COLORS['dim'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel C: Aperture selectivity and potential
    ax = axes[0, 2]
    ax.set_title('C. Selectivity-Potential Relationship', fontweight='bold', color=COLORS['primary'])

    # Plot Phi vs selectivity s
    s = np.linspace(0.01, 1, 100)
    phi = -np.log(s)  # Normalized potential

    ax.plot(s, phi, color=COLORS['aperture'], lw=3)
    ax.fill_between(s, 0, phi, alpha=0.2, color=COLORS['aperture'])

    ax.axhline(y=0, color=COLORS['dim'], lw=1, ls='--')
    ax.axvline(x=1, color=COLORS['dim'], lw=1, ls='--')

    ax.set_xlabel('Selectivity s = Omega_pass/Omega_total', fontsize=9)
    ax.set_ylabel('Potential Phi(a)', fontsize=9)

    ax.text(0.1, 4, 's->0: high barrier', fontsize=9, color=COLORS['secondary'])
    ax.text(0.7, 0.5, 's=1: no barrier', fontsize=9, color=COLORS['tertiary'])

    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.5, 5)

    # Panel D: Bond as aperture
    ax = axes[1, 0]
    ax.set_title('D. Chemical Bond = Aperture', fontweight='bold', color=COLORS['primary'])

    # Two atoms with bond (aperture between them)
    ax.add_patch(Circle((0.3, 0.5), 0.12, fill=True, color=COLORS['secondary'], alpha=0.8))
    ax.add_patch(Circle((0.7, 0.5), 0.12, fill=True, color=COLORS['classical'], alpha=0.8))
    ax.text(0.3, 0.5, 'A', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(0.7, 0.5, 'B', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Bond region = aperture
    ax.add_patch(FancyBboxPatch((0.42, 0.42), 0.16, 0.16, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['aperture'], edgecolor='white', lw=2, alpha=0.5))
    ax.text(0.5, 0.5, 'Bond\n= Aperture', ha='center', va='center', fontsize=8,
           fontweight='bold', color='black')

    # Third molecule trying to approach
    ax.add_patch(Circle((0.5, 0.85), 0.08, fill=True, color=COLORS['quaternary'], alpha=0.8))
    ax.text(0.5, 0.85, 'C', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.75),
               arrowprops=dict(arrowstyle='->', color=COLORS['quaternary'], lw=2))
    ax.text(0.65, 0.72, 'Selected\nby aperture?', fontsize=8, color=COLORS['quaternary'])

    ax.text(0.5, 0.15, 'Bond selects what can approach', ha='center', fontsize=9,
           color=COLORS['highlight'])
    ax.text(0.5, 0.05, 'dH_rxn = Sum(Phi_broken) - Sum(Phi_formed)', ha='center',
           fontsize=9, color=COLORS['dim'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel E: Enzyme catalysis - aperture creation/destruction
    ax = axes[1, 1]
    ax.set_title('E. Enzyme: Balanced Aperture Cycle', fontweight='bold', color=COLORS['primary'])

    # Cycle: E -> E+S -> ES -> EP -> E+P -> E
    states = [(0.5, 0.85, 'E'), (0.15, 0.6, 'E+S'), (0.15, 0.3, 'ES*'),
              (0.85, 0.3, 'EP'), (0.85, 0.6, 'E+P'), (0.5, 0.85, 'E')]

    # Draw cycle
    for i in range(len(states)-1):
        x1, y1, _ = states[i]
        x2, y2, _ = states[i+1]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2,
                                  connectionstyle="arc3,rad=0.2"))

    # Draw state circles
    for x, y, label in states[:-1]:
        ax.add_patch(Circle((x, y), 0.08, fill=True, color=COLORS['dark'],
                           edgecolor=COLORS['accent'], lw=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
               fontweight='bold', color=COLORS['highlight'])

    # Aperture creation/destruction labels
    ax.text(0.08, 0.45, '+Phi\n(create)', fontsize=8, color=COLORS['aperture'])
    ax.text(0.08, 0.2, 'Active\nsite', fontsize=8, color=COLORS['tertiary'])
    ax.text(0.85, 0.45, '-Phi\n(destroy)', fontsize=8, color=COLORS['secondary'])

    ax.text(0.5, 0.05, 'dH_enzyme = +Phi - Phi = 0', ha='center', fontsize=10,
           fontweight='bold', color=COLORS['highlight'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel F: Classical limit
    ax = axes[1, 2]
    ax.set_title('F. Classical Limit: PV from Many Apertures', fontweight='bold', color=COLORS['primary'])

    # Many tiny apertures becoming uniform
    # Left: discrete apertures
    ax.add_patch(Rectangle((0.05, 0.3), 0.35, 0.4, fill=False,
                           edgecolor=COLORS['partition'], lw=2))
    for i in range(5):
        for j in range(4):
            ax.scatter([0.1 + i*0.06], [0.35 + j*0.1], s=30,
                      c=COLORS['aperture'], marker='s', alpha=0.7)
    ax.text(0.22, 0.2, 'n_a apertures\ns < 1', ha='center', fontsize=9, color=COLORS['aperture'])

    # Arrow
    ax.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=3))
    ax.text(0.5, 0.58, 'n->inf\ns->1', ha='center', fontsize=9, color=COLORS['highlight'])

    # Right: uniform pressure
    ax.add_patch(Rectangle((0.6, 0.3), 0.35, 0.4, fill=False,
                           edgecolor=COLORS['classical'], lw=3))
    ax.add_patch(Rectangle((0.6, 0.3), 0.35, 0.4, fill=True,
                           facecolor=COLORS['classical'], alpha=0.2))
    ax.text(0.775, 0.5, 'Uniform\nP', ha='center', va='center', fontsize=12,
           fontweight='bold', color=COLORS['classical'])
    ax.text(0.775, 0.2, 'H = U + PV', ha='center', fontsize=9, color=COLORS['classical'])

    ax.text(0.5, 0.05, 'PV = lim(Sum n_a * Phi_a) as s->1, n->inf', ha='center',
           fontsize=9, color=COLORS['highlight'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'categorical_enthalpy_panel.png', dpi=300,
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'categorical_enthalpy_panel.png'}")


def generate_absolute_zero_boundary_panel():
    """Generate 6-panel chart for absolute zero boundary section."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a12')

    # Panel A: Standard view - T=0 as lowest temperature
    ax = axes[0, 0]
    ax.set_title('A. Standard View: T=0 as Temperature', fontweight='bold', color=COLORS['primary'])

    # Temperature scale
    temps = [0, 1, 2, 3, 4, 5]
    y_pos = 0.5

    for i, t in enumerate(temps):
        x = 0.15 + i * 0.14
        color = COLORS['classical'] if t > 0 else COLORS['boundary']
        ax.scatter([x], [y_pos], s=200, c=color, marker='o')
        ax.text(x, y_pos - 0.15, f'{t}K', ha='center', fontsize=10, color=color)

    ax.plot([0.1, 0.9], [y_pos, y_pos], color=COLORS['dim'], lw=2)
    ax.annotate('', xy=(0.9, y_pos), xytext=(0.85, y_pos),
               arrowprops=dict(arrowstyle='->', color=COLORS['dim'], lw=2))

    ax.text(0.15, y_pos + 0.15, 'T=0\n"lowest"', ha='center', fontsize=9, color=COLORS['boundary'])
    ax.text(0.5, 0.15, 'Standard: T=0 is on the scale', ha='center', fontsize=10,
           color=COLORS['dim'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel B: Categorical view - T=0 as boundary
    ax = axes[0, 1]
    ax.set_title('B. Categorical View: T=0 as Boundary', fontweight='bold', color=COLORS['primary'])

    # Temperature scale with boundary
    temps = [1, 2, 3, 4, 5]
    y_pos = 0.5

    # Boundary line
    ax.axvline(x=0.2, color=COLORS['boundary'], lw=4, ls='-')
    ax.text(0.2, 0.75, 'BOUNDARY', ha='center', fontsize=10, fontweight='bold',
           color=COLORS['boundary'], rotation=90)

    for i, t in enumerate(temps):
        x = 0.3 + i * 0.12
        ax.scatter([x], [y_pos], s=150, c=COLORS['classical'], marker='o')
        ax.text(x, y_pos - 0.12, f'{t}K', ha='center', fontsize=9, color=COLORS['classical'])

    ax.plot([0.25, 0.9], [y_pos, y_pos], color=COLORS['dim'], lw=2)

    # Left of boundary = no time
    ax.add_patch(Rectangle((0, 0.3), 0.2, 0.4, fill=True,
                           facecolor=COLORS['boundary'], alpha=0.2))
    ax.text(0.1, 0.5, 'No\nTime', ha='center', va='center', fontsize=10,
           fontweight='bold', color=COLORS['boundary'])

    ax.text(0.5, 0.15, 'T=0 is NOT on the scale', ha='center', fontsize=10,
           color=COLORS['highlight'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel C: Process cannot reach timeless destination
    ax = axes[0, 2]
    ax.set_title('C. Process Cannot Reach Timeless Point', fontweight='bold', color=COLORS['primary'])

    # Process trajectory approaching but never reaching
    t = np.linspace(0, 10, 100)
    T = 5 * np.exp(-t/3) + 0.1  # Asymptotic to T_min > 0

    ax.plot(t/10, T/5, color=COLORS['time'], lw=3)

    # Boundary
    ax.axhline(y=0, color=COLORS['boundary'], lw=3, ls='-')
    ax.fill_between([0, 1], [-0.1, -0.1], [0, 0], color=COLORS['boundary'], alpha=0.3)
    ax.text(0.5, -0.05, 'T=0: Timeless', ha='center', fontsize=9, color=COLORS['boundary'])

    # Asymptotic approach
    ax.annotate('', xy=(0.95, 0.05), xytext=(0.7, 0.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['tertiary'], lw=2,
                              connectionstyle="arc3,rad=-0.3"))
    ax.text(0.85, 0.25, 'Never\narrives', fontsize=9, color=COLORS['tertiary'])

    ax.set_xlabel('Process Evolution', fontsize=9)
    ax.set_ylabel('Temperature', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.1)

    ax.text(0.5, 1.0, 'Journey requires time; destination has none',
           ha='center', fontsize=9, color=COLORS['highlight'], style='italic')

    # Panel D: Time jump paradox
    ax = axes[1, 0]
    ax.set_title('D. Time Jump Paradox', fontweight='bold', color=COLORS['primary'])

    # Timeline
    ax.plot([0.1, 0.9], [0.5, 0.5], color=COLORS['time'], lw=2)

    # Universe time markers
    times = ['t1', 't2', 't3', 't4', 't5']
    for i, t in enumerate(times):
        x = 0.15 + i * 0.15
        ax.scatter([x], [0.5], s=100, c=COLORS['time'], marker='|', zorder=10)
        ax.text(x, 0.42, t, ha='center', fontsize=9, color=COLORS['time'])

    # Object at T=0
    ax.add_patch(Circle((0.5, 0.75), 0.08, fill=True, color=COLORS['boundary']))
    ax.text(0.5, 0.75, 'O', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(0.65, 0.75, 'At T=0:\nNo internal time', fontsize=8, color=COLORS['boundary'])

    # Disconnect from timeline
    ax.plot([0.5, 0.5], [0.5, 0.67], color=COLORS['boundary'], lw=2, ls='--')
    ax.text(0.55, 0.58, 'Disconnected', fontsize=8, color=COLORS['dim'])

    ax.text(0.5, 0.2, 'Object experiences instant "jump"\nfrom t1 to... when?', ha='center',
           fontsize=9, color=COLORS['highlight'])
    ax.text(0.5, 0.08, 'Cannot leave T=0 (leaving requires time)', ha='center',
           fontsize=9, color=COLORS['secondary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel E: Poincare incompatibility
    ax = axes[1, 1]
    ax.set_title('E. Poincare Incompatibility', fontweight='bold', color=COLORS['primary'])

    # Recurrence cycle
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.3
    x_cycle = 0.5 + r * np.cos(theta)
    y_cycle = 0.55 + r * np.sin(theta) * 0.8

    ax.plot(x_cycle, y_cycle, color=COLORS['tertiary'], lw=2, ls='--', alpha=0.5)
    ax.text(0.5, 0.55, 'Poincare\nRecurrence', ha='center', va='center', fontsize=9,
           color=COLORS['tertiary'])

    # Start point
    ax.scatter([0.5 + r], [0.55], s=150, c=COLORS['tertiary'], marker='o', zorder=10)
    ax.text(0.85, 0.55, 'Start', fontsize=9, color=COLORS['tertiary'])

    # T=0 trap
    ax.add_patch(Circle((0.5 - r, 0.55), 0.08, fill=True, color=COLORS['boundary']))
    ax.text(0.5 - r, 0.55, 'T=0', ha='center', va='center', fontsize=8,
           fontweight='bold', color='white')

    # Broken cycle
    ax.plot([0.5 - r + 0.08, 0.5 - r + 0.15], [0.55, 0.55], color=COLORS['secondary'], lw=3)
    ax.text(0.5 - r + 0.12, 0.48, 'X', fontsize=14, fontweight='bold', color=COLORS['secondary'])

    ax.text(0.5, 0.15, 'At T=0: no evolution, no return', ha='center', fontsize=9,
           color=COLORS['highlight'])
    ax.text(0.5, 0.05, 'Poincare requires time; T=0 has none', ha='center',
           fontsize=9, color=COLORS['secondary'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel F: Boundary equivalence
    ax = axes[1, 2]
    ax.set_title('F. Boundary Equivalence', fontweight='bold', color=COLORS['primary'])

    # Four equivalent descriptions of one boundary
    labels = ['T = 0', 'S = 0', 'tau = undef', 'Singularity']
    colors_eq = [COLORS['boundary'], COLORS['tertiary'], COLORS['time'], COLORS['accent']]
    descriptions = ['No temperature', 'No arrangements', 'No time', '|C| = 1']

    # Central equivalence
    ax.add_patch(Circle((0.5, 0.5), 0.15, fill=True, color=COLORS['dark'],
                        edgecolor=COLORS['highlight'], lw=3))
    ax.text(0.5, 0.5, 'ONE\nBOUNDARY', ha='center', va='center', fontsize=10,
           fontweight='bold', color=COLORS['highlight'])

    # Four views
    positions = [(0.15, 0.8), (0.85, 0.8), (0.15, 0.2), (0.85, 0.2)]
    for (x, y), label, color, desc in zip(positions, labels, colors_eq, descriptions):
        ax.add_patch(FancyBboxPatch((x-0.1, y-0.08), 0.2, 0.16, boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='white', lw=1, alpha=0.7))
        ax.text(x, y + 0.02, label, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white')
        ax.text(x, y - 0.04, desc, ha='center', va='center', fontsize=7, color='white')

        # Connect to center
        ax.plot([x, 0.5], [y - 0.08 if y > 0.5 else y + 0.08,
                          0.5 + 0.15 if y > 0.5 else 0.5 - 0.15],
               color=color, lw=1.5, ls='--', alpha=0.5)

    ax.text(0.5, 0.02, 'Four views of one unreachable boundary', ha='center',
           fontsize=10, color=COLORS['highlight'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'absolute_zero_boundary_panel.png', dpi=300,
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'absolute_zero_boundary_panel.png'}")


if __name__ == "__main__":
    print("Generating Enthalpy and Absolute Zero Panels...")
    print("=" * 60)

    generate_categorical_enthalpy_panel()
    generate_absolute_zero_boundary_panel()

    print("=" * 60)
    print("All panels generated successfully!")

