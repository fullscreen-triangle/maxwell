"""
Generate panel visualizations for the Kelvin Paradox resolution paper.
Each section gets its own informative panel chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Wedge, Rectangle, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "kelvin-paradox" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'primary': '#1a5f7a',
    'secondary': '#c84b31',
    'tertiary': '#57837b',
    'quaternary': '#f5eedc',
    'accent': '#e8d5b7',
    'dark': '#2c3639',
    'light': '#f5f5f5',
    'oscillation': '#3498db',
    'category': '#e74c3c',
    'entropy': '#2ecc71',
    'dark_matter': '#9b59b6',
    'singularity': '#f39c12'
}


def generate_oscillatory_reality_panel():
    """Generate panel for oscillatory reality section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Bounded system phase space (Poincaré recurrence)
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 20*np.pi, 2000)
    r = 1 + 0.3*np.sin(3*theta) + 0.1*np.sin(7*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax1.plot(x, y, color=COLORS['oscillation'], alpha=0.7, linewidth=0.5)
    ax1.scatter([x[0]], [y[0]], color=COLORS['secondary'], s=100, zorder=5, label='Initial')
    ax1.scatter([x[-1]], [y[-1]], color=COLORS['entropy'], s=100, zorder=5, marker='s', label='Return')
    circle = Circle((0, 0), 1.5, fill=False, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax1.add_patch(circle)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_title('(A) Poincaré Recurrence\nin Bounded Phase Space', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    # Panel B: Quantum wavefunction oscillation
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 4*np.pi, 500)
    psi1 = np.sin(x) * np.exp(-0.1*x)
    psi2 = np.sin(2*x) * np.exp(-0.1*x)
    psi_total = psi1 + 0.5*psi2
    ax2.fill_between(x, 0, np.abs(psi_total)**2, alpha=0.3, color=COLORS['oscillation'])
    ax2.plot(x, psi_total, color=COLORS['primary'], linewidth=2, label='$\\psi(x,t)$')
    ax2.plot(x, np.abs(psi_total)**2, color=COLORS['secondary'], linewidth=2, label='$|\\psi|^2$')
    ax2.axhline(y=0, color=COLORS['dark'], linewidth=0.5)
    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title('(B) Quantum Wavefunction\nOscillation', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, 4*np.pi)
    
    # Panel C: Molecular vibrational modes
    ax3 = fig.add_subplot(gs[0, 2])
    t = np.linspace(0, 2*np.pi, 100)
    # Draw molecule (diatomic)
    ax3.scatter([0.3, 0.7], [0.7, 0.7], s=500, color=COLORS['tertiary'], zorder=5)
    ax3.plot([0.3, 0.7], [0.7, 0.7], color=COLORS['dark'], linewidth=3)
    # Vibration arrows
    for i, amp in enumerate([0.1, 0.15, 0.12]):
        ax3.annotate('', xy=(0.3 - amp, 0.7), xytext=(0.3 + amp, 0.7),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2))
        ax3.annotate('', xy=(0.7 - amp, 0.7), xytext=(0.7 + amp, 0.7),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2))
    # Energy levels
    for i, y in enumerate([0.2, 0.3, 0.4, 0.5]):
        ax3.axhline(y=y, xmin=0.1, xmax=0.4, color=COLORS['oscillation'], linewidth=2)
        ax3.text(0.05, y, f'$n={i}$', fontsize=9, va='center')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('(C) Molecular Vibrational\nModes at $T > 0$', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Vibrational configuration transitions
    ax4 = fig.add_subplot(gs[1, 0])
    # Show transitions between vibrational states
    states = ['$(0,0,1)$', '$(0,1,0)$', '$(1,0,0)$', '$(0,1,1)$', '$(1,0,1)$']
    y_pos = np.linspace(0.8, 0.2, 5)
    for i, (state, y) in enumerate(zip(states, y_pos)):
        ax4.add_patch(FancyBboxPatch((0.2, y-0.05), 0.6, 0.1, boxstyle="round,pad=0.02",
                                     facecolor=COLORS['quaternary'], edgecolor=COLORS['primary'], linewidth=2))
        ax4.text(0.5, y, state, ha='center', va='center', fontsize=11, fontweight='bold')
        if i < len(states) - 1:
            ax4.annotate('', xy=(0.5, y_pos[i+1]+0.08), xytext=(0.5, y-0.08),
                        arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Vibrational Configuration\nTransitions', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Temperature vs oscillation persistence
    ax5 = fig.add_subplot(gs[1, 1])
    T = np.linspace(0.01, 300, 100)
    E_vib = 0.5 * 1.38e-23 * T * 1e23  # Normalized energy per mode
    ax5.fill_between(T, 0, E_vib, alpha=0.3, color=COLORS['entropy'])
    ax5.plot(T, E_vib, color=COLORS['entropy'], linewidth=3, label='$\\langle E_{vib} \\rangle$')
    ax5.axhline(y=0, color=COLORS['dark'], linestyle='--', linewidth=1)
    ax5.axvline(x=2.7, color=COLORS['secondary'], linestyle=':', linewidth=2, label='CMB (2.7K)')
    ax5.fill_betweenx([0, E_vib.max()], 0, 2.7, alpha=0.2, color=COLORS['secondary'])
    ax5.set_xlabel('Temperature (K)', fontsize=10)
    ax5.set_ylabel('Vibrational Energy (arb.)', fontsize=10)
    ax5.set_title('(E) Oscillation Persists\nat All $T > 0$', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.set_xlim(0, 300)
    
    # Panel F: Third Law barrier
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    # Barrier at T=0
    r_barrier = np.ones_like(theta) * 0.1
    r_current = np.ones_like(theta) * 0.7
    ax6.fill_between(theta, 0, r_barrier, alpha=0.8, color=COLORS['secondary'], label='T=0 (Unreachable)')
    ax6.fill_between(theta, r_barrier, r_current, alpha=0.3, color=COLORS['entropy'], label='Accessible')
    ax6.plot(theta, r_current, color=COLORS['entropy'], linewidth=3)
    ax6.plot(theta, r_barrier, color=COLORS['secondary'], linewidth=3, linestyle='--')
    ax6.set_rticks([])
    ax6.set_title('(F) Third Law Barrier\nPrevents T=0', fontsize=11, fontweight='bold', pad=20)
    ax6.legend(loc='lower right', fontsize=8, bbox_to_anchor=(1.2, 0))
    
    plt.suptitle('Oscillatory Foundation of Physical Reality', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'oscillatory_reality_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'oscillatory_reality_panel.png'}")


def generate_topology_categories_panel():
    """Generate panel for topology of categories section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Partial order structure
    ax1 = fig.add_subplot(gs[0, 0])
    # Draw Hasse diagram
    nodes = {
        'C0': (0.5, 0.1),
        'C1': (0.2, 0.35),
        'C2': (0.5, 0.35),
        'C3': (0.8, 0.35),
        'C4': (0.35, 0.6),
        'C5': (0.65, 0.6),
        'C6': (0.5, 0.85)
    }
    edges = [('C0', 'C1'), ('C0', 'C2'), ('C0', 'C3'), 
             ('C1', 'C4'), ('C2', 'C4'), ('C2', 'C5'), ('C3', 'C5'),
             ('C4', 'C6'), ('C5', 'C6')]
    for (n1, n2) in edges:
        ax1.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], 
                color=COLORS['primary'], linewidth=2)
    for name, (x, y) in nodes.items():
        ax1.scatter([x], [y], s=400, color=COLORS['tertiary'], zorder=5, edgecolor=COLORS['dark'], linewidth=2)
        ax1.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('(A) Partial Order\n(Completion Precedence)', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Tri-dimensional S-space
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    # Draw coordinate axes
    ax2.quiver(0, 0, 0, 1, 0, 0, color=COLORS['oscillation'], arrow_length_ratio=0.1, linewidth=3)
    ax2.quiver(0, 0, 0, 0, 1, 0, color=COLORS['entropy'], arrow_length_ratio=0.1, linewidth=3)
    ax2.quiver(0, 0, 0, 0, 0, 1, color=COLORS['secondary'], arrow_length_ratio=0.1, linewidth=3)
    ax2.text(1.1, 0, 0, '$\\mathcal{S}_k$', fontsize=12, color=COLORS['oscillation'])
    ax2.text(0, 1.1, 0, '$\\mathcal{S}_t$', fontsize=12, color=COLORS['entropy'])
    ax2.text(0, 0, 1.1, '$\\mathcal{S}_e$', fontsize=12, color=COLORS['secondary'])
    # Draw a point
    ax2.scatter([0.5], [0.5], [0.5], s=200, color=COLORS['singularity'], edgecolor=COLORS['dark'])
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0, 1.2)
    ax2.set_zlim(0, 1.2)
    ax2.set_title('(B) Tri-Dimensional\nS-Space', fontsize=11, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    
    # Panel C: 3^k branching tree
    ax3 = fig.add_subplot(gs[0, 2])
    def draw_tree(ax, x, y, level, max_level, width):
        if level >= max_level:
            return
        colors = [COLORS['oscillation'], COLORS['entropy'], COLORS['secondary']]
        labels = ['k', 't', 'e']
        dx = width / 3
        for i in range(3):
            new_x = x + (i - 1) * dx
            new_y = y - 0.25
            ax.plot([x, new_x], [y, new_y], color=colors[i], linewidth=2)
            ax.scatter([new_x], [new_y], s=100, color=colors[i], zorder=5)
            if level == 0:
                ax.text(new_x, new_y - 0.08, labels[i], ha='center', fontsize=9)
            draw_tree(ax, new_x, new_y, level + 1, max_level, width / 3)
    ax3.scatter([0.5], [0.9], s=200, color=COLORS['primary'], zorder=5)
    ax3.text(0.5, 0.95, '$\\mathcal{C}$', ha='center', fontsize=11, fontweight='bold')
    draw_tree(ax3, 0.5, 0.9, 0, 3, 0.8)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('(C) $3^k$ Branching\nStructure', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Scale ambiguity
    ax4 = fig.add_subplot(gs[1, 0])
    # Draw two identical structures at different "scales"
    for i, (cx, cy, scale, label) in enumerate([(0.25, 0.5, 0.15, 'Level n'), (0.75, 0.5, 0.15, 'Level n+1')]):
        # Draw triangle
        triangle = np.array([[cx, cy + scale], [cx - scale, cy - scale], [cx + scale, cy - scale], [cx, cy + scale]])
        ax4.plot(triangle[:, 0], triangle[:, 1], color=COLORS['primary'], linewidth=2)
        ax4.scatter([cx, cx - scale, cx + scale], [cy + scale, cy - scale, cy - scale], s=100, color=COLORS['tertiary'])
        ax4.text(cx, cy - 0.25, label, ha='center', fontsize=10, fontweight='bold')
    # Arrow showing isometry
    ax4.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=3))
    ax4.text(0.5, 0.55, '$\\Psi_n$', ha='center', fontsize=12, fontweight='bold', color=COLORS['secondary'])
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Scale Ambiguity:\nIdentical Structure', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Completion trajectory
    ax5 = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 10, 100)
    gamma = 1 - np.exp(-0.3 * t)  # Normalized completion fraction
    ax5.fill_between(t, 0, gamma, alpha=0.3, color=COLORS['entropy'])
    ax5.plot(t, gamma, color=COLORS['entropy'], linewidth=3, label='$|\\gamma(t)|/|\\mathcal{C}|$')
    ax5.axhline(y=1, color=COLORS['secondary'], linestyle='--', linewidth=2, label='Complete')
    ax5.scatter([0], [0], s=100, color=COLORS['primary'], zorder=5)
    ax5.scatter([10], [gamma[-1]], s=100, color=COLORS['entropy'], zorder=5)
    ax5.set_xlabel('Time', fontsize=10)
    ax5.set_ylabel('Fraction Completed', fontsize=10)
    ax5.set_title('(E) Completion Trajectory\n$\\gamma(t)$ Expanding', fontsize=11, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=8)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 1.1)
    
    # Panel F: Asymptotic slowing
    ax6 = fig.add_subplot(gs[1, 2])
    t = np.linspace(0.1, 10, 100)
    rate = 0.3 * np.exp(-0.3 * t)  # Completion rate
    ax6.fill_between(t, 0, rate, alpha=0.3, color=COLORS['secondary'])
    ax6.plot(t, rate, color=COLORS['secondary'], linewidth=3, label='$\\dot{C}(t)$')
    ax6.axhline(y=0, color=COLORS['dark'], linestyle='-', linewidth=1)
    ax6.axvline(x=10, color=COLORS['primary'], linestyle=':', linewidth=2, label='$T$ (completion)')
    ax6.set_xlabel('Time', fontsize=10)
    ax6.set_ylabel('Completion Rate', fontsize=10)
    ax6.set_title('(F) Asymptotic Slowing\n$\\dot{C}(t) \\to 0$', fontsize=11, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.set_xlim(0, 10)
    
    plt.suptitle('Topology of Categorical Spaces', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'topology_categories_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'topology_categories_panel.png'}")


def generate_observer_boundary_panel():
    """Generate panel for observer boundary section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Observer making distinctions
    ax1 = fig.add_subplot(gs[0, 0])
    # Draw observer
    observer = Circle((0.2, 0.5), 0.1, color=COLORS['primary'], zorder=5)
    ax1.add_patch(observer)
    ax1.text(0.2, 0.35, 'Observer', ha='center', fontsize=9)
    # Draw reality
    reality_x = np.linspace(0.4, 0.95, 50)
    for i, x in enumerate(reality_x):
        ax1.scatter([x], [0.5 + 0.1*np.sin(10*x)], s=30, 
                   color=COLORS['quaternary'] if i % 3 == 0 else COLORS['entropy'] if i % 3 == 1 else COLORS['secondary'])
    # Distinction rays
    for angle in [-30, 0, 30]:
        rad = np.radians(angle)
        ax1.annotate('', xy=(0.4, 0.5 + 0.2*np.sin(rad)), xytext=(0.3, 0.5),
                    arrowprops=dict(arrowstyle='->', color=COLORS['oscillation'], lw=1.5))
    ax1.text(0.7, 0.75, 'Reality', ha='center', fontsize=10, style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('(A) Observer Making\nCategorical Distinctions', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Termination requirement
    ax2 = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 4*np.pi, 200)
    # Non-terminated event
    y1 = np.sin(t) * (1 + 0.05*t)
    ax2.plot(t[:150], y1[:150], color=COLORS['tertiary'], linewidth=2, alpha=0.5, linestyle='--', label='Non-terminated')
    # Terminated event
    y2 = np.sin(t) * np.exp(-0.2*t)
    ax2.plot(t, y2, color=COLORS['entropy'], linewidth=2, label='Terminated')
    ax2.scatter([t[np.argmax(np.abs(y2) < 0.1)]], [0], s=100, color=COLORS['secondary'], zorder=5, marker='*')
    ax2.axhline(y=0, color=COLORS['dark'], linewidth=0.5)
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Event Amplitude', fontsize=10)
    ax2.set_title('(B) Termination Requirement:\nOnly Completed Events Observable', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    
    # Panel C: Infinity - x structure
    ax3 = fig.add_subplot(gs[0, 2])
    # Pie chart showing accessible vs inaccessible
    sizes = [84, 16]  # Approximate dark:ordinary ratio
    colors_pie = [COLORS['dark_matter'], COLORS['entropy']]
    explode = (0.05, 0)
    wedges, texts, autotexts = ax3.pie(sizes, explode=explode, colors=colors_pie, autopct='%1.0f%%',
                                        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax3.text(0, -0.15, '$x$\n(Inaccessible)', ha='center', fontsize=10, color=COLORS['dark_matter'])
    ax3.text(0.8, 0.5, '$\\infty - x$\n(Accessible)', ha='center', fontsize=10, color=COLORS['entropy'])
    ax3.set_title('(C) The $\\infty - x$ Structure', fontsize=11, fontweight='bold')
    
    # Panel D: Observer network
    ax4 = fig.add_subplot(gs[1, 0])
    # Draw network of observers
    np.random.seed(42)
    n_observers = 7
    positions = np.random.rand(n_observers, 2) * 0.6 + 0.2
    for i in range(n_observers):
        circle = Circle(positions[i], 0.05, color=COLORS['primary'], zorder=5)
        ax4.add_patch(circle)
        ax4.text(positions[i][0], positions[i][1], f'$O_{i+1}$', ha='center', va='center', 
                fontsize=8, color='white', fontweight='bold')
    # Draw connections
    for i in range(n_observers):
        for j in range(i+1, n_observers):
            if np.random.rand() > 0.3:
                ax4.plot([positions[i][0], positions[j][0]], 
                        [positions[i][1], positions[j][1]], 
                        color=COLORS['oscillation'], linewidth=1, alpha=0.5)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Observer Network\nExchanging Information', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Tetration growth
    ax5 = fig.add_subplot(gs[1, 1])
    levels = np.arange(0, 6)
    # Simplified representation of tetration (log scale)
    log_values = [0, 84, 84*84, 84*84*84, 84*84*84*84, 84*84*84*84*84]  # Simplified
    log_log_values = [np.log10(v + 1) for v in log_values]
    ax5.bar(levels, log_log_values, color=COLORS['category'], edgecolor=COLORS['dark'], linewidth=2)
    ax5.set_xlabel('Recursion Level $t$', fontsize=10)
    ax5.set_ylabel('$\\log_{10}(C(t))$ (arb.)', fontsize=10)
    ax5.set_title('(E) Tetration Growth:\n$C(t+1) = n^{C(t)}$', fontsize=11, fontweight='bold')
    ax5.set_xticks(levels)
    
    # Panel F: Conservation - bathtub analogy
    ax6 = fig.add_subplot(gs[1, 2])
    # Draw bathtub shape
    bathtub_x = [0.1, 0.1, 0.2, 0.8, 0.9, 0.9]
    bathtub_y = [0.8, 0.3, 0.2, 0.2, 0.3, 0.8]
    ax6.fill(bathtub_x, bathtub_y, color=COLORS['quaternary'], edgecolor=COLORS['dark'], linewidth=3)
    # Water level
    water_x = [0.15, 0.15, 0.85, 0.85]
    water_y = [0.6, 0.25, 0.25, 0.6]
    ax6.fill(water_x, water_y, color=COLORS['oscillation'], alpha=0.5)
    # Arrow showing no drain
    ax6.text(0.5, 0.1, 'No Drain', ha='center', fontsize=10, fontweight='bold', color=COLORS['secondary'])
    ax6.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.05),
                arrowprops=dict(arrowstyle='-[', color=COLORS['secondary'], lw=2))
    ax6.text(0.5, 0.45, 'Categories', ha='center', fontsize=10, color='white', fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('(F) Conservation:\nInformation Cannot Be Destroyed', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle('Observer-Dependent Categorical Enumeration', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'observer_boundary_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'observer_boundary_panel.png'}")


def generate_heat_death_panel():
    """Generate panel for heat death section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Temperature asymptote
    ax1 = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 100, 200)
    T = 2.7 * np.exp(-0.02 * t) + 0.01  # Asymptotic to small positive value
    ax1.plot(t, T, color=COLORS['secondary'], linewidth=3)
    ax1.axhline(y=0, color=COLORS['dark'], linestyle='--', linewidth=2, label='$T=0$ (Unreachable)')
    ax1.axhline(y=0.01, color=COLORS['entropy'], linestyle=':', linewidth=2, label='Asymptote')
    ax1.fill_between(t, 0, T, alpha=0.2, color=COLORS['secondary'])
    ax1.set_xlabel('Cosmic Time (arb.)', fontsize=10)
    ax1.set_ylabel('Temperature (K)', fontsize=10)
    ax1.set_title('(A) Temperature Asymptote:\n$T \\to 0^+$, Never Reaches 0', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(-0.5, 3)
    
    # Panel B: Maximum particle separation
    ax2 = fig.add_subplot(gs[0, 1])
    np.random.seed(42)
    n_particles = 30
    # Initial clustered
    x_init = np.random.randn(n_particles) * 0.15 + 0.25
    y_init = np.random.randn(n_particles) * 0.15 + 0.5
    # Final separated
    x_final = np.random.rand(n_particles) * 0.4 + 0.55
    y_final = np.random.rand(n_particles) * 0.8 + 0.1
    ax2.scatter(x_init, y_init, s=50, color=COLORS['primary'], alpha=0.7, label='Initial')
    ax2.scatter(x_final, y_final, s=50, color=COLORS['entropy'], alpha=0.7, label='Heat Death')
    ax2.annotate('', xy=(0.5, 0.5), xytext=(0.35, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=3))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('(B) Maximum Particle\nSeparation', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.axis('off')
    
    # Panel C: Vibrational transitions in static config
    ax3 = fig.add_subplot(gs[0, 2])
    # Draw fixed molecule
    ax3.scatter([0.5], [0.5], s=500, color=COLORS['tertiary'], zorder=5, edgecolor=COLORS['dark'], linewidth=2)
    ax3.text(0.5, 0.35, 'Fixed Position', ha='center', fontsize=9)
    # Vibrational states changing
    states = ['$v_1$', '$v_2$', '$v_3$', '$v_4$']
    for i, (angle, state) in enumerate(zip(np.linspace(0, 2*np.pi, 5)[:-1], states)):
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        ax3.annotate('', xy=(x, y), xytext=(0.5, 0.5),
                    arrowprops=dict(arrowstyle='->', color=COLORS['oscillation'], lw=1.5))
        ax3.text(x + 0.1*np.cos(angle), y + 0.1*np.sin(angle), state, ha='center', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('(C) Vibrational Transitions\nat Fixed Position', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Categorical enumeration from heat death
    ax4 = fig.add_subplot(gs[1, 0])
    # Show exponential tower
    levels = ['$10^{80}$', '$(10^{84})^{10^{80}}$', '...', '$N_{max}$']
    y_pos = [0.2, 0.45, 0.65, 0.85]
    for y, label in zip(y_pos, levels):
        ax4.add_patch(FancyBboxPatch((0.2, y-0.08), 0.6, 0.12, boxstyle="round,pad=0.02",
                                     facecolor=COLORS['category'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.7))
        ax4.text(0.5, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    for i in range(len(y_pos)-1):
        ax4.annotate('', xy=(0.5, y_pos[i+1]-0.1), xytext=(0.5, y_pos[i]+0.06),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
    ax4.text(0.5, 0.05, 'Heat Death Base', ha='center', fontsize=10, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Categorical Enumeration\nBegins at Heat Death', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Kinetic vs categorical activity
    ax5 = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 10, 100)
    # Before heat death: both active
    kinetic = np.ones_like(t[:50]) * 0.8
    categorical = np.linspace(0.2, 0.5, 50)
    # After heat death: only categorical
    kinetic_after = np.exp(-0.5*(t[50:]-5)) * 0.8
    categorical_after = 0.5 + 0.4*(1 - np.exp(-0.3*(t[50:]-5)))
    ax5.plot(t[:50], kinetic, color=COLORS['secondary'], linewidth=3, label='Kinetic')
    ax5.plot(t[50:], kinetic_after, color=COLORS['secondary'], linewidth=3)
    ax5.plot(t[:50], categorical, color=COLORS['entropy'], linewidth=3, label='Categorical')
    ax5.plot(t[50:], categorical_after, color=COLORS['entropy'], linewidth=3)
    ax5.axvline(x=5, color=COLORS['dark'], linestyle='--', linewidth=2)
    ax5.text(5, 0.95, 'Heat Death', ha='center', fontsize=9, fontweight='bold')
    ax5.set_xlabel('Cosmic Time', fontsize=10)
    ax5.set_ylabel('Activity Level', fontsize=10)
    ax5.set_title('(E) Kinetic Stasis vs.\nCategorical Hyperactivity', fontsize=11, fontweight='bold')
    ax5.legend(loc='center right', fontsize=8)
    ax5.set_ylim(0, 1)
    
    # Panel F: Progression to singularity
    ax6 = fig.add_subplot(gs[1, 2])
    # Draw timeline
    stages = ['Big Bang', 'Expansion', 'Heat Death', 'Cat. Completion', 'Singularity']
    x_pos = np.linspace(0.1, 0.9, 5)
    ax6.plot(x_pos, [0.5]*5, color=COLORS['dark'], linewidth=3)
    for i, (x, stage) in enumerate(zip(x_pos, stages)):
        color = COLORS['singularity'] if i in [0, 4] else COLORS['entropy'] if i == 2 else COLORS['primary']
        ax6.scatter([x], [0.5], s=200, color=color, zorder=5, edgecolor=COLORS['dark'], linewidth=2)
        ax6.text(x, 0.35 if i % 2 == 0 else 0.65, stage, ha='center', fontsize=9, fontweight='bold', rotation=30)
    # Arrow showing cycle
    ax6.annotate('', xy=(0.15, 0.7), xytext=(0.85, 0.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2,
                               connectionstyle="arc3,rad=0.3"))
    ax6.text(0.5, 0.85, 'Cyclic Return', ha='center', fontsize=10, fontweight='bold', color=COLORS['secondary'])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('(F) Progression:\nHeat Death to Singularity', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle('Heat Death as Categorical Initiation', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'heat_death_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'heat_death_panel.png'}")


def generate_entropy_emergence_panel():
    """Generate panel for entropy emergence section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Categorical entropy increasing
    ax1 = fig.add_subplot(gs[0, 0])
    C = np.linspace(1, 100, 100)
    S_cat = np.log(C)
    ax1.fill_between(C, 0, S_cat, alpha=0.3, color=COLORS['entropy'])
    ax1.plot(C, S_cat, color=COLORS['entropy'], linewidth=3, label='$S_{cat} = k_B \\log |\\gamma|$')
    ax1.set_xlabel('Completed Categories $|\\gamma|$', fontsize=10)
    ax1.set_ylabel('Categorical Entropy $S_{cat}$', fontsize=10)
    ax1.set_title('(A) Categorical Entropy\nIncreases with $|\\gamma|$', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    
    # Panel B: Kinetic vs categorical entropy over time
    ax2 = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 20, 200)
    t_hd = 10  # Heat death time
    S_kin = np.where(t < t_hd, 0.5*(1 - np.exp(-0.3*t)), 0.5)
    S_cat = np.where(t < t_hd, 0.1*t, 1 + 0.3*(t - t_hd))
    ax2.plot(t, S_kin, color=COLORS['secondary'], linewidth=3, label='$S_{kin}$')
    ax2.plot(t, S_cat, color=COLORS['entropy'], linewidth=3, label='$S_{cat}$')
    ax2.plot(t, S_kin + S_cat, color=COLORS['primary'], linewidth=3, linestyle='--', label='$S_{total}$')
    ax2.axvline(x=t_hd, color=COLORS['dark'], linestyle=':', linewidth=2)
    ax2.text(t_hd, 7, 'Heat\nDeath', ha='center', fontsize=9)
    ax2.set_xlabel('Cosmic Time', fontsize=10)
    ax2.set_ylabel('Entropy', fontsize=10)
    ax2.set_title('(B) $S_{kin}$ Saturates,\n$S_{cat}$ Continues', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    
    # Panel C: Entropy decomposition
    ax3 = fig.add_subplot(gs[0, 2])
    # Stacked area chart
    t = np.linspace(0, 10, 100)
    S_kin = 0.4 * (1 - np.exp(-0.5*t))
    S_cat = 0.1 * t
    ax3.fill_between(t, 0, S_kin, alpha=0.7, color=COLORS['secondary'], label='$S_{kin}$')
    ax3.fill_between(t, S_kin, S_kin + S_cat, alpha=0.7, color=COLORS['entropy'], label='$S_{cat}$')
    ax3.plot(t, S_kin + S_cat, color=COLORS['primary'], linewidth=3)
    ax3.set_xlabel('Time', fontsize=10)
    ax3.set_ylabel('Entropy', fontsize=10)
    ax3.set_title('(C) Decomposition:\n$S_{total} = S_{kin} + S_{cat}$', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    
    # Panel D: Zero free energy categorical completion
    ax4 = fig.add_subplot(gs[1, 0])
    # Energy diagram
    levels_y = [0.3, 0.35, 0.4, 0.45]
    for y in levels_y:
        ax4.axhline(y=y, xmin=0.2, xmax=0.8, color=COLORS['oscillation'], linewidth=2)
    # Transitions
    for i in range(len(levels_y)-1):
        ax4.annotate('', xy=(0.5, levels_y[i+1]-0.01), xytext=(0.5, levels_y[i]+0.01),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2))
    ax4.text(0.5, 0.55, '$\\Delta F = 0$', ha='center', fontsize=14, fontweight='bold', color=COLORS['secondary'])
    ax4.text(0.5, 0.65, '$\\Delta S_{cat} > 0$', ha='center', fontsize=14, fontweight='bold', color=COLORS['entropy'])
    ax4.text(0.5, 0.2, 'Vibrational Transitions\nConserve Energy', ha='center', fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Categorical Completion\nat Zero Free Energy', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Shortest path interpretation
    ax5 = fig.add_subplot(gs[1, 1])
    # Draw graph with paths
    nodes = {'A': (0.2, 0.5), 'B': (0.4, 0.7), 'C': (0.4, 0.3), 
             'D': (0.6, 0.6), 'E': (0.6, 0.4), 'T': (0.85, 0.5)}
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'T'), ('E', 'T'), ('B', 'E')]
    for e in edges:
        ax5.plot([nodes[e[0]][0], nodes[e[1]][0]], [nodes[e[0]][1], nodes[e[1]][1]], 
                color=COLORS['tertiary'], linewidth=2, alpha=0.5)
    # Highlight shortest path
    shortest = [('A', 'C'), ('C', 'E'), ('E', 'T')]
    for e in shortest:
        ax5.plot([nodes[e[0]][0], nodes[e[1]][0]], [nodes[e[0]][1], nodes[e[1]][1]], 
                color=COLORS['entropy'], linewidth=4)
    for name, (x, y) in nodes.items():
        color = COLORS['primary'] if name == 'A' else COLORS['singularity'] if name == 'T' else COLORS['tertiary']
        ax5.scatter([x], [y], s=300, color=color, zorder=5, edgecolor=COLORS['dark'], linewidth=2)
        ax5.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax5.text(0.2, 0.35, 'Start', ha='center', fontsize=9)
    ax5.text(0.85, 0.35, 'Termination', ha='center', fontsize=9)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('(E) Entropy = Shortest Path\nto Termination', fontsize=11, fontweight='bold')
    ax5.axis('off')
    
    # Panel F: Arrow of time from categorical irreversibility
    ax6 = fig.add_subplot(gs[1, 2])
    t = np.linspace(0, 10, 100)
    gamma = 1 - np.exp(-0.3 * t)
    ax6.plot(t, gamma, color=COLORS['entropy'], linewidth=3)
    ax6.fill_between(t, 0, gamma, alpha=0.3, color=COLORS['entropy'])
    # Arrow
    ax6.annotate('', xy=(9, 0.9), xytext=(1, 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=4))
    ax6.text(5, 0.3, 'Arrow of Time', ha='center', fontsize=12, fontweight='bold', 
            color=COLORS['secondary'], rotation=35)
    ax6.text(5, 0.7, '$|\\gamma(t)|$ monotonic', ha='center', fontsize=10)
    ax6.set_xlabel('Time', fontsize=10)
    ax6.set_ylabel('$|\\gamma(t)|/|\\mathcal{C}|$', fontsize=10)
    ax6.set_title('(F) Arrow of Time from\nCategorical Irreversibility', fontsize=11, fontweight='bold')
    
    plt.suptitle('Entropy Emergence from Categorical Completion', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'entropy_emergence_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'entropy_emergence_panel.png'}")


def generate_geometric_ratio_panel():
    """Generate panel for geometric ratio section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Oscillation around inaccessible centre
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.3
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    ax1.plot(x, y, color=COLORS['oscillation'], linewidth=3, label='Oscillation (Accessible)')
    ax1.fill(x, y, color=COLORS['oscillation'], alpha=0.2)
    # Centre
    ax1.scatter([0.5], [0.5], s=300, color=COLORS['dark_matter'], zorder=5, 
               edgecolor=COLORS['dark'], linewidth=2, label='Centre (Inaccessible)')
    ax1.text(0.5, 0.5, '$x$', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('(A) Oscillation Around\nInaccessible Centre', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    # Panel B: Tri-dimensional centre-to-surface ratio
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    # Draw sphere (surface = oscillation)
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    r = 0.4
    x = r * np.outer(np.cos(u), np.sin(v)) + 0.5
    y = r * np.outer(np.sin(u), np.sin(v)) + 0.5
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + 0.5
    ax2.plot_surface(x, y, z, color=COLORS['oscillation'], alpha=0.3)
    # Centre point
    ax2.scatter([0.5], [0.5], [0.5], s=200, color=COLORS['dark_matter'], zorder=5)
    ax2.set_title('(B) 3D: Volume (Centre)\nvs Surface (Oscillation)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    
    # Panel C: Recursive accumulation through 3^k
    ax3 = fig.add_subplot(gs[0, 2])
    levels = np.arange(0, 8)
    # Accumulated ratio
    ratio = np.array([sum([(1/3)**i for i in range(k+1)]) * 3/2 for k in levels])
    ax3.bar(levels, ratio, color=COLORS['dark_matter'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.7)
    ax3.axhline(y=1.5, color=COLORS['secondary'], linestyle='--', linewidth=2, label='Asymptote (1.5)')
    ax3.set_xlabel('Recursion Level $k$', fontsize=10)
    ax3.set_ylabel('Centre/Oscillation Ratio', fontsize=10)
    ax3.set_title('(C) Recursive Accumulation\nThrough $3^k$ Branching', fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    
    # Panel D: Information-theoretic derivation
    ax4 = fig.add_subplot(gs[1, 0])
    # Golden ratio visualization
    phi = (1 + np.sqrt(5)) / 2
    # Draw golden rectangle
    rect1 = Rectangle((0.1, 0.2), 0.5, 0.5/phi, facecolor=COLORS['entropy'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.5)
    rect2 = Rectangle((0.1 + 0.5 - 0.5/phi, 0.2), 0.5/phi, 0.5/phi, facecolor=COLORS['dark_matter'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.5)
    ax4.add_patch(rect1)
    ax4.add_patch(rect2)
    ax4.text(0.35, 0.35, '$\\infty - x$', ha='center', fontsize=12, fontweight='bold', color=COLORS['entropy'])
    ax4.text(0.5, 0.35, '$x$', ha='center', fontsize=12, fontweight='bold', color=COLORS['dark_matter'])
    ax4.text(0.5, 0.75, f'$\\phi^2 \\approx {phi**2:.3f}$', ha='center', fontsize=12)
    ax4.text(0.5, 0.65, 'Fixed Point Condition', ha='center', fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Information-Theoretic\nFixed Point', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Comparison of theoretical vs observed
    ax5 = fig.add_subplot(gs[1, 1])
    methods = ['Geometric\n(4.73)', 'Info-Theoretic\n(5.2)', 'Observed\n(5.4)']
    values = [4.73, 5.2, 5.4]
    colors_bar = [COLORS['primary'], COLORS['tertiary'], COLORS['secondary']]
    bars = ax5.bar(methods, values, color=colors_bar, edgecolor=COLORS['dark'], linewidth=2)
    ax5.axhline(y=5.4, color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_ylabel('Dark Matter / Baryonic Matter', fontsize=10)
    ax5.set_title('(E) Theoretical vs.\nObserved Ratios', fontsize=11, fontweight='bold')
    ax5.set_ylim(0, 6.5)
    # Add percentage difference
    for bar, val in zip(bars, values):
        diff = abs(val - 5.4) / 5.4 * 100
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{diff:.1f}%', ha='center', fontsize=9)
    
    # Panel F: Why dark matter is undetectable
    ax6 = fig.add_subplot(gs[1, 2])
    # Observer trying to detect dark matter
    observer = Circle((0.2, 0.5), 0.08, color=COLORS['primary'], zorder=5)
    ax6.add_patch(observer)
    ax6.text(0.2, 0.35, 'Observer', ha='center', fontsize=9)
    # Detection rays (blocked)
    for angle in np.linspace(-20, 20, 5):
        rad = np.radians(angle)
        ax6.plot([0.28, 0.5], [0.5 + 0.1*np.sin(rad), 0.5 + 0.2*np.sin(rad)], 
                color=COLORS['oscillation'], linewidth=2, alpha=0.5)
        ax6.scatter([0.5], [0.5 + 0.2*np.sin(rad)], s=50, color=COLORS['secondary'], marker='x', zorder=5)
    # Dark matter region
    dm_circle = Circle((0.7, 0.5), 0.15, color=COLORS['dark_matter'], alpha=0.3)
    ax6.add_patch(dm_circle)
    ax6.text(0.7, 0.5, 'Dark\nMatter\n(Nothing)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax6.text(0.5, 0.15, 'No Categorical Structure\n→ No Interaction', ha='center', fontsize=10, color=COLORS['secondary'])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('(F) Dark Matter Undetectable:\nNo Categories to Interact', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle('Geometric Origin of the Dark Matter Ratio', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'geometric_ratio_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'geometric_ratio_panel.png'}")


def generate_unified_category_panel():
    """Generate panel for unified category section."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Dimensional equivalence
    ax1 = fig.add_subplot(gs[0, 0])
    # Three representations of 0D
    for i, (x, label, color) in enumerate([(0.2, 'Point', COLORS['primary']), 
                                           (0.5, 'Nothing', COLORS['dark_matter']),
                                           (0.8, 'Singularity', COLORS['singularity'])]):
        ax1.scatter([x], [0.5], s=400, color=color, zorder=5, edgecolor=COLORS['dark'], linewidth=3)
        ax1.text(x, 0.3, label, ha='center', fontsize=11, fontweight='bold')
        ax1.text(x, 0.7, '0D', ha='center', fontsize=14, fontweight='bold', color=color)
    # Equivalence signs
    ax1.text(0.35, 0.5, '$\\equiv$', ha='center', fontsize=20, fontweight='bold')
    ax1.text(0.65, 0.5, '$\\equiv$', ha='center', fontsize=20, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('(A) Dimensional Equivalence:\nAll Three Are 0D', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Categorical equivalence
    ax2 = fig.add_subplot(gs[0, 1])
    # Three boxes showing zero distinctions
    for i, (x, label) in enumerate([(0.2, 'Point'), (0.5, 'Nothing'), (0.8, 'Singularity')]):
        rect = FancyBboxPatch((x-0.12, 0.35), 0.24, 0.3, boxstyle="round,pad=0.02",
                              facecolor=COLORS['quaternary'], edgecolor=COLORS['dark'], linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, 0.5, '∅', ha='center', va='center', fontsize=24, color=COLORS['dark'])
        ax2.text(x, 0.25, label, ha='center', fontsize=10, fontweight='bold')
    ax2.text(0.5, 0.75, 'Zero Internal Distinctions', ha='center', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('(B) Categorical Equivalence:\nNo Internal Structure', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # Panel C: Topological equivalence - oscillation around point = around nothing
    ax3 = fig.add_subplot(gs[0, 2])
    theta = np.linspace(0, 2*np.pi, 100)
    # Left: oscillation around point
    x1 = 0.25 + 0.15 * np.cos(theta)
    y1 = 0.5 + 0.15 * np.sin(theta)
    ax3.plot(x1, y1, color=COLORS['oscillation'], linewidth=3)
    ax3.scatter([0.25], [0.5], s=100, color=COLORS['primary'], zorder=5)
    ax3.text(0.25, 0.25, 'Around Point', ha='center', fontsize=10)
    # Right: oscillation around nothing
    x2 = 0.75 + 0.15 * np.cos(theta)
    y2 = 0.5 + 0.15 * np.sin(theta)
    ax3.plot(x2, y2, color=COLORS['oscillation'], linewidth=3)
    ax3.scatter([0.75], [0.5], s=100, color=COLORS['dark_matter'], zorder=5, marker='o', facecolors='none', linewidths=3)
    ax3.text(0.75, 0.25, 'Around Nothing', ha='center', fontsize=10)
    # Equivalence
    ax3.text(0.5, 0.5, '=', ha='center', fontsize=24, fontweight='bold')
    ax3.text(0.5, 0.8, 'Both Create Same Topology', ha='center', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('(C) Topological Equivalence:\nCircling Point = Circling Nothing', fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Category filling toward singularity
    ax4 = fig.add_subplot(gs[1, 0])
    # Funnel visualization
    for i, (y, width, alpha) in enumerate([(0.9, 0.8, 0.3), (0.7, 0.6, 0.4), (0.5, 0.4, 0.5), (0.3, 0.2, 0.6), (0.15, 0.05, 0.8)]):
        rect = FancyBboxPatch((0.5 - width/2, y - 0.08), width, 0.12, boxstyle="round,pad=0.01",
                              facecolor=COLORS['category'], edgecolor=COLORS['dark'], linewidth=1, alpha=alpha)
        ax4.add_patch(rect)
    # Singularity at bottom
    ax4.scatter([0.5], [0.05], s=300, color=COLORS['singularity'], zorder=5, edgecolor=COLORS['dark'], linewidth=3)
    ax4.text(0.5, 0.05, '$C_{sing}$', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    # Arrow
    ax4.annotate('', xy=(0.5, 0.1), xytext=(0.5, 0.85),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=3))
    ax4.text(0.8, 0.5, 'Categories\nFilling', ha='center', fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('(D) Category Filling\nToward Singularity', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Panel E: Cyclic recurrence
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.ones_like(theta)
    ax5.plot(theta, r, color=COLORS['secondary'], linewidth=4)
    # Mark stages
    stages = ['Big Bang', 'Expansion', 'Heat Death', 'Completion', 'Singularity']
    angles = np.linspace(0, 2*np.pi, 6)[:-1]
    for angle, stage in zip(angles, stages):
        color = COLORS['singularity'] if 'Sing' in stage or 'Bang' in stage else COLORS['entropy']
        ax5.scatter([angle], [1], s=200, color=color, zorder=5, edgecolor=COLORS['dark'], linewidth=2)
        ax5.text(angle, 1.25, stage, ha='center', fontsize=8, fontweight='bold', rotation=np.degrees(angle)-90)
    # Arrow showing direction
    ax5.annotate('', xy=(angles[1], 1.05), xytext=(angles[0], 1.05),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax5.set_rticks([])
    ax5.set_title('(E) Cyclic Recurrence\nfrom Categorical Necessity', fontsize=11, fontweight='bold', pad=20)
    
    # Panel F: Complete cosmic cycle
    ax6 = fig.add_subplot(gs[1, 2])
    # Draw infinity symbol / lemniscate
    t = np.linspace(0, 2*np.pi, 200)
    a = 0.35
    x = a * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1) + 0.5
    y = a * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1) + 0.5
    ax6.plot(x, y, color=COLORS['secondary'], linewidth=4)
    # Mark key points
    ax6.scatter([0.5 - a*np.sqrt(2)], [0.5], s=200, color=COLORS['singularity'], zorder=5, 
               edgecolor=COLORS['dark'], linewidth=2)
    ax6.scatter([0.5 + a*np.sqrt(2)], [0.5], s=200, color=COLORS['entropy'], zorder=5,
               edgecolor=COLORS['dark'], linewidth=2)
    ax6.text(0.5 - a*np.sqrt(2), 0.35, 'Singularity', ha='center', fontsize=9, fontweight='bold')
    ax6.text(0.5 + a*np.sqrt(2), 0.35, 'Heat Death', ha='center', fontsize=9, fontweight='bold')
    ax6.text(0.5, 0.85, 'Eternal Cycle', ha='center', fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('(F) The Eternal Cosmic Cycle:\nBig Bang ↔ Singularity', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle('The Unified Category: Point ≡ Nothing ≡ Singularity', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'unified_category_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'unified_category_panel.png'}")


def generate_partition_lag_panel():
    """Generate panel for partition lag and origin of nothingness."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Static observer on moving number line
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('(A) Static Observer,\nMoving Reality', fontsize=11, fontweight='bold')
    
    # Observer window (static)
    window_rect = Rectangle((0.2, 0.5), 0.6, 0.3, facecolor=COLORS['quaternary'],
                            edgecolor=COLORS['primary'], linewidth=3)
    ax1.add_patch(window_rect)
    ax1.text(0.5, 0.75, 'Observer Window', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(0.5, 0.62, '(k partitions)', ha='center', va='center', fontsize=9, style='italic')
    
    # Number line moving below
    for i, t in enumerate([0, 1, 2]):
        offset = t * 0.15
        color_alpha = 1 - t * 0.3
        # Number line
        y_pos = 0.35 - t * 0.08
        ax1.annotate('', xy=(0.9 + offset, y_pos), xytext=(0.1 + offset, y_pos),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['secondary'], alpha=color_alpha))
        # Numbers on line
        for j, x in enumerate(np.linspace(0.15, 0.85, 5)):
            ax1.text(x + offset, y_pos - 0.04, str(j + i*5), ha='center', fontsize=8, 
                    alpha=color_alpha, color=COLORS['dark'])
        # Time label
        ax1.text(0.05, y_pos, f't={t}', fontsize=8, color=COLORS['tertiary'])
    
    # Arrow showing movement
    ax1.annotate('', xy=(0.85, 0.15), xytext=(0.15, 0.15),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['entropy']))
    ax1.text(0.5, 0.08, 'Reality moves', ha='center', fontsize=10, fontweight='bold', color=COLORS['entropy'])
    
    # Panel B: Partition lag concept
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('(B) Partition Lag:\nTime Passed During Partition', fontsize=11, fontweight='bold')
    
    # Timeline showing partition process
    ax2.plot([0.1, 0.9], [0.7, 0.7], 'k-', linewidth=3)
    ax2.scatter([0.1], [0.7], s=100, color=COLORS['primary'], zorder=5)
    ax2.scatter([0.9], [0.7], s=100, color=COLORS['secondary'], zorder=5)
    ax2.text(0.1, 0.75, r'$t_0$', ha='center', fontsize=10)
    ax2.text(0.9, 0.75, r'$t_0 + k\tau_p$', ha='center', fontsize=10)
    ax2.text(0.5, 0.63, 'Partition Time', ha='center', fontsize=9, style='italic')
    
    # Show what observer sees at start vs end
    # Start
    ax2.add_patch(Rectangle((0.15, 0.35), 0.3, 0.15, facecolor=COLORS['oscillation'], alpha=0.5))
    ax2.text(0.3, 0.43, 'Reality at start', ha='center', fontsize=8)
    ax2.text(0.1, 0.28, 'Partitioned:', fontsize=8, fontweight='bold')
    
    # End - shifted
    ax2.add_patch(Rectangle((0.55, 0.35), 0.3, 0.15, facecolor=COLORS['entropy'], alpha=0.5))
    ax2.text(0.7, 0.43, 'Reality at end', ha='center', fontsize=8)
    ax2.text(0.55, 0.28, 'But now:', fontsize=8, fontweight='bold')
    
    # Gap indicator
    ax2.annotate('', xy=(0.55, 0.42), xytext=(0.45, 0.42),
                arrowprops=dict(arrowstyle='<->', lw=2, color=COLORS['secondary']))
    ax2.text(0.5, 0.48, r'$\Delta$', ha='center', fontsize=12, color=COLORS['secondary'], fontweight='bold')
    
    ax2.text(0.5, 0.12, 'Lag = what moved during partitioning', ha='center', fontsize=9, 
             fontweight='bold', color=COLORS['secondary'])
    
    # Panel C: Undetermined residue
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('(C) Undetermined Residue\n= Nothingness', fontsize=11, fontweight='bold')
    
    # Three states of being
    states = [
        ('Not Absent', 'It existed at start', COLORS['oscillation'], 0.85),
        ('Not Present', 'It moved away', COLORS['secondary'], 0.55),
        ('Not Determinable', 'Never partitioned', COLORS['dark_matter'], 0.25)
    ]
    
    for state, desc, color, y in states:
        ax3.add_patch(Rectangle((0.15, y - 0.08), 0.7, 0.18, facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=2))
        ax3.text(0.5, y + 0.03, state, ha='center', fontsize=10, fontweight='bold')
        ax3.text(0.5, y - 0.04, desc, ha='center', fontsize=9, style='italic')
    
    ax3.text(0.5, 0.05, 'This IS nothingness', ha='center', fontsize=11, 
             fontweight='bold', color=COLORS['singularity'])
    
    # Panel D: Edge indeterminacy
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('(D) Edge Indeterminacy:\nBoundaries Cannot Be Fixed', fontsize=11, fontweight='bold')
    
    # Observer window with fuzzy edges
    center_x, center_y = 0.5, 0.6
    width, height = 0.6, 0.25
    
    # Core region (certain)
    ax4.add_patch(Rectangle((center_x - width/2 + 0.08, center_y - height/2), 
                            width - 0.16, height, facecolor=COLORS['quaternary'], alpha=0.8))
    ax4.text(center_x, center_y, 'Determined', ha='center', va='center', fontsize=10)
    
    # Fuzzy edges (uncertain)
    for i in range(5):
        alpha = 0.4 - i * 0.07
        offset = i * 0.02
        # Left edge
        ax4.add_patch(Rectangle((center_x - width/2 + offset, center_y - height/2),
                                0.08, height, facecolor=COLORS['dark_matter'], alpha=alpha))
        # Right edge
        ax4.add_patch(Rectangle((center_x + width/2 - 0.08 - offset, center_y - height/2),
                                0.08, height, facecolor=COLORS['dark_matter'], alpha=alpha))
    
    ax4.text(center_x - width/2 - 0.02, center_y, '?', ha='center', va='center', fontsize=20, 
             color=COLORS['secondary'], fontweight='bold')
    ax4.text(center_x + width/2 + 0.02, center_y, '?', ha='center', va='center', fontsize=20,
             color=COLORS['secondary'], fontweight='bold')
    
    ax4.text(center_x, 0.25, r'$\delta x_{min} = v \cdot \tau_p^{min} > 0$', ha='center', fontsize=11)
    ax4.text(center_x, 0.12, 'Minimum edge uncertainty', ha='center', fontsize=9, style='italic')
    
    # Panel E: Pastness of observation
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('(E) Observation Is Always Past:\nPresent Is Never Partitioned', fontsize=11, fontweight='bold')
    
    # Timeline
    times = ['Past', 'Past', 'Past', 'Present']
    x_positions = [0.15, 0.35, 0.55, 0.85]
    
    ax5.plot([0.1, 0.9], [0.6, 0.6], 'k-', linewidth=2)
    
    for x, label in zip(x_positions, times):
        if label == 'Present':
            color = COLORS['entropy']
            marker = 's'
            ax5.scatter([x], [0.6], s=150, color=color, zorder=5, marker=marker)
            ax5.text(x, 0.7, 'NOW', ha='center', fontsize=9, fontweight='bold', color=color)
        else:
            color = COLORS['tertiary']
            ax5.scatter([x], [0.6], s=100, color=color, zorder=5, alpha=0.7)
    
    # Partition labels
    ax5.text(0.15, 0.45, r'$C_1$', ha='center', fontsize=10)
    ax5.text(0.35, 0.45, r'$C_2$', ha='center', fontsize=10)
    ax5.text(0.55, 0.45, r'$C_k$', ha='center', fontsize=10)
    
    # Bracket for partitioned region
    ax5.plot([0.1, 0.6], [0.35, 0.35], 'k-', linewidth=2)
    ax5.plot([0.1, 0.1], [0.35, 0.38], 'k-', linewidth=2)
    ax5.plot([0.6, 0.6], [0.35, 0.38], 'k-', linewidth=2)
    ax5.text(0.35, 0.28, 'All partitions are of THE PAST', ha='center', fontsize=9, 
             fontweight='bold', color=COLORS['secondary'])
    
    ax5.text(0.85, 0.45, 'Never\npartitioned!', ha='center', fontsize=9, 
             color=COLORS['entropy'], fontweight='bold')
    
    # Panel F: Dark matter accumulation
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('(F) Dark Matter = Accumulated\nPartition Lag Residue', fontsize=11, fontweight='bold')
    
    # Show accumulation over time
    n_steps = 6
    for i in range(n_steps):
        y = 0.9 - i * 0.13
        width_observed = 0.5 - i * 0.04
        width_residue = 0.3 + i * 0.05
        
        # Observed (shrinking)
        ax6.add_patch(Rectangle((0.1, y - 0.04), width_observed, 0.08, 
                                facecolor=COLORS['oscillation'], alpha=0.6))
        # Residue (growing)
        ax6.add_patch(Rectangle((0.1 + width_observed, y - 0.04), width_residue, 0.08,
                                facecolor=COLORS['dark_matter'], alpha=0.6))
        
        if i == 0:
            ax6.text(0.1 + width_observed/2, y, 'Observed', ha='center', va='center', fontsize=7)
            ax6.text(0.1 + width_observed + width_residue/2, y, 'Residue', ha='center', va='center', fontsize=7)
        
        ax6.text(0.02, y, f't={i}', fontsize=7, va='center')
    
    # Arrow showing growth direction
    ax6.annotate('', xy=(0.8, 0.2), xytext=(0.8, 0.85),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['secondary']))
    ax6.text(0.88, 0.5, 'Time', ha='center', va='center', fontsize=9, rotation=90)
    
    # Ratio
    ax6.text(0.5, 0.05, r'$\frac{x}{\infty - x} \to 5.4$ (dark matter ratio)', 
             ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Partition Lag: The Origin of Nothingness and Dark Matter', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'partition_lag_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'partition_lag_panel.png'}")


def main():
    """Generate all panel visualizations for the Kelvin Paradox paper."""
    print("Generating panel visualizations for Kelvin Paradox paper...")
    print("=" * 60)
    
    generate_oscillatory_reality_panel()
    generate_topology_categories_panel()
    generate_observer_boundary_panel()
    generate_heat_death_panel()
    generate_entropy_emergence_panel()
    generate_geometric_ratio_panel()
    generate_unified_category_panel()
    generate_partition_lag_panel()
    
    print("=" * 60)
    print(f"All panels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

