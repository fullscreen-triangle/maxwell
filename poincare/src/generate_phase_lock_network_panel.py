"""
Generate panel chart showing phase-lock networks and their evolution.
6 network diagrams showing progression of phase-locking over categorical time.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "kelvin-paradox" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Also save to resolution figures
resolution_dir = Path(__file__).parent.parent.parent / "docs" / "resolution" / "figures"
resolution_dir.mkdir(parents=True, exist_ok=True)

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 11,
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
    'node_inactive': '#4a4a6a',
    'node_active': '#7eb8da',
    'node_locked': '#4ade80',
    'edge_weak': '#5a5a7a',
    'edge_forming': '#dac27e',
    'edge_locked': '#4ade80',
    'highlight': '#ffffff',
    'dim': '#3a3a5a',
    'background': '#12121f',
    'accent': '#b87eda',
}


def create_phase_lock_network(n_nodes, lock_probability, seed=None):
    """Create a network with given phase-lock probability."""
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.Graph()
    
    # Add nodes with positions in a circle
    for i in range(n_nodes):
        angle = 2 * np.pi * i / n_nodes
        G.add_node(i, pos=(np.cos(angle), np.sin(angle)))
    
    # Add edges based on lock probability
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < lock_probability:
                # Edge strength based on "how locked" they are
                strength = np.random.uniform(0.5, 1.0)
                G.add_edge(i, j, weight=strength)
    
    return G


def draw_network(ax, G, title, stage_info, show_phases=True):
    """Draw a phase-lock network on the given axes."""
    pos = nx.get_node_attributes(G, 'pos')
    
    # Calculate node properties
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) // 2
    lock_ratio = n_edges / max_edges if max_edges > 0 else 0
    
    # Node colors based on degree (how many phase-locks)
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_colors = []
    for node in G.nodes():
        d = degrees[node]
        if d == 0:
            node_colors.append(COLORS['node_inactive'])
        elif d < max_degree * 0.5:
            node_colors.append(COLORS['node_active'])
        else:
            node_colors.append(COLORS['node_locked'])
    
    # Draw edges with varying thickness and color based on weight
    edge_colors = []
    edge_widths = []
    for (u, v, data) in G.edges(data=True):
        weight = data.get('weight', 0.5)
        if weight < 0.6:
            edge_colors.append(COLORS['edge_weak'])
            edge_widths.append(1)
        elif weight < 0.8:
            edge_colors.append(COLORS['edge_forming'])
            edge_widths.append(2)
        else:
            edge_colors.append(COLORS['edge_locked'])
            edge_widths.append(3)
    
    # Draw network
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                               width=edge_widths, alpha=0.7)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=300, edgecolors='white', linewidths=1.5)
    
    # Add phase indicators if requested
    if show_phases and n_nodes <= 12:
        for node in G.nodes():
            x, y = pos[node]
            # Phase shown as small arc
            phase = np.random.uniform(0, 2*np.pi)  # Random phase for visualization
            if degrees[node] > 0:
                # Locked nodes show consistent phase
                phase = (node * 0.5) % (2*np.pi)
            
            # Draw small phase indicator
            arc_r = 0.15
            theta = np.linspace(0, phase, 20)
            arc_x = x + arc_r * 0.3 * np.cos(theta)
            arc_y = y + arc_r * 0.3 * np.sin(theta)
            if len(arc_x) > 1:
                ax.plot(arc_x, arc_y, color=COLORS['accent'], lw=1.5, alpha=0.8)
    
    # Title and info
    ax.set_title(title, fontweight='bold', color=COLORS['highlight'], pad=10)
    
    # Stage info text
    info_text = f"Edges: {n_edges}/{max_edges}\nLock ratio: {lock_ratio:.1%}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', color=COLORS['dim'],
           bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                    edgecolor=COLORS['dim'], alpha=0.8))
    
    # Category/entropy indicator
    ax.text(0.98, 0.02, stage_info, transform=ax.transAxes, fontsize=9,
           horizontalalignment='right', color=COLORS['accent'], fontweight='bold')
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')


def generate_phase_lock_panel():
    """Generate 6-panel chart showing phase-lock network evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.patch.set_facecolor('#0a0a12')
    
    # Main title
    fig.suptitle('Phase-Lock Network Evolution Through Categorical Time', 
                fontsize=14, fontweight='bold', color=COLORS['highlight'], y=0.98)
    
    n_nodes = 10  # Number of oscillators/molecules
    
    # Stage parameters: (title, lock_probability, stage_info, seed)
    stages = [
        ('A. Initial State: Independent Oscillators', 0.0, 'C = 0', 42),
        ('B. Early Phase-Locking: First Connections', 0.15, 'C = C₀', 42),
        ('C. Growing Network: Cascade Effect', 0.35, 'C = 3C₀', 42),
        ('D. Dense Network: Many Phase-Locks', 0.55, 'C = 10C₀', 42),
        ('E. Near-Complete: Network Saturation', 0.75, 'C → C_max', 42),
        ('F. Categorical Completion: Equilibrium', 0.95, 'S = S_eq', 42),
    ]
    
    for idx, (title, prob, stage_info, seed) in enumerate(stages):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        G = create_phase_lock_network(n_nodes, prob, seed=seed + idx)
        draw_network(ax, G, title, stage_info)
    
    # Add legend at bottom
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['node_inactive'],
                  markersize=10, label='Unlocked oscillator', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['node_active'],
                  markersize=10, label='Partially locked', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['node_locked'],
                  markersize=10, label='Highly locked', linestyle='None'),
        plt.Line2D([0], [0], color=COLORS['edge_weak'], lw=2, label='Weak coupling'),
        plt.Line2D([0], [0], color=COLORS['edge_forming'], lw=2, label='Forming lock'),
        plt.Line2D([0], [0], color=COLORS['edge_locked'], lw=3, label='Strong phase-lock'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
              fontsize=9, frameon=True, facecolor='#12121f', edgecolor='#3a3a5a')
    
    # Add explanatory text
    fig.text(0.5, 0.02, 
            'Phase-lock networks form when oscillators synchronize. Each edge represents a completed categorical connection. '
            'Network density = entropy. Equilibrium = maximum phase-locking.',
            ha='center', fontsize=9, color=COLORS['dim'], style='italic')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save to both locations
    plt.savefig(output_dir / 'phase_lock_network_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.savefig(resolution_dir / 'phase_lock_network_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase_lock_network_panel.png'}")
    print(f"Saved: {resolution_dir / 'phase_lock_network_panel.png'}")


def generate_phase_lock_mechanism_panel():
    """Generate detailed panel showing the mechanism of phase-locking."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.patch.set_facecolor('#0a0a12')
    
    fig.suptitle('Phase-Lock Mechanism: From Oscillation to Network', 
                fontsize=14, fontweight='bold', color=COLORS['highlight'], y=0.98)
    
    # Panel A: Two independent oscillators
    ax = axes[0, 0]
    ax.set_title('A. Independent Oscillators', fontweight='bold', color=COLORS['highlight'])
    
    t = np.linspace(0, 4*np.pi, 200)
    y1 = np.sin(t)
    y2 = np.sin(t + np.pi/3)  # Different phase
    
    ax.plot(t/(4*np.pi), y1 * 0.3 + 0.65, color=COLORS['node_active'], lw=2, label='Osc 1')
    ax.plot(t/(4*np.pi), y2 * 0.3 + 0.35, color=COLORS['accent'], lw=2, label='Osc 2')
    
    # Phase difference indicator
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.25, 0.35),
               arrowprops=dict(arrowstyle='<->', color=COLORS['edge_weak'], lw=2))
    ax.text(0.28, 0.5, 'Phase\ndiff', fontsize=8, color=COLORS['edge_weak'])
    
    ax.text(0.5, 0.08, 'No coupling: phases drift independently', ha='center', 
           fontsize=9, color=COLORS['dim'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel B: Coupling begins
    ax = axes[0, 1]
    ax.set_title('B. Coupling Interaction', fontweight='bold', color=COLORS['highlight'])
    
    # Two oscillators with coupling shown
    ax.add_patch(plt.Circle((0.3, 0.5), 0.15, fill=True, color=COLORS['node_active'], alpha=0.8))
    ax.add_patch(plt.Circle((0.7, 0.5), 0.15, fill=True, color=COLORS['accent'], alpha=0.8))
    
    # Coupling spring/interaction
    spring_x = np.linspace(0.45, 0.55, 20)
    spring_y = 0.5 + 0.05 * np.sin(np.linspace(0, 6*np.pi, 20))
    ax.plot(spring_x, spring_y, color=COLORS['edge_forming'], lw=3)
    
    # Phase arrows
    for cx, angle, color in [(0.3, np.pi/4, COLORS['node_active']), 
                              (0.7, np.pi/6, COLORS['accent'])]:
        ax.annotate('', xy=(cx + 0.12*np.cos(angle), 0.5 + 0.12*np.sin(angle)),
                   xytext=(cx, 0.5),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax.text(0.5, 0.2, 'Interaction enables phase information exchange', ha='center', 
           fontsize=9, color=COLORS['dim'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel C: Phase synchronization
    ax = axes[0, 2]
    ax.set_title('C. Phase Synchronization', fontweight='bold', color=COLORS['highlight'])
    
    t = np.linspace(0, 4*np.pi, 200)
    # Phases converging
    phase_diff = np.pi/3 * np.exp(-t/5)
    y1 = np.sin(t)
    y2 = np.sin(t + phase_diff)
    
    ax.plot(t/(4*np.pi), y1 * 0.3 + 0.65, color=COLORS['node_active'], lw=2)
    ax.plot(t/(4*np.pi), y2 * 0.3 + 0.35, color=COLORS['accent'], lw=2)
    
    # Convergence arrow
    ax.annotate('', xy=(0.9, 0.5), xytext=(0.5, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['edge_locked'], lw=3))
    ax.text(0.7, 0.55, 'Phases\nconverge', fontsize=8, color=COLORS['edge_locked'])
    
    ax.text(0.5, 0.08, 'Coupling drives phase alignment', ha='center', 
           fontsize=9, color=COLORS['dim'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel D: Locked state
    ax = axes[1, 0]
    ax.set_title('D. Phase-Locked State', fontweight='bold', color=COLORS['highlight'])
    
    t = np.linspace(0, 4*np.pi, 200)
    y1 = np.sin(t)
    y2 = np.sin(t)  # Same phase now
    
    ax.plot(t/(4*np.pi), y1 * 0.3 + 0.5, color=COLORS['node_locked'], lw=3, label='Both')
    
    # Single line showing locked state
    ax.add_patch(plt.Circle((0.15, 0.5), 0.08, fill=True, color=COLORS['node_locked']))
    ax.plot([0.23, 0.35], [0.5, 0.5], color=COLORS['edge_locked'], lw=4)
    ax.add_patch(plt.Circle((0.43, 0.5), 0.08, fill=True, color=COLORS['node_locked']))
    
    ax.text(0.29, 0.6, 'LOCKED', fontsize=10, fontweight='bold', color=COLORS['edge_locked'])
    
    ax.text(0.5, 0.15, 'Phase-lock = categorical completion', ha='center', 
           fontsize=9, color=COLORS['highlight'])
    ax.text(0.5, 0.05, 'This connection is now a completed category', ha='center', 
           fontsize=8, color=COLORS['dim'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel E: Network grows through cascade
    ax = axes[1, 1]
    ax.set_title('E. Cascade Effect', fontweight='bold', color=COLORS['highlight'])
    
    # Show how one lock enables others
    positions = {
        0: (0.2, 0.7), 1: (0.5, 0.8), 2: (0.8, 0.7),
        3: (0.2, 0.4), 4: (0.5, 0.3), 5: (0.8, 0.4)
    }
    
    # Draw nodes
    for i, (x, y) in positions.items():
        color = COLORS['node_locked'] if i in [0, 1, 3, 4] else COLORS['node_active']
        ax.add_patch(plt.Circle((x, y), 0.08, fill=True, color=color, alpha=0.9))
        ax.text(x, y, str(i), ha='center', va='center', fontsize=10, 
               fontweight='bold', color='white')
    
    # Existing locks
    locked_edges = [(0, 1), (0, 3), (1, 4), (3, 4)]
    for i, j in locked_edges:
        ax.plot([positions[i][0], positions[j][0]], 
               [positions[i][1], positions[j][1]], 
               color=COLORS['edge_locked'], lw=3)
    
    # Forming locks (cascade)
    forming_edges = [(1, 2), (4, 5)]
    for i, j in forming_edges:
        ax.plot([positions[i][0], positions[j][0]], 
               [positions[i][1], positions[j][1]], 
               color=COLORS['edge_forming'], lw=2, ls='--')
    
    ax.text(0.5, 0.08, 'Locks enable new locks: autocatalytic growth', ha='center', 
           fontsize=9, color=COLORS['highlight'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel F: Entropy = network density
    ax = axes[1, 2]
    ax.set_title('F. Entropy = Network Density', fontweight='bold', color=COLORS['highlight'])
    
    # Graph showing relationship
    densities = [0, 0.1, 0.2, 0.35, 0.55, 0.75, 0.9, 1.0]
    entropies = [0, 0.15, 0.35, 0.55, 0.72, 0.85, 0.95, 1.0]
    
    ax.plot(densities, entropies, color=COLORS['edge_locked'], lw=3, marker='o', markersize=8)
    ax.fill_between(densities, 0, entropies, alpha=0.2, color=COLORS['edge_locked'])
    
    ax.set_xlabel('Network Density (locks/max_locks)', fontsize=9)
    ax.set_ylabel('Categorical Entropy S/S_max', fontsize=9)
    
    # Annotations
    ax.annotate('Sparse:\nlow S', xy=(0.1, 0.15), fontsize=8, color=COLORS['dim'])
    ax.annotate('Dense:\nhigh S', xy=(0.75, 0.85), fontsize=8, color=COLORS['edge_locked'])
    
    ax.text(0.5, -0.15, 'More phase-locks = more completed categories = higher entropy', 
           ha='center', fontsize=9, color=COLORS['highlight'], transform=ax.transAxes)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Save to both locations
    plt.savefig(output_dir / 'phase_lock_mechanism_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.savefig(resolution_dir / 'phase_lock_mechanism_panel.png', dpi=300, 
                facecolor='#0a0a12', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'phase_lock_mechanism_panel.png'}")
    print(f"Saved: {resolution_dir / 'phase_lock_mechanism_panel.png'}")


if __name__ == "__main__":
    print("Generating Phase-Lock Network Panels...")
    print("=" * 60)
    
    generate_phase_lock_panel()
    generate_phase_lock_mechanism_panel()
    
    print("=" * 60)
    print("All panels generated successfully!")

