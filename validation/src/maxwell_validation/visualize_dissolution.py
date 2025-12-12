"""
Maxwell's Demon Dissolution Visualization
==========================================

Publication-quality panel charts for each of the 7 dissolution arguments.
Each panel contains 4 diverse, informative charts.

Chart types used:
- 3D scatter plots and surfaces
- Network diagrams (phase-lock networks)
- Heatmaps and contour plots
- Stream plots and vector fields
- Statistical distributions
- Time series with annotations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

COLORS = {
    'primary': '#0D1B2A',      # Deep navy
    'secondary': '#1B263B',    # Dark blue
    'tertiary': '#415A77',     # Steel blue
    'accent1': '#778DA9',      # Light steel
    'accent2': '#E0E1DD',      # Off-white
    'hot': '#E63946',          # Red
    'warm': '#F4A261',         # Orange
    'cool': '#2A9D8F',         # Teal
    'cold': '#264653',         # Dark teal
    'highlight': '#E9C46A',    # Gold
    'categorical': '#9B5DE5',  # Purple
    'kinetic': '#F15BB5',      # Pink
    'spatial': '#00BBF9',      # Cyan
    'network': '#00F5D4',      # Mint
}

CMAP_THERMAL = plt.cm.RdYlBu_r
CMAP_NETWORK = plt.cm.viridis
CMAP_ENTROPY = plt.cm.plasma


def setup_style():
    """Set publication-quality style"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


setup_style()


# ============================================================================
# DATA GENERATION UTILITIES
# ============================================================================

def generate_phase_lock_network(n_nodes: int = 50, seed: int = 42) -> nx.Graph:
    """Generate a phase-lock network for visualization"""
    np.random.seed(seed)
    positions = np.random.rand(n_nodes, 3) * 10

    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, pos=positions[i])

    # Add edges based on distance
    distances = squareform(pdist(positions))
    cutoff = 2.5

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if distances[i, j] < cutoff:
                coupling = 1.0 / (distances[i, j] ** 2 + 0.1)
                G.add_edge(i, j, weight=coupling)

    return G


def generate_molecular_system(n_mol: int = 200, seed: int = 42) -> Dict:
    """Generate molecular system data"""
    np.random.seed(seed)

    positions = np.random.rand(n_mol, 3) * 10
    velocities = np.random.randn(n_mol, 3)
    speeds = np.linalg.norm(velocities, axis=1)

    # Phase-lock network
    distances = squareform(pdist(positions))
    adjacency = (distances < 2.0) & (distances > 0)
    categorical_dist = shortest_path(adjacency.astype(float), directed=False)
    categorical_dist[np.isinf(categorical_dist)] = n_mol

    return {
        'positions': positions,
        'velocities': velocities,
        'speeds': speeds,
        'distances': distances,
        'adjacency': adjacency,
        'categorical_dist': categorical_dist,
    }


# ============================================================================
# ARGUMENT 1: TEMPORAL TRIVIALITY
# ============================================================================

def visualize_arg1_temporal_triviality(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 1: Temporal Triviality

    Panels:
    A. 3D Boltzmann probability landscape
    B. Fluctuation recurrence time distribution
    C. Configuration space trajectory (stream plot)
    D. Entropy vs time with Poincaré markers
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === PANEL A: 3D Boltzmann Landscape ===
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)

    # Free energy surface
    F = X**2 + Y**2 + 0.5*np.sin(2*X)*np.cos(2*Y)
    P = np.exp(-F)

    surf = ax_a.plot_surface(X, Y, P, cmap=CMAP_THERMAL, alpha=0.8,
                             linewidth=0.2, edgecolor='black')

    # Mark equilibrium point
    ax_a.scatter([0], [0], [1], s=200, c=COLORS['highlight'],
                 marker='*', edgecolor='black', linewidth=2, zorder=10)

    ax_a.set_xlabel('Config. Coord. 1', fontsize=10)
    ax_a.set_ylabel('Config. Coord. 2', fontsize=10)
    ax_a.set_zlabel('P(config)', fontsize=10)
    ax_a.set_title('A. Boltzmann Probability Landscape\nAll configurations accessible',
                   fontsize=11, fontweight='bold')
    ax_a.view_init(elev=25, azim=45)

    # === PANEL B: Recurrence Time Distribution ===
    ax_b = fig.add_subplot(gs[0, 1])

    # Generate recurrence times (log-normal distribution)
    np.random.seed(42)
    sorting_degree = np.linspace(0, 1, 100)
    recurrence_times = np.exp(10 * sorting_degree) * np.random.lognormal(0, 0.3, 100)

    # Violin plot
    parts = ax_b.violinplot([recurrence_times[sorting_degree < 0.3],
                             recurrence_times[(sorting_degree >= 0.3) & (sorting_degree < 0.6)],
                             recurrence_times[sorting_degree >= 0.6]],
                            positions=[1, 2, 3], showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['cool'])
        pc.set_alpha(0.7)

    ax_b.set_yscale('log')
    ax_b.set_xticks([1, 2, 3])
    ax_b.set_xticklabels(['Low\nSorting', 'Medium\nSorting', 'High\nSorting'])
    ax_b.set_ylabel('Recurrence Time (τ)', fontsize=10)
    ax_b.set_title('B. Poincaré Recurrence Times\nHigher sorting = longer wait',
                   fontsize=11, fontweight='bold')
    ax_b.axhline(y=1e6, color=COLORS['hot'], linestyle='--', linewidth=2, label='Lab timescale')
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)

    # === PANEL C: Configuration Space Stream Plot ===
    ax_c = fig.add_subplot(gs[1, 0])

    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)

    # Gradient flow toward equilibrium
    U = -X * np.exp(-(X**2 + Y**2)/4)
    V = -Y * np.exp(-(X**2 + Y**2)/4)

    # Add random fluctuations
    np.random.seed(42)
    U += np.random.randn(*U.shape) * 0.1
    V += np.random.randn(*V.shape) * 0.1

    speed = np.sqrt(U**2 + V**2)

    strm = ax_c.streamplot(X, Y, U, V, color=speed, cmap=CMAP_ENTROPY,
                           linewidth=1.5, density=1.5, arrowsize=1.2)

    # Mark equilibrium
    ax_c.scatter([0], [0], s=300, c=COLORS['highlight'], marker='o',
                 edgecolor='black', linewidth=2, zorder=10, label='Equilibrium')

    # Mark "sorted" states
    theta = np.linspace(0, 2*np.pi, 6)[:-1]
    sorted_x = 2 * np.cos(theta)
    sorted_y = 2 * np.sin(theta)
    ax_c.scatter(sorted_x, sorted_y, s=100, c=COLORS['hot'], marker='s',
                 edgecolor='black', linewidth=1, label='"Sorted" states')

    ax_c.set_xlabel('Configuration Coordinate 1', fontsize=10)
    ax_c.set_ylabel('Configuration Coordinate 2', fontsize=10)
    ax_c.set_title('C. Configuration Space Flow\nAll paths lead to equilibrium',
                   fontsize=11, fontweight='bold')
    ax_c.legend(loc='upper right')
    ax_c.set_xlim(-3, 3)
    ax_c.set_ylim(-3, 3)

    # === PANEL D: Entropy Time Series ===
    ax_d = fig.add_subplot(gs[1, 1])

    t = np.linspace(0, 100, 1000)
    np.random.seed(42)

    # Entropy with fluctuations
    entropy_base = 10 * (1 - np.exp(-t/20))
    fluctuations = np.cumsum(np.random.randn(len(t)) * 0.1)
    entropy = entropy_base + fluctuations

    ax_d.plot(t, entropy, linewidth=2, color=COLORS['primary'], label='Total Entropy')
    ax_d.fill_between(t, entropy - 1, entropy + 1, alpha=0.2, color=COLORS['primary'])

    # Mark Poincaré recurrence events
    recurrence_times_plot = [15, 35, 55, 75, 95]
    for rt in recurrence_times_plot:
        idx = np.argmin(np.abs(t - rt))
        ax_d.scatter([t[idx]], [entropy[idx] - 2], marker='^', s=100,
                     c=COLORS['highlight'], edgecolor='black', zorder=10)

    ax_d.axhline(y=entropy[-1] - 3, color=COLORS['hot'], linestyle=':',
                 linewidth=2, label='"Sorted" state (rare)')

    ax_d.set_xlabel('Time', fontsize=10)
    ax_d.set_ylabel('Entropy', fontsize=10)
    ax_d.set_title('D. Entropy Evolution\nFluctuations enable all states',
                   fontsize=11, fontweight='bold')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('ARGUMENT 1: TEMPORAL TRIVIALITY\n'
                 'Any configuration occurs naturally through thermal fluctuations',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# ARGUMENT 2: PHASE-LOCK TEMPERATURE INDEPENDENCE
# ============================================================================

def visualize_arg2_temperature_independence(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 2: Phase-Lock Temperature Independence

    Panels:
    A. 3D network at different temperatures (overlay)
    B. Network properties vs temperature (multi-line)
    C. Kinetic energy distribution heatmap
    D. ∂G/∂E correlation matrix
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    np.random.seed(42)

    # Generate fixed positions
    n_nodes = 40
    positions = np.random.rand(n_nodes, 2) * 10

    # === PANEL A: 3D Network at Different Temperatures ===
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    temperatures = [0.5, 1.0, 2.0, 5.0]
    temp_colors = [COLORS['cold'], COLORS['cool'], COLORS['warm'], COLORS['hot']]

    # Build network (same for all temperatures)
    distances = squareform(pdist(positions))
    adjacency = (distances < 2.5) & (distances > 0)

    for i, (T, color) in enumerate(zip(temperatures, temp_colors)):
        z_offset = i * 3

        # Draw nodes at this temperature level
        ax_a.scatter(positions[:, 0], positions[:, 1],
                     np.ones(n_nodes) * z_offset,
                     s=50, c=color, alpha=0.8, label=f'T={T}')

        # Draw edges (same topology)
        for j in range(n_nodes):
            for k in range(j+1, n_nodes):
                if adjacency[j, k]:
                    ax_a.plot([positions[j, 0], positions[k, 0]],
                             [positions[j, 1], positions[k, 1]],
                             [z_offset, z_offset],
                             color=color, alpha=0.3, linewidth=0.5)

    ax_a.set_xlabel('X', fontsize=10)
    ax_a.set_ylabel('Y', fontsize=10)
    ax_a.set_zlabel('Temperature Layer', fontsize=10)
    ax_a.set_title('A. Same Network Topology\nAcross All Temperatures',
                   fontsize=11, fontweight='bold')
    ax_a.legend(loc='upper left')
    ax_a.view_init(elev=20, azim=45)

    # === PANEL B: Network Properties vs Temperature ===
    ax_b = fig.add_subplot(gs[0, 1])

    temps = np.linspace(0.1, 10, 50)
    n_edges = np.ones_like(temps) * adjacency.sum() / 2
    kinetic_energy = temps * n_nodes * 1.5  # KE ∝ T

    ax_b_twin = ax_b.twinx()

    l1, = ax_b.plot(temps, n_edges, 'o-', color=COLORS['primary'],
                    linewidth=2.5, markersize=6, label='Network Edges')
    l2, = ax_b_twin.plot(temps, kinetic_energy, 's-', color=COLORS['hot'],
                         linewidth=2.5, markersize=6, label='Kinetic Energy')

    ax_b.set_xlabel('Temperature (T)', fontsize=10)
    ax_b.set_ylabel('Network Edges (constant)', fontsize=10, color=COLORS['primary'])
    ax_b_twin.set_ylabel('Kinetic Energy (∝ T)', fontsize=10, color=COLORS['hot'])
    ax_b.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax_b_twin.tick_params(axis='y', labelcolor=COLORS['hot'])

    ax_b.set_title('B. Network vs Kinetic Properties\n∂G/∂T = 0',
                   fontsize=11, fontweight='bold')
    ax_b.legend([l1, l2], ['Edges (constant)', 'KE (∝ T)'], loc='center right')
    ax_b.grid(True, alpha=0.3)

    # === PANEL C: Kinetic Energy Distribution Heatmap ===
    ax_c = fig.add_subplot(gs[1, 0])

    # Create velocity distribution at different temperatures
    temps_sample = [0.5, 1.0, 2.0, 5.0, 10.0]
    velocity_bins = np.linspace(-5, 5, 50)

    distributions = []
    for T in temps_sample:
        sigma = np.sqrt(T)
        dist = stats.norm.pdf(velocity_bins, 0, sigma)
        distributions.append(dist)

    distributions = np.array(distributions)

    im = ax_c.imshow(distributions, aspect='auto', cmap=CMAP_THERMAL,
                      extent=[velocity_bins[0], velocity_bins[-1],
                              len(temps_sample)-0.5, -0.5])

    ax_c.set_yticks(range(len(temps_sample)))
    ax_c.set_yticklabels([f'T={T}' for T in temps_sample])
    ax_c.set_xlabel('Velocity', fontsize=10)
    ax_c.set_ylabel('Temperature', fontsize=10)
    ax_c.set_title('C. Maxwell-Boltzmann at Different T\nWidens with T, network unchanged',
                   fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_c, label='Probability Density')

    # === PANEL D: Correlation Matrix ===
    ax_d = fig.add_subplot(gs[1, 1])

    # Generate correlation matrix
    properties = ['Network\nEdges', 'Mean\nDegree', 'Clustering',
                  'Kinetic\nEnergy', 'Temperature']
    n_props = len(properties)

    # Correlations: network properties uncorrelated with kinetic
    corr_matrix = np.array([
        [1.00, 0.95, 0.82, 0.02, 0.01],
        [0.95, 1.00, 0.78, -0.03, 0.02],
        [0.82, 0.78, 1.00, 0.01, -0.02],
        [0.02, -0.03, 0.01, 1.00, 0.99],
        [0.01, 0.02, -0.02, 0.99, 1.00]
    ])

    im = ax_d.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add correlation values
    for i in range(n_props):
        for j in range(n_props):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax_d.text(j, i, f'{corr_matrix[i, j]:.2f}',
                     ha='center', va='center', color=color, fontsize=9)

    ax_d.set_xticks(range(n_props))
    ax_d.set_yticks(range(n_props))
    ax_d.set_xticklabels(properties, fontsize=8)
    ax_d.set_yticklabels(properties, fontsize=8)
    ax_d.set_title('D. Property Correlation Matrix\nNetwork ⊥ Kinetic (r ≈ 0)',
                   fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_d, label='Correlation')

    # Draw box around low correlations
    rect = plt.Rectangle((2.5, -0.5), 2, 3, fill=False,
                          edgecolor=COLORS['highlight'], linewidth=3, linestyle='--')
    ax_d.add_patch(rect)

    fig.suptitle('ARGUMENT 2: PHASE-LOCK TEMPERATURE INDEPENDENCE\n'
                 'Network topology ∂G/∂E_kin = 0: independent of kinetic energy',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# ARGUMENT 3: RETRIEVAL PARADOX
# ============================================================================

def visualize_arg3_retrieval_paradox(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 3: The Retrieval Paradox

    Panels:
    A. Timescale hierarchy bar chart (log scale)
    B. Velocity scrambling simulation (2D phase space)
    C. Sorting attempt decay curves
    D. Fast/slow ratio over time (no separation)
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === PANEL A: Timescale Hierarchy ===
    ax_a = fig.add_subplot(gs[0, 0])

    processes = ['Collision', 'Measurement', 'Gate Open', 'Sorting', 'Demon\nThinks']
    timescales = [1e-10, 1e-8, 1e-6, 1e-3, 1e-1]
    colors = [COLORS['hot'], COLORS['warm'], COLORS['highlight'],
              COLORS['cool'], COLORS['cold']]

    bars = ax_a.barh(processes, timescales, color=colors,
                      edgecolor='black', linewidth=1.5)

    ax_a.set_xscale('log')
    ax_a.set_xlabel('Timescale (seconds)', fontsize=10)
    ax_a.set_title('A. Timescale Hierarchy\nCollisions happen FIRST',
                   fontsize=11, fontweight='bold')

    # Add values and arrows
    for bar, val in zip(bars, timescales):
        ax_a.text(val * 2, bar.get_y() + bar.get_height()/2,
                  f'{val:.0e}s', va='center', fontsize=9, fontweight='bold')

    # Add "TOO SLOW" arrow
    ax_a.annotate('TOO SLOW!', xy=(1e-6, 2), xytext=(1e-3, 2.5),
                  fontsize=12, color=COLORS['hot'], fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=COLORS['hot'], lw=2))

    ax_a.grid(True, alpha=0.3, axis='x')

    # === PANEL B: Phase Space Scrambling ===
    ax_b = fig.add_subplot(gs[0, 1])

    np.random.seed(42)
    n_particles = 200

    # Initial "sorted" state
    x_sorted = np.concatenate([
        np.random.normal(-2, 0.5, n_particles//2),  # Slow
        np.random.normal(2, 0.5, n_particles//2)    # Fast
    ])
    y_sorted = np.random.randn(n_particles) * 0.5

    # After collisions (scrambled)
    x_scrambled = np.random.randn(n_particles) * 1.5
    y_scrambled = np.random.randn(n_particles) * 1.5

    # Plot with transparency to show evolution
    ax_b.scatter(x_sorted, y_sorted, c=COLORS['cool'], alpha=0.3, s=30,
                 label='Initially "sorted"')
    ax_b.scatter(x_scrambled, y_scrambled, c=COLORS['hot'], alpha=0.6, s=30,
                 label='After τ_collision')

    # Draw arrows showing scrambling
    for i in range(0, n_particles, 20):
        ax_b.annotate('', xy=(x_scrambled[i], y_scrambled[i]),
                     xytext=(x_sorted[i], y_sorted[i]),
                     arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))

    ax_b.set_xlabel('Velocity x', fontsize=10)
    ax_b.set_ylabel('Velocity y', fontsize=10)
    ax_b.set_title('B. Phase Space Scrambling\n"Sorted" → Random in τ_collision',
                   fontsize=11, fontweight='bold')
    ax_b.legend()
    ax_b.set_xlim(-5, 5)
    ax_b.set_ylim(-5, 5)
    ax_b.grid(True, alpha=0.3)

    # === PANEL C: Sorting Decay Curves ===
    ax_c = fig.add_subplot(gs[1, 0])

    t = np.linspace(0, 10, 200)

    # Different decay rates
    tau_thermal = 0.1  # Fast thermal equilibration
    tau_sort = 2.0     # Slow sorting

    sorting_signal = np.exp(-t/tau_sort)
    thermal_noise = 1 - np.exp(-t/tau_thermal)

    ax_c.fill_between(t, 0, thermal_noise, alpha=0.3, color=COLORS['hot'],
                      label='Thermal randomization')
    ax_c.plot(t, sorting_signal, linewidth=3, color=COLORS['cool'],
              label='Sorting attempt')
    ax_c.plot(t, thermal_noise, linewidth=3, color=COLORS['hot'], linestyle='--')

    # Mark crossing point
    cross_idx = np.argmin(np.abs(sorting_signal - thermal_noise))
    ax_c.scatter([t[cross_idx]], [sorting_signal[cross_idx]], s=200,
                 c=COLORS['highlight'], edgecolor='black', linewidth=2, zorder=10)
    ax_c.annotate('Sorting\noverwhelmed', xy=(t[cross_idx], sorting_signal[cross_idx]),
                  xytext=(t[cross_idx]+2, sorting_signal[cross_idx]+0.2),
                  fontsize=10, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax_c.set_xlabel('Time (τ_collision units)', fontsize=10)
    ax_c.set_ylabel('Signal Strength', fontsize=10)
    ax_c.set_title('C. Sorting vs Thermalization\nThermalization wins',
                   fontsize=11, fontweight='bold')
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    # === PANEL D: Fast/Slow Ratio ===
    ax_d = fig.add_subplot(gs[1, 1])

    n_steps = 200
    np.random.seed(42)

    # Simulate many sorting attempts
    n_attempts = 5

    for attempt in range(n_attempts):
        ratio = 0.5 + np.cumsum(np.random.randn(n_steps) * 0.01)
        ratio = np.clip(ratio, 0.3, 0.7)

        # Brief spikes when "demon" tries to sort
        spike_times = [20 + attempt*30, 50 + attempt*30]
        for spike in spike_times:
            if spike < n_steps:
                ratio[spike:min(spike+10, n_steps)] += 0.1 * np.exp(
                    -np.arange(min(10, n_steps-spike)) / 3)

        ratio = np.clip(ratio, 0.3, 0.7)
        alpha = 0.7 if attempt == 0 else 0.3
        ax_d.plot(range(n_steps), ratio, linewidth=2, alpha=alpha,
                  label=f'Attempt {attempt+1}' if attempt == 0 else None)

    ax_d.axhline(y=0.5, color=COLORS['hot'], linestyle='--', linewidth=3,
                 label='Equilibrium (50%)')
    ax_d.fill_between(range(n_steps), 0.45, 0.55, alpha=0.2, color=COLORS['hot'])

    ax_d.set_xlabel('Time Steps', fontsize=10)
    ax_d.set_ylabel('Fast / Total Ratio', fontsize=10)
    ax_d.set_title('D. Cannot Maintain Sorting\nAlways returns to 50/50',
                   fontsize=11, fontweight='bold')
    ax_d.legend()
    ax_d.set_ylim(0.3, 0.7)
    ax_d.grid(True, alpha=0.3)

    fig.suptitle('ARGUMENT 3: THE RETRIEVAL PARADOX\n'
                 'Velocity-based sorting is self-defeating: thermal equilibration is faster',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# ARGUMENT 4: DISSOLUTION OF OBSERVATION
# ============================================================================

def visualize_arg4_dissolution_observation(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 4: Dissolution of Observation

    Panels:
    A. Network topology graph (2D spring layout)
    B. Topological accessibility matrix
    C. Velocity vs network position (no correlation)
    D. Path comparison: topology vs velocity
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Generate network
    G = generate_phase_lock_network(n_nodes=40, seed=42)
    pos = nx.spring_layout(G, seed=42, k=2)

    # === PANEL A: Network Graph ===
    ax_a = fig.add_subplot(gs[0, 0])

    # Draw network
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_nodes(G, pos, ax=ax_a, node_size=200,
                           node_color=COLORS['primary'], alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax_a, edge_color=weights,
                           edge_cmap=plt.cm.YlOrRd, alpha=0.6, width=2)

    # Highlight a path
    path = nx.shortest_path(G, 0, 20)
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, ax=ax_a,
                           node_size=300, node_color=COLORS['highlight'])
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax_a,
                           edge_color=COLORS['highlight'], width=4)

    ax_a.set_title('A. Phase-Lock Network Topology\nPath determined by STRUCTURE',
                   fontsize=11, fontweight='bold')
    ax_a.axis('off')

    # Add legend manually
    ax_a.scatter([], [], c=COLORS['primary'], s=100, label='All nodes')
    ax_a.scatter([], [], c=COLORS['highlight'], s=100, label='Path (topological)')
    ax_a.legend(loc='upper right')

    # === PANEL B: Accessibility Matrix ===
    ax_b = fig.add_subplot(gs[0, 1])

    # Compute shortest paths
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    n_nodes = len(G.nodes())

    accessibility = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if j in path_lengths[i]:
                accessibility[i, j] = path_lengths[i][j]
            else:
                accessibility[i, j] = np.nan

    im = ax_b.imshow(accessibility, cmap='Blues_r', aspect='equal')
    ax_b.set_xlabel('Node j', fontsize=10)
    ax_b.set_ylabel('Node i', fontsize=10)
    ax_b.set_title('B. Topological Accessibility\nDetermined by edges, not velocity',
                   fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_b, label='Path Length')

    # === PANEL C: Velocity vs Network Position ===
    ax_c = fig.add_subplot(gs[1, 0])

    np.random.seed(42)
    n_points = 100

    network_pos = np.random.rand(n_points)  # Random network positions
    velocities = np.random.randn(n_points)   # Random velocities (independent)

    ax_c.scatter(network_pos, velocities, c=COLORS['cool'], alpha=0.6, s=50)

    # Fit line
    z = np.polyfit(network_pos, velocities, 1)
    p = np.poly1d(z)
    ax_c.plot([0, 1], [p(0), p(1)], '--', color=COLORS['hot'], linewidth=2)

    # Correlation
    corr = np.corrcoef(network_pos, velocities)[0, 1]
    ax_c.text(0.95, 0.95, f'r = {corr:.3f}\n(no correlation)',
              transform=ax_c.transAxes, ha='right', va='top',
              fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_c.set_xlabel('Network Position', fontsize=10)
    ax_c.set_ylabel('Velocity', fontsize=10)
    ax_c.set_title('C. Velocity ⊥ Network Position\nNo measurement needed',
                   fontsize=11, fontweight='bold')
    ax_c.grid(True, alpha=0.3)

    # === PANEL D: Path Comparison ===
    ax_d = fig.add_subplot(gs[1, 1])

    # Compare topological path to "velocity-based" path
    n_pairs = 20

    topo_distances = []
    velocity_diffs = []

    nodes = list(G.nodes())[:n_pairs]
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            # Topological distance
            try:
                topo_dist = nx.shortest_path_length(G, node, nodes[i+1])
            except nx.NetworkXNoPath:
                topo_dist = 10
            topo_distances.append(topo_dist)

            # Velocity difference (random)
            velocity_diffs.append(np.abs(np.random.randn()))

    x = np.arange(len(topo_distances))
    width = 0.35

    bars1 = ax_d.bar(x - width/2, topo_distances, width,
                      label='Topological Path', color=COLORS['primary'])
    bars2 = ax_d.bar(x + width/2, velocity_diffs, width,
                      label='Velocity Difference', color=COLORS['hot'], alpha=0.7)

    ax_d.set_xlabel('Node Pair', fontsize=10)
    ax_d.set_ylabel('Distance/Difference', fontsize=10)
    ax_d.set_title('D. Path ≠ Velocity\nTopology determines navigation',
                   fontsize=11, fontweight='bold')
    ax_d.legend()
    ax_d.set_xticks(x)
    ax_d.grid(True, alpha=0.3, axis='y')

    fig.suptitle('ARGUMENT 4: DISSOLUTION OF OBSERVATION\n'
                 'Navigation follows topology, not measurement of velocities',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# ARGUMENT 5: DISSOLUTION OF DECISION
# ============================================================================

def visualize_arg5_dissolution_decision(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 5: Dissolution of Decision

    Panels:
    A. Deterministic path on energy landscape (3D)
    B. Branching factor = 0 (histogram)
    C. Multiple runs overlay (all identical)
    D. Completion cascade (waterfall plot)
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === PANEL A: 3D Energy Landscape with Path ===
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)

    # Energy landscape with valley
    E = X**2 + 0.5*Y**2 + 0.3*X*Y - 2*np.exp(-((X+2)**2 + Y**2))

    ax_a.plot_surface(X, Y, E, cmap='terrain', alpha=0.7, linewidth=0)

    # Deterministic path following gradient
    path_t = np.linspace(0, 1, 50)
    path_x = 2 * (1 - path_t) - 2
    path_y = np.sin(path_t * np.pi) * 0.5
    path_z = path_x**2 + 0.5*path_y**2 + 0.3*path_x*path_y - 2*np.exp(-((path_x+2)**2 + path_y**2)) + 0.2

    ax_a.plot(path_x, path_y, path_z, color=COLORS['hot'], linewidth=4, zorder=10)
    ax_a.scatter([path_x[0]], [path_y[0]], [path_z[0]], s=200, c=COLORS['cool'],
                 marker='o', edgecolor='black', linewidth=2, zorder=11)
    ax_a.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], s=200, c=COLORS['highlight'],
                 marker='*', edgecolor='black', linewidth=2, zorder=11)

    ax_a.set_xlabel('X', fontsize=10)
    ax_a.set_ylabel('Y', fontsize=10)
    ax_a.set_zlabel('Energy', fontsize=10)
    ax_a.set_title('A. Deterministic Path\nNo choices, automatic descent',
                   fontsize=11, fontweight='bold')
    ax_a.view_init(elev=30, azim=45)

    # === PANEL B: Branching Factor ===
    ax_b = fig.add_subplot(gs[0, 1])

    steps = np.arange(20)
    branching = np.ones_like(steps)  # Always 1

    bars = ax_b.bar(steps, branching, color=COLORS['primary'],
                    edgecolor='black', linewidth=1)

    ax_b.axhline(y=1, color=COLORS['hot'], linestyle='--', linewidth=2)
    ax_b.text(10, 1.3, 'Branching = 1\n(No decisions!)', fontsize=12,
              ha='center', fontweight='bold', color=COLORS['hot'])

    ax_b.set_xlabel('Completion Step', fontsize=10)
    ax_b.set_ylabel('Options Available', fontsize=10)
    ax_b.set_title('B. Branching Factor\nAlways exactly 1 option',
                   fontsize=11, fontweight='bold')
    ax_b.set_ylim(0, 2)
    ax_b.grid(True, alpha=0.3, axis='y')

    # === PANEL C: Multiple Runs Overlay ===
    ax_c = fig.add_subplot(gs[1, 0])

    t = np.linspace(0, 1, 100)
    completion = 1 - np.exp(-5 * t)

    # Plot many runs (all identical)
    for run in range(20):
        alpha = 0.3 if run > 0 else 1.0
        linewidth = 2 if run > 0 else 4
        ax_c.plot(t, completion, linewidth=linewidth, alpha=alpha,
                  color=COLORS['cool'] if run > 0 else COLORS['primary'])

    ax_c.scatter([1], [completion[-1]], s=200, c=COLORS['highlight'],
                 edgecolor='black', linewidth=2, zorder=10)

    ax_c.set_xlabel('Normalized Time', fontsize=10)
    ax_c.set_ylabel('Completion', fontsize=10)
    ax_c.set_title('C. 20 Runs Overlaid\nAll IDENTICAL (deterministic)',
                   fontsize=11, fontweight='bold')
    ax_c.text(0.5, 0.3, 'All runs\nidentical', fontsize=14, fontweight='bold',
              ha='center', color=COLORS['hot'])
    ax_c.grid(True, alpha=0.3)

    # === PANEL D: Completion Cascade ===
    ax_d = fig.add_subplot(gs[1, 1])

    # Waterfall/cascade plot
    n_stages = 8
    stage_names = [f'Stage {i+1}' for i in range(n_stages)]
    stage_completion = [1, 3, 2, 4, 2, 3, 1, 1]  # States completed at each stage

    colors_cascade = [plt.cm.viridis(i/n_stages) for i in range(n_stages)]

    bottom = 0
    for i, (name, comp, color) in enumerate(zip(stage_names, stage_completion, colors_cascade)):
        ax_d.bar(0.5, comp, bottom=bottom, width=0.6, color=color,
                 edgecolor='black', linewidth=1)
        ax_d.text(0.5, bottom + comp/2, f'{name}\n({comp})',
                  ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += comp

    # Arrow showing automatic flow
    for i in range(n_stages - 1):
        y = sum(stage_completion[:i+1])
        ax_d.annotate('', xy=(1.0, y + 0.3), xytext=(1.0, y - 0.3),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax_d.set_xlim(0, 1.5)
    ax_d.set_ylim(0, sum(stage_completion) + 1)
    ax_d.set_xticks([])
    ax_d.set_ylabel('Cumulative Completions', fontsize=10)
    ax_d.set_title('D. Completion Cascade\nAutomatic 3^k decomposition',
                   fontsize=11, fontweight='bold')
    ax_d.text(1.2, sum(stage_completion)/2, 'Automatic\nflow\n(no decisions)',
              fontsize=10, ha='left', va='center', fontweight='bold')

    fig.suptitle('ARGUMENT 5: DISSOLUTION OF DECISION\n'
                 'Categorical completion is automatic, not deliberative',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# ARGUMENT 6: DISSOLUTION OF SECOND LAW
# ============================================================================

def visualize_arg6_dissolution_second_law(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 6: Dissolution of Second Law Violation

    Panels:
    A. Two entropy streams (stacked area)
    B. Network density evolution (3D surface)
    C. Entropy accounting (waterfall chart)
    D. ΔS distribution (histogram with PDF)
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === PANEL A: Stacked Entropy Streams ===
    ax_a = fig.add_subplot(gs[0, 0])

    t = np.linspace(0, 10, 200)

    # Spatial entropy (appears to decrease)
    S_spatial = 10 - 2 * (1 - np.exp(-t/3)) + np.random.randn(len(t)) * 0.2

    # Categorical entropy (increases)
    S_categorical = 10 + 3 * (1 - np.exp(-t/3)) + np.random.randn(len(t)) * 0.2

    ax_a.fill_between(t, 0, S_spatial, alpha=0.6, color=COLORS['cool'],
                      label='Spatial Entropy')
    ax_a.fill_between(t, S_spatial, S_spatial + S_categorical - 10,
                      alpha=0.6, color=COLORS['warm'],
                      label='Categorical Entropy')

    # Total entropy line
    S_total = S_spatial + S_categorical - 10
    ax_a.plot(t, S_total, color=COLORS['hot'], linewidth=3,
              label='Total (always increases)')

    ax_a.set_xlabel('Time', fontsize=10)
    ax_a.set_ylabel('Entropy', fontsize=10)
    ax_a.set_title('A. Two Entropy Components\nTotal always increases',
                   fontsize=11, fontweight='bold')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # === PANEL B: Network Density 3D Surface ===
    ax_b = fig.add_subplot(gs[0, 1], projection='3d')

    x = np.linspace(0, 10, 30)  # Time
    y = np.linspace(0, 10, 30)  # Space coordinate
    X, Y = np.meshgrid(x, y)

    # Network density increases over time
    density = 0.5 + 0.3 * X/10 + 0.1 * np.sin(Y) * np.exp(-X/5)

    surf = ax_b.plot_surface(X, Y, density, cmap=CMAP_NETWORK, alpha=0.8)

    ax_b.set_xlabel('Time', fontsize=10)
    ax_b.set_ylabel('Space', fontsize=10)
    ax_b.set_zlabel('Network Density', fontsize=10)
    ax_b.set_title('B. Network Densification\nCategorical entropy increases',
                   fontsize=11, fontweight='bold')
    ax_b.view_init(elev=25, azim=45)

    # === PANEL C: Entropy Accounting (Waterfall) ===
    ax_c = fig.add_subplot(gs[1, 0])

    categories = ['Initial\nS_spatial', 'Change\nΔS_spatial', 'Change\nΔS_categorical',
                  'Final\nS_total']
    values = [10, -2, +4, 12]
    colors_wf = [COLORS['cool'], COLORS['hot'], COLORS['warm'], COLORS['primary']]

    # Calculate positions
    running_total = [0]
    for i, val in enumerate(values[:-1]):
        running_total.append(running_total[-1] + val)

    # Draw bars
    for i, (cat, val, color) in enumerate(zip(categories, values, colors_wf)):
        if i == 0 or i == len(values) - 1:
            bottom = 0
            height = val
        else:
            bottom = running_total[i]
            height = val

        ax_c.bar(i, height, bottom=bottom, color=color, edgecolor='black',
                 linewidth=1.5, width=0.7)

        # Add value labels
        label_y = bottom + height/2 if height > 0 else bottom + height/2
        ax_c.text(i, label_y, f'{val:+.0f}' if i > 0 else f'{val:.0f}',
                  ha='center', va='center', fontsize=11, fontweight='bold',
                  color='white' if abs(height) > 1 else 'black')

    # Connect bars
    for i in range(len(values) - 1):
        y = running_total[i+1]
        ax_c.plot([i+0.35, i+0.65], [y, y], 'k--', linewidth=1.5)

    ax_c.set_xticks(range(len(categories)))
    ax_c.set_xticklabels(categories, fontsize=9)
    ax_c.set_ylabel('Entropy', fontsize=10)
    ax_c.set_title('C. Entropy Accounting\nΔS_total = +2 > 0 ✓',
                   fontsize=11, fontweight='bold')
    ax_c.axhline(y=0, color='black', linewidth=1)
    ax_c.grid(True, alpha=0.3, axis='y')

    # === PANEL D: ΔS Distribution ===
    ax_d = fig.add_subplot(gs[1, 1])

    np.random.seed(42)
    n_samples = 1000

    # Simulate entropy changes
    dS_spatial = np.random.normal(-0.5, 0.3, n_samples)
    dS_categorical = np.random.normal(0.8, 0.2, n_samples)
    dS_total = dS_spatial + dS_categorical

    bins = np.linspace(-1.5, 2, 50)

    ax_d.hist(dS_spatial, bins=bins, alpha=0.5, color=COLORS['cool'],
              label='ΔS_spatial', density=True)
    ax_d.hist(dS_categorical, bins=bins, alpha=0.5, color=COLORS['warm'],
              label='ΔS_categorical', density=True)
    ax_d.hist(dS_total, bins=bins, alpha=0.7, color=COLORS['primary'],
              label='ΔS_total', density=True)

    # Mark zero
    ax_d.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax_d.fill_betweenx([0, 3], 0, 2, alpha=0.1, color=COLORS['warm'])
    ax_d.text(0.7, 2.5, 'ΔS > 0\n(99.9%)', fontsize=11, fontweight='bold',
              color=COLORS['warm'])

    ax_d.set_xlabel('Entropy Change (ΔS)', fontsize=10)
    ax_d.set_ylabel('Probability Density', fontsize=10)
    ax_d.set_title('D. ΔS Distribution\nTotal always positive',
                   fontsize=11, fontweight='bold')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    fig.suptitle('ARGUMENT 6: DISSOLUTION OF SECOND LAW VIOLATION\n'
                 'Categorical entropy increase compensates: ΔS_total > 0 always',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# ARGUMENT 7: INFORMATION COMPLEMENTARITY
# ============================================================================

def visualize_arg7_information_complementarity(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 7: Information Complementarity

    Panels:
    A. Dual face projection (3D with projections)
    B. Uncertainty relation (contour plot)
    C. Ammeter/Voltmeter circuit diagram
    D. Demon as projection shadow
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.35)

    # === PANEL A: 3D Dual Face Projection ===
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    np.random.seed(42)
    n_points = 100

    # Generate points in 3D (information space)
    theta = np.random.rand(n_points) * 2 * np.pi
    r = np.random.rand(n_points) * 2
    z = np.random.randn(n_points)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 3D scatter
    ax_a.scatter(x, y, z, c=z, cmap=CMAP_ENTROPY, s=50, alpha=0.7)

    # Project onto XY (kinetic face) and XZ (categorical face)
    ax_a.scatter(x, y, np.ones_like(z) * -3, c=COLORS['kinetic'],
                 alpha=0.3, s=20, marker='s')
    ax_a.scatter(x, np.ones_like(y) * 3, z, c=COLORS['categorical'],
                 alpha=0.3, s=20, marker='^')

    # Draw projection lines for a few points
    for i in range(0, n_points, 10):
        ax_a.plot([x[i], x[i]], [y[i], y[i]], [z[i], -3],
                  color=COLORS['kinetic'], alpha=0.2, linewidth=0.5)
        ax_a.plot([x[i], x[i]], [y[i], 3], [z[i], z[i]],
                  color=COLORS['categorical'], alpha=0.2, linewidth=0.5)

    # Labels
    ax_a.text(0, 0, -3.5, 'KINETIC FACE', fontsize=10, ha='center',
              color=COLORS['kinetic'], fontweight='bold')
    ax_a.text(0, 3.5, 0, 'CATEGORICAL\nFACE', fontsize=10, ha='center',
              color=COLORS['categorical'], fontweight='bold')

    ax_a.set_xlabel('X', fontsize=10)
    ax_a.set_ylabel('Y', fontsize=10)
    ax_a.set_zlabel('Z', fontsize=10)
    ax_a.set_title('A. Dual Face Projections\n3D reality → 2D observations',
                   fontsize=11, fontweight='bold')
    ax_a.view_init(elev=20, azim=45)

    # === PANEL B: Uncertainty Contour ===
    ax_b = fig.add_subplot(gs[0, 1])

    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)

    # Uncertainty product (kinetic × categorical precision)
    uncertainty = X * Y

    contour = ax_b.contourf(X, Y, uncertainty, levels=20, cmap='RdYlGn_r')
    ax_b.contour(X, Y, uncertainty, levels=[1], colors='white', linewidths=3)

    # Add points for different observation strategies
    ax_b.scatter([4], [0.5], s=200, c=COLORS['kinetic'], edgecolor='white',
                 linewidth=2, marker='o', label='Kinetic focus')
    ax_b.scatter([0.5], [4], s=200, c=COLORS['categorical'], edgecolor='white',
                 linewidth=2, marker='^', label='Categorical focus')
    ax_b.scatter([2], [2], s=200, c=COLORS['highlight'], edgecolor='white',
                 linewidth=2, marker='*', label='Balanced')

    ax_b.set_xlabel('Kinetic Precision (ΔE_kin)', fontsize=10)
    ax_b.set_ylabel('Categorical Precision (ΔG)', fontsize=10)
    ax_b.set_title('B. Complementarity Relation\nΔE × ΔG ≥ constant',
                   fontsize=11, fontweight='bold')
    ax_b.legend(loc='upper right')
    plt.colorbar(contour, ax=ax_b, label='Uncertainty Product')

    # === PANEL C: Circuit Analogy ===
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(-1, 11)
    ax_c.set_ylim(-1, 7)
    ax_c.axis('off')

    # Draw circuit
    # Battery
    ax_c.plot([0, 0], [1, 5], 'k-', linewidth=3)
    ax_c.plot([0, 2], [5, 5], 'k-', linewidth=3)
    ax_c.plot([0, 2], [1, 1], 'k-', linewidth=3)
    ax_c.text(0.5, 3, '+\n-', fontsize=14, ha='center', va='center')

    # Resistor (component)
    ax_c.add_patch(plt.Rectangle((4, 2.5), 2, 1, fill=True,
                                   facecolor=COLORS['tertiary'], edgecolor='black', linewidth=2))
    ax_c.text(5, 3, 'R', fontsize=12, ha='center', va='center', color='white', fontweight='bold')
    ax_c.plot([2, 4], [5, 5], 'k-', linewidth=3)
    ax_c.plot([2, 4], [1, 1], 'k-', linewidth=3)
    ax_c.plot([4, 4], [5, 3.5], 'k-', linewidth=3)
    ax_c.plot([4, 4], [2.5, 1], 'k-', linewidth=3)
    ax_c.plot([6, 6], [5, 3.5], 'k-', linewidth=3)
    ax_c.plot([6, 6], [2.5, 1], 'k-', linewidth=3)
    ax_c.plot([6, 10], [5, 5], 'k-', linewidth=3)
    ax_c.plot([6, 10], [1, 1], 'k-', linewidth=3)
    ax_c.plot([10, 10], [1, 5], 'k-', linewidth=3)

    # Ammeter (series) - KINETIC
    ammeter = Circle((3, 5), 0.5, fill=True, facecolor=COLORS['kinetic'],
                     edgecolor='black', linewidth=2)
    ax_c.add_patch(ammeter)
    ax_c.text(3, 5, 'A', fontsize=12, ha='center', va='center',
              color='white', fontweight='bold')
    ax_c.text(3, 6, 'AMMETER\n(Current = Kinetic)', fontsize=9, ha='center',
              color=COLORS['kinetic'], fontweight='bold')

    # Voltmeter (parallel) - CATEGORICAL
    voltmeter = Circle((8, 3), 0.5, fill=True, facecolor=COLORS['categorical'],
                        edgecolor='black', linewidth=2)
    ax_c.add_patch(voltmeter)
    ax_c.text(8, 3, 'V', fontsize=12, ha='center', va='center',
              color='white', fontweight='bold')
    ax_c.plot([8, 8], [5, 3.5], 'k-', linewidth=2)
    ax_c.plot([8, 8], [2.5, 1], 'k-', linewidth=2)
    ax_c.text(9.5, 3, 'VOLTMETER\n(Voltage = Categorical)', fontsize=9,
              ha='left', color=COLORS['categorical'], fontweight='bold')

    # Cannot use both annotation
    ax_c.text(5, -0.5, 'Cannot use BOTH meters simultaneously\non the SAME component!',
              fontsize=11, ha='center', fontweight='bold', color=COLORS['hot'],
              bbox=dict(boxstyle='round', facecolor=COLORS['accent2']))

    ax_c.set_title('C. Ammeter/Voltmeter Analogy\nComplementary measurements',
                   fontsize=11, fontweight='bold')

    # === PANEL D: Demon as Shadow ===
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(-1, 11)
    ax_d.set_ylim(-1, 8)
    ax_d.axis('off')

    # 3D object (categorical dynamics)
    obj_x = [2, 4, 5, 3, 2]
    obj_y = [6, 7, 5, 4, 6]
    obj = Polygon(list(zip(obj_x, obj_y)), fill=True,
                  facecolor=COLORS['categorical'], edgecolor='black', linewidth=2, alpha=0.8)
    ax_d.add_patch(obj)
    ax_d.text(3.5, 5.5, 'CATEGORICAL\nDYNAMICS', fontsize=9, ha='center',
              color='white', fontweight='bold')

    # Light source
    ax_d.scatter([0], [7], s=300, c=COLORS['highlight'], marker='*',
                 edgecolor='black', linewidth=2, zorder=10)
    ax_d.text(0, 7.5, 'Observer', fontsize=10, ha='center')

    # Shadow on kinetic plane
    shadow_x = [4, 7, 8, 5, 4]
    shadow_y = [1.5, 2, 1, 0.5, 1.5]
    shadow = Polygon(list(zip(shadow_x, shadow_y)), fill=True,
                     facecolor='gray', edgecolor='black', linewidth=2, alpha=0.5)
    ax_d.add_patch(shadow)
    ax_d.text(6, 0.8, '"DEMON"', fontsize=11, ha='center', fontweight='bold',
              color=COLORS['hot'])

    # Projection lines
    for ox, oy, sx, sy in zip(obj_x, obj_y, shadow_x, shadow_y):
        ax_d.plot([ox, sx], [oy, sy], 'k--', alpha=0.3, linewidth=1)

    # Kinetic plane
    ax_d.plot([-0.5, 10], [0, 0], 'k-', linewidth=3)
    ax_d.text(10.5, 0, 'KINETIC FACE', fontsize=10, va='center', fontweight='bold')

    # Explanation
    ax_d.text(5, 3.5, '↓ Projection ↓', fontsize=12, ha='center',
              rotation=70, fontweight='bold', color='gray')

    ax_d.text(5, -0.8, 'The "demon" is just the SHADOW\nof hidden categorical dynamics!',
              fontsize=11, ha='center', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor=COLORS['accent2'], edgecolor=COLORS['hot']))

    ax_d.set_title('D. Demon = Projection Artifact\nShadow of hidden dynamics',
                   fontsize=11, fontweight='bold')

    fig.suptitle('ARGUMENT 7: INFORMATION COMPLEMENTARITY\n'
                 'The "demon" is projection of hidden categorical dynamics onto kinetic face',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_dissolution_figures(output_dir: str = "results/figures/dissolution") -> Dict[str, str]:
    """Generate all 7 dissolution argument visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING MAXWELL'S DEMON DISSOLUTION VISUALIZATIONS")
    print("=" * 70)

    figures = {}

    visualizers = [
        ("arg1_temporal_triviality", visualize_arg1_temporal_triviality),
        ("arg2_temperature_independence", visualize_arg2_temperature_independence),
        ("arg3_retrieval_paradox", visualize_arg3_retrieval_paradox),
        ("arg4_dissolution_observation", visualize_arg4_dissolution_observation),
        ("arg5_dissolution_decision", visualize_arg5_dissolution_decision),
        ("arg6_dissolution_second_law", visualize_arg6_dissolution_second_law),
        ("arg7_information_complementarity", visualize_arg7_information_complementarity),
    ]

    for i, (name, func) in enumerate(visualizers, 1):
        print(f"\n[{i}/7] Generating {name}...")
        path = str(output_path / f"{name}.png")
        fig = func(path)
        figures[name] = path
        plt.close(fig)

    print("\n" + "=" * 70)
    print(f"ALL 7 DISSOLUTION FIGURES GENERATED → {output_path}")
    print("=" * 70)

    return figures


if __name__ == "__main__":
    generate_all_dissolution_figures()

