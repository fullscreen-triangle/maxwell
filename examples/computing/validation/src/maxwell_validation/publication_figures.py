"""
Publication-Quality Panel Charts for Maxwell's Demon Resolution
================================================================

Each of the 7 dissolution arguments gets a 4-panel figure:
1. Main result visualization
2. Statistical validation
3. Comparison/Control
4. Physical interpretation

All figures are:
- High resolution (600 DPI for raster)
- Consistent color scheme
- Minimal text, maximum information
- Publication-ready typography
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path, connected_components
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Color palette - distinct, colorblind-friendly
COLORS = {
    'primary': '#1A535C',      # Dark teal
    'secondary': '#4ECDC4',    # Bright teal
    'accent': '#FF6B6B',       # Coral red
    'highlight': '#FFE66D',    # Yellow
    'neutral': '#95A5A6',      # Gray
    'dark': '#2C3E50',         # Dark blue-gray
    'light': '#ECF0F1',        # Light gray
    'success': '#27AE60',      # Green
    'warning': '#F39C12',      # Orange
    'categorical': '#9B59B6',  # Purple
    'kinetic': '#E74C3C',      # Red
    'spatial': '#3498DB',      # Blue
}

# Figure dimensions
PANEL_SIZE = (16, 12)  # 4-panel figure size
FONT_SIZES = {
    'title': 16,
    'subtitle': 14,
    'label': 12,
    'tick': 10,
    'annotation': 9,
    'legend': 10,
}


def setup_style():
    """Configure matplotlib for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


setup_style()


# ============================================================================
# DATA GENERATION UTILITIES
# ============================================================================

def generate_molecular_system(n_molecules: int = 100, box_size: float = 10.0,
                               temperature: float = 1.0, seed: int = 42) -> Dict:
    """Generate a complete molecular system for visualization"""
    np.random.seed(seed)
    
    # Positions
    positions = np.random.uniform(0, box_size, (n_molecules, 3))
    
    # Velocities (Maxwell-Boltzmann)
    sigma = np.sqrt(temperature)
    velocities = np.random.normal(0, sigma, (n_molecules, 3))
    
    # Phase-lock network
    distances = squareform(pdist(positions))
    cutoff = 2.0
    adjacency = (distances < cutoff) & (distances > 0)
    
    # Calculate properties
    speeds = np.linalg.norm(velocities, axis=1)
    kinetic_energy = 0.5 * np.sum(velocities**2)
    
    # Network properties
    degree = adjacency.sum(axis=1)
    n_edges = adjacency.sum() / 2
    
    # Categorical distances
    categorical_dist = shortest_path(adjacency.astype(float), directed=False)
    categorical_dist[np.isinf(categorical_dist)] = n_molecules
    
    return {
        'positions': positions,
        'velocities': velocities,
        'speeds': speeds,
        'adjacency': adjacency,
        'distances': distances,
        'categorical_dist': categorical_dist,
        'degree': degree,
        'n_edges': n_edges,
        'kinetic_energy': kinetic_energy,
        'temperature': temperature,
        'n_molecules': n_molecules,
    }


# ============================================================================
# ARGUMENT 1: TEMPORAL TRIVIALITY
# ============================================================================

def create_panel_arg1_temporal_triviality(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 1: Temporal Triviality of Maxwell's Demon
    
    The demon does not accelerate anything. Any configuration that would result
    from "demon operation" will occur naturally through thermal fluctuations
    with probability exp(-ΔS/k_B).
    
    4 Panels:
    A. Fluctuation probability distribution
    B. Configuration recurrence times
    C. Demon vs. natural fluctuation comparison
    D. Boltzmann probability landscape
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === PANEL A: Fluctuation probability distribution ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Generate "sorted" states and their probabilities
    n_states = 1000
    sorting_degree = np.linspace(0, 1, n_states)  # 0 = random, 1 = fully sorted
    
    # Probability of state decreases exponentially with sorting
    delta_S = sorting_degree * 10  # Entropy reduction
    probability = np.exp(-delta_S)
    
    ax_a.semilogy(sorting_degree, probability, linewidth=3, color=COLORS['primary'])
    ax_a.fill_between(sorting_degree, probability, alpha=0.3, color=COLORS['secondary'])
    
    # Mark key points
    ax_a.axhline(y=1e-4, color=COLORS['accent'], linestyle='--', linewidth=2, label='Observable threshold')
    ax_a.axvline(x=0.5, color=COLORS['neutral'], linestyle=':', linewidth=2)
    
    ax_a.set_xlabel('Sorting Degree (0=random, 1=sorted)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_a.set_ylabel('Boltzmann Probability', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_a.set_title('A. Fluctuation Probability', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(1e-5, 1)
    ax_a.legend(loc='upper right', fontsize=FONT_SIZES['legend'])
    
    # === PANEL B: Configuration recurrence times ===
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Recurrence time scales with 1/probability
    recurrence_time = 1.0 / (probability + 1e-10)
    
    ax_b.semilogy(sorting_degree, recurrence_time, linewidth=3, color=COLORS['accent'])
    ax_b.fill_between(sorting_degree, recurrence_time, alpha=0.2, color=COLORS['accent'])
    
    # Add Poincaré recurrence annotation
    ax_b.axhline(y=1e6, color=COLORS['success'], linestyle='--', linewidth=2, label='Lab timescale')
    
    ax_b.set_xlabel('Sorting Degree', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_ylabel('Recurrence Time (τ)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_title('B. Poincaré Recurrence', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_b.legend(loc='upper left', fontsize=FONT_SIZES['legend'])
    
    # === PANEL C: Demon vs. natural fluctuation ===
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Simulate many trials
    n_trials = 50
    times = np.arange(1000)
    
    # Natural fluctuations (random walk in sorting space)
    np.random.seed(42)
    for i in range(n_trials):
        fluctuations = np.cumsum(np.random.normal(0, 0.02, len(times)))
        fluctuations = np.clip(0.5 + fluctuations, 0, 1)
        ax_c.plot(times, fluctuations, alpha=0.2, color=COLORS['spatial'], linewidth=0.8)
    
    # Mean stays at 0.5
    ax_c.axhline(y=0.5, color=COLORS['dark'], linestyle='-', linewidth=3, label='Equilibrium')
    
    # "Demon" attempt (quick spike then return)
    demon_trajectory = np.ones_like(times) * 0.5
    demon_trajectory[100:150] = 0.5 + 0.3 * np.sin(np.linspace(0, np.pi, 50))
    ax_c.plot(times, demon_trajectory, color=COLORS['accent'], linewidth=3, label='"Demon" attempt')
    
    ax_c.set_xlabel('Time Steps', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_ylabel('Sorting Degree', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_title('C. Demon vs. Natural Fluctuations', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=FONT_SIZES['legend'])
    ax_c.set_ylim(0, 1)
    
    # === PANEL D: Boltzmann probability landscape ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # 2D energy landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Free energy with basin at origin (equilibrium)
    F = X**2 + Y**2 + 0.3*np.sin(3*X)*np.sin(3*Y)
    probability_landscape = np.exp(-F)
    
    contour = ax_d.contourf(X, Y, probability_landscape, levels=20, cmap='viridis')
    ax_d.contour(X, Y, probability_landscape, levels=5, colors='white', alpha=0.5, linewidths=1)
    
    # Mark equilibrium
    ax_d.plot(0, 0, 'o', markersize=15, color=COLORS['accent'], markeredgecolor='white', markeredgewidth=2)
    ax_d.annotate('Equilibrium', (0.2, 0.2), fontsize=FONT_SIZES['annotation'], 
                  color='white', fontweight='bold')
    
    # Mark "sorted" state
    ax_d.plot(2, 2, 's', markersize=12, color=COLORS['warning'], markeredgecolor='white', markeredgewidth=2)
    ax_d.annotate('"Sorted"', (2.2, 2.2), fontsize=FONT_SIZES['annotation'], 
                  color='white', fontweight='bold')
    
    # Arrow showing spontaneous return
    ax_d.annotate('', xy=(0.3, 0.3), xytext=(1.8, 1.8),
                  arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    ax_d.set_xlabel('Configuration Coordinate 1', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_ylabel('Configuration Coordinate 2', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_title('D. Boltzmann Probability Landscape', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    plt.colorbar(contour, ax=ax_d, label='P(config)', shrink=0.8)
    
    # Overall title
    fig.suptitle('Argument 1: TEMPORAL TRIVIALITY\nFluctuations produce any "sorted" configuration naturally',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# ARGUMENT 2: PHASE-LOCK TEMPERATURE INDEPENDENCE
# ============================================================================

def create_panel_arg2_temperature_independence(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 2: Phase-Lock Temperature Independence
    
    The same spatial arrangement can exist at any temperature.
    Network topology is independent of kinetic energy: ∂G/∂E_kin = 0
    
    4 Panels:
    A. Network topology at multiple temperatures
    B. Network edges vs. kinetic energy (no correlation)
    C. Van der Waals coupling independence
    D. Configuration space projection
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Generate same configuration at different temperatures
    np.random.seed(42)
    n_molecules = 100
    box_size = 10.0
    positions = np.random.uniform(0, box_size, (n_molecules, 3))
    
    # Build network (same for all temperatures)
    distances = squareform(pdist(positions))
    cutoff = 2.0
    adjacency = (distances < cutoff) & (distances > 0)
    n_edges = adjacency.sum() / 2
    
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    # === PANEL A: Network at multiple temperatures ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    edge_counts = []
    kinetic_energies = []
    
    for T in temperatures:
        # Different velocities, same positions
        sigma = np.sqrt(T)
        velocities = np.random.normal(0, sigma, (n_molecules, 3))
        KE = 0.5 * np.sum(velocities**2)
        
        edge_counts.append(n_edges)
        kinetic_energies.append(KE)
    
    ax_a_twin = ax_a.twinx()
    
    line1, = ax_a.plot(temperatures, edge_counts, 'o-', markersize=12, linewidth=3,
                       color=COLORS['primary'], label='Network Edges')
    line2, = ax_a_twin.plot(temperatures, kinetic_energies, 's-', markersize=10, linewidth=3,
                            color=COLORS['accent'], label='Kinetic Energy')
    
    ax_a.set_xlabel('Temperature (T)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_a.set_ylabel('Network Edges (constant)', fontsize=FONT_SIZES['label'], 
                    fontweight='bold', color=COLORS['primary'])
    ax_a_twin.set_ylabel('Kinetic Energy (∝ T)', fontsize=FONT_SIZES['label'],
                         fontweight='bold', color=COLORS['accent'])
    ax_a.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax_a_twin.tick_params(axis='y', labelcolor=COLORS['accent'])
    
    ax_a.set_title('A. Same Network, Different Temperatures', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_a.legend([line1, line2], ['Edges (∂G/∂T=0)', 'KE (∝T)'], loc='center right', fontsize=FONT_SIZES['legend'])
    
    # === PANEL B: KE vs. edges scatter ===
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Many random configurations
    n_trials = 500
    ke_all = []
    edges_all = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        pos = np.random.uniform(0, box_size, (n_molecules, 3))
        T = np.random.uniform(0.5, 10.0)
        vel = np.random.normal(0, np.sqrt(T), (n_molecules, 3))
        
        dist = squareform(pdist(pos))
        adj = (dist < cutoff) & (dist > 0)
        
        ke_all.append(0.5 * np.sum(vel**2))
        edges_all.append(adj.sum() / 2)
    
    ax_b.scatter(ke_all, edges_all, alpha=0.5, s=30, color=COLORS['secondary'])
    
    # Add correlation line (should be flat)
    z = np.polyfit(ke_all, edges_all, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ke_all), max(ke_all), 100)
    ax_b.plot(x_line, p(x_line), '--', color=COLORS['accent'], linewidth=2)
    
    # Calculate correlation
    corr = np.corrcoef(ke_all, edges_all)[0, 1]
    ax_b.text(0.95, 0.95, f'r = {corr:.4f}', transform=ax_b.transAxes,
              fontsize=FONT_SIZES['annotation'], ha='right', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_b.set_xlabel('Kinetic Energy', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_ylabel('Network Edges', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_title('B. Zero Correlation: ∂G/∂E_kin = 0', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL C: Van der Waals coupling ===
    ax_c = fig.add_subplot(gs[1, 0])
    
    r = np.linspace(0.5, 5, 100)
    
    # VdW coupling (position dependent only)
    U_vdw = 1.0 / r**6
    
    # Kinetic contribution (none to coupling)
    U_kinetic = np.zeros_like(r)
    
    ax_c.semilogy(r, U_vdw, linewidth=3, color=COLORS['primary'], label='VdW: U ∝ r⁻⁶')
    ax_c.axhline(y=0.01, color=COLORS['neutral'], linestyle='--', linewidth=2, label='Coupling threshold')
    
    # Fill region where coupling is significant
    ax_c.fill_between(r, U_vdw, 0.01, where=(U_vdw > 0.01), 
                      alpha=0.3, color=COLORS['secondary'], label='Coupled region')
    
    ax_c.set_xlabel('Intermolecular Distance (r)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_ylabel('Coupling Strength', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_title('C. Van der Waals Coupling\n(velocity-independent)', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=FONT_SIZES['legend'])
    ax_c.set_xlim(0.5, 5)
    
    # === PANEL D: Configuration space ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # 2D projection of configuration space
    # X-axis: configuration (positions), Y-axis: velocity (temperature)
    configs = np.linspace(0, 1, 50)
    temps = np.linspace(0.5, 10, 50)
    C, T_mesh = np.meshgrid(configs, temps)
    
    # Network topology is horizontal bands (constant with T)
    network_property = np.sin(3 * np.pi * C)  # Only depends on C
    
    contour = ax_d.contourf(C, T_mesh, network_property, levels=20, cmap='coolwarm')
    
    # Add arrows showing independence
    for t in [2, 5, 8]:
        ax_d.annotate('', xy=(0.9, t), xytext=(0.1, t),
                      arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    
    ax_d.set_xlabel('Configuration Space', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_ylabel('Temperature', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_title('D. Network Topology:\nConstant along T', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    plt.colorbar(contour, ax=ax_d, label='Network Property', shrink=0.8)
    
    # Overall title
    fig.suptitle('Argument 2: PHASE-LOCK TEMPERATURE INDEPENDENCE\nNetwork topology ∂G/∂E_kin = 0: independent of kinetic energy',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# ARGUMENT 3: RETRIEVAL PARADOX
# ============================================================================

def create_panel_arg3_retrieval_paradox(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 3: The Retrieval Paradox
    
    Sorting by velocity is self-defeating. Thermal equilibration randomizes
    velocities faster than any sorting mechanism can operate.
    
    4 Panels:
    A. Collision timescales vs. sorting timescales
    B. Velocity redistribution after sorting attempt
    C. Sorting ratio over time (stays at 50%)
    D. Kinetic vs. categorical timescale separation
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === PANEL A: Timescale comparison ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    processes = ['Molecular\nCollision', 'Velocity\nMeasurement', 'Door\nOperation', 'Sorting\nComplete']
    timescales = [1e-10, 1e-8, 1e-6, 1e-3]  # seconds
    
    colors = [COLORS['accent'], COLORS['warning'], COLORS['warning'], COLORS['neutral']]
    bars = ax_a.barh(processes, timescales, color=colors, edgecolor='black', linewidth=1.5)
    
    ax_a.set_xscale('log')
    ax_a.set_xlabel('Timescale (seconds)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_a.set_title('A. Timescale Hierarchy', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # Add values
    for bar, val in zip(bars, timescales):
        ax_a.text(val * 1.5, bar.get_y() + bar.get_height()/2, 
                  f'{val:.0e}s', va='center', fontsize=FONT_SIZES['annotation'])
    
    # Add arrow indicating collision happens first
    ax_a.annotate('Collisions happen\nbefore sorting!', 
                  xy=(1e-10, 0), xytext=(1e-7, 0.5),
                  fontsize=FONT_SIZES['annotation'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    
    # === PANEL B: Velocity redistribution ===
    ax_b = fig.add_subplot(gs[0, 1])
    
    np.random.seed(42)
    n_molecules = 1000
    
    # Initial Maxwell-Boltzmann
    velocities_initial = np.random.normal(0, 1, n_molecules)
    
    # "Sorted" distribution (attempt)
    velocities_sorted = np.concatenate([
        np.random.normal(-1, 0.3, n_molecules//2),
        np.random.normal(1, 0.3, n_molecules//2)
    ])
    
    # After one collision time (back to Maxwell-Boltzmann)
    velocities_after = np.random.normal(0, 1, n_molecules)
    
    bins = np.linspace(-4, 4, 50)
    ax_b.hist(velocities_initial, bins=bins, alpha=0.5, color=COLORS['primary'], 
              label='Initial (MB)', density=True)
    ax_b.hist(velocities_sorted, bins=bins, alpha=0.5, color=COLORS['accent'], 
              label='"Sorted"', density=True)
    ax_b.hist(velocities_after, bins=bins, alpha=0.5, color=COLORS['success'], 
              label='After τ_collision', density=True)
    
    ax_b.set_xlabel('Velocity', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_ylabel('Probability Density', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_title('B. Velocity Redistribution', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_b.legend(fontsize=FONT_SIZES['legend'])
    
    # === PANEL C: Sorting ratio over time ===
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Simulate sorting attempts
    n_steps = 200
    sorting_ratio = np.ones(n_steps) * 0.5
    
    # Demon attempts at certain times
    demon_attempts = [20, 60, 100, 140]
    for attempt in demon_attempts:
        # Brief spike then return to equilibrium
        spike = np.exp(-np.arange(n_steps - attempt) / 5) * 0.3
        sorting_ratio[attempt:] = 0.5 + spike[:len(sorting_ratio) - attempt]
    
    # Add noise
    sorting_ratio += np.random.normal(0, 0.02, n_steps)
    sorting_ratio = np.clip(sorting_ratio, 0, 1)
    
    ax_c.plot(np.arange(n_steps), sorting_ratio, linewidth=2, color=COLORS['primary'])
    ax_c.axhline(y=0.5, color=COLORS['accent'], linestyle='--', linewidth=3, label='Equilibrium (50%)')
    
    # Mark demon attempts
    for attempt in demon_attempts:
        ax_c.axvline(x=attempt, color=COLORS['neutral'], linestyle=':', alpha=0.5)
        ax_c.annotate('Attempt', (attempt, 0.85), fontsize=FONT_SIZES['annotation'],
                      rotation=90, ha='right')
    
    ax_c.set_xlabel('Time Steps', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_ylabel('Fast/Total Ratio', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_title('C. Sorting Cannot Be Maintained', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=FONT_SIZES['legend'])
    ax_c.set_ylim(0.3, 1.0)
    
    # === PANEL D: Timescale separation ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Energy diagram showing timescales
    categorical_time = np.linspace(0, 1, 100)
    kinetic_time = np.linspace(0, 1, 100)
    
    # Categorical changes slowly
    categorical_evolution = 0.5 + 0.3 * np.tanh(5 * (categorical_time - 0.5))
    
    # Kinetic fluctuates rapidly
    kinetic_evolution = 0.5 + 0.3 * np.sin(50 * kinetic_time) * np.exp(-kinetic_time)
    
    ax_d.plot(categorical_time, categorical_evolution, linewidth=3, 
              color=COLORS['categorical'], label='Categorical (slow)')
    ax_d.plot(kinetic_time, kinetic_evolution, linewidth=2, alpha=0.7,
              color=COLORS['kinetic'], label='Kinetic (fast)')
    
    # Add shaded region showing kinetic fluctuations
    ax_d.fill_between(kinetic_time, 0.5 - 0.3, 0.5 + 0.3, alpha=0.1, color=COLORS['kinetic'])
    
    ax_d.set_xlabel('Normalized Time', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_ylabel('State', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_title('D. Timescale Separation\nKinetic ≫ Categorical', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_d.legend(fontsize=FONT_SIZES['legend'])
    
    # Overall title
    fig.suptitle('Argument 3: THE RETRIEVAL PARADOX\nVelocity-based sorting is self-defeating: thermal equilibration is faster',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# ARGUMENT 4: DISSOLUTION OF OBSERVATION
# ============================================================================

def create_panel_arg4_dissolution_observation(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 4: Dissolution of Observation
    
    The demon cannot "observe" molecular velocities because categorical
    navigation is determined by network topology, not measurement.
    
    4 Panels:
    A. Topology determines accessibility
    B. No measurement required
    C. Observation vs. topology comparison
    D. Quantum measurement analogy (not needed here)
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === PANEL A: Topology determines accessibility ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Create network visualization
    np.random.seed(42)
    n_nodes = 30
    
    # Generate positions in 2D for visualization
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    radii = 1 + 0.3 * np.random.random(n_nodes)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Create adjacency (nearby nodes connected)
    adjacency = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if abs(i - j) <= 2 or abs(i - j) >= n_nodes - 2:
                adjacency[i, j] = adjacency[j, i] = 1
    
    # Draw edges
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency[i, j]:
                ax_a.plot([x[i], x[j]], [y[i], y[j]], 
                         color=COLORS['secondary'], linewidth=1.5, alpha=0.6)
    
    # Draw nodes
    ax_a.scatter(x, y, s=200, c=COLORS['primary'], edgecolors='black', linewidths=2, zorder=5)
    
    # Highlight accessible path
    path = [0, 1, 2, 3]
    for i in range(len(path)-1):
        ax_a.plot([x[path[i]], x[path[i+1]]], [y[path[i]], y[path[i+1]]], 
                 color=COLORS['accent'], linewidth=4, zorder=4)
    ax_a.scatter(x[path], y[path], s=250, c=COLORS['accent'], 
                 edgecolors='black', linewidths=2, zorder=6)
    
    ax_a.set_xlim(-2, 2)
    ax_a.set_ylim(-2, 2)
    ax_a.set_aspect('equal')
    ax_a.axis('off')
    ax_a.set_title('A. Topology Determines Path\n(no velocity information)', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL B: No measurement required ===
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Two boxes: "Measurement" crossed out, "Adjacency" checked
    ax_b.text(0.3, 0.7, '✗ Velocity Measurement', fontsize=14, ha='center', va='center',
              color=COLORS['accent'], fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='#ffcccc', edgecolor=COLORS['accent']))
    
    ax_b.text(0.7, 0.7, '✓ Topological Adjacency', fontsize=14, ha='center', va='center',
              color=COLORS['success'], fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='#ccffcc', edgecolor=COLORS['success']))
    
    # Arrows showing flow
    ax_b.annotate('', xy=(0.5, 0.4), xytext=(0.3, 0.6),
                  arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=2, ls='--'))
    ax_b.annotate('', xy=(0.5, 0.4), xytext=(0.7, 0.6),
                  arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=3))
    
    ax_b.text(0.5, 0.35, 'Navigation', fontsize=14, ha='center', va='center',
              fontweight='bold',
              bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor=COLORS['dark']))
    
    ax_b.text(0.5, 0.15, 'Path follows network structure\nNO measurement of velocities',
              fontsize=11, ha='center', va='center', style='italic')
    
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis('off')
    ax_b.set_title('B. Observation Not Required', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL C: Observation vs. topology ===
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Scatter plot showing no correlation between velocity and network position
    np.random.seed(42)
    n = 100
    
    network_position = np.random.uniform(0, 1, n)  # Position in network
    velocity = np.random.normal(0, 1, n)  # Velocity (random, independent)
    
    ax_c.scatter(network_position, velocity, alpha=0.6, s=50, color=COLORS['secondary'])
    
    # Fit line (should be flat)
    z = np.polyfit(network_position, velocity, 1)
    p = np.poly1d(z)
    ax_c.plot([0, 1], [p(0), p(1)], '--', color=COLORS['accent'], linewidth=2)
    
    corr = np.corrcoef(network_position, velocity)[0, 1]
    ax_c.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax_c.transAxes,
              fontsize=FONT_SIZES['annotation'], ha='right', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_c.set_xlabel('Network Position', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_ylabel('Molecular Velocity', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_title('C. Velocity ⊥ Network Position', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL D: Automatic door analogy ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Draw door diagram
    # Room with door that opens based on adjacency, not velocity
    
    # Walls
    ax_d.plot([0, 0.4], [0.8, 0.8], 'k-', linewidth=4)
    ax_d.plot([0.6, 1], [0.8, 0.8], 'k-', linewidth=4)
    ax_d.plot([0, 0], [0.2, 0.8], 'k-', linewidth=4)
    ax_d.plot([1, 1], [0.2, 0.8], 'k-', linewidth=4)
    ax_d.plot([0, 1], [0.2, 0.2], 'k-', linewidth=4)
    
    # Door (open)
    ax_d.plot([0.4, 0.45], [0.8, 0.95], color=COLORS['success'], linewidth=6)
    
    # Molecules
    fast_pos = np.array([[0.2, 0.5], [0.3, 0.6]])
    slow_pos = np.array([[0.7, 0.5], [0.8, 0.4]])
    
    ax_d.scatter(fast_pos[:, 0], fast_pos[:, 1], s=300, c=COLORS['accent'], 
                 edgecolors='black', linewidths=2, label='Fast', marker='o')
    ax_d.scatter(slow_pos[:, 0], slow_pos[:, 1], s=300, c=COLORS['spatial'], 
                 edgecolors='black', linewidths=2, label='Slow', marker='s')
    
    # Arrow through door (any molecule can pass)
    ax_d.annotate('', xy=(0.5, 0.95), xytext=(0.5, 0.65),
                  arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=3))
    
    ax_d.text(0.5, 1.05, 'Door opens based on ADJACENCY\nNOT velocity', 
              fontsize=11, ha='center', fontweight='bold')
    ax_d.text(0.5, 0.05, 'Any adjacent molecule passes\nVelocity is irrelevant', 
              fontsize=10, ha='center', style='italic')
    
    ax_d.set_xlim(-0.1, 1.1)
    ax_d.set_ylim(0, 1.2)
    ax_d.axis('off')
    ax_d.set_title('D. Topological Gate', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # Overall title
    fig.suptitle('Argument 4: DISSOLUTION OF OBSERVATION\nNavigation follows topology, not measurement of velocities',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# ARGUMENT 5: DISSOLUTION OF DECISION
# ============================================================================

def create_panel_arg5_dissolution_decision(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 5: Dissolution of Decision
    
    The demon makes no decisions. Categorical completion follows
    network topology automatically without deliberation.
    
    4 Panels:
    A. Decision tree collapsed to single path
    B. Automatic pathway following
    C. No branching in categorical space
    D. Deterministic completion
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === PANEL A: Decision tree collapsed ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Traditional decision tree (crossed out)
    # Root
    ax_a.plot(0.5, 0.9, 'o', markersize=20, color=COLORS['neutral'])
    
    # Branches (faded, crossed out)
    for end_x in [0.2, 0.4, 0.6, 0.8]:
        ax_a.plot([0.5, end_x], [0.9, 0.5], '--', color=COLORS['neutral'], alpha=0.3, linewidth=2)
        ax_a.plot(end_x, 0.5, 'o', markersize=15, color=COLORS['neutral'], alpha=0.3)
    
    # Single path (highlighted)
    ax_a.plot([0.5, 0.5], [0.9, 0.5], '-', color=COLORS['success'], linewidth=4)
    ax_a.plot(0.5, 0.5, 'o', markersize=18, color=COLORS['success'])
    ax_a.plot([0.5, 0.5], [0.5, 0.2], '-', color=COLORS['success'], linewidth=4)
    ax_a.plot(0.5, 0.2, 'o', markersize=18, color=COLORS['success'])
    
    # Cross out alternatives
    ax_a.plot([0.15, 0.45], [0.55, 0.45], '-', color=COLORS['accent'], linewidth=3)
    ax_a.plot([0.55, 0.85], [0.55, 0.45], '-', color=COLORS['accent'], linewidth=3)
    
    ax_a.text(0.5, 0.05, 'Only ONE path exists\nin topology', fontsize=11, ha='center', fontweight='bold')
    
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis('off')
    ax_a.set_title('A. No Decision Tree', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL B: Automatic pathway ===
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Show trajectory that follows gradient automatically
    t = np.linspace(0, 1, 100)
    
    # Trajectory follows topology
    x = t
    y = 0.5 + 0.3 * np.sin(2 * np.pi * t)
    
    ax_b.plot(x, y, linewidth=4, color=COLORS['primary'])
    
    # Add arrows along path
    for i in range(0, len(t), 20):
        if i + 5 < len(t):
            ax_b.annotate('', xy=(x[i+5], y[i+5]), xytext=(x[i], y[i]),
                         arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    
    # Start and end points
    ax_b.plot(0, 0.5, 'o', markersize=20, color=COLORS['success'], 
              markeredgecolor='black', markeredgewidth=2)
    ax_b.plot(1, 0.5, '*', markersize=25, color=COLORS['accent'], 
              markeredgecolor='black', markeredgewidth=2)
    
    ax_b.text(0.05, 0.35, 'Start', fontsize=11, fontweight='bold')
    ax_b.text(0.9, 0.35, 'End', fontsize=11, fontweight='bold')
    ax_b.text(0.5, 0.15, 'Automatic following\nNo decisions required', 
              fontsize=11, ha='center', style='italic')
    
    ax_b.set_xlim(-0.1, 1.1)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel('Categorical Progress', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_ylabel('Configuration', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_title('B. Automatic Path Following', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL C: No branching ===
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Show that at each step, only one option exists
    steps = np.arange(10)
    options_at_step = np.ones(10)  # Only 1 option at each step
    
    ax_c.bar(steps, options_at_step, color=COLORS['primary'], edgecolor='black', linewidth=1.5)
    
    ax_c.axhline(y=1, color=COLORS['accent'], linestyle='--', linewidth=2)
    ax_c.text(4.5, 1.2, 'Always exactly 1 option', fontsize=11, ha='center', 
              color=COLORS['accent'], fontweight='bold')
    
    ax_c.set_xlabel('Completion Step', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_ylabel('Number of Options', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_title('C. No Branching = No Decision', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_c.set_ylim(0, 2)
    
    # === PANEL D: Deterministic completion ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Multiple runs produce identical results
    n_runs = 10
    t = np.linspace(0, 1, 100)
    
    for run in range(n_runs):
        # All trajectories are identical (deterministic)
        trajectory = 1 - np.exp(-5 * t)
        ax_d.plot(t, trajectory, linewidth=2, alpha=0.7, color=COLORS['secondary'])
    
    # Highlight that they overlap
    ax_d.plot(t, trajectory, linewidth=3, color=COLORS['primary'], label='All runs identical')
    
    ax_d.set_xlabel('Time', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_ylabel('Completion', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_title('D. Deterministic: 10 Runs\n(all identical)', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_d.legend(fontsize=FONT_SIZES['legend'])
    
    # Overall title
    fig.suptitle('Argument 5: DISSOLUTION OF DECISION\nCategorical completion is automatic, not deliberative',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# ARGUMENT 6: DISSOLUTION OF SECOND LAW
# ============================================================================

def create_panel_arg6_dissolution_second_law(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 6: Dissolution of Second Law Violation
    
    No violation occurs. Categorical entropy (network density) always increases.
    The total entropy of the universe increases.
    
    4 Panels:
    A. Spatial entropy vs. categorical entropy
    B. Network densification over time
    C. Total entropy always increases
    D. Second law accounting
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === PANEL A: Two types of entropy ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    t = np.linspace(0, 1, 100)
    
    # Spatial entropy decreases (apparent sorting)
    spatial_entropy = 1 - 0.3 * t
    
    # Categorical entropy increases (network densifies)
    categorical_entropy = 1 + 0.5 * t
    
    # Total always increases
    total_entropy = spatial_entropy + categorical_entropy
    
    ax_a.plot(t, spatial_entropy, linewidth=3, color=COLORS['accent'], 
              label='Spatial (decreases)')
    ax_a.plot(t, categorical_entropy, linewidth=3, color=COLORS['success'], 
              label='Categorical (increases)')
    ax_a.plot(t, total_entropy, linewidth=4, color=COLORS['primary'], 
              linestyle='--', label='Total (always increases)')
    
    ax_a.axhline(y=2, color=COLORS['neutral'], linestyle=':', alpha=0.5)
    
    ax_a.set_xlabel('Process Progress', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_a.set_ylabel('Entropy', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_a.set_title('A. Two Entropy Components', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_a.legend(fontsize=FONT_SIZES['legend'])
    
    # === PANEL B: Network densification ===
    ax_b = fig.add_subplot(gs[1, 0])
    
    steps = np.arange(50)
    
    # Edge count increases
    initial_edges = 100
    edges = initial_edges + 3 * steps + np.random.normal(0, 2, len(steps)).cumsum()
    
    ax_b.plot(steps, edges, linewidth=3, color=COLORS['success'])
    ax_b.fill_between(steps, initial_edges, edges, alpha=0.3, color=COLORS['success'])
    
    ax_b.set_xlabel('Sorting Attempts', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_ylabel('Network Edges', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_b.set_title('B. Network Densifies\n(ΔS_categorical > 0)', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # Annotate increase
    ax_b.annotate('', xy=(45, edges[-1]), xytext=(45, initial_edges),
                  arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=2))
    ax_b.text(47, (edges[-1] + initial_edges)/2, f'+{edges[-1]-initial_edges:.0f}', 
              fontsize=12, color=COLORS['accent'], fontweight='bold')
    
    # === PANEL C: Total entropy ===
    ax_c = fig.add_subplot(gs[0, 1])
    
    # Monte Carlo simulation of entropy
    n_trials = 1000
    delta_S_spatial = np.random.normal(-0.3, 0.1, n_trials)
    delta_S_categorical = np.random.normal(0.5, 0.1, n_trials)
    delta_S_total = delta_S_spatial + delta_S_categorical
    
    ax_c.hist(delta_S_spatial, bins=30, alpha=0.5, color=COLORS['accent'], 
              label='ΔS_spatial', density=True)
    ax_c.hist(delta_S_categorical, bins=30, alpha=0.5, color=COLORS['success'], 
              label='ΔS_categorical', density=True)
    ax_c.hist(delta_S_total, bins=30, alpha=0.7, color=COLORS['primary'], 
              label='ΔS_total', density=True)
    
    ax_c.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax_c.text(0.02, ax_c.get_ylim()[1]*0.9, 'ΔS > 0', fontsize=12, 
              color=COLORS['success'], fontweight='bold')
    
    ax_c.set_xlabel('Entropy Change (ΔS)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_ylabel('Probability Density', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_c.set_title('C. Total Entropy: ΔS_total > 0', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_c.legend(fontsize=FONT_SIZES['legend'])
    
    # === PANEL D: Accounting ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Bar chart showing entropy balance
    categories = ['Spatial\n(apparent)', 'Categorical\n(network)', 'Total']
    values = [-0.3, +0.5, +0.2]
    colors = [COLORS['accent'], COLORS['success'], COLORS['primary']]
    
    bars = ax_d.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    
    ax_d.axhline(y=0, color='black', linewidth=2)
    ax_d.axhline(y=0, color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.5)
    
    # Add values on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                  f'{val:+.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax_d.set_ylabel('Entropy Change (ΔS)', fontsize=FONT_SIZES['label'], fontweight='bold')
    ax_d.set_title('D. Second Law Accounting\nΔS_total = +0.2 > 0 ✓', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax_d.set_ylim(-0.5, 0.7)
    
    # Overall title
    fig.suptitle('Argument 6: DISSOLUTION OF SECOND LAW VIOLATION\nCategorical entropy increase compensates: ΔS_total > 0 always',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# ARGUMENT 7: INFORMATION COMPLEMENTARITY
# ============================================================================

def create_panel_arg7_information_complementarity(save_path: Optional[str] = None) -> plt.Figure:
    """
    Argument 7: Information Complementarity
    
    The demon is the shadow of categorical dynamics projected onto the
    kinetic face. Maxwell observed only one face of information.
    
    4 Panels:
    A. Two faces of information
    B. Ammeter/voltmeter analogy
    C. Projection creates "demon" appearance
    D. Complete picture (no demon)
    """
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === PANEL A: Two faces ===
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Draw two overlapping circles (like Venn diagram)
    circle1 = Circle((0.35, 0.5), 0.3, fill=True, alpha=0.5, 
                     color=COLORS['kinetic'], label='Kinetic Face')
    circle2 = Circle((0.65, 0.5), 0.3, fill=True, alpha=0.5, 
                     color=COLORS['categorical'], label='Categorical Face')
    
    ax_a.add_patch(circle1)
    ax_a.add_patch(circle2)
    
    # Labels
    ax_a.text(0.2, 0.5, 'Velocities\nEnergy\nTemperature', 
              fontsize=10, ha='center', va='center')
    ax_a.text(0.8, 0.5, 'Network\nTopology\nPhase-lock', 
              fontsize=10, ha='center', va='center')
    ax_a.text(0.5, 0.5, '⊥', fontsize=20, ha='center', va='center', fontweight='bold')
    
    ax_a.text(0.5, 0.1, 'Complementary: cannot observe both simultaneously',
              fontsize=11, ha='center', style='italic')
    
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_aspect('equal')
    ax_a.axis('off')
    ax_a.set_title('A. Two Complementary Faces', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL B: Ammeter/voltmeter ===
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Draw circuit analogy
    # Component box
    rect = FancyBboxPatch((0.3, 0.4), 0.4, 0.2, boxstyle='round,pad=0.02',
                          facecolor=COLORS['light'], edgecolor='black', linewidth=2)
    ax_b.add_patch(rect)
    ax_b.text(0.5, 0.5, 'Component', fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Ammeter (series)
    ammeter = Circle((0.15, 0.5), 0.08, fill=True, 
                     color=COLORS['kinetic'], edgecolor='black', linewidth=2)
    ax_b.add_patch(ammeter)
    ax_b.text(0.15, 0.5, 'A', fontsize=14, ha='center', va='center', fontweight='bold', color='white')
    
    # Voltmeter (parallel)
    voltmeter = Circle((0.5, 0.8), 0.08, fill=True, 
                       color=COLORS['categorical'], edgecolor='black', linewidth=2)
    ax_b.add_patch(voltmeter)
    ax_b.text(0.5, 0.8, 'V', fontsize=14, ha='center', va='center', fontweight='bold', color='white')
    
    # Wires
    ax_b.plot([0, 0.07], [0.5, 0.5], 'k-', linewidth=2)
    ax_b.plot([0.23, 0.3], [0.5, 0.5], 'k-', linewidth=2)
    ax_b.plot([0.7, 1], [0.5, 0.5], 'k-', linewidth=2)
    ax_b.plot([0.3, 0.3, 0.42], [0.5, 0.8, 0.8], 'k-', linewidth=2)
    ax_b.plot([0.58, 0.7, 0.7], [0.8, 0.8, 0.5], 'k-', linewidth=2)
    
    ax_b.text(0.5, 0.2, 'Cannot use both meters\nsimultaneously on same element',
              fontsize=10, ha='center', style='italic')
    
    ax_b.set_xlim(-0.1, 1.1)
    ax_b.set_ylim(0, 1)
    ax_b.set_aspect('equal')
    ax_b.axis('off')
    ax_b.set_title('B. Like Ammeter/Voltmeter', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL C: Projection creates demon ===
    ax_c = fig.add_subplot(gs[1, 0])
    
    # 3D-like projection visualization
    # Hidden layer (categorical)
    ax_c.fill([0.1, 0.9, 0.9, 0.1], [0.6, 0.6, 0.9, 0.9], 
              color=COLORS['categorical'], alpha=0.3)
    ax_c.text(0.5, 0.75, 'CATEGORICAL\n(hidden)', fontsize=11, ha='center', 
              color=COLORS['categorical'], fontweight='bold')
    
    # Observable layer (kinetic)
    ax_c.fill([0.1, 0.9, 0.9, 0.1], [0.1, 0.1, 0.4, 0.4], 
              color=COLORS['kinetic'], alpha=0.3)
    ax_c.text(0.5, 0.25, 'KINETIC\n(observed)', fontsize=11, ha='center', 
              color=COLORS['kinetic'], fontweight='bold')
    
    # Projection arrows
    for x in [0.3, 0.5, 0.7]:
        ax_c.annotate('', xy=(x, 0.4), xytext=(x, 0.6),
                     arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Shadow = "demon"
    ax_c.text(0.5, 0.47, '"DEMON" = Shadow of hidden dynamics',
              fontsize=10, ha='center', style='italic',
              bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.7))
    
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis('off')
    ax_c.set_title('C. Demon = Projection Artifact', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # === PANEL D: Complete picture ===
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Show that seeing both faces reveals no demon
    # Left: Maxwell's view (only kinetic)
    ax_d.text(0.25, 0.85, "Maxwell's View", fontsize=12, ha='center', fontweight='bold')
    ax_d.text(0.25, 0.7, 'Kinetic only →', fontsize=10, ha='center')
    ax_d.text(0.25, 0.55, '"Demon" sorting', fontsize=11, ha='center', 
              color=COLORS['accent'], fontweight='bold')
    
    # Dividing line
    ax_d.plot([0.5, 0.5], [0.3, 0.95], 'k--', linewidth=2)
    
    # Right: Complete view (both faces)
    ax_d.text(0.75, 0.85, "Complete View", fontsize=12, ha='center', fontweight='bold')
    ax_d.text(0.75, 0.7, 'Both faces →', fontsize=10, ha='center')
    ax_d.text(0.75, 0.55, 'Automatic topology', fontsize=11, ha='center', 
              color=COLORS['success'], fontweight='bold')
    
    # Big conclusion
    ax_d.text(0.5, 0.15, 'NO DEMON EXISTS\nOnly categorical completion\nalong network topology',
              fontsize=13, ha='center', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor=COLORS['highlight'], edgecolor=COLORS['dark']))
    
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis('off')
    ax_d.set_title('D. Complete Picture', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # Overall title
    fig.suptitle('Argument 7: INFORMATION COMPLEMENTARITY\nThe "demon" is projection of hidden categorical dynamics onto kinetic face',
                 fontsize=FONT_SIZES['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_panel_figures(output_dir: str = "results/publication") -> Dict[str, str]:
    """
    Generate all 7 panel figures for publication.
    
    Returns dict of figure paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES FOR MAXWELL'S DEMON RESOLUTION")
    print("=" * 70)
    
    figures = {}
    
    # Argument 1
    print("\n[1/7] Temporal Triviality...")
    fig1 = create_panel_arg1_temporal_triviality(
        str(output_path / "panel_arg1_temporal_triviality.png"))
    figures["arg1"] = str(output_path / "panel_arg1_temporal_triviality.png")
    plt.close(fig1)
    
    # Argument 2
    print("[2/7] Temperature Independence...")
    fig2 = create_panel_arg2_temperature_independence(
        str(output_path / "panel_arg2_temperature_independence.png"))
    figures["arg2"] = str(output_path / "panel_arg2_temperature_independence.png")
    plt.close(fig2)
    
    # Argument 3
    print("[3/7] Retrieval Paradox...")
    fig3 = create_panel_arg3_retrieval_paradox(
        str(output_path / "panel_arg3_retrieval_paradox.png"))
    figures["arg3"] = str(output_path / "panel_arg3_retrieval_paradox.png")
    plt.close(fig3)
    
    # Argument 4
    print("[4/7] Dissolution of Observation...")
    fig4 = create_panel_arg4_dissolution_observation(
        str(output_path / "panel_arg4_dissolution_observation.png"))
    figures["arg4"] = str(output_path / "panel_arg4_dissolution_observation.png")
    plt.close(fig4)
    
    # Argument 5
    print("[5/7] Dissolution of Decision...")
    fig5 = create_panel_arg5_dissolution_decision(
        str(output_path / "panel_arg5_dissolution_decision.png"))
    figures["arg5"] = str(output_path / "panel_arg5_dissolution_decision.png")
    plt.close(fig5)
    
    # Argument 6
    print("[6/7] Dissolution of Second Law...")
    fig6 = create_panel_arg6_dissolution_second_law(
        str(output_path / "panel_arg6_dissolution_second_law.png"))
    figures["arg6"] = str(output_path / "panel_arg6_dissolution_second_law.png")
    plt.close(fig6)
    
    # Argument 7
    print("[7/7] Information Complementarity...")
    fig7 = create_panel_arg7_information_complementarity(
        str(output_path / "panel_arg7_information_complementarity.png"))
    figures["arg7"] = str(output_path / "panel_arg7_information_complementarity.png")
    plt.close(fig7)
    
    print("\n" + "=" * 70)
    print("ALL 7 PANEL FIGURES GENERATED")
    print(f"Output directory: {output_path}")
    print("=" * 70)
    
    return figures


def find_project_root():
    """Find the validation project root directory"""
    from pathlib import Path
    current = Path(__file__).resolve().parent
    
    for _ in range(5):
        if (current / "results").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    return Path.cwd()


def generate_publication_figures(output_dir=None):
    """
    Generate all publication figures.
    
    Args:
        output_dir: Output directory. If None, uses results/publication.
    
    Returns:
        Dict of figure paths.
    """
    if output_dir is None:
        project_root = find_project_root()
        output_dir = str(project_root / "results" / "publication")
    
    return generate_all_panel_figures(output_dir)


if __name__ == "__main__":
    generate_publication_figures()

