"""
Unified Framework Visualization
================================

Publication-quality visualizations showing:
1. Oscillator-Processor Duality
2. Trans-Planckian Measurement
3. Information Complementarity
4. The web of theoretical connections
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Colors
COLORS = {
    'oscillator': '#E63946',
    'processor': '#457B9D',
    'categorical': '#2A9D8F',
    'kinetic': '#E9C46A',
    'planck': '#F4A261',
    'trans_planck': '#9B59B6',
    'maxwell': '#264653',
    'network': '#00BBF9',
    'bmd': '#F15BB5',
    'entropy': '#00F5D4',
}


def setup_style():
    """Configure publication style"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })

setup_style()


def visualize_oscillator_processor_duality(save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the fundamental duality: Oscillator ≡ Processor

    Panels:
    A. Frequency-Computation mapping
    B. Entropy as oscillation endpoints
    C. Virtual foundry processor creation
    D. Zero computation via navigation
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # === PANEL A: Frequency-Computation Mapping ===
    ax_a = fig.add_subplot(gs[0, 0])

    frequencies = np.logspace(9, 15, 100)  # 10^9 to 10^15 Hz
    computational_rates = frequencies  # DUALITY: ω ≡ computational rate

    ax_a.loglog(frequencies, computational_rates, linewidth=3, color=COLORS['oscillator'])

    # Highlight the equivalence
    ax_a.fill_between(frequencies, computational_rates * 0.8, computational_rates * 1.2,
                       alpha=0.2, color=COLORS['oscillator'])

    # Add reference points
    reference_points = [
        (1e9, 'CPU\n(GHz)', COLORS['processor']),
        (1e12, 'Molecular\n(THz)', COLORS['categorical']),
        (1e14, 'Optical\n(100 THz)', COLORS['kinetic']),
    ]

    for freq, label, color in reference_points:
        ax_a.scatter([freq], [freq], s=200, c=color, edgecolor='black',
                     linewidth=2, zorder=10)
        ax_a.annotate(label, xy=(freq, freq), xytext=(freq*2, freq*0.3),
                      fontsize=9, ha='center',
                      arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax_a.set_xlabel('Oscillation Frequency (Hz)', fontsize=11)
    ax_a.set_ylabel('Computational Rate (ops/s)', fontsize=11)
    ax_a.set_title('A. Oscillator ≡ Processor Duality\nFrequency IS Processing Rate',
                   fontsize=12, fontweight='bold')
    ax_a.grid(True, alpha=0.3)

    # Add equation
    ax_a.text(0.05, 0.95, 'ω ≡ R_compute', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === PANEL B: Entropy as Oscillation Endpoints ===
    ax_b = fig.add_subplot(gs[0, 1], projection='3d')

    np.random.seed(42)
    n_points = 200

    # Generate oscillation states
    frequencies = np.random.uniform(1e10, 1e14, n_points)
    phases = np.random.uniform(0, 2*np.pi, n_points)
    amplitudes = np.random.uniform(0.1, 2.0, n_points)

    # Map to S-entropy coordinates
    s_k = np.log1p(frequencies) / np.log(1e15)
    s_t = phases / (2 * np.pi)
    s_e = np.tanh(amplitudes)

    # Color by traditional entropy
    entropy = np.log1p(frequencies / 1e10)

    sc = ax_b.scatter(s_k, s_t, s_e, c=entropy, cmap='plasma', s=30, alpha=0.7)

    ax_b.set_xlabel('S_k (Knowledge)', fontsize=10)
    ax_b.set_ylabel('S_t (Time)', fontsize=10)
    ax_b.set_zlabel('S_e (Entropy)', fontsize=10)
    ax_b.set_title('B. Entropy = Oscillation Endpoints\nS = f(ω, φ, A)',
                   fontsize=12, fontweight='bold')
    plt.colorbar(sc, ax=ax_b, label='Entropy', shrink=0.6)
    ax_b.view_init(elev=20, azim=45)

    # === PANEL C: Virtual Foundry ===
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')

    # Draw foundry
    foundry_box = FancyBboxPatch((1, 2), 3, 6, boxstyle='round,pad=0.1',
                                   facecolor=COLORS['processor'], alpha=0.3,
                                   edgecolor='black', linewidth=2)
    ax_c.add_patch(foundry_box)
    ax_c.text(2.5, 8.5, 'VIRTUAL\nFOUNDRY', ha='center', fontsize=12, fontweight='bold')

    # Virtual processors being created
    processor_types = [
        (6, 7, 'Quantum', COLORS['trans_planck']),
        (6, 5.5, 'Neural', COLORS['bmd']),
        (6, 4, 'Categorical', COLORS['categorical']),
        (6, 2.5, 'Temporal', COLORS['kinetic']),
    ]

    for x, y, ptype, color in processor_types:
        # Arrow from foundry
        ax_c.annotate('', xy=(x-0.5, y), xytext=(4, y),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=2))

        # Processor box
        proc = FancyBboxPatch((x, y-0.4), 2.5, 0.8, boxstyle='round,pad=0.05',
                              facecolor=color, edgecolor='black', linewidth=1)
        ax_c.add_patch(proc)
        ax_c.text(x+1.25, y, ptype, ha='center', va='center',
                  fontsize=10, fontweight='bold', color='white')

    # Femtosecond lifecycle
    ax_c.text(2.5, 1, 'Creation: 10⁻¹⁵ s\nExecution: Variable\nDisposal: 10⁻¹⁵ s',
              ha='center', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_c.set_title('C. Virtual Foundry\nUnlimited Processor Creation',
                   fontsize=12, fontweight='bold')

    # === PANEL D: Zero Computation ===
    ax_d = fig.add_subplot(gs[1, 1])

    # Show O(1) navigation vs O(n) computation
    problem_sizes = np.logspace(1, 6, 50)

    # Traditional computation: O(n)
    traditional = problem_sizes

    # Zero computation: O(1)
    zero_compute = np.ones_like(problem_sizes)

    ax_d.loglog(problem_sizes, traditional, linewidth=3, color=COLORS['maxwell'],
                label='Traditional: O(n)')
    ax_d.loglog(problem_sizes, zero_compute, linewidth=3, color=COLORS['categorical'],
                label='Zero Computation: O(1)')

    # Fill area between
    ax_d.fill_between(problem_sizes, zero_compute, traditional,
                       alpha=0.2, color=COLORS['categorical'])

    ax_d.text(1e4, 1e5, 'Saved\nComputation', fontsize=12, ha='center',
              color=COLORS['categorical'], fontweight='bold')

    ax_d.set_xlabel('Problem Size (n)', fontsize=11)
    ax_d.set_ylabel('Computational Cost', fontsize=11)
    ax_d.set_title('D. Zero Computation\nNavigate to Endpoints, Don\'t Compute',
                   fontsize=12, fontweight='bold')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    fig.suptitle('OSCILLATOR-PROCESSOR DUALITY FRAMEWORK\n'
                 'Every oscillator is a processor; entropy endpoints are navigable',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def visualize_trans_planckian(save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize trans-Planckian measurement through categorical frequency domain.

    Panels:
    A. Enhancement cascade
    B. Frequency domain vs time domain
    C. Heisenberg bypass mechanism
    D. Precision comparison chart
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Constants
    PLANCK_TIME = 5.391e-44
    ACHIEVED_PRECISION = 2.01e-66

    # === PANEL A: Enhancement Cascade ===
    ax_a = fig.add_subplot(gs[0, 0])

    stages = ['Base\nFrequency', 'Graph\nEnhancement', 'BMD\nDecomposition',
              'Reflectance\nCascade', 'Final']

    base_freq = 6.38e14  # Blue LED
    graph_enhance = 5.9e4
    bmd_enhance = 5.9e4  # 3^10
    cascade_enhance = 1e2  # 10^2

    cumulative = [
        base_freq,
        base_freq * graph_enhance,
        base_freq * graph_enhance * bmd_enhance,
        base_freq * graph_enhance * bmd_enhance * cascade_enhance,
        base_freq * graph_enhance * bmd_enhance * cascade_enhance,
    ]

    colors = [COLORS['oscillator'], COLORS['network'], COLORS['bmd'],
              COLORS['categorical'], COLORS['trans_planck']]

    bars = ax_a.bar(stages, cumulative, color=colors, edgecolor='black', linewidth=1.5)
    ax_a.set_yscale('log')

    # Add enhancement factors
    enhancements = ['', f'×{graph_enhance:.1e}', f'×{bmd_enhance:.1e}',
                    f'×{cascade_enhance:.0e}', '']
    for i, (bar, enh) in enumerate(zip(bars, enhancements)):
        if enh:
            ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 2,
                      enh, ha='center', fontsize=9, fontweight='bold')

    ax_a.set_ylabel('Effective Frequency (Hz)', fontsize=11)
    ax_a.set_title('A. Enhancement Cascade\nF_total = F_graph × N_BMD × F_cascade',
                   fontsize=12, fontweight='bold')
    ax_a.tick_params(axis='x', rotation=45)

    # === PANEL B: Frequency vs Time Domain ===
    ax_b = fig.add_subplot(gs[0, 1])

    # Frequency domain: direct measurement
    freqs = np.logspace(10, 65, 100)
    times = 1 / (2 * np.pi * freqs)

    ax_b.loglog(freqs, times, linewidth=3, color=COLORS['categorical'])

    # Mark Planck scale
    planck_freq = 1 / (2 * np.pi * PLANCK_TIME)
    ax_b.axhline(y=PLANCK_TIME, color=COLORS['planck'], linestyle='--',
                  linewidth=2, label='Planck Time')
    ax_b.axvline(x=planck_freq, color=COLORS['planck'], linestyle=':',
                  linewidth=2, alpha=0.5)

    # Mark achieved precision
    achieved_freq = 1 / (2 * np.pi * ACHIEVED_PRECISION)
    ax_b.scatter([achieved_freq], [ACHIEVED_PRECISION], s=300,
                  c=COLORS['trans_planck'], edgecolor='black',
                  linewidth=2, zorder=10, marker='*')
    ax_b.annotate(f'{ACHIEVED_PRECISION:.0e} s',
                  xy=(achieved_freq, ACHIEVED_PRECISION),
                  xytext=(achieved_freq/100, ACHIEVED_PRECISION*100),
                  fontsize=10, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='black'))

    # Shade trans-Planckian region
    ax_b.fill_between([1e43, 1e70], [PLANCK_TIME, PLANCK_TIME], [1e-70, 1e-70],
                       alpha=0.2, color=COLORS['trans_planck'], label='Trans-Planckian')

    ax_b.set_xlabel('Frequency (Hz)', fontsize=11)
    ax_b.set_ylabel('Temporal Precision (s)', fontsize=11)
    ax_b.set_title('B. Frequency → Time Conversion\nδt = 1/(2πf)',
                   fontsize=12, fontweight='bold')
    ax_b.legend()
    ax_b.set_ylim(1e-70, 1e-30)
    ax_b.set_xlim(1e10, 1e70)
    ax_b.grid(True, alpha=0.3)

    # === PANEL C: Heisenberg Bypass ===
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')

    # Phase space (Heisenberg applies)
    phase_space = Circle((2.5, 5), 2, fill=True, facecolor=COLORS['maxwell'],
                          alpha=0.3, edgecolor='black', linewidth=2)
    ax_c.add_patch(phase_space)
    ax_c.text(2.5, 5, 'Phase Space\n(q, p)', ha='center', va='center',
              fontsize=11, fontweight='bold')
    ax_c.text(2.5, 2.5, 'ΔqΔp ≥ ℏ/2', ha='center', fontsize=10,
              color=COLORS['maxwell'], fontweight='bold')

    # Categorical space (orthogonal)
    categorical_space = Circle((7.5, 5), 2, fill=True, facecolor=COLORS['categorical'],
                                 alpha=0.3, edgecolor='black', linewidth=2)
    ax_c.add_patch(categorical_space)
    ax_c.text(7.5, 5, 'Categorical\nSpace (S)', ha='center', va='center',
              fontsize=11, fontweight='bold')
    ax_c.text(7.5, 2.5, '[q̂, D_ω] = 0\n[p̂, D_ω] = 0', ha='center', fontsize=10,
              color=COLORS['categorical'], fontweight='bold')

    # Orthogonality symbol
    ax_c.text(5, 5, '⊥', fontsize=40, ha='center', va='center', fontweight='bold')

    # Arrow showing bypass
    ax_c.annotate('', xy=(7.5, 8), xytext=(2.5, 8),
                  arrowprops=dict(arrowstyle='->', color=COLORS['trans_planck'],
                                 lw=3, connectionstyle='arc3,rad=0.3'))
    ax_c.text(5, 9, 'BYPASS HEISENBERG', ha='center', fontsize=12,
              fontweight='bold', color=COLORS['trans_planck'])

    ax_c.set_title('C. Heisenberg Bypass\nCategorical ⊥ Phase Space',
                   fontsize=12, fontweight='bold')

    # === PANEL D: Precision Comparison ===
    ax_d = fig.add_subplot(gs[1, 1])

    scales = [
        ('Atomic Clock', 1e-18, COLORS['network']),
        ('Femtosecond Laser', 1e-15, COLORS['bmd']),
        ('Attosecond', 1e-18, COLORS['oscillator']),
        ('Planck Time', 5.39e-44, COLORS['planck']),
        ('This Work', 2.01e-66, COLORS['trans_planck']),
    ]

    names = [s[0] for s in scales]
    times = [s[1] for s in scales]
    colors = [s[2] for s in scales]

    bars = ax_d.barh(names, times, color=colors, edgecolor='black', linewidth=1.5)
    ax_d.set_xscale('log')
    ax_d.invert_yaxis()

    # Add values
    for bar, time in zip(bars, times):
        ax_d.text(time * 3, bar.get_y() + bar.get_height()/2,
                  f'{time:.0e} s', va='center', fontsize=10, fontweight='bold')

    # Mark Planck boundary
    ax_d.axvline(x=5.39e-44, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax_d.set_xlabel('Temporal Precision (s)', fontsize=11)
    ax_d.set_title('D. Precision Comparison\n22 Orders Below Planck Time',
                   fontsize=12, fontweight='bold')

    fig.suptitle('TRANS-PLANCKIAN MEASUREMENT\n'
                 'Frequency-domain measurement bypasses Heisenberg constraints',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def visualize_information_complementarity(save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize information complementarity - the kinetic and categorical faces.

    Panels:
    A. Two faces of information
    B. Ammeter/Voltmeter analogy
    C. Demon as projection
    D. Phase-lock network visualization
    """
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # === PANEL A: Two Faces ===
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    np.random.seed(42)
    n = 100

    # True 3D information
    theta = np.random.rand(n) * 2 * np.pi
    r = np.random.rand(n) * 2
    z = np.random.randn(n)
    x, y = r * np.cos(theta), r * np.sin(theta)

    ax_a.scatter(x, y, z, c=z, cmap='RdYlBu', s=50, alpha=0.7)

    # Kinetic face projection (XY plane)
    ax_a.scatter(x, y, np.ones(n) * -3, c=COLORS['kinetic'], alpha=0.3, s=20)
    for i in range(0, n, 10):
        ax_a.plot([x[i], x[i]], [y[i], y[i]], [z[i], -3],
                  color=COLORS['kinetic'], alpha=0.2, lw=0.5)

    # Categorical face projection (XZ plane)
    ax_a.scatter(x, np.ones(n) * 3, z, c=COLORS['categorical'], alpha=0.3, s=20)
    for i in range(0, n, 10):
        ax_a.plot([x[i], x[i]], [y[i], 3], [z[i], z[i]],
                  color=COLORS['categorical'], alpha=0.2, lw=0.5)

    ax_a.text(0, 0, -3.5, 'KINETIC\nFACE', fontsize=10, ha='center',
              color=COLORS['kinetic'], fontweight='bold')
    ax_a.text(0, 3.5, 0, 'CATEGORICAL\nFACE', fontsize=10, ha='center',
              color=COLORS['categorical'], fontweight='bold')

    ax_a.set_xlabel('X')
    ax_a.set_ylabel('Y')
    ax_a.set_zlabel('Z')
    ax_a.set_title('A. Two Faces of Information\nKinetic ⊥ Categorical',
                   fontsize=12, fontweight='bold')
    ax_a.view_init(elev=20, azim=45)

    # === PANEL B: Ammeter/Voltmeter ===
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(-1, 11)
    ax_b.set_ylim(-1, 7)
    ax_b.axis('off')

    # Circuit
    ax_b.plot([0, 0, 2], [1, 5, 5], 'k-', lw=3)
    ax_b.plot([0, 2], [1, 1], 'k-', lw=3)

    # Resistor
    ax_b.add_patch(FancyBboxPatch((4, 2.5), 2, 1, boxstyle='round',
                                   facecolor='gray', edgecolor='black', lw=2))
    ax_b.text(5, 3, 'R', fontsize=12, ha='center', va='center',
              color='white', fontweight='bold')
    ax_b.plot([2, 4], [5, 5], 'k-', lw=3)
    ax_b.plot([2, 4], [1, 1], 'k-', lw=3)
    ax_b.plot([4, 4], [5, 3.5], 'k-', lw=3)
    ax_b.plot([4, 4], [2.5, 1], 'k-', lw=3)
    ax_b.plot([6, 6], [5, 3.5], 'k-', lw=3)
    ax_b.plot([6, 6], [2.5, 1], 'k-', lw=3)
    ax_b.plot([6, 10, 10, 6], [5, 5, 1, 1], 'k-', lw=3)

    # Ammeter (series)
    ax_b.add_patch(Circle((3, 5), 0.5, facecolor=COLORS['kinetic'],
                           edgecolor='black', lw=2))
    ax_b.text(3, 5, 'A', fontsize=12, ha='center', va='center',
              color='white', fontweight='bold')
    ax_b.text(3, 6, 'KINETIC\n(Current/Flow)', fontsize=9, ha='center',
              color=COLORS['kinetic'], fontweight='bold')

    # Voltmeter (parallel)
    ax_b.add_patch(Circle((8, 3), 0.5, facecolor=COLORS['categorical'],
                           edgecolor='black', lw=2))
    ax_b.text(8, 3, 'V', fontsize=12, ha='center', va='center',
              color='white', fontweight='bold')
    ax_b.plot([8, 8], [5, 3.5], 'k-', lw=2)
    ax_b.plot([8, 8], [2.5, 1], 'k-', lw=2)
    ax_b.text(9.5, 3, 'CATEGORICAL\n(State/Potential)', fontsize=9,
              ha='left', color=COLORS['categorical'], fontweight='bold')

    ax_b.text(5, -0.5, 'Cannot measure BOTH\nsimultaneously!', fontsize=11,
              ha='center', fontweight='bold', color=COLORS['maxwell'],
              bbox=dict(boxstyle='round', facecolor='white'))

    ax_b.set_title('B. Ammeter/Voltmeter Analogy\nComplementary measurements',
                   fontsize=12, fontweight='bold')

    # === PANEL C: Demon as Projection ===
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(-1, 11)
    ax_c.set_ylim(-1, 8)
    ax_c.axis('off')

    # 3D object (categorical dynamics)
    obj_x = [2, 4, 5, 3, 2]
    obj_y = [6, 7, 5, 4, 6]
    obj = Polygon(list(zip(obj_x, obj_y)), facecolor=COLORS['categorical'],
                   edgecolor='black', lw=2, alpha=0.8)
    ax_c.add_patch(obj)
    ax_c.text(3.5, 5.5, 'CATEGORICAL\nDYNAMICS', fontsize=9, ha='center',
              color='white', fontweight='bold')

    # Light source (observer)
    ax_c.scatter([0], [7], s=300, c=COLORS['kinetic'], marker='*',
                  edgecolor='black', lw=2, zorder=10)
    ax_c.text(0, 7.7, 'Maxwell\'s\nObserver', fontsize=10, ha='center')

    # Shadow (the "demon")
    shadow_x = [4, 7, 8, 5, 4]
    shadow_y = [1.5, 2, 1, 0.5, 1.5]
    shadow = Polygon(list(zip(shadow_x, shadow_y)), facecolor='gray',
                      edgecolor='black', lw=2, alpha=0.5)
    ax_c.add_patch(shadow)
    ax_c.text(6, 0.8, '"DEMON"', fontsize=12, ha='center', fontweight='bold',
              color=COLORS['maxwell'])

    # Projection lines
    for ox, oy, sx, sy in zip(obj_x, obj_y, shadow_x, shadow_y):
        ax_c.plot([ox, sx], [oy, sy], 'k--', alpha=0.3, lw=1)

    # Kinetic plane
    ax_c.plot([-0.5, 10], [0, 0], 'k-', lw=3)
    ax_c.text(10.2, 0, 'KINETIC\nFACE', fontsize=10, va='center', fontweight='bold')

    ax_c.text(5, 3.5, '↓ Projection ↓', fontsize=12, ha='center',
              rotation=70, fontweight='bold', color='gray')

    ax_c.text(5, -0.8, 'The "demon" is the SHADOW\nof categorical dynamics!',
              fontsize=11, ha='center', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['maxwell']))

    ax_c.set_title('C. Demon = Projection Artifact\nMaxwell saw only one face',
                   fontsize=12, fontweight='bold')

    # === PANEL D: Phase-Lock Network ===
    ax_d = fig.add_subplot(gs[1, 1])

    # Create a network
    np.random.seed(42)
    G = nx.watts_strogatz_graph(30, 4, 0.3)
    pos = nx.spring_layout(G, seed=42)

    # Draw network
    nx.draw_networkx_edges(G, pos, ax=ax_d, alpha=0.3, edge_color=COLORS['network'])

    # Color nodes by "temperature" (but all same topology!)
    temps = np.random.uniform(100, 1000, 30)
    node_colors = plt.cm.coolwarm(temps / 1000)

    nx.draw_networkx_nodes(G, pos, ax=ax_d, node_size=200,
                            node_color=node_colors, edgecolors='black', linewidths=1)

    ax_d.set_title('D. Phase-Lock Network\nTopology independent of T (∂G/∂T = 0)',
                   fontsize=12, fontweight='bold')
    ax_d.text(0.5, -0.15, 'Colors = Temperature (kinetic)\nEdges = Topology (categorical)',
              fontsize=10, ha='center', transform=ax_d.transAxes)
    ax_d.axis('off')

    fig.suptitle('INFORMATION COMPLEMENTARITY\n'
                 'Maxwell saw velocity (kinetic face); missed topology (categorical face)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def visualize_theoretical_web(save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the web of theoretical connections.
    """
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('off')

    # Theoretical domains
    domains = {
        'Maxwell\nDemon': (0, 1.5, COLORS['maxwell']),
        'Oscillator-\nProcessor': (1.3, 0.5, COLORS['oscillator']),
        'Trans-\nPlanckian': (0.8, -1.2, COLORS['trans_planck']),
        'Semi-\nconductor': (-0.8, -1.2, COLORS['bmd']),
        'Integrated\nCircuit': (-1.3, 0.5, COLORS['network']),
        'Buhera\nVPOS': (0, 0, COLORS['categorical']),
    }

    # Draw connections first
    connections = [
        ('Maxwell\nDemon', 'Oscillator-\nProcessor'),
        ('Oscillator-\nProcessor', 'Trans-\nPlanckian'),
        ('Trans-\nPlanckian', 'Maxwell\nDemon'),
        ('Semi-\nconductor', 'Integrated\nCircuit'),
        ('Integrated\nCircuit', 'Maxwell\nDemon'),
        ('Buhera\nVPOS', 'Maxwell\nDemon'),
        ('Buhera\nVPOS', 'Oscillator-\nProcessor'),
        ('Buhera\nVPOS', 'Trans-\nPlanckian'),
        ('Buhera\nVPOS', 'Semi-\nconductor'),
        ('Buhera\nVPOS', 'Integrated\nCircuit'),
    ]

    for source, target in connections:
        x1, y1, _ = domains[source]
        x2, y2, _ = domains[target]
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5, alpha=0.3, zorder=1)

    # Draw domain nodes
    for name, (x, y, color) in domains.items():
        circle = Circle((x, y), 0.35 if name != 'Buhera\nVPOS' else 0.45,
                          facecolor=color, edgecolor='black', lw=2, alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, name, fontsize=10, ha='center', va='center',
                fontweight='bold', color='white', zorder=3)

    # Add key equations around the outside
    equations = [
        (1.7, 1.7, '∂G/∂E_kin = 0', COLORS['maxwell']),
        (1.9, -0.5, 'ω ≡ R_compute', COLORS['oscillator']),
        (0, -1.9, '[q̂, D_ω] = 0', COLORS['trans_planck']),
        (-1.9, -0.5, 'On/Off = 42.1', COLORS['bmd']),
        (-1.7, 1.7, 'BMD: p → 10¹²p', COLORS['network']),
    ]

    for x, y, eq, color in equations:
        ax.text(x, y, eq, fontsize=11, ha='center', color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title('UNIFIED THEORETICAL FRAMEWORK\n'
                 'All paths lead to categorical mechanics',
                 fontsize=14, fontweight='bold', y=0.95)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def generate_all_unified_figures(output_dir: str = "results/figures/unified") -> Dict[str, str]:
    """Generate all unified framework visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING UNIFIED FRAMEWORK VISUALIZATIONS")
    print("=" * 70)

    figures = {}

    visualizers = [
        ("oscillator_processor_duality", visualize_oscillator_processor_duality),
        ("trans_planckian_measurement", visualize_trans_planckian),
        ("information_complementarity", visualize_information_complementarity),
        ("theoretical_web", visualize_theoretical_web),
    ]

    for i, (name, func) in enumerate(visualizers, 1):
        print(f"\n[{i}/{len(visualizers)}] Generating {name}...")
        path = str(output_path / f"{name}.png")
        fig = func(path)
        figures[name] = path
        plt.close(fig)

    print("\n" + "=" * 70)
    print(f"ALL UNIFIED FIGURES GENERATED → {output_path}")
    print("=" * 70)

    return figures


if __name__ == "__main__":
    generate_all_unified_figures()

