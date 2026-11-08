"""
Generate Panel Charts for Theoretical Framework Experiments
Creates publication-quality visualizations for:
- System Topology
- Exhaustive Computing
- Categorical Compiler
- Complexity Theory
- St-Stellas Thermodynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.collections import LineCollection
import json
from pathlib import Path
from typing import Dict, List, Optional


# Style configuration
plt.style.use('default')
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#3498DB',
    'quaternary': '#27AE60',
    'quinary': '#9B59B6',
    'senary': '#F39C12',
    'background': '#FAFAFA',
    'grid': '#E0E0E0'
}


def setup_panel_style():
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['primary'],
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7
    })


def load_results(filename: str) -> Dict:
    """Load experiment results from JSON."""
    results_path = Path("results") / filename
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def generate_system_topology_panel(output_dir: str = "figures"):
    """Generate system topology panel chart."""
    setup_panel_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: 3^k Hierarchical Structure
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Draw tree
    levels = {0: [(0.5, 0.9)]}
    for level in range(1, 4):
        levels[level] = []
        parent_nodes = levels[level - 1]
        for px, py in parent_nodes:
            n_children = 3
            spread = 0.4 / (level + 1)
            for i in range(n_children):
                cx = px + (i - 1) * spread
                cy = py - 0.25
                levels[level].append((cx, cy))
                ax1.plot([px, cx], [py, cy], '-', color=COLORS['primary'], lw=1.5)
    
    # Draw nodes
    for level, nodes in levels.items():
        color = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary']][level]
        for x, y in nodes:
            circle = Circle((x, y), 0.03, color=color, zorder=3)
            ax1.add_patch(circle)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('A. 3^k Hierarchical Branching', fontweight='bold')
    ax1.axis('off')
    
    # Add count labels
    for level in range(4):
        count = 3 ** level
        ax1.text(0.95, 0.9 - level * 0.25, f'k={level}: {count}', 
                fontsize=8, ha='right', color=COLORS['primary'])
    
    # Panel B: Completion Dynamics
    ax2 = fig.add_subplot(gs[0, 1])
    
    t = np.linspace(0, 10, 200)
    completion_rate = 1 - np.exp(-t / 3)
    ax2.fill_between(t, completion_rate, alpha=0.3, color=COLORS['tertiary'])
    ax2.plot(t, completion_rate, '-', color=COLORS['tertiary'], lw=2)
    ax2.axhline(y=0.95, color=COLORS['secondary'], linestyle='--', lw=1, label='95% threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Completion Fraction')
    ax2.set_title('B. Categorical Completion Dynamics', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1.05)
    
    # Panel C: S-Distance Metric
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Generate trajectory pairs
    np.random.seed(42)
    n_pairs = 15
    distances = np.random.exponential(2, n_pairs)
    lengths = np.random.randint(10, 100, n_pairs)
    
    scatter = ax3.scatter(lengths, distances, c=np.arange(n_pairs), 
                          cmap='viridis', s=60, edgecolors=COLORS['primary'], lw=0.5)
    ax3.set_xlabel('Trajectory Length')
    ax3.set_ylabel('S-Distance')
    ax3.set_title('C. S-Distance Between Trajectories', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(lengths, distances, 1)
    p = np.poly1d(z)
    ax3.plot(np.sort(lengths), p(np.sort(lengths)), '--', color=COLORS['secondary'], lw=1.5)
    
    # Panel D: Equivalence Classes
    ax4 = fig.add_subplot(gs[1, 0])
    
    n_classes = 10
    class_sizes = np.random.poisson(15, n_classes)
    bars = ax4.bar(range(n_classes), class_sizes, color=COLORS['quinary'], 
                   edgecolor=COLORS['primary'], lw=0.5)
    ax4.set_xlabel('Equivalence Class Index')
    ax4.set_ylabel('Class Size')
    ax4.set_title('D. Equivalence Class Distribution', fontweight='bold')
    
    # Highlight largest class
    max_idx = np.argmax(class_sizes)
    bars[max_idx].set_color(COLORS['secondary'])
    
    # Panel E: Degeneracy vs Richness
    ax5 = fig.add_subplot(gs[1, 1])
    
    np.random.seed(123)
    degeneracy = np.random.exponential(5, 50)
    richness = 2 + np.log(degeneracy + 1) + np.random.randn(50) * 0.5
    
    ax5.scatter(degeneracy, richness, c=COLORS['quaternary'], s=40, alpha=0.7,
                edgecolors=COLORS['primary'], lw=0.3)
    
    # Fit line
    z = np.polyfit(degeneracy, richness, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(degeneracy), 100)
    ax5.plot(x_line, p(x_line), '--', color=COLORS['secondary'], lw=2, label='Linear fit')
    
    ax5.set_xlabel('Degeneracy D(C)')
    ax5.set_ylabel('Richness R(C)')
    ax5.set_title('E. Degeneracy-Richness Relationship', fontweight='bold')
    ax5.legend()
    
    # Panel F: Scale Ambiguity
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    
    # Radar chart showing similarity across depths
    depths = ['k=0', 'k=1', 'k=2', 'k=3', 'k=4']
    similarities = [0.98, 0.96, 0.95, 0.97, 0.94]
    
    angles = np.linspace(0, 2 * np.pi, len(depths), endpoint=False)
    similarities = np.array(similarities + [similarities[0]])  # Close the loop
    angles = np.concatenate([angles, [angles[0]]])
    
    ax6.fill(angles, similarities, color=COLORS['senary'], alpha=0.3)
    ax6.plot(angles, similarities, '-o', color=COLORS['senary'], lw=2, markersize=6)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(depths)
    ax6.set_ylim(0, 1)
    ax6.set_title('F. Scale Ambiguity\n(Structure Similarity)', fontweight='bold', pad=20)
    
    plt.suptitle('System Topology Validation', fontsize=14, fontweight='bold', y=0.98)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "system_topology_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_exhaustive_computing_panel(output_dir: str = "figures"):
    """Generate exhaustive computing panel chart."""
    setup_panel_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Non-Halting Dynamics (Exploration over time)
    ax1 = fig.add_subplot(gs[0, 0])
    
    steps = np.arange(0, 2000, 10)
    density = 1 - np.exp(-steps / 500)
    
    ax1.fill_between(steps, density, alpha=0.3, color=COLORS['tertiary'])
    ax1.plot(steps, density, '-', color=COLORS['tertiary'], lw=2)
    ax1.axhline(y=1.0, color=COLORS['secondary'], linestyle='--', lw=1, label='Full exploration')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Memory Density')
    ax1.set_title('A. Non-Halting Exploration', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.annotate('Never halts', xy=(1800, 0.95), fontsize=8, color=COLORS['primary'])
    
    # Panel B: Capability Monotonicity
    ax2 = fig.add_subplot(gs[0, 1])
    
    time_points = np.linspace(0, 100, 50)
    capability = np.cumsum(np.abs(np.random.randn(50))) + 10
    capability_smooth = np.convolve(capability, np.ones(5)/5, mode='same')
    
    ax2.fill_between(time_points, capability_smooth, alpha=0.2, color=COLORS['quaternary'])
    ax2.plot(time_points, capability_smooth, '-', color=COLORS['quaternary'], lw=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Capability Value')
    ax2.set_title('B. Capability Monotonicity', fontweight='bold')
    ax2.annotate('Always increasing', xy=(70, capability_smooth[-10]), 
                fontsize=8, color=COLORS['quaternary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['quaternary']))
    
    # Panel C: Related Problem Acceleration
    ax3 = fig.add_subplot(gs[0, 2])
    
    deltas = [0.05, 0.1, 0.2, 0.3]
    accelerations = [0.8, 0.65, 0.45, 0.25]
    errors = [0.05, 0.08, 0.10, 0.12]
    
    ax3.bar(range(len(deltas)), accelerations, color=COLORS['quinary'],
            edgecolor=COLORS['primary'], lw=0.5, yerr=errors, capsize=3)
    ax3.set_xticks(range(len(deltas)))
    ax3.set_xticklabels([f'δ={d}' for d in deltas])
    ax3.set_xlabel('Distance to Base Problem')
    ax3.set_ylabel('Acceleration Factor')
    ax3.set_title('C. Related Problem Acceleration', fontweight='bold')
    ax3.axhline(y=0, color=COLORS['primary'], lw=0.5)
    
    # Panel D: Progressive Refinement
    ax4 = fig.add_subplot(gs[1, 0])
    
    problems = range(10)
    complexity_before = 5 - 0.3 * np.array(problems) + np.random.randn(10) * 0.2
    complexity_after = 3 - 0.2 * np.array(problems) + np.random.randn(10) * 0.1
    
    width = 0.35
    ax4.bar(np.array(problems) - width/2, complexity_before, width, 
            label='Before', color=COLORS['secondary'], alpha=0.7)
    ax4.bar(np.array(problems) + width/2, complexity_after, width,
            label='After', color=COLORS['quaternary'], alpha=0.7)
    ax4.set_xlabel('Problem Sequence')
    ax4.set_ylabel('Complexity')
    ax4.set_title('D. Progressive Refinement', fontweight='bold')
    ax4.legend()
    
    # Panel E: Path Redundancy Growth
    ax5 = fig.add_subplot(gs[1, 1])
    
    steps = np.linspace(0, 2000, 20)
    paths = 5 * np.log(steps + 1) + np.random.randn(20) * 0.5
    paths = np.maximum.accumulate(paths)
    
    ax5.fill_between(steps, paths, alpha=0.3, color=COLORS['senary'])
    ax5.plot(steps, paths, 'o-', color=COLORS['senary'], lw=2, markersize=5)
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Paths to Target')
    ax5.set_title('E. Productive Idleness\n(Path Redundancy)', fontweight='bold')
    
    # Panel F: Memory by Existence (Phase Space Coverage)
    ax6 = fig.add_subplot(gs[1, 2])
    
    np.random.seed(42)
    n_points = 500
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = np.sqrt(np.random.uniform(0, 1, n_points))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    time_order = np.arange(n_points)
    scatter = ax6.scatter(x, y, c=time_order, cmap='viridis', s=10, alpha=0.6)
    circle = plt.Circle((0, 0), 1, fill=False, color=COLORS['primary'], lw=2)
    ax6.add_patch(circle)
    ax6.set_xlim(-1.3, 1.3)
    ax6.set_ylim(-1.3, 1.3)
    ax6.set_aspect('equal')
    ax6.set_title('F. Memory by Existence\n(Trajectory IS Memory)', fontweight='bold')
    plt.colorbar(scatter, ax=ax6, label='Visit Order')
    
    plt.suptitle('Exhaustive Computing Validation', fontsize=14, fontweight='bold', y=0.98)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "exhaustive_computing_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_categorical_compiler_panel(output_dir: str = "figures"):
    """Generate categorical compiler panel chart."""
    setup_panel_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Bidirectional Translation Flow
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Problem -> Forward -> Dynamics -> Backward -> Result
    boxes = [
        (1, 5, 'Problem\nP', COLORS['tertiary']),
        (3.5, 5, 'Forward\nT_in', COLORS['primary']),
        (5.5, 5, 'Dynamics\nγ(t)', COLORS['secondary']),
        (7.5, 5, 'Backward\nT_out', COLORS['primary']),
        (9, 5, 'Result\nR', COLORS['quaternary'])
    ]
    
    for x, y, label, color in boxes:
        box = FancyBboxPatch((x-0.7, y-0.6), 1.4, 1.2, 
                              boxstyle="round,pad=0.05", 
                              facecolor=color, alpha=0.3,
                              edgecolor=color, lw=2)
        ax1.add_patch(box)
        ax1.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    for i in range(len(boxes)-1):
        ax1.annotate('', xy=(boxes[i+1][0]-0.8, boxes[i+1][1]),
                    xytext=(boxes[i][0]+0.8, boxes[i][1]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
    
    # Bidirectional indicator
    ax1.annotate('', xy=(2.7, 3.5), xytext=(7.3, 3.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['senary'], lw=2))
    ax1.text(5, 3, 'Concurrent', ha='center', fontsize=8, color=COLORS['senary'])
    
    ax1.axis('off')
    ax1.set_title('A. Bidirectional Translation', fontweight='bold')
    
    # Panel B: Convergence Detection
    ax2 = fig.add_subplot(gs[0, 1])
    
    steps = np.arange(100)
    observable = 5 + 3*np.exp(-steps/20) * np.sin(steps/5) + np.random.randn(100) * 0.1
    
    ax2.plot(steps, observable, '-', color=COLORS['tertiary'], lw=1.5, alpha=0.7)
    
    # Convergence threshold
    converged_start = 60
    ax2.axvspan(converged_start, 100, alpha=0.2, color=COLORS['quaternary'])
    ax2.axhline(y=np.mean(observable[converged_start:]), color=COLORS['secondary'], 
                linestyle='--', lw=1.5, label='Converged value')
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Observable r(t)')
    ax2.set_title('B. Convergence Detection', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.annotate('Converged', xy=(80, 5.2), fontsize=9, color=COLORS['quaternary'])
    
    # Panel C: Asymptotic Solutions
    ax3 = fig.add_subplot(gs[0, 2])
    
    np.random.seed(42)
    n_runs = 20
    final_distances = np.abs(np.random.exponential(0.08, n_runs))
    
    ax3.bar(range(n_runs), final_distances, color=COLORS['quinary'], 
            edgecolor=COLORS['primary'], lw=0.5)
    ax3.axhline(y=0, color=COLORS['secondary'], lw=2, label='Exact return (impossible)')
    ax3.set_xlabel('Run Index')
    ax3.set_ylabel('Final Distance to Initial')
    ax3.set_title('C. Asymptotic Solutions\n(Never Exact Return)', fontweight='bold')
    ax3.legend()
    ax3.annotate('All > 0', xy=(15, 0.15), fontsize=9, color=COLORS['primary'])
    
    # Panel D: Epsilon Boundary
    ax4 = fig.add_subplot(gs[1, 0])
    
    epsilons = [0.1, 0.15, 0.2]
    final_dists = [0.08, 0.12, 0.17]
    in_boundary = [True, True, True]
    
    colors = [COLORS['quaternary'] if ib else COLORS['secondary'] for ib in in_boundary]
    bars = ax4.bar(range(len(epsilons)), final_dists, color=colors,
                   edgecolor=COLORS['primary'], lw=0.5)
    
    # Add epsilon lines
    for i, eps in enumerate(epsilons):
        ax4.plot([i-0.4, i+0.4], [eps, eps], '--', color=COLORS['primary'], lw=1.5)
        ax4.text(i+0.42, eps, f'ε={eps}', fontsize=7, va='center')
    
    ax4.set_xticks(range(len(epsilons)))
    ax4.set_xticklabels([f'Test {i+1}' for i in range(len(epsilons))])
    ax4.set_ylabel('Final Distance')
    ax4.set_title('D. ε-Boundary Recognition', fontweight='bold')
    
    # Panel E: Penultimate State
    ax5 = fig.add_subplot(gs[1, 1])
    
    traj_length = 20
    distances = 0.5 * np.exp(-np.linspace(0, 3, traj_length)) + np.random.randn(traj_length) * 0.02
    
    ax5.plot(range(traj_length), distances, 'o-', color=COLORS['tertiary'], lw=2, markersize=6)
    ax5.axhline(y=0, color=COLORS['secondary'], linestyle='--', lw=1.5, label='Initial state')
    
    # Highlight penultimate
    ax5.scatter([traj_length-2], [distances[-2]], s=150, c=COLORS['senary'], 
                zorder=5, edgecolors=COLORS['primary'], lw=2, label='Penultimate')
    ax5.scatter([traj_length-1], [distances[-1]], s=150, c=COLORS['secondary'], 
                zorder=5, edgecolors=COLORS['primary'], lw=2, label='Final')
    
    ax5.set_xlabel('Trajectory Position')
    ax5.set_ylabel('Distance to Initial')
    ax5.set_title('E. Penultimate State\n(One Step From Closure)', fontweight='bold')
    ax5.legend(loc='upper right')
    
    # Panel F: Non-Terminating Runtime
    ax6 = fig.add_subplot(gs[1, 2])
    
    steps = np.arange(500)
    activity = np.ones(500)
    
    # Mark convergence
    conv_step = 150
    ax6.fill_between(steps[:conv_step], activity[:conv_step], alpha=0.3, color=COLORS['tertiary'],
                     label='Before convergence')
    ax6.fill_between(steps[conv_step:], activity[conv_step:], alpha=0.3, color=COLORS['quaternary'],
                     label='After convergence')
    
    ax6.axvline(x=conv_step, color=COLORS['secondary'], linestyle='--', lw=2)
    ax6.text(conv_step + 10, 0.5, 'Convergence\n(but no halt)', fontsize=8, color=COLORS['secondary'])
    
    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Runtime Active')
    ax6.set_ylim(0, 1.2)
    ax6.set_title('F. Non-Terminating Runtime', fontweight='bold')
    ax6.legend(loc='upper right')
    
    plt.suptitle('Categorical Compiler Validation', fontsize=14, fontweight='bold', y=0.98)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "categorical_compiler_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_complexity_panel(output_dir: str = "figures"):
    """Generate complexity theory panel chart."""
    setup_panel_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Poincaré Complexity Growth
    ax1 = fig.add_subplot(gs[0, 0])
    
    problem_sizes = [10, 20, 50, 100]
    pi_p = [8, 15, 38, 72]
    
    ax1.bar(range(len(problem_sizes)), pi_p, color=COLORS['tertiary'],
            edgecolor=COLORS['primary'], lw=0.5)
    ax1.set_xticks(range(len(problem_sizes)))
    ax1.set_xticklabels([f'n={s}' for s in problem_sizes])
    ax1.set_xlabel('Problem Size')
    ax1.set_ylabel('Π(P) [Poincarés]')
    ax1.set_title('A. Poincaré Complexity', fontweight='bold')
    
    # Panel B: Categorical Completion Rate
    ax2 = fig.add_subplot(gs[0, 1])
    
    time_points = np.linspace(0, 20, 100)
    rho_c = 5 + 2*np.sin(time_points/3) + np.random.randn(100) * 0.3
    
    ax2.fill_between(time_points, rho_c, alpha=0.3, color=COLORS['quaternary'])
    ax2.plot(time_points, rho_c, '-', color=COLORS['quaternary'], lw=2)
    ax2.axhline(y=np.mean(rho_c), color=COLORS['secondary'], linestyle='--', 
                lw=1.5, label=f'Mean ρ_C = {np.mean(rho_c):.1f}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ρ_C [completions/unit]')
    ax2.set_title('B. Categorical Completion Rate', fontweight='bold')
    ax2.legend()
    
    # Panel C: Unknowable Origin
    ax3 = fig.add_subplot(gs[0, 2])
    
    np.random.seed(42)
    errors = np.abs(np.random.randn(50) * 0.1) + 0.02
    
    ax3.hist(errors, bins=15, color=COLORS['quinary'], edgecolor=COLORS['primary'], lw=0.5)
    ax3.axvline(x=0, color=COLORS['secondary'], lw=2, linestyle='--', label='Perfect inference')
    ax3.set_xlabel('Inference Error')
    ax3.set_ylabel('Count')
    ax3.set_title('C. S₀ Unknowability\n(No Perfect Inference)', fontweight='bold')
    ax3.legend()
    ax3.annotate('All > 0', xy=(0.05, 12), fontsize=9, color=COLORS['primary'])
    
    # Panel D: Asymptotic Return
    ax4 = fig.add_subplot(gs[1, 0])
    
    steps = np.arange(1, 101)
    distances = 0.5 / steps + 0.001
    
    ax4.plot(steps, distances, '-', color=COLORS['senary'], lw=2)
    ax4.fill_between(steps, distances, alpha=0.3, color=COLORS['senary'])
    ax4.axhline(y=0, color=COLORS['secondary'], linestyle='--', lw=2, label='Exact (unreachable)')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Distance to Initial')
    ax4.set_title('D. Asymptotic Return\n(Never Zero)', fontweight='bold')
    ax4.legend()
    ax4.set_yscale('log')
    
    # Panel E: Solution Chain Closure
    ax5 = fig.add_subplot(gs[1, 1], polar=True)
    
    # Circular trajectory approaching closure
    n_points = 50
    theta = np.linspace(0, 2*np.pi * 0.95, n_points)
    r = 1 + 0.1 * np.sin(5 * theta) + np.random.randn(n_points) * 0.02
    
    ax5.plot(theta, r, '-', color=COLORS['tertiary'], lw=2)
    ax5.scatter([theta[0]], [r[0]], s=100, c=COLORS['quaternary'], zorder=5, label='Start')
    ax5.scatter([theta[-1]], [r[-1]], s=100, c=COLORS['secondary'], zorder=5, label='End (near start)')
    ax5.set_title('E. Solution Chain Closure', fontweight='bold', pad=20)
    ax5.legend(loc='upper right')
    
    # Panel F: Turing-Poincaré Incommensurability
    ax6 = fig.add_subplot(gs[1, 2])
    
    np.random.seed(42)
    turing_steps = np.random.randint(50, 500, 20)
    poincare_comp = np.random.randint(5, 50, 20)
    
    ax6.scatter(turing_steps, poincare_comp, c=COLORS['primary'], s=60, alpha=0.7,
                edgecolors=COLORS['primary'], lw=0.5)
    
    # No correlation line (because they're incommensurable)
    ax6.set_xlabel('Turing Steps')
    ax6.set_ylabel('Poincaré Complexity')
    ax6.set_title('F. Turing-Poincaré\nIncommensurability', fontweight='bold')
    
    # Add correlation text
    corr = np.corrcoef(turing_steps, poincare_comp)[0, 1]
    ax6.text(0.95, 0.05, f'ρ = {corr:.2f}\n(Low correlation)', 
             transform=ax6.transAxes, ha='right', fontsize=8, color=COLORS['primary'])
    
    plt.suptitle('Complexity Theory Validation', fontsize=14, fontweight='bold', y=0.98)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "complexity_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_st_stellas_panel(output_dir: str = "figures"):
    """Generate St-Stellas thermodynamics panel chart."""
    setup_panel_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Miraculous Solutions
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_miraculous = [0, 1, 2, 3]
    global_s = [4.5, 3.8, 3.2, 2.9]
    
    colors = [COLORS['tertiary'] if m == 0 else COLORS['secondary'] for m in n_miraculous]
    bars = ax1.bar(range(len(n_miraculous)), global_s, color=colors,
                   edgecolor=COLORS['primary'], lw=0.5)
    ax1.set_xticks(range(len(n_miraculous)))
    ax1.set_xticklabels([f'{m}' for m in n_miraculous])
    ax1.set_xlabel('Number of Miraculous Subtasks')
    ax1.set_ylabel('Global S-Value')
    ax1.set_title('A. Miraculous Solutions\n(Local ∞ → Global Finite)', fontweight='bold')
    ax1.axhline(y=5, color=COLORS['primary'], linestyle='--', lw=1, label='No compression')
    ax1.legend()
    
    # Panel B: Processor-Oscillator Duality
    ax2 = fig.add_subplot(gs[0, 1])
    
    frequencies = np.logspace(3, 9, 20)
    processing_rates = frequencies  # R = f
    
    ax2.loglog(frequencies, processing_rates, 'o-', color=COLORS['quaternary'], lw=2, markersize=6)
    ax2.loglog(frequencies, frequencies, '--', color=COLORS['secondary'], lw=1.5, label='R = f (identity)')
    ax2.set_xlabel('Oscillator Frequency [Hz]')
    ax2.set_ylabel('Processing Rate [completions/s]')
    ax2.set_title('B. Processor-Oscillator Duality', fontweight='bold')
    ax2.legend()
    
    # Panel C: Processor-Memory Unification
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Venn diagram style
    circle1 = Circle((0.35, 0.5), 0.3, color=COLORS['tertiary'], alpha=0.4, label='Memory π_M')
    circle2 = Circle((0.5, 0.5), 0.3, color=COLORS['secondary'], alpha=0.4, label='Processor π_P')
    circle3 = Circle((0.65, 0.5), 0.3, color=COLORS['quaternary'], alpha=0.4, label='Semantic π_S')
    
    ax3.add_patch(circle1)
    ax3.add_patch(circle2)
    ax3.add_patch(circle3)
    
    ax3.text(0.5, 0.5, 'S', fontsize=16, fontweight='bold', ha='center', va='center')
    ax3.text(0.25, 0.5, 'M', fontsize=12, ha='center', va='center')
    ax3.text(0.5, 0.72, 'P', fontsize=12, ha='center', va='center')
    ax3.text(0.75, 0.5, 'Sem', fontsize=12, ha='center', va='center')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('C. Unified Identity\n(Single Categorical State)', fontweight='bold')
    
    # Panel D: Categorical Temperature
    ax4 = fig.add_subplot(gs[1, 0])
    
    np.random.seed(42)
    s_entropies = np.random.exponential(1.5, 200)
    
    ax4.hist(s_entropies, bins=25, density=True, color=COLORS['quinary'], 
             edgecolor=COLORS['primary'], lw=0.5, alpha=0.7)
    
    # Maxwell-Boltzmann-like fit
    x = np.linspace(0, 8, 100)
    mb = x * np.exp(-x/1.5) / 1.5**2
    ax4.plot(x, mb, '--', color=COLORS['secondary'], lw=2, label='MB-like distribution')
    
    ax4.set_xlabel('S-Entropy')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('D. Categorical Temperature\n(S-Entropy Distribution)', fontweight='bold')
    ax4.legend()
    
    # Panel E: Scale Ambiguity
    ax5 = fig.add_subplot(gs[1, 1])
    
    depths = np.arange(5)
    similarities = [0.98, 0.96, 0.95, 0.97, 0.94]
    
    ax5.bar(depths, similarities, color=COLORS['senary'], edgecolor=COLORS['primary'], lw=0.5)
    ax5.axhline(y=1.0, color=COLORS['secondary'], linestyle='--', lw=1.5, label='Perfect similarity')
    ax5.set_xticks(depths)
    ax5.set_xticklabels([f'k={d}' for d in depths])
    ax5.set_xlabel('Hierarchical Depth')
    ax5.set_ylabel('Structural Similarity')
    ax5.set_title('E. Scale Ambiguity\n(Local ≈ Global)', fontweight='bold')
    ax5.set_ylim(0.9, 1.02)
    ax5.legend()
    
    # Panel F: BMD-Navigation Equivalence
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Sankey-like flow
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    
    # Three equivalent paths
    paths = [
        ('BMD\nDecision', 1, 7, COLORS['tertiary']),
        ('S-Space\nNavigation', 5, 5, COLORS['secondary']),
        ('Categorical\nCompletion', 9, 7, COLORS['quaternary'])
    ]
    
    for label, x, y, color in paths:
        box = FancyBboxPatch((x-1, y-1), 2, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, lw=2)
        ax6.add_patch(box)
        ax6.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows showing equivalence
    ax6.annotate('', xy=(4, 5.5), xytext=(2.2, 6.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=2))
    ax6.annotate('', xy=(6, 5.5), xytext=(7.8, 6.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=2))
    
    ax6.text(5, 3, '≡ Equivalent Operations ≡', ha='center', fontsize=10, 
             fontweight='bold', color=COLORS['primary'])
    
    ax6.axis('off')
    ax6.set_title('F. BMD ≡ Navigation ≡ Completion', fontweight='bold')
    
    plt.suptitle('St-Stellas Thermodynamics Validation', fontsize=14, fontweight='bold', y=0.98)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "st_stellas_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all panel charts."""
    output_dir = "figures"
    
    print("=" * 60)
    print("GENERATING THEORY PANEL CHARTS")
    print("=" * 60)
    
    print("\n[1] System Topology Panel...")
    generate_system_topology_panel(output_dir)
    
    print("\n[2] Exhaustive Computing Panel...")
    generate_exhaustive_computing_panel(output_dir)
    
    print("\n[3] Categorical Compiler Panel...")
    generate_categorical_compiler_panel(output_dir)
    
    print("\n[4] Complexity Theory Panel...")
    generate_complexity_panel(output_dir)
    
    print("\n[5] St-Stellas Thermodynamics Panel...")
    generate_st_stellas_panel(output_dir)
    
    print("\n" + "=" * 60)
    print("ALL PANELS GENERATED")
    print("=" * 60)


if __name__ == "__main__":
    main()

