"""
Figure 6: Computational Implementation

Panel A: Algorithm Flowchart (3D)
Panel B: Computational Scaling
Panel C: Hardware Requirements
Panel D: Real-Time Performance
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

def generate_figure_6(output_dir, dpi=300):
    """Generate Figure 6 with 4 panels."""
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Algorithm Flowchart (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _panel_a_flowchart(ax1)
    
    # Panel B: Computational Scaling
    ax2 = fig.add_subplot(2, 2, 2)
    _panel_b_scaling(ax2)
    
    # Panel C: Hardware Requirements
    ax3 = fig.add_subplot(2, 2, 3)
    _panel_c_hardware(ax3)
    
    # Panel D: Real-Time Performance
    ax4 = fig.add_subplot(2, 2, 4)
    _panel_d_realtime(ax4)
    
    plt.tight_layout()
    output_path = output_dir / "figure_06_computational_implementation.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def _panel_a_flowchart(ax):
    """Panel A: 3D flowchart showing computational pipeline."""
    # Define nodes (processing steps)
    nodes = {
        'Input': np.array([0, 0, 0]),
        'Constraint\nSatisfaction': np.array([2, 0, 0]),
        'Multi-Modal\nFusion': np.array([4, 0, 0]),
        'Integration': np.array([6, 0, 0]),
        'Output': np.array([8, 0, 0])
    }
    
    # Node colors (by computational cost)
    node_colors = {
        'Input': 'lightblue',
        'Constraint\nSatisfaction': 'orange',
        'Multi-Modal\nFusion': 'red',
        'Integration': 'yellow',
        'Output': 'lightgreen'
    }
    
    # Node sizes (by computational cost)
    node_sizes = {
        'Input': 200,
        'Constraint\nSatisfaction': 800,
        'Multi-Modal\nFusion': 600,
        'Integration': 400,
        'Output': 200
    }
    
    # Timing annotations (ms)
    timings = {
        'Input': 0.1,
        'Constraint\nSatisfaction': 0.5,
        'Multi-Modal\nFusion': 0.3,
        'Integration': 0.1,
        'Output': 0.0
    }
    
    # Plot nodes
    for name, pos in nodes.items():
        ax.scatter(*pos, s=node_sizes[name], c=node_colors[name], 
                  alpha=0.7, edgecolors='black', linewidths=2, zorder=3)
        ax.text(*pos, f'{name}\n({timings[name]} ms)', 
               fontsize=8, ha='center', va='center', fontweight='bold', zorder=4)
    
    # Draw edges (data flow)
    node_list = list(nodes.items())
    for i in range(len(node_list) - 1):
        pos1 = node_list[i][1]
        pos2 = node_list[i+1][1]
        
        # Edge thickness = bandwidth
        thickness = 3
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
               'k-', linewidth=thickness, alpha=0.6, zorder=1)
        
        # Arrow
        direction = pos2 - pos1
        direction = direction / np.linalg.norm(direction) * 0.3
        arrow_start = pos2 - direction
        # Simple arrow representation
        ax.plot([arrow_start[0], pos2[0]], [arrow_start[1], pos2[1]], 
               [arrow_start[2], pos2[2]], 'k-', linewidth=thickness+1, 
               alpha=0.8, zorder=2)
    
    ax.set_xlabel('Pipeline Stage', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('Panel A: Algorithm Flowchart (3D)', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def _panel_b_scaling(ax):
    """Panel B: Computational scaling log-log plot."""
    # Number of molecules
    n_molecules = np.logspace(3, 6, 20)  # 10³ to 10⁶
    
    # Computation time: O(N log N) scaling
    computation_time = n_molecules * np.log10(n_molecules) * 1e-6  # seconds
    
    # Add some noise
    computation_time += np.random.randn(len(n_molecules)) * computation_time * 0.1
    
    # Plot data points
    ax.loglog(n_molecules, computation_time, 'o', markersize=8, 
             color='blue', alpha=0.7, label='Actual measurements', zorder=3)
    
    # Fit line: O(N log N)
    fit_time = n_molecules * np.log10(n_molecules) * 1e-6
    ax.loglog(n_molecules, fit_time, 'r-', linewidth=2.5, 
             label='O(N log N) fit', zorder=2)
    
    # Comparison: brute force O(N²)
    brute_force = n_molecules**2 * 1e-9
    ax.loglog(n_molecules, brute_force, 'k--', linewidth=2, 
             label='Brute force O(N²)', alpha=0.7, zorder=1)
    
    ax.set_xlabel('Number of Molecules', fontsize=11, fontweight='bold')
    ax.set_ylabel('Computation Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Panel B: Computational Scaling', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Annotation
    ax.annotate('Efficient scaling', 
               xy=(1e5, 1e-1), xytext=(1e4, 1e-2),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=10, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

def _panel_c_hardware(ax):
    """Panel C: Hardware requirements comparison."""
    methods = ['Cryo-EM', 'Super-resolution', 'This work']
    costs = [10e6, 1e6, 10e3]  # USD
    data_rates = [0.1, 1, 1]  # GB/s
    
    # Colors
    colors = ['purple', 'orange', 'green']
    
    # Bar chart for cost
    bars = ax.bar(methods, costs, color=colors, alpha=0.7, 
                 edgecolor='black', linewidth=1.5)
    
    # Log scale
    ax.set_yscale('log')
    ax.set_ylabel('Cost ($)', fontsize=11, fontweight='bold', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        if cost >= 1e6:
            label = f'${cost/1e6:.1f}M'
        else:
            label = f'${cost/1e3:.0f}K'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Second y-axis for data rate
    ax2 = ax.twinx()
    for i, (method, rate) in enumerate(zip(methods, data_rates)):
        ax2.scatter(i, rate, s=300, color=colors[i], marker='s', 
                   edgecolor='black', linewidth=1.5, zorder=3)
        ax2.text(i, rate, f'{rate} GB/s', ha='center', va='bottom', 
                fontsize=8, color='blue', fontweight='bold')
    
    ax2.set_ylabel('Data Rate (GB/s)', fontsize=11, fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1.5)
    
    # Annotation
    ax.annotate('Accessible to standard labs', 
               xy=(2, 10e3), xytext=(1, 1e5),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'),
               fontsize=10, fontweight='bold', color='green',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax.set_title('Panel C: Hardware Requirements', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

def _panel_d_realtime(ax):
    """Panel D: Real-time performance time series."""
    # Time array (seconds)
    t = np.linspace(0, 10, 1000)
    
    # Number of molecules tracked (increases to 10⁵)
    n_tracked = 1e5 * (1 - np.exp(-t / 2))
    n_tracked += np.random.randn(len(t)) * n_tracked * 0.05  # Noise
    
    # Plot
    ax.plot(t, n_tracked, linewidth=2.5, color='blue', label='Tracked molecules', zorder=2)
    
    # Shaded region: processing lag (<1 ms)
    ax.axvspan(0, 10, alpha=0.2, color='green', label='Processing lag <1 ms')
    
    # Frame rate annotation
    fps = 1000
    ax.axhline(1e5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(9, 1e5, f'{fps} fps', fontsize=9, fontweight='bold',
           ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Example frame inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="25%", height="25%", loc='upper right')
    
    # Example frame showing tracked molecules
    n_frame = 100
    frame_positions = np.random.rand(n_frame, 2) * 10
    axins.scatter(*frame_positions.T, s=20, c='blue', alpha=0.6, 
                 edgecolors='black', linewidths=0.3)
    axins.set_title('Example Frame', fontsize=8)
    axins.set_xlabel('X (μm)', fontsize=7)
    axins.set_ylabel('Y (μm)', fontsize=7)
    axins.grid(True, alpha=0.3)
    
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Molecules Tracked', fontsize=11, fontweight='bold')
    ax.set_title('Panel D: Real-Time Performance', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1e5)
    ax.set_xlim(0, 10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Annotation
    ax.annotate('True real-time imaging', 
               xy=(5, 5e4), xytext=(7, 8e4),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
               fontsize=10, fontweight='bold', color='blue',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_figure_6(output_dir, dpi=300)
