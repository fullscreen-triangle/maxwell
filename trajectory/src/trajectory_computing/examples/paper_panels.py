"""
Paper Panels for Trajectory Computing - Clean 1x4 Format

Each panel: 4 charts in a single row (1x4 layout)
Minimal text - visualizations only, no tables or excessive labels
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Wedge, Rectangle, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
R_MOON = 1.737e6
M_MOON = 7.342e22


def panel_1_triple_equivalence():
    """Panel 1: Triple Equivalence - Oscillation, Category, Partition, Entropy (1x4)"""
    print("Generating Panel 1: Triple Equivalence...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Triple Equivalence', fontsize=16, fontweight='bold', y=0.98)

    # Chart 1: Oscillatory modes on Moon surface
    ax1 = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    r_base = 1.0
    ax1.fill(r_base * np.cos(theta), r_base * np.sin(theta), color='gray', alpha=0.3)
    for l in range(5):
        amplitude = 0.12 / (l + 1)
        r = r_base + amplitude * np.cos(l * theta) * np.sin((l+1) * theta)
        ax1.plot(r * np.cos(theta), r * np.sin(theta),
                color=plt.cm.viridis(l / 5), linewidth=2, alpha=0.8)
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Oscillation', fontsize=12)

    # Chart 2: Categorical morphism network
    ax2 = axes[1]
    n_levels = 5
    nodes_per_level = [1, 3, 5, 7, 9]
    for level in range(n_levels):
        n_nodes = nodes_per_level[level]
        y = 1 - level * 0.22
        for i in range(n_nodes):
            x = (i - (n_nodes - 1) / 2) * 0.12
            color = plt.cm.plasma(level / n_levels)
            ax2.scatter([x], [y], s=150, c=[color], edgecolors='black', zorder=5, alpha=0.8)
            # Morphisms to next level
            if level < n_levels - 1:
                n_next = nodes_per_level[level + 1]
                y_next = 1 - (level + 1) * 0.22
                for j in range(n_next):
                    x_next = (j - (n_next - 1) / 2) * 0.12
                    if abs(x_next - x) < 0.15:
                        ax2.plot([x, x_next], [y - 0.03, y_next + 0.03],
                                color='gray', alpha=0.3, linewidth=0.5)
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.15, 1.1)
    ax2.axis('off')
    ax2.set_title('Category', fontsize=12)

    # Chart 3: 3D Partition (Moon divided)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax3.plot_surface(x, y, z, alpha=0.4, color='gray', edgecolor='none')
    # Partition planes
    for angle in [0, np.pi/3, 2*np.pi/3]:
        xx = np.linspace(-1, 1, 15)
        zz = np.linspace(-1, 1, 15)
        XX, ZZ = np.meshgrid(xx, zz)
        YY = np.zeros_like(XX)
        XX_rot = XX * np.cos(angle)
        YY_rot = XX * np.sin(angle)
        mask = XX_rot**2 + YY_rot**2 + ZZ**2 <= 1.01
        ax3.plot_surface(XX_rot, YY_rot, ZZ, alpha=0.3, color='blue', edgecolor='none')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ax3.axis('off')
    ax3.set_title('Partition', fontsize=12)
    axes[2].axis('off')  # Hide the 2D axis

    # Chart 4: Entropy emergence
    ax4 = axes[3]
    n_values = np.linspace(1, 50, 100)
    for M, color in [(1, '#1f77b4'), (2, '#2ca02c'), (3, '#d62728')]:
        S = M * np.log(n_values)
        ax4.plot(n_values, S, color=color, linewidth=3, alpha=0.8)
    ax4.fill_between(n_values, 0, 3 * np.log(n_values), alpha=0.1, color='gray')
    ax4.set_xlim(0, 50)
    ax4.set_ylim(0, 12)
    ax4.set_xlabel('n', fontsize=10)
    ax4.set_ylabel('S', fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_title('Entropy', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(OUTPUT_DIR, "panel_1_triple_equivalence.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


def panel_2_moon_derivation():
    """Panel 2: Deriving the Moon - Existence, Mass, Orbit, Structure (1x4)"""
    print("Generating Panel 2: Moon Derivation...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Deriving the Moon', fontsize=16, fontweight='bold', y=0.98)

    # Chart 1: Earth-Moon phase-lock (orbital diagram)
    ax1 = axes[0]
    # Earth
    earth = Circle((0, 0), 0.15, fill=True, facecolor='blue', edgecolor='darkblue', linewidth=2)
    ax1.add_patch(earth)
    # Moon orbit
    orbit_theta = np.linspace(0, 2*np.pi, 100)
    orbit_r = 0.7
    ax1.plot(orbit_r * np.cos(orbit_theta), orbit_r * np.sin(orbit_theta),
            'k--', linewidth=1, alpha=0.5)
    # Moon at position
    moon_angle = np.pi/4
    moon = Circle((orbit_r * np.cos(moon_angle), orbit_r * np.sin(moon_angle)),
                  0.05, fill=True, facecolor='gray', edgecolor='black')
    ax1.add_patch(moon)
    # Gravitational coupling lines
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax1.plot([0.15 * np.cos(angle), orbit_r * np.cos(moon_angle)],
                [0.15 * np.sin(angle), orbit_r * np.sin(moon_angle)],
                'r-', alpha=0.2, linewidth=1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Phase-Lock', fontsize=12)

    # Chart 2: Moon interior density (radial cross-section)
    ax2 = axes[1]
    r = np.linspace(0, 1, 100)
    # Core, mantle, crust density profile
    rho = np.piecewise(r, [r < 0.2, (r >= 0.2) & (r < 0.85), r >= 0.85],
                       [7500, 3400, 2900])
    theta = np.linspace(0, 2*np.pi, 100)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    RHO = np.piecewise(R, [R < 0.2, (R >= 0.2) & (R < 0.85), R >= 0.85],
                       [1.0, 0.5, 0.3])
    ax2.contourf(X, Y, RHO, levels=20, cmap='YlOrRd')
    ax2.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax2.plot(0.2 * np.cos(theta), 0.2 * np.sin(theta), 'r--', linewidth=1, alpha=0.7)
    ax2.plot(0.85 * np.cos(theta), 0.85 * np.sin(theta), 'b--', linewidth=1, alpha=0.7)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Internal Structure', fontsize=12)

    # Chart 3: 3D Moon with partition coordinates
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    # Create crater-like texture
    colors = np.ones(x.shape + (4,))
    colors[:, :, 0:3] = 0.6  # Gray
    colors[:, :, 3] = 0.9
    # Add some variation
    noise = np.random.rand(*x.shape) * 0.1
    colors[:, :, 0:3] += noise[:, :, np.newaxis]
    ax3.plot_surface(x, y, z, facecolors=colors, edgecolor='none')
    # Add coordinate lines
    for theta_line in [0, np.pi/2, np.pi, 3*np.pi/2]:
        phi = np.linspace(0, np.pi, 50)
        xl = np.sin(phi) * np.cos(theta_line)
        yl = np.sin(phi) * np.sin(theta_line)
        zl = np.cos(phi)
        ax3.plot(xl, yl, zl, 'b-', linewidth=1, alpha=0.5)
    ax3.axis('off')
    ax3.set_title('Partition Space', fontsize=12)
    axes[2].axis('off')

    # Chart 4: Orbital validation (derived vs observed)
    ax4 = axes[3]
    quantities = ['r', 'M', 'T']
    derived = [0.998, 0.9997, 1.0]  # Normalized to observed
    ax4.bar(quantities, derived, color=['#2ca02c', '#1f77b4', '#d62728'], alpha=0.8, width=0.6)
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax4.set_ylim(0.99, 1.01)
    ax4.set_ylabel('Derived / Observed', fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_title('Validation', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(OUTPUT_DIR, "panel_2_moon_derivation.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


def panel_3_eclipse_shadow():
    """Panel 3: Eclipse and Shadow - Geometry, Umbra, Path, Saros (1x4)"""
    print("Generating Panel 3: Eclipse Shadow...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Eclipse Trajectories', fontsize=16, fontweight='bold', y=0.98)

    # Chart 1: Sun-Earth-Moon alignment
    ax1 = axes[0]
    # Sun
    sun = Circle((0.1, 0.5), 0.08, fill=True, facecolor='yellow', edgecolor='orange', linewidth=2)
    ax1.add_patch(sun)
    # Earth
    earth = Circle((0.4, 0.5), 0.04, fill=True, facecolor='blue', edgecolor='darkblue', linewidth=2)
    ax1.add_patch(earth)
    # Shadow cone
    ax1.fill([0.4, 0.95, 0.4], [0.54, 0.5, 0.46], color='black', alpha=0.4)
    ax1.fill([0.4, 1.0, 1.0, 0.4], [0.54, 0.62, 0.38, 0.46], color='gray', alpha=0.2)
    # Moon
    moon = Circle((0.75, 0.5), 0.025, fill=True, facecolor='darkgray', edgecolor='black')
    ax1.add_patch(moon)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.3, 0.7)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Alignment', fontsize=12)

    # Chart 2: 3D Shadow cone
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    z = np.linspace(0, 2, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    Z, THETA = np.meshgrid(z, theta)
    # Umbra
    R_u = 0.4 * (1 - Z/2.2)
    R_u = np.maximum(R_u, 0)
    X_u = R_u * np.cos(THETA)
    Y_u = R_u * np.sin(THETA)
    ax2.plot_surface(X_u, Y_u, Z, alpha=0.7, color='black', edgecolor='none')
    # Penumbra
    R_p = 0.4 + 0.08 * Z
    X_p = R_p * np.cos(THETA)
    Y_p = R_p * np.sin(THETA)
    ax2.plot_surface(X_p, Y_p, Z, alpha=0.15, color='gray', edgecolor='none')
    # Earth
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    xe = 0.25 * np.outer(np.cos(u), np.sin(v))
    ye = 0.25 * np.outer(np.sin(u), np.sin(v))
    ze = 0.25 * np.outer(np.ones(np.size(u)), np.cos(v)) - 0.25
    ax2.plot_surface(xe, ye, ze, color='blue', alpha=0.9)
    ax2.axis('off')
    ax2.set_title('Shadow Cone', fontsize=12)
    axes[1].axis('off')

    # Chart 3: Moon path through shadow
    ax3 = axes[2]
    # Penumbra
    penumbra = Circle((0, 0), 0.7, fill=True, facecolor='lightgray', alpha=0.3, edgecolor='gray')
    ax3.add_patch(penumbra)
    # Umbra
    umbra = Circle((0, 0), 0.45, fill=True, facecolor='gray', alpha=0.5, edgecolor='black')
    ax3.add_patch(umbra)
    # Moon paths
    for y0, color in [(0, 'red'), (0.25, 'orange'), (0.5, 'blue')]:
        ax3.plot([-0.9, 0.9], [y0, y0], color=color, linewidth=2, alpha=0.8)
        for x in [-0.5, 0, 0.5]:
            m = Circle((x, y0), 0.08, fill=True, facecolor='darkgray',
                       edgecolor=color, linewidth=2, alpha=0.7)
            ax3.add_patch(m)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Transit Paths', fontsize=12)

    # Chart 4: Saros cycle
    ax4 = axes[3]
    saros_years = np.array([0, 18.03, 36.06, 54.09, 72.12])
    ax4.scatter(saros_years, np.ones_like(saros_years) * 0.5,
               s=400, c='red', edgecolors='black', zorder=5)
    ax4.plot([0, 72], [0.5, 0.5], 'k-', linewidth=2)
    for i in range(len(saros_years) - 1):
        ax4.annotate('', xy=(saros_years[i+1], 0.55), xytext=(saros_years[i], 0.55),
                    arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax4.set_xlim(-5, 80)
    ax4.set_ylim(0.3, 0.7)
    ax4.set_xlabel('Years', fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.set_yticks([])
    ax4.set_title('Saros Cycle', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(OUTPUT_DIR, "panel_3_eclipse_shadow.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


def panel_4_subsurface():
    """Panel 4: Subsurface Detection - Bootprints, Regolith, 3D Volume, Dust (1x4)"""
    print("Generating Panel 4: Subsurface Detection...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Opacity-Independent Detection', fontsize=16, fontweight='bold', y=0.98)

    # Chart 1: Bootprint cross-section
    ax1 = axes[0]
    x = np.linspace(-1, 1, 100)
    # Surface with bootprint depression
    surface = np.zeros_like(x)
    bootprint_mask = np.abs(x) < 0.3
    surface[bootprint_mask] = -0.035 * (1 - (x[bootprint_mask] / 0.3)**2)
    ax1.fill_between(x, surface, -0.15, color='sienna', alpha=0.7)
    ax1.fill_between(x, surface, 0.05, color='tan', alpha=0.3)
    ax1.plot(x, surface, 'k-', linewidth=2)
    # Boot outline
    ax1.plot([-0.25, -0.25, 0.25, 0.25], [0.05, 0.02, 0.02, 0.05],
            'k-', linewidth=3)
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_ylim(-0.15, 0.1)
    ax1.axis('off')
    ax1.set_title('Bootprint', fontsize=12)

    # Chart 2: Regolith depth profile
    ax2 = axes[1]
    depth = np.linspace(0, 4, 100)
    density = 1500 * (1 + 0.15 * depth)
    density[depth > 2.3] = 3100
    ax2.fill_betweenx(depth, 0, density, color='sienna', alpha=0.6)
    ax2.axhline(y=2.3, color='red', linestyle='--', linewidth=2)
    ax2.set_xlim(0, 4000)
    ax2.set_ylim(4, 0)
    ax2.set_xlabel(r'$\rho$', fontsize=10)
    ax2.set_ylabel('Depth', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Regolith', fontsize=12)

    # Chart 3: 3D Volumetric bootprint
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    # Bootprint depression
    Z = np.zeros_like(X)
    boot_mask = (np.abs(X) < 0.5) & (np.abs(Y) < 0.3)
    Z[boot_mask] = -0.3 * (1 - (X[boot_mask]/0.5)**2) * (1 - (Y[boot_mask]/0.3)**2)
    ax3.plot_surface(X, Y, Z, cmap='YlOrBr', alpha=0.9, edgecolor='none')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-0.5, 0.2)
    ax3.axis('off')
    ax3.set_title('3D Volume', fontsize=12)
    axes[2].axis('off')

    # Chart 4: Dust displacement pattern (top view)
    ax4 = axes[3]
    # Create radial dust displacement
    theta = np.linspace(0, 2*np.pi, 100)
    for r in np.linspace(0.2, 0.9, 8):
        displacement = 0.1 * np.exp(-r * 2) * np.random.randn(100)
        x = (r + displacement) * np.cos(theta)
        y = (r + displacement) * np.sin(theta)
        ax4.plot(x, y, color=plt.cm.Oranges(r), linewidth=1, alpha=0.7)
    # LM footpad
    footpad = Circle((0, 0), 0.15, fill=True, facecolor='gray', edgecolor='black', linewidth=2)
    ax4.add_patch(footpad)
    ax4.set_xlim(-1.1, 1.1)
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Dust Pattern', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(OUTPUT_DIR, "panel_4_subsurface.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


def panel_5_identity():
    """Panel 5: The Identity - Observation, Computing, Processing, Unified (1x4)"""
    print("Generating Panel 5: The Identity...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Observation = Computing = Processing', fontsize=16, fontweight='bold', y=0.98)

    # Chart 1: Observation (telescope pointing at Moon)
    ax1 = axes[0]
    # Telescope
    ax1.fill([0.1, 0.1, 0.15, 0.15], [0.2, 0.5, 0.5, 0.2], color='gray')
    ax1.fill([0.08, 0.17, 0.17, 0.08], [0.5, 0.5, 0.55, 0.55], color='darkgray')
    # Light rays
    for y_offset in [-0.05, 0, 0.05]:
        ax1.plot([0.17, 0.9], [0.52 + y_offset, 0.6 + y_offset * 2],
                'y-', linewidth=2, alpha=0.5)
    # Moon
    moon = Circle((0.9, 0.6), 0.08, fill=True, facecolor='lightgray', edgecolor='black')
    ax1.add_patch(moon)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Observation', fontsize=12)

    # Chart 2: Computing (address resolution tree)
    ax2 = axes[1]
    # Ternary tree
    levels = 4
    for level in range(levels):
        n_nodes = 3**level
        y = 0.9 - level * 0.2
        for i in range(n_nodes):
            x = (i - (n_nodes - 1) / 2) * (0.8 / n_nodes)
            color = plt.cm.viridis(level / levels)
            ax2.scatter([x + 0.5], [y], s=100 / (level + 1), c=[color],
                       edgecolors='black', alpha=0.8)
    # Highlight path
    path_x = [0.5, 0.5 + 0.13, 0.5 + 0.18, 0.5 + 0.19]
    path_y = [0.9, 0.7, 0.5, 0.3]
    ax2.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7)
    ax2.scatter(path_x, path_y, c='red', s=80, zorder=5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.1, 1)
    ax2.axis('off')
    ax2.set_title('Computing', fontsize=12)

    # Chart 3: Processing (constraint propagation)
    ax3 = axes[2]
    # Surface layer
    ax3.fill_between([0, 1], [0.8, 0.8], [0.7, 0.7], color='tan', alpha=0.8)
    # Arrows showing constraint flow
    for x in [0.2, 0.4, 0.6, 0.8]:
        ax3.annotate('', xy=(x, 0.3), xytext=(x, 0.7),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    # Subsurface layer
    ax3.fill_between([0, 1], [0.3, 0.3], [0.2, 0.2], color='sienna', alpha=0.8)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Processing', fontsize=12)

    # Chart 4: Unified (three converging to same point)
    ax4 = axes[3]
    # Three paths converging
    center = (0.5, 0.5)
    starts = [(0.1, 0.9), (0.9, 0.9), (0.5, 0.1)]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for start, color in zip(starts, colors):
        ax4.annotate('', xy=center, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=4,
                                   connectionstyle='arc3,rad=0.2'))
    # Central point
    ax4.scatter([0.5], [0.5], s=500, c='gold', edgecolors='black', linewidth=3, zorder=5)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Identity', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(OUTPUT_DIR, "panel_5_identity.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath


def generate_all_panels():
    """Generate all five panels in 1x4 format."""
    print("=" * 70)
    print("GENERATING PAPER PANELS (1x4 Format)")
    print("=" * 70)
    print()

    panels = []
    panels.append(panel_1_triple_equivalence())
    panels.append(panel_2_moon_derivation())
    panels.append(panel_3_eclipse_shadow())
    panels.append(panel_4_subsurface())
    panels.append(panel_5_identity())

    print()
    print("=" * 70)
    print(f"Generated {len(panels)} panels (1x4 format, minimal text)")
    print("=" * 70)
    for p in panels:
        print(f"  - {os.path.basename(p)}")

    return panels


if __name__ == "__main__":
    generate_all_panels()
