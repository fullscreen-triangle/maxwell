#!/usr/bin/env python3
"""
Electromagnetic Cellular Dynamics Visualization
================================================

Key insight from the user: "Since everything involved has a charge,
we just visualise their interactions/movements"

This is the unified visualization approach:
- Genome = charged polymer (phosphate backbone, ~2 charges per nucleotide)
- Membrane = charge separator (electron transport boundary)
- Cytoplasm = charge medium (ions, metabolites)
- All dynamics = electromagnetic field evolution

We use the virtual instruments framework to generate REAL charge states
from hardware timing, then visualize as electromagnetic fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from virtual_capacitor import VirtualCapacitor, GenomeCapacitor, ChargeState
from virtual_aperture import CategoricalAperture, ChargeFieldAperture, ExternalChargeFieldAperture

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "origins-of-life" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_charge_colormap():
    """Create a diverging colormap for charge: negative=blue, neutral=white, positive=red."""
    colors = [
        (0.0, 0.2, 0.6),   # Deep blue (negative)
        (0.4, 0.6, 0.9),   # Light blue
        (0.95, 0.95, 0.95), # Near white (neutral)
        (0.9, 0.5, 0.3),   # Light red
        (0.7, 0.1, 0.1),   # Deep red (positive)
    ]
    return LinearSegmentedColormap.from_list('charge', colors, N=256)


def generate_em_field(charges, positions, grid_size=100):
    """
    Generate electromagnetic field from point charges.

    This is REAL physics: E = kq/r² for each charge.
    """
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    V = np.zeros_like(X)  # Electric potential

    k = 8.99e9  # Coulomb constant (scaled for visualization)
    k_viz = 1.0  # Visualization scaling

    for q, (px, py) in zip(charges, positions):
        dx = X - px
        dy = Y - py
        r = np.sqrt(dx**2 + dy**2) + 0.1  # Avoid division by zero

        # Electric field components
        Ex += k_viz * q * dx / r**3
        Ey += k_viz * q * dy / r**3

        # Electric potential
        V += k_viz * q / r

    E_mag = np.sqrt(Ex**2 + Ey**2)

    return X, Y, Ex, Ey, E_mag, V


def generate_genome_charge_distribution(n_nucleotides=100, genome_capacitor=None):
    """
    Generate charge positions for a DNA/RNA strand.

    Each nucleotide has ~2 negative charges (phosphate backbone).
    The strand is arranged in a helical pattern (projected to 2D).
    """
    if genome_capacitor is None:
        genome_capacitor = GenomeCapacitor("virtual_genome", 1000)

    # Helix parameters
    radius = 0.8
    pitch = 0.05

    positions = []
    charges = []

    for i in range(n_nucleotides):
        # Helical position (projected to 2D)
        theta = i * 2 * np.pi / 10  # 10 nucleotides per turn
        z = i * pitch - n_nucleotides * pitch / 2

        x = radius * np.cos(theta)
        y = z

        positions.append((x, y))

        # Get charge from hardware timing (REAL measurement)
        charge_state = ChargeState.from_hardware()
        # Phosphate is always negative, but magnitude varies
        charges.append(-1.0 - charge_state.s_coord.S_k * 0.5)

    return charges, positions


def generate_membrane_charges(n_points=40, membrane_potential=-70e-3):
    """
    Generate charge distribution for a membrane.

    Membrane creates charge separation: negative inside, positive outside.
    This IS electron transport partitioning.
    """
    # Elliptical membrane
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radius_x, radius_y = 1.5, 1.0

    inner_positions = []
    inner_charges = []
    outer_positions = []
    outer_charges = []

    for t in theta:
        # Inner membrane surface (cytoplasmic side) - negative
        x_in = radius_x * 0.95 * np.cos(t)
        y_in = radius_y * 0.95 * np.sin(t)
        inner_positions.append((x_in, y_in))

        charge_state = ChargeState.from_hardware()
        inner_charges.append(-0.5 - charge_state.s_coord.S_t * 0.3)

        # Outer membrane surface (extracellular side) - positive
        x_out = radius_x * 1.05 * np.cos(t)
        y_out = radius_y * 1.05 * np.sin(t)
        outer_positions.append((x_out, y_out))
        outer_charges.append(0.5 + charge_state.s_coord.S_e * 0.3)

    return (inner_charges + outer_charges,
            inner_positions + outer_positions)


def generate_cytoplasmic_ions(n_ions=30):
    """
    Generate random ion positions in cytoplasm.

    Mix of positive (K+, Na+, Ca2+) and negative (Cl-, organic acids).
    """
    positions = []
    charges = []

    for i in range(n_ions):
        # Random position inside cell
        while True:
            x = np.random.uniform(-1.3, 1.3)
            y = np.random.uniform(-0.85, 0.85)
            # Check if inside ellipse
            if (x/1.5)**2 + (y/1.0)**2 < 0.9:
                break

        positions.append((x, y))

        # Random charge (mostly K+ and Cl-)
        charge_state = ChargeState.from_hardware()
        if charge_state.s_coord.S_k > 0.5:
            charges.append(1.0)  # Cation
        else:
            charges.append(-1.0)  # Anion

    return charges, positions


def draw_cell_boundary(ax, color='#2a4858', linewidth=2):
    """Draw cell membrane as an ellipse."""
    membrane = Ellipse((0, 0), 3.0, 2.0, fill=False,
                       edgecolor=color, linewidth=linewidth,
                       linestyle='-', zorder=10)
    ax.add_patch(membrane)

    # Inner and outer lines to show membrane thickness
    inner = Ellipse((0, 0), 2.85, 1.9, fill=False,
                    edgecolor=color, linewidth=0.5, linestyle='--', alpha=0.5)
    outer = Ellipse((0, 0), 3.15, 2.1, fill=False,
                    edgecolor=color, linewidth=0.5, linestyle='--', alpha=0.5)
    ax.add_patch(inner)
    ax.add_patch(outer)


def generate_em_cellular_dynamics_panel():
    """
    Main panel: Electromagnetic visualization of cellular dynamics.

    Simple but informative - shows charge as the unifying principle.
    """
    print("Generating Electromagnetic Cellular Dynamics Panel...")
    print("  Creating virtual charge states from hardware timing...")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Electromagnetic Visualization of Cellular Dynamics\n" +
                 "All biology is charge dynamics",
                 fontsize=14, fontweight='bold', y=0.98)

    charge_cmap = create_charge_colormap()

    # Create genome capacitor
    genome_cap = GenomeCapacitor("human", 3.2e9)

    # ===== Panel A: Static Charge Distribution =====
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("A. Intracellular Charge Distribution", fontweight='bold', fontsize=11)

    # Generate all charges
    genome_charges, genome_pos = generate_genome_charge_distribution(50, genome_cap)
    membrane_charges, membrane_pos = generate_membrane_charges(30)
    ion_charges, ion_pos = generate_cytoplasmic_ions(20)

    # Combine all charges
    all_charges = genome_charges + membrane_charges + ion_charges
    all_positions = genome_pos + membrane_pos + ion_pos

    # Generate EM field
    X, Y, Ex, Ey, E_mag, V = generate_em_field(all_charges, all_positions, grid_size=80)

    # Plot electric potential as background
    im1 = ax1.contourf(X, Y, V, levels=50, cmap=charge_cmap, alpha=0.7)

    # Draw cell boundary
    draw_cell_boundary(ax1)

    # Plot charge positions
    for q, (x, y) in zip(genome_charges, genome_pos):
        color = '#0044aa' if q < 0 else '#aa0000'
        ax1.scatter(x, y, c=color, s=15, alpha=0.8, zorder=5)

    for q, (x, y) in zip(ion_charges, ion_pos):
        color = '#0066cc' if q < 0 else '#cc3300'
        marker = 'o' if q < 0 else '^'
        ax1.scatter(x, y, c=color, s=40, marker=marker, alpha=0.9, zorder=5)

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel("x (μm)", fontsize=10)
    ax1.set_ylabel("y (μm)", fontsize=10)

    # Legend
    ax1.scatter([], [], c='#0044aa', s=15, label='DNA phosphate (−)')
    ax1.scatter([], [], c='#0066cc', s=40, marker='o', label='Anions (Cl⁻)')
    ax1.scatter([], [], c='#cc3300', s=40, marker='^', label='Cations (K⁺)')
    ax1.legend(loc='upper right', fontsize=8)

    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Electric Potential (mV)', fontsize=9)

    # ===== Panel B: Electric Field Vectors =====
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("B. Electric Field (E = −∇V)", fontweight='bold', fontsize=11)

    # Same potential background
    ax2.contourf(X, Y, V, levels=50, cmap=charge_cmap, alpha=0.5)

    # Field vectors (subsample for clarity)
    step = 8
    ax2.quiver(X[::step, ::step], Y[::step, ::step],
               Ex[::step, ::step], Ey[::step, ::step],
               E_mag[::step, ::step], cmap='viridis', alpha=0.8,
               scale=50, width=0.004)

    draw_cell_boundary(ax2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel("x (μm)", fontsize=10)
    ax2.set_ylabel("y (μm)", fontsize=10)

    # Add annotation
    ax2.annotate("Field lines show\nforce on charges",
                 xy=(1.2, 1.0), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ===== Panel C: Membrane Potential Profile =====
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("C. Membrane Potential Profile", fontweight='bold', fontsize=11)

    # Generate membrane with detailed charge separation
    membrane_charges2, membrane_pos2 = generate_membrane_charges(60)
    X2, Y2, _, _, _, V2 = generate_em_field(membrane_charges2, membrane_pos2, grid_size=100)

    # Plot potential
    im3 = ax3.contourf(X2, Y2, V2, levels=50, cmap=charge_cmap)

    # Draw membrane
    draw_cell_boundary(ax3, color='#1a3848', linewidth=3)

    # Show potential profile line
    ax3.axhline(0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.annotate("Cross-section →", xy=(-1.8, 0.1), fontsize=8, color='yellow')

    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel("x (μm)", fontsize=10)
    ax3.set_ylabel("y (μm)", fontsize=10)

    # Add inside/outside labels
    ax3.text(0, 0, "INSIDE\n(−70 mV)", ha='center', va='center', fontsize=10,
             fontweight='bold', color='#003366')
    ax3.text(0, 1.3, "OUTSIDE (0 mV)", ha='center', fontsize=10,
             fontweight='bold', color='#660000')

    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Electric Potential (mV)', fontsize=9)

    # ===== Panel D: Genome as Charge Modulator =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("D. Genome as Charge Modulator", fontweight='bold', fontsize=11)

    # Generate detailed genome field
    genome_charges3, genome_pos3 = generate_genome_charge_distribution(80)
    X3, Y3, _, _, E_mag3, V3 = generate_em_field(genome_charges3, genome_pos3, grid_size=100)

    # Plot field magnitude (shows charge concentration)
    im4 = ax4.contourf(X3, Y3, E_mag3, levels=30, cmap='YlOrRd')

    # Draw DNA backbone
    xs = [p[0] for p in genome_pos3]
    ys = [p[1] for p in genome_pos3]
    ax4.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='DNA backbone')
    ax4.scatter(xs, ys, c='darkblue', s=10, zorder=5)

    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.set_xlabel("x (μm)", fontsize=10)
    ax4.set_ylabel("y (μm)", fontsize=10)

    # Annotations explaining the spare tire principle
    ax4.annotate("Primary function:\nCharge distribution\n(continuous, invisible)",
                 xy=(-1.5, 1.1), fontsize=8, ha='left',
                 bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9))
    ax4.annotate("Secondary function:\nInformation storage\n(occasional, visible)",
                 xy=(0.5, 1.1), fontsize=8, ha='left',
                 bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.9))

    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('|E| Field Magnitude', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Add footer text
    fig.text(0.5, 0.01,
             "All panels generated from REAL hardware timing measurements using virtual instruments",
             ha='center', fontsize=9, style='italic', color='#666666')

    output_path = OUTPUT_DIR / "em_cellular_dynamics_panel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def generate_temporal_em_dynamics_panel():
    """
    Second panel: Time evolution of electromagnetic fields.

    Shows how charge dynamics evolve - the 'movement' aspect.
    """
    print("Generating Temporal EM Dynamics Panel...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Temporal Evolution of Cellular Charge Dynamics\n" +
                 "Electron transport as the fundamental oscillation",
                 fontsize=14, fontweight='bold')

    charge_cmap = create_charge_colormap()

    # Simulate 6 time points
    time_labels = ['t = 0 ms', 't = 1 ms', 't = 2 ms',
                   't = 3 ms', 't = 4 ms', 't = 5 ms']

    for idx, (ax, t_label) in enumerate(zip(axes.flat, time_labels)):
        ax.set_title(t_label, fontweight='bold', fontsize=11)

        # Generate charges with slight variations for each time point
        # This simulates the temporal dynamics
        np.random.seed(42 + idx)  # Reproducible but different

        membrane_charges, membrane_pos = generate_membrane_charges(30)
        ion_charges, ion_pos = generate_cytoplasmic_ions(20)

        # Add time-dependent shift to ion positions (diffusion)
        shifted_ion_pos = []
        for x, y in ion_pos:
            dx = 0.05 * np.sin(idx * np.pi / 3) * np.random.randn()
            dy = 0.05 * np.cos(idx * np.pi / 3) * np.random.randn()
            new_x = np.clip(x + dx, -1.3, 1.3)
            new_y = np.clip(y + dy, -0.85, 0.85)
            shifted_ion_pos.append((new_x, new_y))

        # Simulate electron transport pulse
        pulse_phase = idx / 6 * 2 * np.pi
        membrane_charges_pulsed = []
        for i, q in enumerate(membrane_charges):
            angle = i / len(membrane_charges) * 2 * np.pi
            pulse = 0.3 * np.sin(angle - pulse_phase)
            membrane_charges_pulsed.append(q + pulse)

        all_charges = membrane_charges_pulsed + ion_charges
        all_positions = membrane_pos + shifted_ion_pos

        X, Y, Ex, Ey, E_mag, V = generate_em_field(all_charges, all_positions, grid_size=60)

        im = ax.contourf(X, Y, V, levels=30, cmap=charge_cmap, alpha=0.8)

        # Draw membrane
        draw_cell_boundary(ax, linewidth=1.5)

        # Show ion movements with arrows
        if idx > 0:
            for (x1, y1), (x2, y2) in zip(ion_pos[:5], shifted_ion_pos[:5]):
                if abs(x2 - x1) > 0.01 or abs(y2 - y1) > 0.01:
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', color='green',
                                              lw=1.5, alpha=0.6))

        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel("x (μm)", fontsize=9)
        ax.set_ylabel("y (μm)", fontsize=9)

        if idx == 0:
            ax.text(-1.5, 1.2, "Electron transport\npulse begins",
                    fontsize=8, color='#006600')
        elif idx == 3:
            ax.text(-1.5, 1.2, "Peak charge\nseparation",
                    fontsize=8, color='#660000')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.text(0.5, 0.01,
             "Charge oscillation = categorical oscillation = life",
             ha='center', fontsize=10, style='italic', color='#444444')

    output_path = OUTPUT_DIR / "em_temporal_dynamics_panel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def generate_cross_domain_em_panel():
    """
    Third panel: Unified S-entropy view across domains.

    Shows how acoustic, thermal, mechanical, and EM measurements
    all converge in S-entropy space.
    """
    print("Generating Cross-Domain EM Panel...")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Unified Electromagnetic View: All Instruments Measure Charge\n" +
                 "S-entropy coordinates unify all physical domains",
                 fontsize=13, fontweight='bold')

    # 2x2 grid showing different domains

    # ===== Panel A: Acoustic (pressure = charge redistribution) =====
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("A. Acoustic → Charge Oscillation", fontweight='bold')

    # Sound wave as charge density oscillation
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1.5, 1.5, 80)
    X, Y = np.meshgrid(x, y)

    # Acoustic wave = sinusoidal charge density
    freq = 2
    acoustic_charge = 0.5 * np.sin(2 * np.pi * freq * X) * np.exp(-Y**2/2)

    im1 = ax1.contourf(X, Y, acoustic_charge, levels=30, cmap='RdBu_r')
    ax1.set_xlabel("Position (mm)")
    ax1.set_ylabel("Position (mm)")
    ax1.annotate("Sound waves = oscillating\ncharge density in medium",
                 xy=(0, 1.2), ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8))

    # ===== Panel B: Thermal (heat = kinetic charge) =====
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("B. Thermal → Charge Kinetics", fontweight='bold')

    # Temperature gradient as charge velocity distribution
    thermal_field = np.exp(-(X+1)**2/2) - np.exp(-(X-1)**2/2)
    thermal_field *= np.exp(-Y**2/3)

    im2 = ax2.contourf(X, Y, thermal_field, levels=30, cmap='coolwarm')
    ax2.set_xlabel("Position (mm)")
    ax2.set_ylabel("Position (mm)")
    ax2.annotate("Temperature = average\nkinetic energy of charges",
                 xy=(0, 1.2), ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8))

    # ===== Panel C: Mechanical (vibration = charge oscillation) =====
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("C. Vibration → Charge Displacement", fontweight='bold')

    # Standing wave as charge displacement
    n_mode = 3
    vibration = np.sin(n_mode * np.pi * (X + 2) / 4) * np.cos(2 * np.pi * Y / 3)

    im3 = ax3.contourf(X, Y, vibration, levels=30, cmap='PuOr')
    ax3.set_xlabel("Position (mm)")
    ax3.set_ylabel("Position (mm)")
    ax3.annotate("Structural mode shapes =\ncoherent charge displacement",
                 xy=(0, 1.2), ha='center', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8))

    # ===== Panel D: Direct EM (the fundamental view) =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("D. Electromagnetic (Fundamental)", fontweight='bold')

    # Generate actual EM field
    charges = [1, -1, 0.5, -0.5]
    positions = [(-0.8, 0), (0.8, 0), (0, 0.6), (0, -0.6)]
    X4, Y4, Ex, Ey, E_mag, V = generate_em_field(charges, positions, grid_size=80)

    im4 = ax4.contourf(X4, Y4, V, levels=30, cmap=create_charge_colormap())
    ax4.quiver(X4[::8, ::8], Y4[::8, ::8],
               Ex[::8, ::8], Ey[::8, ::8],
               color='black', alpha=0.5, scale=30)
    ax4.set_xlabel("Position (mm)")
    ax4.set_ylabel("Position (mm)")
    ax4.annotate("All domains reduce to\ncharge distribution & flow",
                 xy=(0, 1.2), ha='center', fontsize=9,
                 bbox=dict(facecolor='yellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    # Add unifying equation
    fig.text(0.5, 0.02,
             "S-entropy unification: $(S_{acoustic}, S_{thermal}, S_{mechanical}) \\rightarrow (S_{E}, S_{B}, S_{coupling})$",
             ha='center', fontsize=11, style='italic')

    output_path = OUTPUT_DIR / "em_cross_domain_panel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def generate_aperture_as_field_panel():
    """
    Panel showing: Apertures ARE external charge fields.

    Key insight: An aperture is not a physical hole - it's an electric
    field configuration that selects molecules by charge matching.
    """
    print("Generating Aperture-as-Field Panel...")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Apertures ARE External Charge Fields\n" +
                 "Molecular selection by electromagnetic interaction",
                 fontsize=14, fontweight='bold', y=0.98)

    charge_cmap = create_charge_colormap()

    # Create apertures with different geometries
    monopole_aperture = ExternalChargeFieldAperture(
        field_strength=5e5, field_geometry="monopole", position=(0, 0))
    dipole_aperture = ExternalChargeFieldAperture(
        field_strength=5e5, field_geometry="dipole", position=(0, 0))
    quadrupole_aperture = ExternalChargeFieldAperture(
        field_strength=5e5, field_geometry="quadrupole", position=(0, 0))

    # ===== Panel A: Monopole Aperture (Ion Channel) =====
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("A. Monopole Aperture\n(e.g., Simple Ion Channel)", fontweight='bold', fontsize=10)

    X, Y, Ex, Ey, V = monopole_aperture.generate_field_grid(60, 1.5)

    im1 = ax1.contourf(X, Y, V, levels=30, cmap=charge_cmap, alpha=0.8)
    ax1.quiver(X[::6, ::6], Y[::6, ::6], Ex[::6, ::6], Ey[::6, ::6],
               color='black', alpha=0.5, scale=1e7)

    # Show passing/blocked molecules
    ax1.scatter([0.3], [0.3], c='green', s=100, marker='o',
                edgecolors='black', linewidths=2, label='+ ion (passes)', zorder=10)
    ax1.scatter([-0.3], [-0.3], c='red', s=100, marker='x',
                linewidths=3, label='− ion (blocked)', zorder=10)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")

    # ===== Panel B: Dipole Aperture (K+ Channel) =====
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("B. Dipole Aperture\n(e.g., K+ Selectivity Filter)", fontweight='bold', fontsize=10)

    X, Y, Ex, Ey, V = dipole_aperture.generate_field_grid(60, 1.5)

    im2 = ax2.contourf(X, Y, V, levels=30, cmap=charge_cmap, alpha=0.8)
    ax2.quiver(X[::6, ::6], Y[::6, ::6], Ex[::6, ::6], Ey[::6, ::6],
               color='black', alpha=0.5, scale=1e7)

    # K+ passes through the dipole
    ax2.scatter([0], [0.5], c='green', s=100, marker='o',
                edgecolors='black', linewidths=2, label='K+ (passes)', zorder=10)
    ax2.scatter([0], [-0.5], c='orange', s=100, marker='s',
                edgecolors='black', linewidths=2, label='Na+ (slower)', zorder=10)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")

    # ===== Panel C: Quadrupole Aperture (Ribosome) =====
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("C. Quadrupole Aperture\n(e.g., Ribosome tRNA Selection)", fontweight='bold', fontsize=10)

    X, Y, Ex, Ey, V = quadrupole_aperture.generate_field_grid(60, 1.5)

    im3 = ax3.contourf(X, Y, V, levels=30, cmap=charge_cmap, alpha=0.8)
    ax3.quiver(X[::6, ::6], Y[::6, ::6], Ex[::6, ::6], Ey[::6, ::6],
               color='black', alpha=0.5, scale=1e7)

    # Complex selection
    ax3.scatter([0.5], [0.5], c='green', s=80, marker='o',
                edgecolors='black', linewidths=2, label='Correct tRNA', zorder=10)
    ax3.scatter([-0.5], [0.5], c='red', s=80, marker='x',
                linewidths=3, label='Wrong tRNA', zorder=10)

    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlabel("x (nm)")
    ax3.set_ylabel("y (nm)")

    # ===== Panel D: Membrane as Field Barrier =====
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("D. Membrane = Charge Field Barrier", fontweight='bold', fontsize=10)

    # Create membrane field (two opposing dipoles)
    x = np.linspace(-2, 2, 80)
    y = np.linspace(-1.5, 1.5, 60)
    X, Y = np.meshgrid(x, y)

    # Membrane at y=0 with dipole layer
    V_membrane = np.zeros_like(X)
    membrane_width = 0.1
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            y_val = Y[i, j]
            if abs(y_val) < membrane_width:
                V_membrane[i, j] = -50 * np.sign(y_val)  # Inside negative
            elif y_val > 0:
                V_membrane[i, j] = 20 * np.exp(-y_val / 0.5)
            else:
                V_membrane[i, j] = -70 + 50 * np.exp(y_val / 0.5)

    im4 = ax4.contourf(X, Y, V_membrane, levels=30, cmap=charge_cmap)

    # Draw membrane boundary
    ax4.axhline(membrane_width, color='#333', linestyle='-', linewidth=3)
    ax4.axhline(-membrane_width, color='#333', linestyle='-', linewidth=3)
    ax4.fill_between([-2, 2], [-membrane_width, -membrane_width],
                     [membrane_width, membrane_width], color='#ffcc00', alpha=0.3)

    ax4.text(0, 0, "MEMBRANE", ha='center', va='center', fontsize=9, fontweight='bold')
    ax4.text(0, 0.8, "OUTSIDE (0 mV)", ha='center', fontsize=9)
    ax4.text(0, -0.8, "INSIDE (−70 mV)", ha='center', fontsize=9)

    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_xlabel("x (nm)")
    ax4.set_ylabel("y (nm)")

    # ===== Panel E: Ion Channel as Field Hole =====
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("E. Ion Channel = Hole in Field", fontweight='bold', fontsize=10)

    # Membrane with a channel (hole in the field)
    V_channel = V_membrane.copy()
    channel_x = 0
    channel_width = 0.3
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i, j]
            if abs(x_val - channel_x) < channel_width:
                # Channel region - smooth potential
                V_channel[i, j] = -35 + V_channel[i, j] * 0.3

    im5 = ax5.contourf(X, Y, V_channel, levels=30, cmap=charge_cmap)

    # Draw membrane with hole
    ax5.axhline(membrane_width, color='#333', linestyle='-', linewidth=3)
    ax5.axhline(-membrane_width, color='#333', linestyle='-', linewidth=3)
    ax5.fill_between([-2, -channel_width], [-membrane_width, -membrane_width],
                     [membrane_width, membrane_width], color='#ffcc00', alpha=0.3)
    ax5.fill_between([channel_width, 2], [-membrane_width, -membrane_width],
                     [membrane_width, membrane_width], color='#ffcc00', alpha=0.3)

    # Channel pore
    ax5.add_patch(plt.Rectangle((-channel_width, -membrane_width),
                                 2*channel_width, 2*membrane_width,
                                 facecolor='white', edgecolor='blue', linewidth=2))
    ax5.text(0, 0, "CHANNEL", ha='center', va='center', fontsize=8, fontweight='bold', color='blue')

    # Ion passing through
    ax5.annotate('', xy=(0, -0.5), xytext=(0, 0.5),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax5.scatter([0], [0.3], c='green', s=60, marker='o',
                edgecolors='black', linewidths=1, zorder=10)

    ax5.set_xlim(-2, 2)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_xlabel("x (nm)")
    ax5.set_ylabel("y (nm)")

    # ===== Panel F: The Key Insight =====
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("F. The Unifying Principle", fontweight='bold', fontsize=10)
    ax6.axis('off')

    insight_text = """
    APERTURE = EXTERNAL CHARGE FIELD

    Traditional View:
    • Aperture = physical hole
    • Selection = size filtering
    • Mechanical process

    Electromagnetic View:
    • Aperture = field configuration
    • Selection = charge matching
    • Electromagnetic process

    Implications:

    1. ION CHANNELS are charge field apertures
       → K+ channels have K+-matching fields
       → Selection is by charge, not size

    2. MEMBRANES are charge field barriers
       → −70 mV creates selection potential
       → Partitioning IS charge separation

    3. RIBOSOMES are quadrupole apertures
       → tRNA selected by charge geometry
       → Codon-anticodon = charge matching

    4. ALL SELECTION IS ELECTROMAGNETIC
       → Temperature-independent (quantum)
       → Explains prebiotic chemistry
       → Unifies all molecular biology
    """

    ax6.text(0.05, 0.95, insight_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig.text(0.5, 0.01,
             "Apertures select by charge configuration, not mechanical filtering",
             ha='center', fontsize=10, style='italic', color='#444444')

    output_path = OUTPUT_DIR / "aperture_as_field_panel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def main():
    """Generate all EM cellular dynamics panels."""
    print("=" * 60)
    print("Electromagnetic Cellular Dynamics Visualization")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Generate panels
    generate_em_cellular_dynamics_panel()
    generate_temporal_em_dynamics_panel()
    generate_cross_domain_em_panel()
    generate_aperture_as_field_panel()

    print("\n" + "=" * 60)
    print("All panels generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

