#!/usr/bin/env python3
"""
Generate panel charts for the Origins of Life paper.
Visualizes electron transport partitioning, homochirality, and semiconductor origins.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge, Rectangle, Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from pathlib import Path
import os

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "origins-of-life" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_orgels_paradox_panel():
    """Panel 1: Orgel's Paradox and probability comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Orgel's Paradox: Information-First Impossibility", fontsize=14, fontweight='bold')
    
    # Panel A: Circular dependency diagram
    ax = axes[0, 0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(A) Circular Dependency', fontsize=11, fontweight='bold')
    
    # Three nodes in a triangle
    angles = [90, 210, 330]
    labels = ['Information\n(DNA/RNA)', 'Catalysis\n(Enzymes)', 'Metabolism\n(Energy)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    positions = []
    for angle, label, color in zip(angles, labels, colors):
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        positions.append((x, y))
        circle = Circle((x, y), 0.3, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows between nodes
    for i in range(3):
        start = positions[i]
        end = positions[(i+1) % 3]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length * 0.35
        dy_norm = dy / length * 0.35
        ax.annotate('', xy=(end[0]-dx_norm, end[1]-dy_norm), 
                   xytext=(start[0]+dx_norm, start[1]+dy_norm),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.text(0, 0, 'No Entry\nPoint!', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='red', style='italic')
    
    # Panel B: Probability comparison (log scale)
    ax = axes[0, 1]
    scenarios = ['DNA-first', 'RNA World', 'Membrane-first']
    probabilities = [1e-200, 1e-150, 1e-6]
    colors = ['#E74C3C', '#F39C12', '#2ECC71']
    
    bars = ax.barh(scenarios, [np.log10(p) for p in probabilities], color=colors, edgecolor='black')
    ax.set_xlabel('log10(Probability)', fontsize=10)
    ax.set_title('(B) Origin Scenario Probabilities', fontsize=11, fontweight='bold')
    ax.axvline(x=-150, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-220, 0)
    
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
               f'10^{int(np.log10(prob))}', va='center', fontsize=9)
    
    # Panel C: Probability ratio
    ax = axes[0, 2]
    ax.axis('off')
    ax.set_title('(C) Probability Ratio', fontsize=11, fontweight='bold')
    
    ratio_text = r'$\frac{P_{\mathrm{membrane}}}{P_{\mathrm{RNA}}} = \frac{10^{-6}}{10^{-150}} = 10^{144}$'
    ax.text(0.5, 0.6, ratio_text, ha='center', va='center', fontsize=16, 
            transform=ax.transAxes)
    ax.text(0.5, 0.3, 'Membrane-first is', ha='center', va='center', fontsize=12,
            transform=ax.transAxes)
    ax.text(0.5, 0.15, r'$10^{144}$ times more likely', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#2ECC71', transform=ax.transAxes)
    
    # Panel D: Information requires infrastructure
    ax = axes[1, 0]
    ax.axis('off')
    ax.set_title('(D) Virus Proof', fontsize=11, fontweight='bold')
    
    # Virus illustration
    virus_circle = Circle((0.3, 0.5), 0.15, facecolor='#9B59B6', edgecolor='black', 
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(virus_circle)
    ax.text(0.3, 0.5, 'Virus\nDNA/RNA', ha='center', va='center', fontsize=8,
            fontweight='bold', transform=ax.transAxes, color='white')
    
    ax.text(0.55, 0.5, '+', ha='center', va='center', fontsize=20, 
            transform=ax.transAxes)
    
    # No infrastructure
    ax.text(0.75, 0.5, 'No Cell', ha='center', va='center', fontsize=10,
            transform=ax.transAxes, style='italic')
    
    ax.text(0.5, 0.25, '= Zero Function', ha='center', va='center', fontsize=12,
            fontweight='bold', color='red', transform=ax.transAxes)
    ax.text(0.5, 0.1, 'Information is inert without infrastructure', 
            ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Panel E: Thermodynamic favorability
    ax = axes[1, 1]
    processes = ['RNA\nSynthesis', 'Membrane\nAssembly']
    delta_g = [150, -60]  # kJ/mol (approximate)
    colors = ['#E74C3C', '#2ECC71']
    
    bars = ax.bar(processes, delta_g, color=colors, edgecolor='black', width=0.6)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel(r'$\Delta G$ (kJ/mol)', fontsize=10)
    ax.set_title('(E) Thermodynamic Favorability', fontsize=11, fontweight='bold')
    ax.set_ylim(-100, 200)
    
    ax.text(0, 170, 'Unfavorable', ha='center', va='bottom', fontsize=9, color='#E74C3C')
    ax.text(1, -80, 'Favorable', ha='center', va='top', fontsize=9, color='#2ECC71')
    
    # Panel F: Resolution arrow
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('(F) Resolution', fontsize=11, fontweight='bold')
    
    ax.text(0.5, 0.8, 'Traditional:', ha='center', va='center', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.5, 0.65, 'Info -> Enzymes -> Life', ha='center', va='center', 
            fontsize=10, color='#E74C3C', transform=ax.transAxes)
    
    ax.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.55),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               xycoords='axes fraction', textcoords='axes fraction')
    
    ax.text(0.5, 0.35, 'Resolution:', ha='center', va='center', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.5, 0.2, 'Electron Transport -> Partitioning -> Life', 
            ha='center', va='center', fontsize=10, color='#2ECC71', 
            fontweight='bold', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'orgels_paradox_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'orgels_paradox_panel.png'}")


def generate_electron_transport_panel():
    """Panel 2: Electron transport as fundamental partitioning."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Electron Transport Creates Charge Partitioning", fontsize=14, fontweight='bold')
    
    # Panel A: Electron transport creates partition
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(A) Charge Separation', fontsize=11, fontweight='bold')
    
    # Before
    ax.text(2.5, 9, 'Before', ha='center', va='center', fontsize=10, fontweight='bold')
    circle_before = Circle((2.5, 7), 1.5, facecolor='#EEEEEE', edgecolor='black', linewidth=2)
    ax.add_patch(circle_before)
    ax.text(2.5, 7, 'Neutral', ha='center', va='center', fontsize=9)
    
    # Arrow
    ax.annotate('', xy=(6, 7), xytext=(4.5, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.25, 7.5, 'e-', ha='center', va='center', fontsize=10, color='blue')
    
    # After
    ax.text(7.5, 9, 'After', ha='center', va='center', fontsize=10, fontweight='bold')
    # Positive region
    circle_pos = Circle((6.5, 7), 0.8, facecolor='#E74C3C', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(circle_pos)
    ax.text(6.5, 7, '+', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    # Negative region
    circle_neg = Circle((8.5, 7), 0.8, facecolor='#3498DB', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(circle_neg)
    ax.text(8.5, 7, '-', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    ax.text(5, 4.5, 'PARTITION', ha='center', va='center', fontsize=12, 
            fontweight='bold', color='purple')
    
    # Electric field lines
    for y_offset in [-0.5, 0, 0.5]:
        ax.annotate('', xy=(8.2, 7+y_offset), xytext=(6.8, 7+y_offset),
                   arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))
    ax.text(7.5, 5.5, 'Electric Field', ha='center', va='center', fontsize=9, color='gray')
    
    # Panel B: Quantum tunneling
    ax = axes[0, 1]
    x = np.linspace(0, 10, 500)
    barrier_start, barrier_end = 4, 6
    
    # Potential barrier
    potential = np.zeros_like(x)
    potential[(x >= barrier_start) & (x <= barrier_end)] = 1
    ax.fill_between(x, potential, alpha=0.3, color='gray', label='Barrier')
    ax.plot(x, potential, 'k-', linewidth=2)
    
    # Wavefunction (decaying through barrier)
    psi = np.zeros_like(x)
    psi[x < barrier_start] = np.sin(2 * (x[x < barrier_start] - 1)) * np.exp(-0.1 * (barrier_start - x[x < barrier_start]))
    psi[(x >= barrier_start) & (x <= barrier_end)] = 0.5 * np.exp(-2 * (x[(x >= barrier_start) & (x <= barrier_end)] - barrier_start))
    psi[x > barrier_end] = 0.1 * np.sin(2 * (x[x > barrier_end] - barrier_end))
    
    ax.plot(x, psi + 0.5, 'b-', linewidth=2, label=r'$\psi(x)$')
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Energy / Probability', fontsize=10)
    ax.set_title('(B) Quantum Tunneling', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-0.5, 1.5)
    
    # Panel C: Temperature independence
    ax = axes[0, 2]
    temps = [10, 50, 100, 200, 300]
    tunneling_prob = [0.15, 0.15, 0.15, 0.15, 0.15]  # Constant
    encounter_rate = [t/300 for t in temps]  # Proportional to sqrt(T)
    
    ax.plot(temps, tunneling_prob, 'b-o', linewidth=2, markersize=8, label='Tunneling P')
    ax.plot(temps, encounter_rate, 'r--s', linewidth=2, markersize=8, label='Encounter Rate')
    ax.set_xlabel('Temperature (K)', fontsize=10)
    ax.set_ylabel('Relative Value', fontsize=10)
    ax.set_title('(C) Temperature Independence', fontsize=11, fontweight='bold')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(0, 1.2)
    
    # Panel D: Charge field as aperture
    ax = axes[1, 0]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('(D) Charge Field Aperture', fontsize=11, fontweight='bold')
    
    # Equipotential lines
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Dipole potential
    r1 = np.sqrt((X + 1)**2 + Y**2)
    r2 = np.sqrt((X - 1)**2 + Y**2)
    r1[r1 < 0.3] = 0.3
    r2[r2 < 0.3] = 0.3
    phi = 1/r1 - 1/r2
    
    contours = ax.contour(X, Y, phi, levels=[-2, -1, -0.5, 0, 0.5, 1, 2], 
                          colors='gray', alpha=0.5)
    
    # Charges
    ax.plot(-1, 0, 'ro', markersize=15)
    ax.plot(1, 0, 'bo', markersize=15)
    ax.text(-1, -0.5, '+', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(1, -0.5, '-', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Aperture region
    aperture = Ellipse((0, 0), 1, 2.5, facecolor='none', edgecolor='green', 
                        linewidth=3, linestyle='--')
    ax.add_patch(aperture)
    ax.text(0, 2, 'Aperture', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')
    
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # Panel E: Molecular selection
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('(E) Geometric Selection', fontsize=11, fontweight='bold')
    
    # Aperture
    aperture_rect = Rectangle((0.3, 0.3), 0.1, 0.4, facecolor='gray', 
                               edgecolor='black', linewidth=2, transform=ax.transAxes)
    ax.add_patch(aperture_rect)
    ax.text(0.35, 0.75, 'Aperture', ha='center', va='bottom', fontsize=9, transform=ax.transAxes)
    
    # Matching molecule (passes)
    circle1 = Circle((0.15, 0.5), 0.08, facecolor='#2ECC71', edgecolor='black', 
                     linewidth=2, transform=ax.transAxes)
    ax.add_patch(circle1)
    ax.annotate('', xy=(0.55, 0.5), xytext=(0.25, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'),
               xycoords='axes fraction', textcoords='axes fraction')
    ax.text(0.65, 0.5, 'PASS', ha='left', va='center', fontsize=10, 
            color='green', fontweight='bold', transform=ax.transAxes)
    
    # Non-matching molecule (blocked)
    ellipse1 = Ellipse((0.15, 0.2), 0.2, 0.08, facecolor='#E74C3C', edgecolor='black',
                       linewidth=2, transform=ax.transAxes)
    ax.add_patch(ellipse1)
    ax.text(0.35, 0.2, 'X', ha='center', va='center', fontsize=14, 
            color='red', fontweight='bold', transform=ax.transAxes)
    ax.text(0.55, 0.2, 'BLOCK', ha='left', va='center', fontsize=10,
            color='red', fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.05, 'Selection by configuration, not velocity', 
            ha='center', va='center', fontsize=9, style='italic', transform=ax.transAxes)
    
    # Panel F: Energy landscape
    ax = axes[1, 2]
    x = np.linspace(0, 10, 100)
    
    # Standard thermal reaction
    thermal = 1 + np.exp(-((x-3)**2)/0.5) * 2 - (x > 5) * 0.5
    thermal = np.clip(thermal, 0, 3)
    
    # Electron transport pathway
    et_pathway = 1 - 0.3 * np.sin(x * np.pi / 5)
    
    ax.plot(x, thermal, 'r-', linewidth=2, label='Thermal Activation')
    ax.plot(x, et_pathway, 'b--', linewidth=2, label='ET Pathway')
    ax.fill_between(x, thermal, et_pathway, where=thermal > et_pathway, 
                    alpha=0.3, color='green', label='ET Advantage')
    
    ax.set_xlabel('Reaction Coordinate', fontsize=10)
    ax.set_ylabel('Free Energy', fontsize=10)
    ax.set_title('(F) Energy Landscape', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'electron_transport_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'electron_transport_panel.png'}")


def generate_homochirality_panel():
    """Panel 3: Homochirality as proof of partitioning."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Homochirality: Proof of Partitioning Primacy", fontsize=14, fontweight='bold')
    
    # Panel A: Chiral hierarchy
    ax = axes[0, 0]
    ax.axis('off')
    ax.set_title('(A) Hierarchical Chiral Propagation', fontsize=11, fontweight='bold')
    
    levels = ['Electron Spin', 'Amino Acids', 'Sugars', 'DNA Helix', 'Proteins', 'Membranes']
    chiralities = ['Spin', 'L-form', 'D-form', 'Right', 'Right', 'Specific']
    y_positions = np.linspace(0.9, 0.1, len(levels))
    
    for i, (level, chiral, y) in enumerate(zip(levels, chiralities, y_positions)):
        color = '#3498DB' if i == 0 else '#2ECC71'
        ax.text(0.3, y, level, ha='right', va='center', fontsize=10, transform=ax.transAxes)
        ax.text(0.5, y, '->', ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.7, y, chiral, ha='left', va='center', fontsize=10, 
                fontweight='bold', color=color, transform=ax.transAxes)
        if i < len(levels) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + 0.03), xytext=(0.5, y - 0.03),
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray'),
                       xycoords='axes fraction', textcoords='axes fraction')
    
    ax.text(0.5, 0.02, 'Same partition propagates through all levels', 
            ha='center', va='center', fontsize=9, style='italic', transform=ax.transAxes)
    
    # Panel B: L vs D amino acids
    ax = axes[0, 1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(B) L vs D Amino Acids', fontsize=11, fontweight='bold')
    
    # L-form (selected)
    ax.text(-1, 1.5, 'L-form', ha='center', va='center', fontsize=11, fontweight='bold', color='#2ECC71')
    l_circle = Circle((-1, 0.5), 0.6, facecolor='#2ECC71', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(l_circle)
    ax.text(-1, 0.5, 'NH2-C-COOH', ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(-1, -0.3, 'SELECTED', ha='center', va='center', fontsize=10, color='#2ECC71', fontweight='bold')
    
    # D-form (excluded)
    ax.text(1, 1.5, 'D-form', ha='center', va='center', fontsize=11, fontweight='bold', color='#E74C3C')
    d_circle = Circle((1, 0.5), 0.6, facecolor='#E74C3C', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(d_circle)
    ax.text(1, 0.5, 'HOOC-C-NH2', ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(1, -0.3, 'EXCLUDED', ha='center', va='center', fontsize=10, color='#E74C3C', fontweight='bold')
    
    ax.text(0, -1.5, 'Mirror images - same energy, different geometry', 
            ha='center', va='center', fontsize=9, style='italic')
    
    # Panel C: Spin-orbit coupling
    ax = axes[0, 2]
    theta = np.linspace(0, 4*np.pi, 100)
    r = 1
    x = theta / (2*np.pi)
    y = r * np.sin(theta)
    z = r * np.cos(theta)
    
    ax.plot(x, y, 'b-', linewidth=2, label='Electron path')
    ax.fill_between(x, y, alpha=0.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Spin arrows
    for i in range(0, len(x), 20):
        ax.annotate('', xy=(x[i], y[i] + 0.3), xytext=(x[i], y[i]),
                   arrowprops=dict(arrowstyle='->', lw=1, color='red'))
    
    ax.set_xlabel('Helical Progress', fontsize=10)
    ax.set_ylabel('Transverse Position', fontsize=10)
    ax.set_title('(C) Spin-Orbit Coupling', fontsize=11, fontweight='bold')
    ax.text(1, 0.8, 'Spin aligns\nwith helix', ha='center', va='center', fontsize=9, color='red')
    
    # Panel D: Autocatalytic amplification
    ax = axes[1, 0]
    t = np.linspace(0, 10, 100)
    ee_initial = 0.001
    k_auto = 0.5
    
    # Enantiomeric excess evolution
    ee = np.tanh(k_auto * t + np.arctanh(ee_initial))
    
    ax.plot(t, ee, 'b-', linewidth=2)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Complete homochirality')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Racemic')
    
    ax.fill_between(t, ee, alpha=0.2)
    ax.set_xlabel('Time (arbitrary)', fontsize=10)
    ax.set_ylabel('Enantiomeric Excess (ee)', fontsize=10)
    ax.set_title('(D) Autocatalytic Amplification', fontsize=11, fontweight='bold')
    ax.legend(loc='center right', fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    
    # Panel E: Chiral aperture
    ax = axes[1, 1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(E) Chiral Aperture Selection', fontsize=11, fontweight='bold')
    
    # L-aperture
    aperture = Wedge((0, 0), 1, 30, 150, width=0.3, facecolor='#3498DB', 
                     edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(aperture)
    ax.text(0, 0, 'L-shaped\nAperture', ha='center', va='center', fontsize=8)
    
    # L molecule fits
    l_mol = Circle((-1.5, 0.5), 0.2, facecolor='#2ECC71', edgecolor='black', linewidth=2)
    ax.add_patch(l_mol)
    ax.annotate('', xy=(-0.3, 0.5), xytext=(-1.2, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(-1.5, 0.9, 'L-mol', ha='center', va='center', fontsize=8, color='#2ECC71')
    
    # D molecule blocked
    d_mol = Circle((-1.5, -0.5), 0.2, facecolor='#E74C3C', edgecolor='black', linewidth=2)
    ax.add_patch(d_mol)
    ax.text(-0.5, -0.5, 'X', ha='center', va='center', fontsize=14, color='red', fontweight='bold')
    ax.text(-1.5, -0.9, 'D-mol', ha='center', va='center', fontsize=8, color='#E74C3C')
    
    # Panel F: Evidence summary
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('(F) Evidence for Partitioning Primacy', fontsize=11, fontweight='bold')
    
    evidence = [
        'If Information-First:',
        '  -> No chiral mechanism',
        '  -> Predicts racemic or mixed',
        '  -> FALSIFIED',
        '',
        'If Partitioning-First:',
        '  -> Spin-orbit coupling',
        '  -> Predicts homochirality',
        '  -> CONFIRMED'
    ]
    
    for i, line in enumerate(evidence):
        y = 0.9 - i * 0.1
        color = 'red' if 'FALSIFIED' in line else ('green' if 'CONFIRMED' in line else 'black')
        weight = 'bold' if 'FALSIFIED' in line or 'CONFIRMED' in line else 'normal'
        ax.text(0.1, y, line, ha='left', va='center', fontsize=10, 
                color=color, fontweight=weight, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'homochirality_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'homochirality_panel.png'}")


def generate_membrane_scaffolding_panel():
    """Panel 4: Membranes as electron transport scaffolding."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Membranes as Electron Transport Scaffolding", fontsize=14, fontweight='bold')
    
    # Panel A: Membrane charge architecture
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(A) Membrane Charge Architecture', fontsize=11, fontweight='bold')
    
    # Membrane bilayer
    membrane_rect = Rectangle((1, 4), 8, 2, facecolor='#F39C12', 
                               edgecolor='black', linewidth=2, alpha=0.5)
    ax.add_patch(membrane_rect)
    
    # Negative charges on surfaces
    for x in np.linspace(1.5, 8.5, 8):
        ax.text(x, 6.2, '-', ha='center', va='center', fontsize=12, 
                color='blue', fontweight='bold')
        ax.text(x, 3.8, '-', ha='center', va='center', fontsize=12,
                color='blue', fontweight='bold')
    
    ax.text(5, 5, 'Hydrophobic\nCore', ha='center', va='center', fontsize=9)
    ax.text(5, 7, 'Outside (+ ions)', ha='center', va='center', fontsize=9, color='red')
    ax.text(5, 3, 'Inside (neutral)', ha='center', va='center', fontsize=9)
    
    # Electric field arrows
    for x in [3, 5, 7]:
        ax.annotate('', xy=(x, 4.2), xytext=(x, 5.8),
                   arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
    ax.text(9, 5, 'E-field', ha='left', va='center', fontsize=8, color='gray')
    
    # Panel B: Cellular battery
    ax = axes[0, 1]
    ax.axis('off')
    ax.set_title('(B) Cellular Battery', fontsize=11, fontweight='bold')
    
    # Battery diagram
    cathode = Rectangle((0.2, 0.3), 0.2, 0.4, facecolor='blue', 
                        edgecolor='black', linewidth=2, transform=ax.transAxes)
    anode = Rectangle((0.6, 0.3), 0.2, 0.4, facecolor='gray',
                      edgecolor='black', linewidth=2, transform=ax.transAxes)
    ax.add_patch(cathode)
    ax.add_patch(anode)
    
    ax.text(0.3, 0.75, 'Cathode\n(Membrane)', ha='center', va='center', 
            fontsize=9, transform=ax.transAxes)
    ax.text(0.7, 0.75, 'Anode\n(Cytoplasm)', ha='center', va='center',
            fontsize=9, transform=ax.transAxes)
    
    # Potential difference
    ax.annotate('', xy=(0.6, 0.5), xytext=(0.4, 0.5),
               arrowprops=dict(arrowstyle='<->', lw=2, color='red'),
               xycoords='axes fraction', textcoords='axes fraction')
    ax.text(0.5, 0.55, r'$\Delta\Phi$ = 50-100 mV', ha='center', va='bottom',
            fontsize=10, color='red', transform=ax.transAxes)
    
    ax.text(0.5, 0.15, 'Drives electron transport', ha='center', va='center',
            fontsize=10, style='italic', transform=ax.transAxes)
    
    # Panel C: Electron transport chain
    ax = axes[0, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('(C) Membrane ET Chain', fontsize=11, fontweight='bold')
    
    complexes = ['I', 'II', 'III', 'IV']
    x_positions = [1.5, 3.5, 5.5, 7.5]
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6']
    
    # Membrane
    membrane_rect = Rectangle((0.5, 1.5), 9, 2, facecolor='#EEEEEE',
                               edgecolor='black', linewidth=2, alpha=0.5)
    ax.add_patch(membrane_rect)
    
    for x, name, color in zip(x_positions, complexes, colors):
        circle = Circle((x, 2.5), 0.6, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 2.5, name, ha='center', va='center', fontsize=12, 
                fontweight='bold', color='white')
    
    # Electron flow arrows
    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i+1] - 0.7, 2.5), 
                   xytext=(x_positions[i] + 0.7, 2.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    ax.text(5, 0.8, 'e- flow through membrane scaffold', ha='center', 
            va='center', fontsize=10, color='blue')
    ax.text(5, 4.2, 'H+ pumping', ha='center', va='center', fontsize=10, color='red')
    
    # Panel D: Scaffolding vs compartmentalization
    ax = axes[1, 0]
    properties = ['Negative charge', 'ET proteins', 'Proton gradient', 'Compartment']
    scaffolding_scores = [1.0, 1.0, 0.9, 0.3]
    compartment_scores = [0.2, 0.3, 0.4, 1.0]
    
    x = np.arange(len(properties))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, scaffolding_scores, width, label='Scaffolding', color='#3498DB')
    bars2 = ax.bar(x + width/2, compartment_scores, width, label='Compartment', color='#E74C3C')
    
    ax.set_ylabel('Importance Score', fontsize=10)
    ax.set_title('(D) Function Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(properties, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.2)
    
    # Panel E: Thermodynamic drive
    ax = axes[1, 1]
    chain_length = np.arange(8, 20, 2)
    delta_g = -3.5 * chain_length + 20  # Hydrophobic effect
    
    ax.bar(chain_length, -delta_g, color='#2ECC71', edgecolor='black', width=1.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Fatty Acid Chain Length', fontsize=10)
    ax.set_ylabel(r'$-\Delta G$ (kJ/mol)', fontsize=10)
    ax.set_title('(E) Membrane Formation Drive', fontsize=11, fontweight='bold')
    ax.text(14, 40, 'Spontaneous\nAssembly', ha='center', va='center', 
            fontsize=10, color='#2ECC71', fontweight='bold')
    
    # Panel F: Evolution timeline
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('(F) Evolutionary Sequence', fontsize=11, fontweight='bold')
    
    steps = [
        '1. Electron transport\n   (charge partitioning)',
        '2. Amphiphile association\n   (scaffolding stability)',
        '3. Membrane formation\n   (ET optimization)',
        '4. Compartmentalization\n   (secondary benefit)'
    ]
    
    for i, step in enumerate(steps):
        y = 0.85 - i * 0.22
        color = '#3498DB' if i < 3 else '#95A5A6'
        ax.text(0.1, y, step, ha='left', va='center', fontsize=10,
                color=color, transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.15, y - 0.08), xytext=(0.15, y - 0.02),
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray'),
                       xycoords='axes fraction', textcoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'membrane_scaffolding_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'membrane_scaffolding_panel.png'}")


def generate_semiconductor_origins_panel():
    """Panel 5: Semiconductor origins and interstellar chemistry."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Semiconductor Origins: Interstellar Prebiotic Chemistry", fontsize=14, fontweight='bold')
    
    # Panel A: Temperature vs reaction rate paradox
    ax = axes[0, 0]
    temps = np.linspace(10, 300, 100)
    
    # Classical Arrhenius
    ea = 0.5  # eV
    k_b = 8.617e-5  # eV/K
    arrhenius = np.exp(-ea / (k_b * temps))
    arrhenius = arrhenius / arrhenius[-1]  # Normalize
    
    # Quantum tunneling (temperature independent)
    tunneling = np.ones_like(temps) * 0.1
    
    ax.semilogy(temps, arrhenius, 'r-', linewidth=2, label='Classical (Arrhenius)')
    ax.semilogy(temps, tunneling, 'b--', linewidth=2, label='Quantum Tunneling')
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax.text(50, 1e-2, 'Interstellar\nT ~ 10-50 K', ha='center', va='center', fontsize=8)
    
    ax.set_xlabel('Temperature (K)', fontsize=10)
    ax.set_ylabel('Relative Rate', fontsize=10)
    ax.set_title('(A) Kinetic Paradox', fontsize=11, fontweight='bold')
    ax.legend(loc='center right', fontsize=8)
    ax.set_ylim(1e-10, 10)
    
    # Panel B: Mineral semiconductor band diagram
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(B) Mineral Semiconductor', fontsize=11, fontweight='bold')
    
    # Valence band
    vb = Rectangle((1, 1), 8, 2, facecolor='#3498DB', edgecolor='black', 
                   linewidth=2, alpha=0.7)
    ax.add_patch(vb)
    ax.text(5, 2, 'Valence Band', ha='center', va='center', fontsize=10, color='white')
    
    # Band gap
    ax.text(5, 4.5, 'Band Gap', ha='center', va='center', fontsize=10, style='italic')
    ax.annotate('', xy=(5, 3.5), xytext=(5, 5.5),
               arrowprops=dict(arrowstyle='<->', lw=2, color='gray'))
    
    # Conduction band
    cb = Rectangle((1, 6), 8, 2, facecolor='#E74C3C', edgecolor='black',
                   linewidth=2, alpha=0.7)
    ax.add_patch(cb)
    ax.text(5, 7, 'Conduction Band', ha='center', va='center', fontsize=10, color='white')
    
    # Electron excitation
    ax.annotate('', xy=(3, 6.5), xytext=(3, 2.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='yellow'))
    ax.text(2, 4.5, 'UV/\nCosmic\nRay', ha='center', va='center', fontsize=8, color='orange')
    
    # Hole creation
    circle_hole = Circle((3, 2), 0.3, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle_hole)
    ax.text(3.5, 2, 'h+', ha='left', va='center', fontsize=10)
    
    # Panel C: Ice matrix apertures
    ax = axes[0, 2]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('(C) Ice Matrix Apertures', fontsize=11, fontweight='bold')
    
    # Ice structure (hexagonal pattern)
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * 1.2 + (j % 2) * 0.6
            y = j * 1.0
            if x**2 + y**2 < 7:
                circle = Circle((x, y), 0.2, facecolor='#3498DB', 
                               edgecolor='black', linewidth=1, alpha=0.6)
                ax.add_patch(circle)
    
    # Pores (apertures)
    pore_positions = [(0.6, 0.5), (-0.6, -0.5), (0, 1.5)]
    for px, py in pore_positions:
        pore = Circle((px, py), 0.15, facecolor='white', edgecolor='green', linewidth=2)
        ax.add_patch(pore)
    
    ax.text(0, -2.5, 'Pores select by molecular size', ha='center', 
            va='center', fontsize=9, style='italic')
    ax.set_xlabel('x (nm)', fontsize=10)
    ax.set_ylabel('y (nm)', fontsize=10)
    
    # Panel D: Circularly polarized light chiral selection
    ax = axes[1, 0]
    theta = np.linspace(0, 4*np.pi, 200)
    
    # Left-handed CPL
    x_left = np.cos(theta)
    y_left = np.sin(theta)
    z_left = theta / (2*np.pi)
    
    ax.plot(z_left, x_left, 'b-', linewidth=2, label='L-CPL')
    ax.plot(z_left, y_left, 'b--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Propagation', fontsize=10)
    ax.set_ylabel('Field Amplitude', fontsize=10)
    ax.set_title('(D) Circularly Polarized Light', fontsize=11, fontweight='bold')
    ax.text(1, 0.8, 'Helical E-field\nselects chirality', ha='center', 
            va='center', fontsize=9, style='italic')
    
    # Panel E: Delivery to planets
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('(E) Delivery Pathway', fontsize=11, fontweight='bold')
    
    # Molecular cloud
    cloud = Ellipse((0.15, 0.7), 0.2, 0.15, facecolor='#9B59B6', 
                    edgecolor='black', linewidth=2, alpha=0.5, transform=ax.transAxes)
    ax.add_patch(cloud)
    ax.text(0.15, 0.7, 'Cloud', ha='center', va='center', fontsize=8, 
            color='white', transform=ax.transAxes)
    
    # Arrow to comet
    ax.annotate('', xy=(0.35, 0.7), xytext=(0.28, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'),
               xycoords='axes fraction', textcoords='axes fraction')
    
    # Comet
    comet = Ellipse((0.45, 0.7), 0.15, 0.1, facecolor='#3498DB',
                    edgecolor='black', linewidth=2, alpha=0.7, transform=ax.transAxes)
    ax.add_patch(comet)
    ax.text(0.45, 0.7, 'Comet', ha='center', va='center', fontsize=8,
            color='white', transform=ax.transAxes)
    
    # Arrow to meteorite
    ax.annotate('', xy=(0.55, 0.5), xytext=(0.5, 0.62),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'),
               xycoords='axes fraction', textcoords='axes fraction')
    
    # Meteorite
    meteor = Circle((0.6, 0.4), 0.08, facecolor='#7F8C8D',
                   edgecolor='black', linewidth=2, transform=ax.transAxes)
    ax.add_patch(meteor)
    ax.text(0.6, 0.4, 'Met', ha='center', va='center', fontsize=7,
            color='white', transform=ax.transAxes)
    
    # Arrow to planet
    ax.annotate('', xy=(0.75, 0.35), xytext=(0.68, 0.38),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'),
               xycoords='axes fraction', textcoords='axes fraction')
    
    # Planet
    planet = Circle((0.85, 0.3), 0.12, facecolor='#2ECC71',
                   edgecolor='black', linewidth=2, transform=ax.transAxes)
    ax.add_patch(planet)
    ax.text(0.85, 0.3, 'Earth', ha='center', va='center', fontsize=8,
            color='white', transform=ax.transAxes)
    
    ax.text(0.5, 0.1, r'$10^7 - 10^9$ kg/year delivered', ha='center',
            va='center', fontsize=10, transform=ax.transAxes)
    
    # Panel F: Continuity of partitioning
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('(F) Continuous Partitioning', fontsize=11, fontweight='bold')
    
    stages = [
        'Mineral surfaces',
        'Electron transport',
        'Charge partitions',
        'Chiral selection',
        'Delivery to planets',
        'Biological systems'
    ]
    
    for i, stage in enumerate(stages):
        y = 0.9 - i * 0.15
        color = plt.cm.viridis(i / len(stages))
        ax.text(0.5, y, stage, ha='center', va='center', fontsize=10,
                color=color, fontweight='bold', transform=ax.transAxes)
        if i < len(stages) - 1:
            ax.annotate('', xy=(0.5, y - 0.07), xytext=(0.5, y - 0.02),
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray'),
                       xycoords='axes fraction', textcoords='axes fraction')
    
    ax.text(0.5, 0.02, 'Same mechanism throughout', ha='center', va='center',
            fontsize=9, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'semiconductor_origins_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'semiconductor_origins_panel.png'}")


def generate_autocatalytic_panel():
    """Panel 6: Autocatalytic electron transport systems."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Autocatalytic Electron Transport: Self-Referential Systems", fontsize=14, fontweight='bold')
    
    # Panel A: Normal vs autocatalytic
    ax = axes[0, 0]
    ax.axis('off')
    ax.set_title('(A) Normal vs Autocatalytic', fontsize=11, fontweight='bold')
    
    # Normal catalysis
    ax.text(0.25, 0.85, 'Normal Catalysis:', ha='center', va='center', 
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    e_circle = Circle((0.1, 0.65), 0.06, facecolor='#3498DB', edgecolor='black',
                      linewidth=2, transform=ax.transAxes)
    ax.add_patch(e_circle)
    ax.text(0.1, 0.65, 'E', ha='center', va='center', fontsize=8, color='white',
            fontweight='bold', transform=ax.transAxes)
    
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.18, 0.65),
               arrowprops=dict(arrowstyle='->', lw=1, color='black'),
               xycoords='axes fraction', textcoords='axes fraction')
    
    s_circle = Circle((0.32, 0.65), 0.06, facecolor='#E74C3C', edgecolor='black',
                      linewidth=2, transform=ax.transAxes)
    ax.add_patch(s_circle)
    ax.text(0.32, 0.65, 'S', ha='center', va='center', fontsize=8, color='white',
            fontweight='bold', transform=ax.transAxes)
    
    ax.annotate('', xy=(0.47, 0.65), xytext=(0.40, 0.65),
               arrowprops=dict(arrowstyle='->', lw=1, color='black'),
               xycoords='axes fraction', textcoords='axes fraction')
    
    p_circle = Circle((0.54, 0.65), 0.06, facecolor='#2ECC71', edgecolor='black',
                      linewidth=2, transform=ax.transAxes)
    ax.add_patch(p_circle)
    ax.text(0.54, 0.65, 'P', ha='center', va='center', fontsize=8, color='white',
            fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.35, 0.52, 'E unchanged', ha='center', va='center', fontsize=8,
            style='italic', transform=ax.transAxes)
    
    # Autocatalysis
    ax.text(0.25, 0.35, 'Autocatalysis:', ha='center', va='center',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    m_circle = Circle((0.15, 0.2), 0.08, facecolor='#9B59B6', edgecolor='black',
                      linewidth=2, transform=ax.transAxes)
    ax.add_patch(m_circle)
    ax.text(0.15, 0.2, 'M', ha='center', va='center', fontsize=10, color='white',
            fontweight='bold', transform=ax.transAxes)
    
    ax.annotate('', xy=(0.35, 0.2), xytext=(0.25, 0.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple'),
               xycoords='axes fraction', textcoords='axes fraction')
    ax.text(0.30, 0.25, 'e-', ha='center', va='bottom', fontsize=8, 
            color='blue', transform=ax.transAxes)
    
    m2_circle = Circle((0.45, 0.2), 0.08, facecolor='#8E44AD', edgecolor='black',
                       linewidth=2, transform=ax.transAxes)
    ax.add_patch(m2_circle)
    ax.text(0.45, 0.2, "M'", ha='center', va='center', fontsize=10, color='white',
            fontweight='bold', transform=ax.transAxes)
    
    # Feedback arrow
    ax.annotate('', xy=(0.18, 0.12), xytext=(0.42, 0.12),
               arrowprops=dict(arrowstyle='->', lw=2, color='green', 
                             connectionstyle='arc3,rad=-0.3'),
               xycoords='axes fraction', textcoords='axes fraction')
    ax.text(0.30, 0.05, 'enables', ha='center', va='center', fontsize=8,
            color='green', transform=ax.transAxes)
    
    # Panel B: Bistability
    ax = axes[0, 1]
    A_star = np.linspace(0, 1, 100)
    
    # Rate of change for autocatalytic system
    k_et = 0.5
    k_back = 0.3
    k_auto = 0.8
    D = 0.5  # donor concentration
    A = 1 - A_star  # total - activated
    
    dA_dt = k_et * A * D - k_back * A_star + k_auto * A_star * A
    
    ax.plot(A_star, dA_dt, 'b-', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.fill_between(A_star, dA_dt, where=dA_dt > 0, alpha=0.3, color='green')
    ax.fill_between(A_star, dA_dt, where=dA_dt < 0, alpha=0.3, color='red')
    
    # Mark stable points
    ax.plot([0.1], [0], 'go', markersize=12, label='Inactive (stable)')
    ax.plot([0.7], [0], 'g^', markersize=12, label='Active (stable)')
    
    ax.set_xlabel('[A*] (activated fraction)', fontsize=10)
    ax.set_ylabel('d[A*]/dt', fontsize=10)
    ax.set_title('(B) Bistability', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel C: Iron-sulfur cluster
    ax = axes[0, 2]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(C) FeS Cluster (Primordial)', fontsize=11, fontweight='bold')
    
    # Fe atoms
    fe_positions = [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
    for x, y in fe_positions:
        fe = Circle((x, y), 0.3, facecolor='#E74C3C', edgecolor='black', linewidth=2)
        ax.add_patch(fe)
        ax.text(x, y, 'Fe', ha='center', va='center', fontsize=8, 
                color='white', fontweight='bold')
    
    # S atoms
    s_positions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for x, y in s_positions:
        s = Circle((x, y), 0.25, facecolor='#F1C40F', edgecolor='black', linewidth=2)
        ax.add_patch(s)
        ax.text(x, y, 'S', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Electron pathway
    ax.annotate('', xy=(0.5, 0.2), xytext=(-0.5, 0.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue', 
                             connectionstyle='arc3,rad=0.3'))
    ax.text(0, 0, 'e-', ha='center', va='center', fontsize=10, color='blue')
    
    ax.text(0, -1.7, 'Geochemically abundant\nAutocatalytic ET', ha='center',
            va='center', fontsize=9, style='italic')
    
    # Panel D: Environmental coupling
    ax = axes[1, 0]
    # Venn diagram style
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(D) Environmental Coupling', fontsize=11, fontweight='bold')
    
    # System states
    system = Circle((-0.4, 0), 1, facecolor='#3498DB', edgecolor='black',
                   linewidth=2, alpha=0.5)
    ax.add_patch(system)
    ax.text(-0.9, 0, 'System\nStates', ha='center', va='center', fontsize=9)
    
    # Environment permitted
    env = Circle((0.4, 0), 1, facecolor='#2ECC71', edgecolor='black',
                linewidth=2, alpha=0.5)
    ax.add_patch(env)
    ax.text(0.9, 0, 'Env.\nPermitted', ha='center', va='center', fontsize=9)
    
    # Intersection
    ax.text(0, 0, 'Accessible\nStates', ha='center', va='center', fontsize=9,
            fontweight='bold')
    
    ax.text(0, -1.7, 'Self-knowledge = Environmental knowledge', ha='center',
            va='center', fontsize=9, style='italic')
    
    # Panel E: Minimal requirements
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('(E) Minimal Requirements', fontsize=11, fontweight='bold')
    
    requirements = [
        ('1. Electron Donor', '#E74C3C'),
        ('2. Electron Acceptor', '#3498DB'),
        ('3. Coupling Pathway', '#F39C12'),
        ('4. Regeneration', '#2ECC71')
    ]
    
    for i, (req, color) in enumerate(requirements):
        y = 0.8 - i * 0.2
        rect = Rectangle((0.1, y - 0.05), 0.8, 0.12, facecolor=color,
                         edgecolor='black', linewidth=2, alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, y, req, ha='center', va='center', fontsize=10,
                fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.05, 'All 4 necessary and sufficient', ha='center',
            va='center', fontsize=9, style='italic', transform=ax.transAxes)
    
    # Panel F: Self-reference loop
    ax = axes[1, 2]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(F) Self-Reference Loop', fontsize=11, fontweight='bold')
    
    # Circular arrow
    theta = np.linspace(0, 1.8*np.pi, 100)
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'b-', linewidth=3)
    
    # Arrow head
    ax.annotate('', xy=(r*np.cos(1.8*np.pi), r*np.sin(1.8*np.pi)),
               xytext=(r*np.cos(1.7*np.pi), r*np.sin(1.7*np.pi)),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # Labels on the loop
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    labels = ['ET', "M'", 'enables', 'ET']
    for angle, label in zip(angles, labels):
        x_pos = 1.1 * np.cos(angle)
        y_pos = 1.1 * np.sin(angle)
        ax.text(x_pos, y_pos, label, ha='center', va='center', fontsize=10,
                fontweight='bold')
    
    ax.text(0, 0, 'CLOSED\nLOOP', ha='center', va='center', fontsize=12,
            fontweight='bold', color='purple')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'autocatalytic_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'autocatalytic_panel.png'}")


def generate_categorical_oscillation_panel():
    """Panel 7: Categorical oscillation mathematical foundation."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Categorical Oscillation: Mathematical Foundation", fontsize=14, fontweight='bold')
    
    # Panel A: The three axioms
    ax = axes[0, 0]
    ax.axis('off')
    ax.set_title('(A) Three Axioms', fontsize=11, fontweight='bold')
    
    axioms = [
        ('1. Partitioning', r'$C_n = \sum_i c_{n,i}$', '#E74C3C'),
        ('2. Traversal', r'$C_n \to c_{n,1} \to c_{n,2} \to \cdots$', '#3498DB'),
        ('3. Recursion', r'$C_{n+1} = f(C_n, \mathcal{H}_n)$', '#2ECC71')
    ]
    
    for i, (name, eq, color) in enumerate(axioms):
        y = 0.8 - i * 0.28
        rect = Rectangle((0.05, y - 0.08), 0.9, 0.2, facecolor=color,
                         edgecolor='black', linewidth=2, alpha=0.3, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, y + 0.05, name, ha='center', va='center', fontsize=11,
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, y - 0.03, eq, ha='center', va='center', fontsize=10,
                transform=ax.transAxes)
    
    # Panel B: Oscillation emergence
    ax = axes[0, 1]
    ax.axis('off')
    ax.set_title('(B) Oscillation Emergence', fontsize=11, fontweight='bold')
    
    # Show C_n ≈ C_{n+1} but C_n ≠ C_{n+1}
    ax.text(0.5, 0.85, 'Key Insight:', ha='center', va='center', fontsize=11,
            fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.65, r'$C_{n+1} \approx C_n$', ha='center', va='center', 
            fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.55, '(similar structure)', ha='center', va='center',
            fontsize=10, style='italic', color='gray', transform=ax.transAxes)
    
    ax.text(0.5, 0.4, 'BUT', ha='center', va='center', fontsize=12,
            fontweight='bold', color='red', transform=ax.transAxes)
    
    ax.text(0.5, 0.25, r'$C_{n+1} \neq C_n$', ha='center', va='center',
            fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.15, '(different categorical state)', ha='center', va='center',
            fontsize=10, style='italic', color='gray', transform=ax.transAxes)
    
    ax.text(0.5, 0.02, 'Due to partition history!', ha='center', va='center',
            fontsize=10, fontweight='bold', color='purple', transform=ax.transAxes)
    
    # Panel C: Partition sequence visualization
    ax = axes[0, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(C) Partition Sequence', fontsize=11, fontweight='bold')
    
    # C_0 = 4
    ax.text(1, 8.5, r'$C_0 = 4$', ha='center', va='center', fontsize=10)
    for i in range(4):
        circle = Circle((0.5 + i*0.5, 7.5), 0.2, facecolor='#3498DB', 
                        edgecolor='black', linewidth=1)
        ax.add_patch(circle)
    
    # Arrow and partition
    ax.annotate('', xy=(5, 7.5), xytext=(3, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(4, 7.8, 'partition', ha='center', va='bottom', fontsize=8)
    
    # {2, 1, 1}
    ax.text(7, 8.5, r'$\{2,1,1\}$', ha='center', va='center', fontsize=10)
    # 2
    for i in range(2):
        circle = Circle((5.5 + i*0.4, 7.5), 0.18, facecolor='#E74C3C',
                        edgecolor='black', linewidth=1)
        ax.add_patch(circle)
    # 1
    circle = Circle((6.8, 7.5), 0.18, facecolor='#F39C12', edgecolor='black', linewidth=1)
    ax.add_patch(circle)
    # 1
    circle = Circle((7.5, 7.5), 0.18, facecolor='#2ECC71', edgecolor='black', linewidth=1)
    ax.add_patch(circle)
    
    # Arrow down
    ax.annotate('', xy=(5, 5.5), xytext=(5, 6.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.3, 6, 'return', ha='left', va='center', fontsize=8)
    
    # C_1 = 4 (same total, different history)
    ax.text(5, 5, r'$C_1 = 4$', ha='center', va='center', fontsize=10)
    ax.text(5, 4.3, r'$\mathcal{H}_1 = \{2,1,1\}$', ha='center', va='center', 
            fontsize=9, color='purple')
    
    # Arrow down
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.3, 3, 'partition', ha='left', va='center', fontsize=8)
    
    # Different partition
    ax.text(5, 2, r'$C_2 = 4$', ha='center', va='center', fontsize=10)
    ax.text(5, 1.3, r'$\mathcal{H}_2 = \{2,1,1\}, \{3,1\}$', ha='center', va='center',
            fontsize=9, color='purple')
    
    ax.text(5, 0.3, 'Same total, different history = oscillation', ha='center',
            va='center', fontsize=9, style='italic')
    
    # Panel D: Electron transport as categorical oscillation
    ax = axes[1, 0]
    ax.axis('off')
    ax.set_title('(D) ET as Categorical Oscillation', fontsize=11, fontweight='bold')
    
    mappings = [
        ('Charge separation', 'Partitioning'),
        ('Electron movement', 'Traversal'),
        ('Field enables more ET', 'Recursion'),
        ('Same Q, different history', 'Oscillation')
    ]
    
    for i, (et, cat) in enumerate(mappings):
        y = 0.85 - i * 0.2
        ax.text(0.25, y, et, ha='center', va='center', fontsize=9,
                transform=ax.transAxes)
        ax.text(0.5, y, r'$\Leftrightarrow$', ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
        ax.text(0.75, y, cat, ha='center', va='center', fontsize=9,
                fontweight='bold', color='#3498DB', transform=ax.transAxes)
    
    ax.text(0.5, 0.05, 'ET instantiates categorical oscillation', ha='center',
            va='center', fontsize=10, style='italic', transform=ax.transAxes)
    
    # Panel E: Autocatalysis as self-reference
    ax = axes[1, 1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(E) Autocatalysis = Self-Reference', fontsize=11, fontweight='bold')
    
    # Spiral showing enhancement
    theta = np.linspace(0, 4*np.pi, 200)
    r = 0.3 + 0.15 * theta / np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax.plot(x, y, 'b-', linewidth=2)
    ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-5], y[-5]),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Labels
    ax.text(0, 0, 'Start', ha='center', va='center', fontsize=8)
    ax.text(0.8, 0.8, 'Growing\npartition\nhistory', ha='center', va='center', fontsize=8)
    
    ax.text(0, -1.3, r'$P(\text{ET}|\mathcal{H}_n) > P(\text{ET}|\mathcal{H}_0)$', 
            ha='center', va='center', fontsize=10)
    
    # Panel F: Time as derivative
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('(F) Time Emerges from Sequence', fontsize=11, fontweight='bold')
    
    # Sequence
    sequence = [r'$C_0$', r'$C_1$', r'$C_2$', r'$\cdots$', r'$C_n$']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sequence)))
    
    for i, (c, color) in enumerate(zip(sequence, colors)):
        x = 0.1 + i * 0.18
        circle = Circle((x, 0.6), 0.06, facecolor=color, edgecolor='black',
                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x, 0.5, c, ha='center', va='center', fontsize=10, transform=ax.transAxes)
        if i < len(sequence) - 1:
            ax.annotate('', xy=(x + 0.1, 0.6), xytext=(x + 0.07, 0.6),
                       arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                       xycoords='axes fraction', textcoords='axes fraction')
    
    ax.text(0.5, 0.35, r'Index $n$ = emergent time', ha='center', va='center',
            fontsize=11, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.2, r'$\mathcal{H}_{n-1} \subset \mathcal{H}_n$', ha='center',
            va='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.1, 'defines temporal order', ha='center', va='center',
            fontsize=10, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'categorical_oscillation_panel.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'categorical_oscillation_panel.png'}")


def main():
    """Generate all panel charts."""
    print("Generating Origins of Life panel charts...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    generate_categorical_oscillation_panel()
    generate_orgels_paradox_panel()
    generate_electron_transport_panel()
    generate_homochirality_panel()
    generate_membrane_scaffolding_panel()
    generate_semiconductor_origins_panel()
    generate_autocatalytic_panel()
    
    print("\nAll panels generated successfully!")


if __name__ == "__main__":
    main()

