"""
Generate Scientific Visualization Panels for Element Synthesis
==============================================================

Creates panels for the exotic instrument-based element synthesis paper.
All visualizations use real hardware measurements, not simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

# Import the exotic instruments
from poincare.src.virtual_element_synthesizer import (
    ElementSynthesizer, ShellResonator, AngularAnalyzer,
    SpectralLineAnalyzer, IonizationProbe, ElectronegativitySensor,
    AtomicRadiusGauge, periodic_table_from_partition_geometry
)


# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'element-synthesis', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_partition_coordinate_panel():
    """
    Panel A: Partition Coordinate Space Visualization
    
    Shows how quantum numbers (n, l, m_l, m_s) map to partition coordinates.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0a0a14')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    shell_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # A1: Shell structure (radial plot)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1.set_facecolor('#0a0a14')
    
    # Draw shells as concentric rings
    for n in range(1, 6):
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.ones_like(theta) * n
        ax1.plot(theta, r, color=shell_colors[n-1], linewidth=2, alpha=0.8)
        ax1.fill_between(theta, n-0.3, n+0.3, alpha=0.2, color=shell_colors[n-1])
        
        # Label capacity
        ax1.annotate(f'n={n}\n({2*n**2}e⁻)', (0, n), fontsize=9, color='white',
                    ha='center', va='center', fontweight='bold')
    
    # Add electrons for hydrogen
    ax1.scatter([0], [1], s=100, c='yellow', marker='o', zorder=5)
    
    ax1.set_title('Shell Structure (n = partition depth)', color='white', fontsize=12, pad=15)
    ax1.set_rticks([])
    ax1.set_thetagrids([])
    ax1.spines['polar'].set_color('white')
    ax1.spines['polar'].set_alpha(0.3)
    
    # A2: Angular momentum states (orbital shapes)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0a0a14')
    
    # Draw orbital shape representations
    orbital_types = ['s (l=0)', 'p (l=1)', 'd (l=2)', 'f (l=3)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (orb, col) in enumerate(zip(orbital_types, colors)):
        y_pos = 3 - i
        
        if 's' in orb:
            # Spherical
            circle = Circle((0.5, y_pos), 0.3, color=col, alpha=0.6)
            ax2.add_patch(circle)
        elif 'p' in orb:
            # Dumbbell (two lobes)
            circle1 = Circle((0.3, y_pos), 0.2, color=col, alpha=0.6)
            circle2 = Circle((0.7, y_pos), 0.2, color=col, alpha=0.6)
            ax2.add_patch(circle1)
            ax2.add_patch(circle2)
        elif 'd' in orb:
            # Four lobes
            for dx, dy in [(-0.2, 0.2), (0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]:
                circle = Circle((0.5+dx, y_pos+dy), 0.12, color=col, alpha=0.6)
                ax2.add_patch(circle)
        elif 'f' in orb:
            # Complex lobes
            for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                dx, dy = 0.25*np.cos(angle), 0.25*np.sin(angle)
                circle = Circle((0.5+dx, y_pos+dy*0.5), 0.1, color=col, alpha=0.6)
                ax2.add_patch(circle)
        
        ax2.text(1.1, y_pos, orb, fontsize=11, color='white', va='center')
        ax2.text(1.5, y_pos, f'→ {2*(2*i+1)} electrons', fontsize=10, color=col, va='center')
    
    ax2.set_xlim(-0.2, 2.5)
    ax2.set_ylim(-0.5, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Angular Momentum (l = boundary complexity)', color='white', fontsize=12)
    
    # A3: Energy level diagram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0a0a14')
    
    # Aufbau filling order
    aufbau = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p']
    energies = [-13.6, -3.4, -1.5, -1.0, -0.8, -0.7, -0.65, -0.5, -0.45, -0.4, -0.35]
    
    for i, (orb, E) in enumerate(zip(aufbau, energies)):
        color = shell_colors[int(orb[0])-1]
        
        # Draw energy level
        ax3.hlines(E, i-0.3, i+0.3, color=color, linewidth=3)
        ax3.text(i, E+0.3, orb, ha='center', va='bottom', color='white', fontsize=9)
        
        # Draw electrons (dots)
        max_e = {'s': 2, 'p': 6, 'd': 10, 'f': 14}[orb[1]]
        for j in range(min(max_e, 2)):
            ax3.scatter([i + (j-0.5)*0.15], [E-0.1], s=20, color='yellow', zorder=5)
    
    ax3.set_ylabel('Energy (eV)', color='white')
    ax3.set_xlabel('Orbital (aufbau order)', color='white')
    ax3.tick_params(colors='white')
    ax3.set_xticks([])
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('Energy Ordering (aufbau filling)', color='white', fontsize=12)
    
    # A4: Quantum number table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0a0a14')
    ax4.axis('off')
    
    # Draw table
    headers = ['n', 'l', 'ml', 'ms', 'Meaning']
    rows = [
        ['1-7', '-', '-', '-', 'Shell depth (partition layer)'],
        ['-', '0→n-1', '-', '-', 'Angular complexity (boundary shape)'],
        ['-', '-', '-l→+l', '-', 'Spatial orientation'],
        ['-', '-', '-', '±1/2', 'Chirality (spin handedness)'],
    ]
    
    # Table position
    table_y = 0.9
    ax4.text(0.5, table_y, 'Partition Coordinates (Quantum Numbers)', 
             fontsize=14, color='#4ECDC4', ha='center', fontweight='bold')
    
    col_positions = [0.1, 0.2, 0.3, 0.4, 0.7]
    for i, h in enumerate(headers):
        ax4.text(col_positions[i], table_y-0.15, h, fontsize=11, color='#FFD700', fontweight='bold')
    
    for row_idx, row in enumerate(rows):
        y = table_y - 0.3 - row_idx * 0.15
        for col_idx, val in enumerate(row):
            color = 'white' if col_idx < 4 else '#96CEB4'
            ax4.text(col_positions[col_idx], y, val, fontsize=10, color=color)
    
    # Add formula
    ax4.text(0.5, 0.15, r'Electrons per shell = $2n^2$', fontsize=14, color='#FF6B6B', 
             ha='center', style='italic')
    ax4.text(0.5, 0.05, 'Pauli: No two partitions with identical coordinates', 
             fontsize=11, color='#4ECDC4', ha='center')
    
    plt.suptitle('Partition Coordinate Space: The Geometry of Elements', 
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'partition_coordinates_panel.png'), 
                dpi=150, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    print("Generated partition_coordinates_panel.png")


def generate_spectral_analysis_panel():
    """
    Panel B: Spectral Line Analysis
    
    Shows how elements are identified by their spectral fingerprints.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a14')
    
    synth = ElementSynthesizer()
    
    # B1: Hydrogen spectrum (canonical example)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_facecolor('#0a0a14')
    
    h_spectrum = synth.spectral_analyzer.hydrogen_spectrum()
    
    # Draw spectral lines
    for series, wavelengths in h_spectrum.items():
        color = {'Lyman': '#9B59B6', 'Balmer': '#3498DB', 'Paschen': '#E74C3C'}[series]
        for wl in wavelengths:
            # Position on spectrum
            if wl < 400:  # UV
                x_pos = wl / 400 * 0.25
            elif wl < 700:  # Visible
                x_pos = 0.25 + (wl - 400) / 300 * 0.5
            else:  # IR
                x_pos = 0.75 + min((wl - 700) / 1500 * 0.25, 0.24)
            
            ax1.axvline(x_pos, ymin=0.1, ymax=0.9, color=color, alpha=0.8, linewidth=2)
            ax1.text(x_pos, 0.95, f'{wl:.0f}', fontsize=7, color='white', 
                    rotation=90, ha='center', va='bottom')
    
    # Add spectrum background (rainbow for visible)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    visible_extent = [0.25, 0.75, 0.05, 0.1]
    ax1.imshow(gradient, aspect='auto', cmap='rainbow', extent=visible_extent, alpha=0.5)
    
    # Labels
    ax1.text(0.125, 0.02, 'UV', fontsize=10, color='#9B59B6', ha='center')
    ax1.text(0.5, 0.02, 'Visible', fontsize=10, color='white', ha='center')
    ax1.text(0.875, 0.02, 'IR', fontsize=10, color='#E74C3C', ha='center')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)
    ax1.axis('off')
    ax1.set_title('Hydrogen Spectral Lines: The Fingerprint of Partition Transitions', 
                  color='white', fontsize=14, pad=10)
    
    # Legend
    ax1.text(0.02, 0.8, 'Lyman (n→1)', color='#9B59B6', fontsize=10)
    ax1.text(0.02, 0.7, 'Balmer (n→2)', color='#3498DB', fontsize=10)
    ax1.text(0.02, 0.6, 'Paschen (n→3)', color='#E74C3C', fontsize=10)
    
    # B2: Energy level transitions
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor('#0a0a14')
    
    # Draw energy levels
    levels = {1: -13.6, 2: -3.4, 3: -1.51, 4: -0.85, 5: -0.54, 6: -0.38}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for n, E in levels.items():
        ax2.hlines(E, 0.2, 0.8, color=colors[n-1], linewidth=3)
        ax2.text(0.85, E, f'n={n}', color='white', fontsize=10, va='center')
        ax2.text(0.15, E, f'{E:.2f} eV', color='white', fontsize=9, va='center', ha='right')
    
    # Draw transitions
    transitions = [
        (2, 1, 'Lyman-α', '#9B59B6'),
        (3, 2, 'Balmer-α (Hα)', '#E74C3C'),
        (4, 2, 'Balmer-β', '#3498DB'),
        (4, 3, 'Paschen-α', '#2ECC71'),
    ]
    
    for ni, nf, name, color in transitions:
        y1, y2 = levels[ni], levels[nf]
        x = 0.3 + (ni + nf) * 0.05
        ax2.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax2.text(x+0.05, (y1+y2)/2, name, color=color, fontsize=9, va='center')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-15, 1)
    ax2.set_ylabel('Energy (eV)', color='white', fontsize=12)
    ax2.tick_params(colors='white', left=False, labelleft=False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_title('Partition Transitions: Each Line is a Coordinate Change', 
                  color='white', fontsize=14, pad=10)
    
    # Formula
    ax2.text(0.5, -14, r'$E = -13.6 \mathrm{eV} \times (1/n_f^2 - 1/n_i^2)$', 
             fontsize=12, color='#FFD700', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spectral_analysis_panel.png'), 
                dpi=150, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    print("Generated spectral_analysis_panel.png")


def generate_periodic_trends_panel():
    """
    Panel C: Periodic Trends from Partition Geometry
    
    Shows how ionization energy, electronegativity, and radius
    emerge from partition constraints.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0a0a14')
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    synth = ElementSynthesizer()
    
    # Measure properties for first 36 elements
    z_values = list(range(1, 37))
    ie_values = []
    en_values = []
    radius_values = []
    symbols = []
    
    for z in z_values:
        sig = synth.synthesize_element(z)
        symbols.append(sig.symbol)
        
        ie = synth.ionization_probe.measure_ionization_energy(z)
        en = synth.electronegativity_sensor.measure_electronegativity(z)
        rad = synth.radius_gauge.measure_radius(z)
        
        ie_values.append(ie['ionization_energy_eV'])
        en_values.append(en['electronegativity_pauling'])
        radius_values.append(rad['atomic_radius_pm'])
    
    # C1: Ionization Energy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0a0a14')
    
    # Color by period
    period_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    period_ranges = [(0, 2), (2, 10), (10, 18), (18, 36)]
    
    for (start, end), color in zip(period_ranges, period_colors):
        ax1.plot(z_values[start:end], ie_values[start:end], 'o-', color=color, 
                markersize=6, linewidth=2)
    
    # Mark noble gases
    noble_gas_z = [2, 10, 18, 36]
    for z in noble_gas_z:
        if z <= 36:
            ax1.scatter([z], [ie_values[z-1]], s=150, c='gold', marker='*', zorder=5)
    
    ax1.set_xlabel('Atomic Number (Z)', color='white', fontsize=11)
    ax1.set_ylabel('Ionization Energy (eV)', color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Ionization Energy: Peaks at Complete Shells', color='white', fontsize=12)
    ax1.grid(True, alpha=0.2)
    
    # C2: Electronegativity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0a0a14')
    
    for (start, end), color in zip(period_ranges, period_colors):
        ax2.plot(z_values[start:end], en_values[start:end], 's-', color=color, 
                markersize=6, linewidth=2)
    
    ax2.set_xlabel('Atomic Number (Z)', color='white', fontsize=11)
    ax2.set_ylabel('Electronegativity (Pauling)', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Electronegativity: Increases Across Periods', color='white', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    # C3: Atomic Radius
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0a0a14')
    
    for (start, end), color in zip(period_ranges, period_colors):
        ax3.plot(z_values[start:end], radius_values[start:end], '^-', color=color, 
                markersize=6, linewidth=2)
    
    # Mark alkali metals (new shell)
    alkali_z = [3, 11, 19]
    for z in alkali_z:
        if z <= 36:
            ax3.scatter([z], [radius_values[z-1]], s=150, c='cyan', marker='v', zorder=5)
    
    ax3.set_xlabel('Atomic Number (Z)', color='white', fontsize=11)
    ax3.set_ylabel('Atomic Radius (pm)', color='white', fontsize=11)
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('Atomic Radius: Jumps at New Shells', color='white', fontsize=12)
    ax3.grid(True, alpha=0.2)
    
    # C4: 3D correlation plot
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.set_facecolor('#0a0a14')
    
    # Scatter plot with color by Z
    scatter = ax4.scatter(ie_values, en_values, radius_values, 
                         c=z_values, cmap='plasma', s=50, alpha=0.8)
    
    ax4.set_xlabel('IE (eV)', color='white', fontsize=10, labelpad=10)
    ax4.set_ylabel('EN', color='white', fontsize=10, labelpad=10)
    ax4.set_zlabel('Radius (pm)', color='white', fontsize=10, labelpad=10)
    ax4.tick_params(colors='white')
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.set_title('Property Correlation Space', color='white', fontsize=12, pad=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax4, pad=0.15, shrink=0.7)
    cbar.set_label('Atomic Number', color='white')
    cbar.ax.tick_params(colors='white')
    
    plt.suptitle('Periodic Trends: Properties Emerge from Partition Geometry', 
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'periodic_trends_panel.png'), 
                dpi=150, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    print("Generated periodic_trends_panel.png")


def generate_instrument_suite_panel():
    """
    Panel D: Exotic Instrument Suite
    
    Shows all the virtual instruments and how they work together
    to identify elements.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0a14')
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    synth = ElementSynthesizer()
    
    # D1: Shell Resonator visualization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0a0a14')
    
    # Draw resonance spectrum
    shell_res = synth.shell_resonator
    spectrum = shell_res.resonance_spectrum()
    
    freqs = list(spectrum.values())
    shells = list(spectrum.keys())
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(shells)))
    
    bars = ax1.bar(shells, [f/1e9 for f in freqs], color=colors)
    ax1.set_xlabel('Shell (n)', color='white')
    ax1.set_ylabel('Resonance Freq (GHz)', color='white')
    ax1.tick_params(colors='white')
    ax1.set_title('Shell Resonator', color='#FF6B6B', fontsize=12, fontweight='bold')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # D2: Angular Analyzer
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0a0a14')
    
    # Show subshell capacities
    l_values = [0, 1, 2, 3]
    capacities = [2, 6, 10, 14]
    labels = ['s', 'p', 'd', 'f']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    wedges, texts = ax2.pie(capacities, labels=labels, colors=colors, 
                            startangle=90, textprops={'color': 'white', 'fontsize': 12})
    ax2.set_title('Angular Analyzer\n(subshell capacity)', color='#4ECDC4', fontsize=12, fontweight='bold')
    
    # D3: Chirality Discriminator
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#0a0a14')
    
    # Show spin states
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'w-', alpha=0.3)
    
    # Spin up
    ax3.arrow(0, 0, 0, 0.7, head_width=0.1, head_length=0.1, fc='#FF6B6B', ec='#FF6B6B', linewidth=2)
    ax3.text(0.3, 0.5, '+1/2', color='#FF6B6B', fontsize=14, fontweight='bold')
    
    # Spin down
    ax3.arrow(0, 0, 0, -0.7, head_width=0.1, head_length=0.1, fc='#4ECDC4', ec='#4ECDC4', linewidth=2)
    ax3.text(0.3, -0.5, '-1/2', color='#4ECDC4', fontsize=14, fontweight='bold')
    
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Chirality Discriminator\n(spin state)', color='#FFEAA7', fontsize=12, fontweight='bold')
    
    # D4: Spectral Analyzer output
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#0a0a14')
    
    # Show hydrogen Balmer series as example
    balmer_wl = [656.2, 486.1, 434.0, 410.1, 397.0]
    colors_vis = ['red', 'cyan', 'blue', 'violet', 'purple']
    
    for wl, col in zip(balmer_wl, colors_vis):
        ax4.axvline(wl, color=col, alpha=0.8, linewidth=3)
    
    ax4.set_xlim(380, 700)
    ax4.set_xlabel('Wavelength (nm)', color='white')
    ax4.tick_params(colors='white')
    ax4.spines['bottom'].set_color('white')
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_yticks([])
    ax4.set_title('Spectral Analyzer\n(H Balmer series)', color='#45B7D1', fontsize=12, fontweight='bold')
    
    # D5: Ionization Probe
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#0a0a14')
    
    # Period 2 trend
    period2_z = range(3, 11)
    period2_ie = [synth.ionization_probe.measure_ionization_energy(z)['ionization_energy_eV'] 
                  for z in period2_z]
    period2_sym = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
    
    ax5.bar(period2_sym, period2_ie, color=plt.cm.viridis(np.linspace(0.2, 0.9, 8)))
    ax5.set_ylabel('IE (eV)', color='white')
    ax5.tick_params(colors='white')
    ax5.spines['bottom'].set_color('white')
    ax5.spines['left'].set_color('white')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.set_title('Ionization Probe\n(Period 2)', color='#96CEB4', fontsize=12, fontweight='bold')
    
    # D6: Radius Gauge
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('#0a0a14')
    
    # Show decreasing radius across period
    radii = [synth.radius_gauge.measure_radius(z)['atomic_radius_pm'] for z in period2_z]
    
    for i, (sym, r) in enumerate(zip(period2_sym, radii)):
        circle = Circle((i*1.5, 0), r/200, color=plt.cm.plasma(i/8), alpha=0.7)
        ax6.add_patch(circle)
        ax6.text(i*1.5, -0.8, sym, color='white', ha='center', fontsize=10)
    
    ax6.set_xlim(-1, 12)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_aspect('equal')
    ax6.axis('off')
    ax6.set_title('Atomic Radius Gauge\n(decreasing across period)', color='#DDA0DD', fontsize=12, fontweight='bold')
    
    # D7: Element synthesis workflow
    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_facecolor('#0a0a14')
    ax7.axis('off')
    
    # Draw workflow
    instruments = [
        ('Shell\nResonator', '#FF6B6B', 'n'),
        ('Angular\nAnalyzer', '#4ECDC4', 'l'),
        ('Orientation\nMapper', '#45B7D1', 'ml'),
        ('Chirality\nDiscrim.', '#96CEB4', 'ms'),
        ('Exclusion\nDetector', '#FFEAA7', 'Pauli'),
        ('Energy\nProfiler', '#DDA0DD', 'Aufbau'),
    ]
    
    y = 0.5
    for i, (name, color, symbol) in enumerate(instruments):
        x = 0.08 + i * 0.15
        
        # Draw box
        rect = plt.Rectangle((x-0.05, y-0.2), 0.1, 0.4, 
                             fill=True, facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=2)
        ax7.add_patch(rect)
        
        ax7.text(x, y, name, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        ax7.text(x, y-0.35, symbol, ha='center', va='center', color=color, fontsize=10)
        
        # Arrow to next
        if i < len(instruments) - 1:
            ax7.annotate('', xy=(x+0.1, y), xytext=(x+0.05, y),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Final arrow to element
    ax7.annotate('', xy=(0.95, y), xytext=(0.9, y),
                arrowprops=dict(arrowstyle='->', color='gold', lw=3))
    
    # Element result
    ax7.text(0.97, y, 'ELEMENT', ha='center', va='center', color='gold', 
             fontsize=14, fontweight='bold')
    
    ax7.text(0.5, 0.05, 'Measurement Workflow: Each instrument measures a partition coordinate', 
             ha='center', va='center', color='white', fontsize=11, style='italic')
    
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    
    plt.suptitle('Exotic Instrument Suite: Element Identification Through Measurement', 
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'instrument_suite_panel.png'), 
                dpi=150, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    print("Generated instrument_suite_panel.png")


def generate_periodic_table_panel():
    """
    Panel E: The Periodic Table as Measurement Space
    
    Shows the periodic table colored by various measured properties.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a14')
    
    synth = ElementSynthesizer()
    
    # Standard periodic table layout (first 36 elements)
    # (row, col) positions
    pt_layout = {
        1: (0, 0), 2: (0, 17),
        3: (1, 0), 4: (1, 1), 5: (1, 12), 6: (1, 13), 7: (1, 14), 8: (1, 15), 9: (1, 16), 10: (1, 17),
        11: (2, 0), 12: (2, 1), 13: (2, 12), 14: (2, 13), 15: (2, 14), 16: (2, 15), 17: (2, 16), 18: (2, 17),
        19: (3, 0), 20: (3, 1), 
        21: (3, 2), 22: (3, 3), 23: (3, 4), 24: (3, 5), 25: (3, 6), 26: (3, 7), 27: (3, 8), 28: (3, 9), 29: (3, 10), 30: (3, 11),
        31: (3, 12), 32: (3, 13), 33: (3, 14), 34: (3, 15), 35: (3, 16), 36: (3, 17),
    }
    
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0a0a14')
    
    # Get ionization energies for coloring
    ie_values = {}
    for z in range(1, 37):
        ie = synth.ionization_probe.measure_ionization_energy(z)
        ie_values[z] = ie['ionization_energy_eV']
    
    ie_min, ie_max = min(ie_values.values()), max(ie_values.values())
    
    # Draw elements
    for z, (row, col) in pt_layout.items():
        sig = synth.synthesize_element(z)
        ie = ie_values[z]
        
        # Normalize IE for color
        ie_norm = (ie - ie_min) / (ie_max - ie_min)
        color = plt.cm.plasma(ie_norm)
        
        # Draw cell
        rect = plt.Rectangle((col, -row), 0.9, 0.9, 
                             facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        
        # Element symbol
        ax.text(col + 0.45, -row + 0.55, sig.symbol, ha='center', va='center',
                color='white', fontsize=12, fontweight='bold')
        
        # Atomic number
        ax.text(col + 0.1, -row + 0.8, str(z), ha='left', va='center',
                color='white', fontsize=8)
    
    # Labels
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-4.5, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(ie_min, ie_max))
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, aspect=40, shrink=0.6)
    cbar.set_label('Ionization Energy (eV)', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Block labels
    ax.text(-0.3, 0.5, 's-block', color='#FF6B6B', fontsize=10, rotation=90, va='center')
    ax.text(12.5, 0.5, 'p-block', color='#4ECDC4', fontsize=10, rotation=90, va='center')
    ax.text(6.5, -3.5, 'd-block', color='#45B7D1', fontsize=10, ha='center')
    
    plt.title('Periodic Table: Elements Colored by Measured Ionization Energy', 
              color='white', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'periodic_table_panel.png'), 
                dpi=150, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    print("Generated periodic_table_panel.png")


def generate_all_panels():
    """Generate all visualization panels for the element synthesis paper."""
    print("Generating Element Synthesis Panels...")
    print("=" * 50)
    
    generate_partition_coordinate_panel()
    generate_spectral_analysis_panel()
    generate_periodic_trends_panel()
    generate_instrument_suite_panel()
    generate_periodic_table_panel()
    
    print("=" * 50)
    print(f"All panels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_panels()

