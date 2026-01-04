"""
Hardware Oscillation Signature Analysis
2-panel visualization of multi-scale oscillatory signatures
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

# ============================================================================
# HARDWARE OSCILLATION DATA
# ============================================================================
if __name__ == "__main__":

    # Individual oscillation sources
    oscillations = {
        'CPU Frequency': {
            'frequency': 10.00,  # Hz
            'amplitude': 7.15e7,
            'phase': -1.537,  # rad
            'damping': 1.000,
            'symmetry': 0.010,
            'color': '#e74c3c',
        },
        'Thermal': {
            'frequency': 0.100,  # Hz
            'amplitude': 3.559,
            'phase': -1.564,  # rad
            'damping': 0.159,
            'symmetry': 0.006,
            'color': '#f39c12',
        },
        'Electromagnetic': {
            'frequency': 120.00,  # Hz
            'amplitude': 1.458,
            'phase': -1.195,  # rad
            'damping': 1.000,
            'symmetry': 0.002,
            'color': '#3498db',
        },
    }

    # Combined hardware signature
    combined_hardware = {
        'frequency': 29.03,  # Hz
        'amplitude': 3.57e7,
        'phase': -1.477,  # rad
        'damping': 0.748,
        'symmetry': 0.007,
    }

    # Mapped to molecular scale
    molecular_scale = {
        'frequency': 1.00e13,  # Hz (10 THz)
        'amplitude': 6.09e1,
        'phase': -1.477,  # rad
        'damping': 0.000000,
        'symmetry': 0.007,
    }

    # ============================================================================
    # HELPER FUNCTIONS
    # ============================================================================

    def generate_oscillation(freq, amp, phase, damping, duration=1.0, fs=1000):
        """Generate damped oscillation signal"""
        t = np.linspace(0, duration, int(fs * duration))
        signal = amp * np.exp(-damping * t) * np.sin(2 * np.pi * freq * t + phase)
        return t, signal

    def calculate_fft(signal, fs):
        """Calculate FFT of signal"""
        n = len(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/fs)
        
        # Only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft = np.abs(fft[pos_mask])
        
        return freqs, fft

    # ============================================================================
    # CREATE FIGURE
    # ============================================================================

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4,
                height_ratios=[1, 1.2])

    # ============================================================================
    # PANEL A: Individual Oscillation Waveforms
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, :])

    duration = 2.0  # seconds
    fs = 10000  # sampling rate

    for name, params in oscillations.items():
        t, signal = generate_oscillation(
            params['frequency'], 
            params['amplitude'], 
            params['phase'],
            params['damping'],
            duration=duration,
            fs=fs
        )
        
        # Normalize for visualization
        signal_norm = signal / np.abs(signal).max()
        
        ax_a.plot(t, signal_norm, linewidth=1.5, alpha=0.8,
                label=f"{name} ({params['frequency']:.2f} Hz)",
                color=params['color'])

    # Styling
    ax_a.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax_a.set_ylabel('Normalized Amplitude', fontsize=10, fontweight='bold')
    ax_a.set_title('A. Individual Hardware Oscillation Sources', 
                fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, duration)

    # Add zoom inset for high-frequency component
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax_a, width="30%", height="40%", loc='lower left',
                        bbox_to_anchor=(0.05, 0.05, 1, 1),
                        bbox_transform=ax_a.transAxes)

    t_zoom, signal_zoom = generate_oscillation(
        oscillations['Electromagnetic']['frequency'],
        oscillations['Electromagnetic']['amplitude'],
        oscillations['Electromagnetic']['phase'],
        oscillations['Electromagnetic']['damping'],
        duration=0.1,
        fs=fs
    )
    signal_zoom_norm = signal_zoom / np.abs(signal_zoom).max()

    ax_inset.plot(t_zoom, signal_zoom_norm, linewidth=1.5,
                color=oscillations['Electromagnetic']['color'])
    ax_inset.set_title('EM (120 Hz zoom)', fontsize=7)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.tick_params(labelsize=6)

    # ============================================================================
    # PANEL B: Frequency Spectrum (FFT)
    # ============================================================================
    ax_b = fig.add_subplot(gs[1, 0])

    # Generate combined signal
    t_combined = np.linspace(0, 10, int(fs * 10))  # 10 seconds for better freq resolution
    combined_signal = np.zeros_like(t_combined)

    for name, params in oscillations.items():
        _, signal = generate_oscillation(
            params['frequency'],
            params['amplitude'],
            params['phase'],
            params['damping'],
            duration=10,
            fs=fs
        )
        combined_signal += signal

    # Calculate FFT
    freqs, fft_mag = calculate_fft(combined_signal, fs)

    # Plot
    ax_b.semilogy(freqs, fft_mag, 'b-', linewidth=1.5, alpha=0.7)

    # Mark peaks
    for name, params in oscillations.items():
        freq = params['frequency']
        # Find closest frequency in FFT
        idx = np.argmin(np.abs(freqs - freq))
        ax_b.plot(freqs[idx], fft_mag[idx], 'o', markersize=10,
                color=params['color'], markeredgecolor='black',
                markeredgewidth=1.5, zorder=10)
        
        ax_b.annotate(f"{name}\n{freq:.2f} Hz",
                    xy=(freqs[idx], fft_mag[idx]),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=params['color'], alpha=0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    # Mark combined frequency
    combined_freq = combined_hardware['frequency']
    idx_combined = np.argmin(np.abs(freqs - combined_freq))
    ax_b.axvline(combined_freq, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Combined: {combined_freq:.2f} Hz')

    ax_b.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Magnitude', fontsize=10, fontweight='bold')
    ax_b.set_title('B. Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.set_xlim(0, 150)
    ax_b.legend(fontsize=8)

    # ============================================================================
    # PANEL C: Signature Parameters Comparison
    # ============================================================================
    ax_c = fig.add_subplot(gs[1, 1])

    # Prepare data
    sources = list(oscillations.keys()) + ['Combined']
    frequencies = [oscillations[k]['frequency'] for k in oscillations.keys()] + [combined_hardware['frequency']]
    amplitudes = [oscillations[k]['amplitude'] for k in oscillations.keys()] + [combined_hardware['amplitude']]
    dampings = [oscillations[k]['damping'] for k in oscillations.keys()] + [combined_hardware['damping']]
    symmetries = [oscillations[k]['symmetry'] for k in oscillations.keys()] + [combined_hardware['symmetry']]

    # Normalize for comparison
    freq_norm = np.array(frequencies) / max(frequencies)
    amp_norm = np.array(amplitudes) / max(amplitudes)
    damp_norm = np.array(dampings) / max(dampings)
    sym_norm = np.array(symmetries) / max(symmetries)

    x = np.arange(len(sources))
    width = 0.2

    # Grouped bar chart
    bars1 = ax_c.bar(x - 1.5*width, freq_norm, width, label='Frequency',
                    color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax_c.bar(x - 0.5*width, amp_norm, width, label='Amplitude',
                    color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1)
    bars3 = ax_c.bar(x + 0.5*width, damp_norm, width, label='Damping',
                    color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1)
    bars4 = ax_c.bar(x + 1.5*width, sym_norm, width, label='Symmetry',
                    color='#f39c12', alpha=0.7, edgecolor='black', linewidth=1)

    # Highlight combined
    bars1[-1].set_edgecolor('red')
    bars1[-1].set_linewidth(2.5)
    bars2[-1].set_edgecolor('red')
    bars2[-1].set_linewidth(2.5)
    bars3[-1].set_edgecolor('red')
    bars3[-1].set_linewidth(2.5)
    bars4[-1].set_edgecolor('red')
    bars4[-1].set_linewidth(2.5)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(sources, fontsize=8, rotation=45, ha='right')
    ax_c.set_ylabel('Normalized Value', fontsize=10, fontweight='bold')
    ax_c.set_title('C. Signature Parameters (Normalized)', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=8, loc='upper left')
    ax_c.grid(True, alpha=0.3, axis='y')

    # ============================================================================
    # PANEL D: Scale Mapping (Hardware → Molecular)
    # ============================================================================
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.axis('off')

    # Create scale mapping visualization
    y_pos = 0.95

    # Title
    ax_d.text(0.5, y_pos, 'Scale Mapping', 
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=ax_d.transAxes)
    y_pos -= 0.12

    # Hardware scale box
    hw_box = FancyBboxPatch((0.1, y_pos-0.25), 0.8, 0.22,
                            boxstyle="round,pad=0.01",
                            edgecolor='#3498db', facecolor='#3498db',
                            alpha=0.2, linewidth=2,
                            transform=ax_d.transAxes)
    ax_d.add_patch(hw_box)

    ax_d.text(0.5, y_pos-0.02, 'Hardware Scale', 
            ha='center', va='top', fontsize=10, fontweight='bold',
            transform=ax_d.transAxes, color='#3498db')
    y_pos -= 0.06

    hw_text = f"Frequency: {combined_hardware['frequency']:.2f} Hz\n"
    hw_text += f"Amplitude: {combined_hardware['amplitude']:.2e}\n"
    hw_text += f"Phase: {combined_hardware['phase']:.3f} rad\n"
    hw_text += f"Damping: {combined_hardware['damping']:.3f}\n"
    hw_text += f"Symmetry: {combined_hardware['symmetry']:.3f}"

    ax_d.text(0.5, y_pos-0.02, hw_text, 
            ha='center', va='top', fontsize=8,
            transform=ax_d.transAxes, family='monospace')
    y_pos -= 0.28

    # Arrow
    arrow = mpatches.FancyArrowPatch((0.5, y_pos+0.02), (0.5, y_pos-0.08),
                                    transform=ax_d.transAxes,
                                    arrowstyle='->', mutation_scale=30,
                                    linewidth=3, color='red', zorder=10)
    ax_d.add_patch(arrow)

    ax_d.text(0.52, y_pos-0.03, 'Mapping\n×3.4×10¹¹', 
            ha='left', va='center', fontsize=8, fontweight='bold',
            transform=ax_d.transAxes, color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    y_pos -= 0.12

    # Molecular scale box
    mol_box = FancyBboxPatch((0.1, y_pos-0.25), 0.8, 0.22,
                            boxstyle="round,pad=0.01",
                            edgecolor='#2ecc71', facecolor='#2ecc71',
                            alpha=0.2, linewidth=2,
                            transform=ax_d.transAxes)
    ax_d.add_patch(mol_box)

    ax_d.text(0.5, y_pos-0.02, 'Molecular Scale (O₂)', 
            ha='center', va='top', fontsize=10, fontweight='bold',
            transform=ax_d.transAxes, color='#2ecc71')
    y_pos -= 0.06

    mol_text = f"Frequency: {molecular_scale['frequency']:.2e} Hz (10 THz)\n"
    mol_text += f"Amplitude: {molecular_scale['amplitude']:.2e}\n"
    mol_text += f"Phase: {molecular_scale['phase']:.3f} rad\n"
    mol_text += f"Damping: {molecular_scale['damping']:.6f}\n"
    mol_text += f"Symmetry: {molecular_scale['symmetry']:.3f}"

    ax_d.text(0.5, y_pos-0.02, mol_text, 
            ha='center', va='top', fontsize=8,
            transform=ax_d.transAxes, family='monospace')

    ax_d.set_title('D. Hardware → Molecular Mapping', fontsize=12, fontweight='bold')

    # ============================================================================
    # OVERALL TITLE AND SAVE
    # ============================================================================

    fig.suptitle('Hardware Oscillation Signature Analysis: Multi-Scale Coupling', 
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('hardware_oscillation_signatures.pdf', bbox_inches='tight')
    plt.savefig('hardware_oscillation_signatures.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: hardware_oscillation_signatures.pdf/.png")

    plt.show()

    # ============================================================================
    # PRINT SUMMARY STATISTICS
    # ============================================================================

    print("\n" + "="*80)
    print("HARDWARE OSCILLATION SIGNATURE SUMMARY")
    print("="*80)

    print("\nIndividual Sources:")
    print("-" * 80)
    for name, params in oscillations.items():
        print(f"\n{name}:")
        print(f"  Frequency: {params['frequency']:.2f} Hz")
        print(f"  Amplitude: {params['amplitude']:.2e}")
        print(f"  Phase: {params['phase']:.3f} rad ({np.degrees(params['phase']):.1f}°)")
        print(f"  Damping: {params['damping']:.3f}")
        print(f"  Symmetry: {params['symmetry']:.3f}")

    print("\n" + "-" * 80)
    print("Combined Hardware Signature:")
    print("-" * 80)
    for key, value in combined_hardware.items():
        if key == 'phase':
            print(f"  {key.capitalize()}: {value:.3f} rad ({np.degrees(value):.1f}°)")
        else:
            print(f"  {key.capitalize()}: {value:.2e}" if isinstance(value, float) and value > 1000 else f"  {key.capitalize()}: {value:.3f}")

    print("\n" + "-" * 80)
    print("Molecular Scale Mapping:")
    print("-" * 80)
    for key, value in molecular_scale.items():
        if key == 'phase':
            print(f"  {key.capitalize()}: {value:.3f} rad ({np.degrees(value):.1f}°)")
        elif key == 'frequency':
            print(f"  {key.capitalize()}: {value:.2e} Hz ({value/1e12:.1f} THz)")
        else:
            print(f"  {key.capitalize()}: {value:.2e}" if isinstance(value, float) and (value > 1000 or value < 0.001) else f"  {key.capitalize()}: {value:.6f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("✓ CPU frequency dominates amplitude (7.15×10⁷)")
    print("✓ EM field has highest frequency (120 Hz)")
    print("✓ Thermal has lowest damping (0.159) → longest persistence")
    print("✓ Combined signature: 29.03 Hz (weighted average)")
    print("✓ Molecular mapping: 10 THz (O₂ rotational timescale)")
    print("✓ Phase preserved across scales (-1.477 rad)")
    print("✓ Symmetry maintained (0.007)")
    print("\n" + "="*80)
