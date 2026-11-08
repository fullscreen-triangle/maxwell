"""
Multi-Modal Virtual Microscopy for Cell Images
===============================================

Simulates different microscopy modalities and wavelengths:
- Brightfield microscopy
- Phase contrast microscopy
- Fluorescence microscopy (different wavelengths)
- Confocal microscopy
- Darkfield microscopy
- DIC (Differential Interference Contrast)
- Phase difference visualization (scattering and multiple reflections)

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import scipy.ndimage as ndi
from scipy.fft import fft2, ifft2, fftshift
import sys


def load_cell_image(image_path: Path, target_size: int = 512):
    """Load and preprocess cell image."""
    print(f"Loading: {image_path.name}")
    
    img = Image.open(image_path)
    img_array = np.array(img, dtype=float)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        if img_array.shape[2] == 3:
            img_array = 0.299 * img_array[:, :, 0] + \
                       0.587 * img_array[:, :, 1] + \
                       0.114 * img_array[:, :, 2]
    
    # Normalize
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-10)
    
    # Resize if needed
    if max(img_array.shape) != target_size:
        scale = target_size / max(img_array.shape)
        new_shape = (int(img_array.shape[0] * scale), int(img_array.shape[1] * scale))
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_pil = img_pil.resize((new_shape[1], new_shape[0]), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil, dtype=float) / 255.0
    
    return img_array


def brightfield_microscopy(img, wavelength_nm=550):
    """
    Simulate brightfield microscopy.
    Brightfield: transmitted light, absorption contrast.
    """
    # Brightfield shows absorption - darker regions absorb more light
    # Simulate by inverting intensity (dense regions = dark)
    brightfield = 1.0 - img * 0.8  # Absorption contrast
    brightfield = np.clip(brightfield, 0, 1)
    
    # Add wavelength-dependent contrast
    if wavelength_nm < 500:  # Blue
        contrast_factor = 1.2
    elif wavelength_nm < 600:  # Green/Yellow
        contrast_factor = 1.0
    else:  # Red
        contrast_factor = 0.9
    
    brightfield = brightfield * contrast_factor
    return np.clip(brightfield, 0, 1)


def phase_contrast_microscopy(img):
    """
    Simulate phase contrast microscopy.
    Phase contrast: converts phase shifts to intensity variations.
    """
    # Phase contrast enhances edges and phase boundaries
    # Use gradient magnitude to simulate phase shifts
    grad_y, grad_x = np.gradient(img)
    phase_shift = np.sqrt(grad_x**2 + grad_y**2)
    
    # Convert phase shift to intensity
    phase_contrast = 0.5 + 0.5 * np.sin(phase_shift * np.pi * 2)
    
    # Enhance contrast
    phase_contrast = (phase_contrast - phase_contrast.min()) / (phase_contrast.max() - phase_contrast.min() + 1e-10)
    
    return phase_contrast


def fluorescence_microscopy(img, wavelength_nm=488, emission_wavelength_nm=520):
    """
    Simulate fluorescence microscopy at specific wavelength.
    
    Args:
        wavelength_nm: Excitation wavelength (e.g., 488 nm for GFP)
        emission_wavelength_nm: Emission wavelength
    """
    # Fluorescence: specific structures emit light at different wavelengths
    # Simulate by wavelength-dependent response
    
    # Different structures respond to different wavelengths
    if wavelength_nm == 350:  # DAPI (nucleus)
        # Nucleus-like structures (high intensity regions)
        fluorescence = img ** 0.5  # Enhance bright regions
    elif wavelength_nm == 488:  # GFP (general)
        # General fluorescence
        fluorescence = img
    elif wavelength_nm == 555:  # RFP/mCherry
        # Cytoplasm-like structures
        fluorescence = 1.0 - (1.0 - img) ** 0.7
    elif wavelength_nm == 647:  # Far-red
        # Membrane structures
        edges = ndi.sobel(img)
        fluorescence = np.clip(img + edges * 0.3, 0, 1)
    else:
        fluorescence = img
    
    # Add wavelength-dependent efficiency
    efficiency = np.exp(-(wavelength_nm - 500)**2 / (2 * 100**2))  # Gaussian around 500 nm
    fluorescence = fluorescence * (0.5 + 0.5 * efficiency)
    
    # Add noise (photon counting)
    noise = np.random.poisson(fluorescence * 100, size=fluorescence.shape) / 100.0
    fluorescence = np.clip(noise, 0, 1)
    
    return fluorescence


def confocal_microscopy(img, wavelength_nm=488):
    """
    Simulate confocal microscopy.
    Confocal: point-by-point scanning, optical sectioning.
    """
    # Confocal has better resolution and optical sectioning
    # Simulate by sharpening and reducing out-of-focus blur
    
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) * 0.1
    
    confocal = ndi.convolve(img, kernel)
    confocal = np.clip(confocal, 0, 1)
    
    # Optical sectioning: reduce background
    confocal = confocal * 0.7 + 0.3 * img  # Mix with original
    
    # Add wavelength-dependent resolution
    resolution_factor = 200.0 / wavelength_nm  # Better resolution at shorter wavelengths
    confocal = ndi.gaussian_filter(confocal, sigma=1.0 / resolution_factor)
    
    return confocal


def darkfield_microscopy(img):
    """
    Simulate darkfield microscopy.
    Darkfield: only scattered light is visible, bright objects on dark background.
    """
    # Darkfield shows edges and small structures
    # Use high-pass filter
    blurred = ndi.gaussian_filter(img, sigma=5)
    darkfield = img - blurred
    darkfield = np.clip(darkfield, 0, 1)
    
    # Enhance edges
    edges = ndi.sobel(img)
    darkfield = darkfield + edges * 0.5
    darkfield = np.clip(darkfield, 0, 1)
    
    # Invert: bright objects on dark background
    darkfield = 1.0 - darkfield
    
    return darkfield


def dic_microscopy(img):
    """
    Simulate DIC (Differential Interference Contrast) microscopy.
    DIC: gradient-based contrast, 3D-like appearance.
    """
    # DIC shows gradients and refractive index variations
    grad_y, grad_x = np.gradient(img)
    
    # DIC creates shadow-cast effect
    dic = np.abs(grad_x) + np.abs(grad_y)
    dic = (dic - dic.min()) / (dic.max() - dic.min() + 1e-10)
    
    # Add directional lighting effect
    lighting = grad_x * 0.7 + grad_y * 0.3
    dic = dic + lighting * 0.3
    dic = np.clip(dic, 0, 1)
    
    return dic


def phase_difference_visualization(img):
    """
    Visualize phase differences from scattering and multiple reflections.
    
    This shows:
    - Direct transmitted light (phase = 0)
    - Scattered light (phase shifts)
    - Multiple reflections (phase accumulation)
    """
    # Simulate phase map
    # Phase accumulates in dense regions and at interfaces
    phase_map = np.zeros_like(img, dtype=complex)
    
    # Direct transmission (reference phase = 0)
    direct_phase = np.ones_like(img, dtype=complex)
    
    # Scattered light: phase shifts at edges
    grad_y, grad_x = np.gradient(img)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    scattered_phase = np.exp(1j * edge_strength * np.pi)  # Phase shift proportional to edge strength
    
    # Multiple reflections: phase accumulation in dense regions
    # Dense regions (high intensity) cause multiple internal reflections
    reflection_phase = np.exp(1j * img * np.pi * 2)  # Phase proportional to density
    
    # Combine: direct + scattered + reflected
    total_phase = direct_phase + 0.3 * scattered_phase + 0.2 * reflection_phase
    
    # Visualize phase differences
    phase_diff = np.angle(total_phase)  # Phase angle in radians
    phase_diff = (phase_diff + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Also show amplitude (interference pattern)
    amplitude = np.abs(total_phase)
    amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-10)
    
    return phase_diff, amplitude, total_phase


def oxygen_tracker_spectrometry(img, wavelengths_nm=[488, 550, 647]):
    """
    Use oxygen as a tracker through inverse spectrometry.
    
    Calculate expected phase shifts from oxygen absorption at different wavelengths,
    then show the interference patterns that should result.
    
    This validates the framework by:
    1. Estimating oxygen distribution from cell structure
    2. Calculating expected phase shifts from O₂ absorption spectra
    3. Predicting interference patterns
    4. Comparing with observed patterns
    
    Returns:
        oxygen_distribution: Estimated O₂ concentration map
        expected_phase_shifts: Phase shifts from O₂ absorption
        interference_pattern: Resulting interference pattern
        validation_map: Comparison of expected vs observed
    """
    # Step 1: Estimate oxygen distribution
    # Oxygen is consumed in mitochondria (dense regions) and present in cytoplasm
    # Use cell structure to infer O₂ distribution
    
    # Mitochondria-like regions (high metabolic activity = low O₂)
    # These are typically bright, dense regions
    mitochondria_mask = img > 0.7  # High intensity = high metabolic activity
    
    # Cytoplasm (moderate O₂)
    cytoplasm_mask = (img > 0.3) & (img <= 0.7)
    
    # Background/extracellular (high O₂)
    background_mask = img <= 0.3
    
    # Create oxygen distribution map
    oxygen_distribution = np.zeros_like(img)
    oxygen_distribution[background_mask] = 1.0  # High O₂ in background
    oxygen_distribution[cytoplasm_mask] = 0.6   # Moderate O₂ in cytoplasm
    oxygen_distribution[mitochondria_mask] = 0.2  # Low O₂ in mitochondria
    
    # Add spatial correlation (O₂ diffuses)
    oxygen_distribution = ndi.gaussian_filter(oxygen_distribution, sigma=3)
    oxygen_distribution = (oxygen_distribution - oxygen_distribution.min()) / \
                         (oxygen_distribution.max() - oxygen_distribution.min() + 1e-10)
    
    # Step 2: Calculate expected phase shifts from O₂ absorption
    # O₂ has absorption bands at specific wavelengths:
    # - Strong absorption in UV (< 200 nm)
    # - Weak absorption in visible (O₂-A band ~760 nm, O₂-B band ~687 nm)
    # - For visible wavelengths, use refractive index changes
    
    # O₂ absorption coefficient (cm⁻¹) at different wavelengths
    # Simplified model: stronger absorption at longer wavelengths in visible range
    o2_absorption_coeffs = {}
    for wl in wavelengths_nm:
        if wl < 500:
            # Blue: weak O₂ absorption
            o2_absorption_coeffs[wl] = 0.01 * (1.0 - wl / 500.0)
        elif wl < 600:
            # Green/Yellow: moderate
            o2_absorption_coeffs[wl] = 0.02 * (wl - 500) / 100.0
        else:
            # Red: stronger (approaching O₂-B band)
            o2_absorption_coeffs[wl] = 0.03 + 0.02 * (wl - 600) / 100.0
    
    # Phase shift from O₂ absorption
    # Phase shift = (2π / λ) * n * L, where n is refractive index change
    # Refractive index change proportional to O₂ concentration and absorption
    expected_phase_shifts = {}
    for wl in wavelengths_nm:
        # Refractive index change from O₂
        # Δn = α * C, where α is absorption coefficient, C is concentration
        delta_n = o2_absorption_coeffs[wl] * oxygen_distribution
        
        # Path length (assume 1 μm cell thickness)
        path_length = 1.0  # μm
        
        # Phase shift: Δφ = (2π / λ) * Δn * L
        wavelength_um = wl / 1000.0  # Convert nm to μm
        phase_shift = (2 * np.pi / wavelength_um) * delta_n * path_length
        
        expected_phase_shifts[wl] = phase_shift
    
    # Step 3: Calculate interference pattern
    # Interference from multiple O₂ absorption sites
    # Use Fourier transform to see interference fringes
    
    # Average phase shift across wavelengths (weighted by absorption strength)
    total_phase_shift = np.zeros_like(img)
    total_weight = 0
    for wl, phase_shift in expected_phase_shifts.items():
        weight = o2_absorption_coeffs[wl]
        total_phase_shift += phase_shift * weight
        total_weight += weight
    
    if total_weight > 0:
        total_phase_shift /= total_weight
    
    # Create interference pattern
    # Complex field: E = E₀ * exp(i * φ)
    complex_field = np.exp(1j * total_phase_shift)
    
    # Interference pattern (intensity)
    interference_pattern = np.abs(complex_field)**2
    interference_pattern = (interference_pattern - interference_pattern.min()) / \
                          (interference_pattern.max() - interference_pattern.min() + 1e-10)
    
    # Step 4: Validation - compare expected vs observed
    # Observed pattern would come from actual phase measurements
    # For now, simulate "observed" from structure (edges, membranes)
    grad_y, grad_x = np.gradient(img)
    observed_phase = np.sqrt(grad_x**2 + grad_y**2) * np.pi
    observed_pattern = np.abs(np.exp(1j * observed_phase))**2
    observed_pattern = (observed_pattern - observed_pattern.min()) / \
                      (observed_pattern.max() - observed_pattern.min() + 1e-10)
    
    # Correlation between expected and observed
    correlation = np.corrcoef(interference_pattern.flatten(), 
                             observed_pattern.flatten())[0, 1]
    
    # Difference map
    validation_map = np.abs(interference_pattern - observed_pattern)
    
    return {
        'oxygen_distribution': oxygen_distribution,
        'expected_phase_shifts': expected_phase_shifts,
        'interference_pattern': interference_pattern,
        'observed_pattern': observed_pattern,
        'validation_map': validation_map,
        'correlation': correlation
    }


def create_multimodal_figure(img_array, image_name, output_path):
    """
    Create comprehensive multi-modal microscopy figure.
    
    Layout:
    Row 1: Original | Brightfield | Phase Contrast | Fluorescence (488 nm)
    Row 2: Fluorescence (555 nm) | Fluorescence (647 nm) | Confocal | Darkfield
    Row 3: DIC | Phase Difference | Phase Amplitude | Wavelength Comparison
    Row 4: O₂ Tracker (NEW) | O₂ Distribution | Expected Phase | Interference Pattern
    """
    
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3,
                          left=0.05, right=0.95, top=0.97, bottom=0.03)
    
    fig.suptitle(f'Multi-Modal Virtual Microscopy: {image_name}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Row 1: Basic modalities
    ax1 = fig.add_subplot(gs[0, 0])
    plot_image(ax1, img_array, 'Original Cell Image', 'A', 'gray')
    
    ax2 = fig.add_subplot(gs[0, 1])
    brightfield = brightfield_microscopy(img_array, wavelength_nm=550)
    plot_image(ax2, brightfield, 'Brightfield (550 nm)', 'B', 'gray')
    
    ax3 = fig.add_subplot(gs[0, 2])
    phase_contrast = phase_contrast_microscopy(img_array)
    plot_image(ax3, phase_contrast, 'Phase Contrast', 'C', 'viridis')
    
    ax4 = fig.add_subplot(gs[0, 3])
    fluor_488 = fluorescence_microscopy(img_array, wavelength_nm=488, emission_wavelength_nm=520)
    plot_image(ax4, fluor_488, 'Fluorescence (488 nm GFP)', 'D', 'Greens')
    
    # Row 2: Advanced modalities
    ax5 = fig.add_subplot(gs[1, 0])
    fluor_555 = fluorescence_microscopy(img_array, wavelength_nm=555, emission_wavelength_nm=580)
    plot_image(ax5, fluor_555, 'Fluorescence (555 nm RFP)', 'E', 'hot')
    
    ax6 = fig.add_subplot(gs[1, 1])
    fluor_647 = fluorescence_microscopy(img_array, wavelength_nm=647, emission_wavelength_nm=670)
    plot_image(ax6, fluor_647, 'Fluorescence (647 nm Far-red)', 'F', 'magma')
    
    ax7 = fig.add_subplot(gs[1, 2])
    confocal = confocal_microscopy(img_array, wavelength_nm=488)
    plot_image(ax7, confocal, 'Confocal (488 nm)', 'G', 'gray')
    
    ax8 = fig.add_subplot(gs[1, 3])
    darkfield = darkfield_microscopy(img_array)
    plot_image(ax8, darkfield, 'Darkfield', 'H', 'gray')
    
    # Row 3: Phase analysis and comparison
    ax9 = fig.add_subplot(gs[2, 0])
    dic = dic_microscopy(img_array)
    plot_image(ax9, dic, 'DIC (Differential Interference)', 'I', 'gray')
    
    ax10 = fig.add_subplot(gs[2, 1])
    phase_diff, phase_amp, total_phase = phase_difference_visualization(img_array)
    plot_image(ax10, phase_diff, 'Phase Difference\n(Scattering + Reflections)', 'J', 'hsv')
    
    ax11 = fig.add_subplot(gs[2, 2])
    plot_image(ax11, phase_amp, 'Phase Amplitude\n(Interference Pattern)', 'K', 'plasma')
    
    ax12 = fig.add_subplot(gs[2, 3])
    plot_wavelength_comparison(ax12, img_array, 'L')
    
    # Row 4: Oxygen Tracker Spectrometry (NEW)
    o2_results = oxygen_tracker_spectrometry(img_array, wavelengths_nm=[488, 550, 647])
    
    ax13 = fig.add_subplot(gs[3, 0])
    plot_image(ax13, o2_results['oxygen_distribution'], 
              'O₂ Distribution\n(Inferred from Structure)', 'M', 'coolwarm')
    
    ax14 = fig.add_subplot(gs[3, 1])
    # Show expected phase shift at 550 nm (middle wavelength)
    expected_phase_550 = o2_results['expected_phase_shifts'][550]
    expected_phase_norm = (expected_phase_550 - expected_phase_550.min()) / \
                         (expected_phase_550.max() - expected_phase_550.min() + 1e-10)
    plot_image(ax14, expected_phase_norm, 
              'Expected Phase Shift\n(from O₂ Absorption)', 'N', 'hsv')
    
    ax15 = fig.add_subplot(gs[3, 2])
    plot_image(ax15, o2_results['interference_pattern'], 
              'O₂ Interference Pattern\n(Predicted)', 'O', 'plasma')
    
    ax16 = fig.add_subplot(gs[3, 3])
    plot_oxygen_validation(ax16, o2_results, 'P')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_image(ax, data, title, label, cmap='gray', vmin=None, vmax=None):
    """Plot image with title and label."""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'{label}. {title}', fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return im


def plot_wavelength_comparison(ax, img, label):
    """Compare different wavelengths side by side."""
    wavelengths = [350, 488, 555, 647]  # DAPI, GFP, RFP, Far-red
    wavelength_names = ['350 nm\n(DAPI)', '488 nm\n(GFP)', '555 nm\n(RFP)', '647 nm\n(Far-red)']
    
    # Create small sub-images
    n_wl = len(wavelengths)
    comparison = np.zeros((img.shape[0], img.shape[1] * n_wl))
    
    for i, (wl, name) in enumerate(zip(wavelengths, wavelength_names)):
        fluor = fluorescence_microscopy(img, wavelength_nm=wl)
        start_col = i * img.shape[1]
        end_col = (i + 1) * img.shape[1]
        comparison[:, start_col:end_col] = fluor
    
    im = ax.imshow(comparison, cmap='hot', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'{label}. Wavelength Comparison', fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Add wavelength labels
    for i, name in enumerate(wavelength_names):
        x_pos = (i + 0.5) / n_wl
        ax.text(x_pos, 0.95, name, transform=ax.transAxes,
               fontsize=9, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_oxygen_validation(ax, o2_results, label):
    """Plot validation panel comparing expected vs observed interference patterns."""
    # Create 2x2 subplot within this axis
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    subplot_spec = ax.get_subplotspec()
    gs_sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=subplot_spec, hspace=0.3, wspace=0.3)
    
    # Observed pattern
    ax1 = plt.subplot(gs_sub[0, 0])
    im1 = ax1.imshow(o2_results['observed_pattern'], cmap='plasma', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Observed Pattern', fontsize=9, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Expected pattern
    ax2 = plt.subplot(gs_sub[0, 1])
    im2 = ax2.imshow(o2_results['interference_pattern'], cmap='plasma', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Expected Pattern', fontsize=9, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Difference map
    ax3 = plt.subplot(gs_sub[1, 0])
    im3 = ax3.imshow(o2_results['validation_map'], cmap='hot', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Difference Map', fontsize=9, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Statistics
    ax4 = plt.subplot(gs_sub[1, 1])
    ax4.axis('off')
    
    stats_text = f"""
    O₂ Tracker Validation:
    
    Correlation: {o2_results['correlation']:.4f}
    
    Mean Difference: {o2_results['validation_map'].mean():.4f}
    Max Difference: {o2_results['validation_map'].max():.4f}
    
    O₂ Distribution:
      Mean: {o2_results['oxygen_distribution'].mean():.3f}
      Range: [{o2_results['oxygen_distribution'].min():.3f}, 
              {o2_results['oxygen_distribution'].max():.3f}]
    
    Validation: {"✓ PASS" if o2_results['correlation'] > 0.5 else "✗ FAIL"}
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=8, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Main title
    ax.text(0.5, 0.98, f'{label}. O₂ Tracker Validation', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           ha='center', va='top')


def main():
    """Process 3 selected cell images."""
    # Find cell images
    # Script is in: maxwell/publication/multi-modal-virtual-microscopy/figures/
    # Need to go to: maxwell/public/cells/
    script_dir = Path(__file__).parent
    cells_dir = script_dir.parent.parent.parent / "public" / "cells"
    output_dir = script_dir
    
    print(f"Looking for cell images in: {cells_dir}")
    print(f"Output directory: {output_dir}")
    
    if not cells_dir.exists():
        print(f"ERROR: Directory does not exist: {cells_dir}")
        return
    
    # Select 3 diverse images (use files that exist)
    # Get all available images first
    all_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        all_images.extend(list(cells_dir.glob(f'*{ext}')))
        all_images.extend(list(cells_dir.glob(f'*{ext.upper()}')))
    
    if len(all_images) == 0:
        print(f"ERROR: No images found in {cells_dir}")
        return
    
    # Select first 3 images
    image_files = [img.name for img in all_images[:3]]
    print(f"Selected images: {image_files}")
    
    print("="*70)
    print("Multi-Modal Virtual Microscopy Analysis")
    print("="*70)
    
    for img_file in image_files:
        img_path = cells_dir / img_file
        if not img_path.exists():
            # Try to find the actual path
            matching = list(cells_dir.glob(f'*{img_file}*'))
            if matching:
                img_path = matching[0]
            else:
                print(f"Warning: {img_file} not found, skipping...")
                continue
        
        print(f"\nProcessing: {img_file}")
        print("-"*70)
        
        # Load image
        img_array = load_cell_image(img_path, target_size=512)
        
        # Create multi-modal figure
        output_path = output_dir / f"multimodal_microscopy_{img_path.stem}.png"
        create_multimodal_figure(img_array, img_path.stem, output_path)
    
    print("\n" + "="*70)
    print("[OK] Multi-modal microscopy analysis complete!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
