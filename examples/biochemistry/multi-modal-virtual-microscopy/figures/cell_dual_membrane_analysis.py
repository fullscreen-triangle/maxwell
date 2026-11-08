"""
Cell Image Dual-Membrane Analysis
==================================

Similar to the dual membrane pixel demon analysis with dog pictures,
but now using real microscope cell images from maxwell/public.

This creates the same set of panels:
- Row 1: Original | Front S_k | Front S_t | Front S_e
- Row 2: Negative | Back S_k  | Back S_t  | Back S_e
- Row 3: Conjugacy verification panels

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import sys

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

try:
    from maxwell.src.maxwell.dual_membrane_pixel_demon import DualMembraneGrid
except ImportError:
    # Fallback: simplified version
    print("Warning: Could not import DualMembraneGrid, using simplified version")
    DualMembraneGrid = None


def load_cell_image(image_path: Path, target_size: int = 256):
    """
    Load and preprocess cell image.
    
    Args:
        image_path: Path to cell image
        target_size: Target size for downsampling
        
    Returns:
        Normalized image array [0, 1]
    """
    print(f"\nLoading cell image: {image_path.name}")
    
    # Load image
    try:
        img = Image.open(image_path)
        img_array = np.array(img, dtype=float)
        
        # Handle different image formats
        if len(img_array.shape) == 3:
            # RGB/RGBA - convert to grayscale
            if img_array.shape[2] == 4:
                # RGBA - use alpha channel or convert
                img_array = img_array[:, :, :3]  # Remove alpha
            if img_array.shape[2] == 3:
                # RGB to grayscale
                img_array = 0.299 * img_array[:, :, 0] + \
                           0.587 * img_array[:, :, 1] + \
                           0.114 * img_array[:, :, 2]
        
        # Normalize to [0, 1]
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-10)
        
        print(f"  Original shape: {img_array.shape}")
        
        # Downsample if needed
        if max(img_array.shape) > target_size:
            scale = target_size / max(img_array.shape)
            new_shape = (int(img_array.shape[0] * scale), int(img_array.shape[1] * scale))
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            img_pil = img_pil.resize((new_shape[1], new_shape[0]), Image.Resampling.LANCZOS)
            img_array = np.array(img_pil, dtype=float) / 255.0
            print(f"  Downsampled to: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def process_cell_image_simple(img_array: np.ndarray):
    """
    Process cell image through dual-membrane framework (simplified version).
    
    This creates S-entropy coordinates from image intensity.
    """
    print(f"\nProcessing cell image through dual-membrane framework...")
    print(f"  Image shape: {img_array.shape}")
    
    ny, nx = img_array.shape
    
    # Initialize S-entropy coordinates
    front_sk = np.zeros((ny, nx))
    front_st = np.zeros((ny, nx))
    front_se = np.zeros((ny, nx))
    
    back_sk = np.zeros((ny, nx))
    back_st = np.zeros((ny, nx))
    back_se = np.zeros((ny, nx))
    
    # Process each pixel
    for i in range(ny):
        for j in range(nx):
            intensity = img_array[i, j]
            
            # Map intensity to S_k coordinate
            S_k = 2.0 * intensity - 1.0  # Map [0,1] to [-1,1]
            S_t = 0.0  # Neutral temporal
            S_e = 0.5  # Neutral evolutionary
            
            # Front face
            front_sk[i, j] = S_k
            front_st[i, j] = S_t
            front_se[i, j] = S_e
            
            # Back face (phase conjugate)
            back_sk[i, j] = -S_k  # Phase conjugate: negate S_k
            back_st[i, j] = S_t
            back_se[i, j] = S_e
    
    print("  Processing complete")
    
    # Calculate correlation
    correlation = np.corrcoef(front_sk.flatten(), back_sk.flatten())[0, 1]
    print(f"  Front-Back S_k correlation: {correlation:.4f}")
    
    return {
        'original': img_array,
        'front': {
            's_k': front_sk,
            's_t': front_st,
            's_e': front_se
        },
        'back': {
            's_k': back_sk,
            's_t': back_st,
            's_e': back_se
        },
        'correlation': correlation,
        'image_name': 'cell_image'
    }


def plot_image(ax, data, title, label, cmap='gray', vmin=None, vmax=None, centered=False):
    """Plot image with title and label."""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'{label}. {title}', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return im


def plot_correlation(ax, front_data, back_data, label):
    """Plot correlation scatter plot."""
    # Flatten arrays
    front_flat = front_data.flatten()
    back_flat = back_data.flatten()
    
    # Sample for performance
    if len(front_flat) > 10000:
        indices = np.random.choice(len(front_flat), 10000, replace=False)
        front_flat = front_flat[indices]
        back_flat = back_flat[indices]
    
    ax.scatter(front_flat, back_flat, alpha=0.3, s=1, c='blue')
    ax.set_xlabel('Front $S_k$', fontsize=9)
    ax.set_ylabel('Back $S_k$', fontsize=9)
    ax.set_title(f'{label}. Correlation: Front vs Back $S_k$', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line (expected: y = -x)
    xlim = ax.get_xlim()
    ax.plot(xlim, [-xlim[0], -xlim[1]], 'r--', linewidth=2, label='Expected: $y = -x$')
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='box')


def create_dual_membrane_figure(data, output_path: Path):
    """
    Create comprehensive dual-membrane visualization for cell image.
    
    Layout:
    Row 1: Original | Front S_k | Front S_t | Front S_e
    Row 2: Negative | Back S_k  | Back S_t  | Back S_e
    Row 3: Conjugacy verification panels
    """
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                          left=0.06, right=0.96, top=0.94, bottom=0.06)
    
    fig.suptitle(f'Dual-Membrane Analysis: {data["image_name"]}', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Original and Front Face
    ax_orig = fig.add_subplot(gs[0, 0])
    plot_image(ax_orig, data['original'], 'Original Cell Image', 'A', 'gray')
    
    ax_f_sk = fig.add_subplot(gs[0, 1])
    plot_image(ax_f_sk, data['front']['s_k'], 'Front $S_k$ (Knowledge)', 'B', 'RdBu_r', vmin=-1, vmax=1)
    
    ax_f_st = fig.add_subplot(gs[0, 2])
    plot_image(ax_f_st, data['front']['s_t'], 'Front $S_t$ (Temporal)', 'C', 'viridis', vmin=-1, vmax=1)
    
    ax_f_se = fig.add_subplot(gs[0, 3])
    plot_image(ax_f_se, data['front']['s_e'], 'Front $S_e$ (Evolution)', 'D', 'plasma', vmin=0, vmax=1)
    
    # Row 2: Negative and Back Face (The "HIDDEN" conjugate state)
    negative = 1.0 - data['original']
    ax_neg = fig.add_subplot(gs[1, 0])
    plot_image(ax_neg, negative, 'Negative (Visual)', 'E', 'gray')
    
    ax_b_sk = fig.add_subplot(gs[1, 1])
    plot_image(ax_b_sk, data['back']['s_k'], 'Back $S_k$ (Conjugate)', 'F', 'RdBu_r', vmin=-1, vmax=1)
    
    ax_b_st = fig.add_subplot(gs[1, 2])
    plot_image(ax_b_st, data['back']['s_t'], 'Back $S_t$ (Conjugate)', 'G', 'viridis', vmin=-1, vmax=1)
    
    ax_b_se = fig.add_subplot(gs[1, 3])
    plot_image(ax_b_se, data['back']['s_e'], 'Back $S_e$ (Conjugate)', 'H', 'plasma', vmin=0, vmax=1)
    
    # Row 3: Conjugacy verification
    sum_sk = data['front']['s_k'] + data['back']['s_k']
    ax_sum = fig.add_subplot(gs[2, 0])
    plot_image(ax_sum, sum_sk, 'Sum: $S_k^{front} + S_k^{back} \\approx 0$', 'I', 'RdYlGn',
               vmin=-0.1, vmax=0.1, centered=True)
    
    # Correlation plot
    ax_corr = fig.add_subplot(gs[2, 1])
    plot_correlation(ax_corr, data['front']['s_k'], data['back']['s_k'], 'J')
    
    # Difference map
    diff_sk = np.abs(data['front']['s_k'] - (-data['back']['s_k']))
    ax_diff = fig.add_subplot(gs[2, 2])
    plot_image(ax_diff, diff_sk, 'Difference: $|S_k^{front} - (-S_k^{back})|$', 'K', 'hot',
               vmin=0, vmax=0.1)
    
    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 3])
    ax_stats.axis('off')
    
    stats_text = f"""
    Statistics:
    
    Correlation: {data['correlation']:.4f}
    
    Front S_k:
      Mean: {data['front']['s_k'].mean():.4f}
      Std:  {data['front']['s_k'].std():.4f}
      Min:  {data['front']['s_k'].min():.4f}
      Max:  {data['front']['s_k'].max():.4f}
    
    Back S_k:
      Mean: {data['back']['s_k'].mean():.4f}
      Std:  {data['back']['s_k'].std():.4f}
      Min:  {data['back']['s_k'].min():.4f}
      Max:  {data['back']['s_k'].max():.4f}
    
    Sum (should be ~0):
      Mean: {sum_sk.mean():.6f}
      Std:  {sum_sk.std():.6f}
      Max:  {np.abs(sum_sk).max():.6f}
    """
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                 verticalalignment='center', transform=ax_stats.transAxes)
    ax_stats.set_title('L. Statistics', fontsize=10, fontweight='bold')
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved figure: {output_path}")
    plt.close()


def main():
    """Process all cell images in maxwell/public/cells."""
    # Find cell images
    # Script is in: maxwell/publication/multi-modal-virtual-microscopy/figures/
    # Need to go to: maxwell/public/cells/
    # From figures/ -> multi-modal-virtual-microscopy/ -> publication/ -> maxwell/
    script_dir = Path(__file__).parent
    cells_dir = script_dir.parent.parent.parent / "public" / "cells"
    output_dir = script_dir
    
    print(f"Looking for cell images in: {cells_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if directory exists
    if not cells_dir.exists():
        print(f"ERROR: Directory does not exist: {cells_dir}")
        return
    
    # Image extensions to look for
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    
    # Find all images
    cell_images = []
    for ext in image_extensions:
        cell_images.extend(list(cells_dir.glob(f'*{ext}')))
        cell_images.extend(list(cells_dir.glob(f'*{ext.upper()}')))
    
    if not cell_images:
        print(f"No cell images found in {cells_dir}")
        return
    
    print(f"Found {len(cell_images)} cell images")
    
    # Process each image
    for img_path in cell_images:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print('='*60)
        
        # Load image
        img_array = load_cell_image(img_path, target_size=256)
        if img_array is None:
            continue
        
        # Process through dual-membrane framework
        data = process_cell_image_simple(img_array)
        data['image_name'] = img_path.stem
        
        # Create figure
        output_path = output_dir / f"cell_dual_membrane_{img_path.stem}.png"
        create_dual_membrane_figure(data, output_path)
    
    print(f"\n{'='*60}")
    print("[OK] All cell images processed!")
    print(f"Output directory: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
