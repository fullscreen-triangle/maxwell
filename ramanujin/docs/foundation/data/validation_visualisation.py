"""
Thought Library Inspector & Visualizer
Complete analysis of NPZ thought libraries
"""
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

class ThoughtLibraryInspector:
    """Inspect and visualize thought libraries"""
    
    def __init__(self, npz_files, json_files):
        self.npz_files = npz_files
        self.json_files = json_files
        self.thoughts = {}
        self.validations = {}
        
    def load_all(self):
        """Load all NPZ and JSON files"""
        
        print("\n" + "="*70)
        print("LOADING THOUGHT LIBRARIES")
        print("="*70 + "\n")
        
        # Load NPZ files
        for npz_file in self.npz_files:
            print(f"Loading: {npz_file}")
            data = np.load(npz_file)
            
            print(f"  Arrays in file: {list(data.files)}")
            
            thought_data = {}
            for key in data.files:
                arr = data[key]
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
                thought_data[key] = arr
            
            self.thoughts[npz_file] = thought_data
            print()
        
        # Load JSON files
        for json_file in self.json_files:
            print(f"Loading: {json_file}")
            with open(json_file, 'r') as f:
                val_data = json.load(f)
            
            print(f"  Keys: {list(val_data.keys())}")
            print(f"  n_thoughts_captured: {val_data.get('n_thoughts_captured', 'N/A')}")
            
            self.validations[json_file] = val_data
            print()
        
        print("="*70)
        print("✓ ALL FILES LOADED")
        print("="*70 + "\n")
        
        return self.thoughts, self.validations
    
    def analyze_structure(self):
        """Analyze the structure of thought libraries"""
        
        print("\n" + "="*70)
        print("THOUGHT LIBRARY STRUCTURE ANALYSIS")
        print("="*70 + "\n")
        
        for npz_file, thought_data in self.thoughts.items():
            print(f"\nFile: {npz_file}")
            print("-" * 70)
            
            for key, arr in thought_data.items():
                print(f"\n{key}:")
                print(f"  Shape: {arr.shape}")
                print(f"  Dtype: {arr.dtype}")
                print(f"  Min: {arr.min():.6f}")
                print(f"  Max: {arr.max():.6f}")
                print(f"  Mean: {arr.mean():.6f}")
                print(f"  Std: {arr.std():.6f}")
                
                # If it's O2 positions, analyze spatial distribution
                if 'o2_positions' in key.lower():
                    print(f"\n  Spatial Analysis:")
                    
                    # Center of mass
                    com = arr.mean(axis=0)
                    print(f"    Center of mass: [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}]")
                    
                    # Distances from origin
                    distances = np.linalg.norm(arr, axis=1)
                    print(f"    Distance from origin: {distances.min():.3f} to {distances.max():.3f} Å")
                    print(f"    Mean distance: {distances.mean():.3f} Å")
                    
                    # Spread (radius of gyration)
                    rg = np.sqrt(np.mean(np.sum((arr - com)**2, axis=1)))
                    print(f"    Radius of gyration: {rg:.3f} Å")
                    
                    # Pairwise distances
                    pairwise = pdist(arr)
                    print(f"    Pairwise distances: {pairwise.min():.3f} to {pairwise.max():.3f} Å")
                    print(f"    Mean pairwise: {pairwise.mean():.3f} Å")
        
        print("\n" + "="*70)
        
    def analyze_validations(self):
        """Analyze validation data"""
        
        print("\n" + "="*70)
        print("VALIDATION DATA ANALYSIS")
        print("="*70 + "\n")
        
        for json_file, val_data in self.validations.items():
            print(f"\nFile: {json_file}")
            print("-" * 70)
            
            # Pretty print JSON structure
            print(json.dumps(val_data, indent=2))
            
            # Extract key metrics
            if 'similarity' in val_data:
                sim = val_data['similarity']
                if 'mean_similarity' in sim:
                    print(f"\n  Mean similarity: {sim['mean_similarity']:.4f}")
                if 'std_similarity' in sim:
                    print(f"  Std similarity: {sim['std_similarity']:.4f}")
        
        print("\n" + "="*70)
    
    def create_visualization(self, output='thought_library_analysis'):
        """Create comprehensive visualization"""
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Get first thought library for visualization
        first_npz = list(self.thoughts.keys())[0]
        thought_data = self.thoughts[first_npz]
        
        # Find O2 positions array
        o2_key = [k for k in thought_data.keys() if 'o2_positions' in k.lower()][0]
        o2_positions = thought_data[o2_key]
        
        # Panel A: 3D O2 configuration
        ax_a = fig.add_subplot(gs[0, 0], projection='3d')
        self.panel_a_3d_config(ax_a, o2_positions)
        
        # Panel B: Distance distribution
        ax_b = fig.add_subplot(gs[0, 1])
        self.panel_b_distances(ax_b, o2_positions)
        
        # Panel C: Pairwise distance matrix
        ax_c = fig.add_subplot(gs[0, 2])
        self.panel_c_pairwise_matrix(ax_c, o2_positions)
        
        # Panel D: XY projection
        ax_d = fig.add_subplot(gs[1, 0])
        self.panel_d_xy_projection(ax_d, o2_positions)
        
        # Panel E: XZ projection
        ax_e = fig.add_subplot(gs[1, 1])
        self.panel_e_xz_projection(ax_e, o2_positions)
        
        # Panel F: YZ projection
        ax_f = fig.add_subplot(gs[1, 2])
        self.panel_f_yz_projection(ax_f, o2_positions)
        
        # Panel G: Radial distribution
        ax_g = fig.add_subplot(gs[2, 0])
        self.panel_g_radial_distribution(ax_g, o2_positions)
        
        # Panel H: PCA analysis
        ax_h = fig.add_subplot(gs[2, 1])
        self.panel_h_pca(ax_h, o2_positions)
        
        # Panel I: Validation metrics (if available)
        ax_i = fig.add_subplot(gs[2, 2])
        self.panel_i_validation(ax_i)
        
        # Overall title
        fig.suptitle(f'Thought Library Analysis: {Path(first_npz).stem}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        plt.savefig(f'{output}.pdf', bbox_inches='tight')
        plt.savefig(f'{output}.png', bbox_inches='tight')
        print(f"\n✓ Saved: {output}.pdf/.png")
        
        return fig
    
    def panel_a_3d_config(self, ax, o2_positions):
        """Panel A: 3D O2 configuration"""
        
        # Calculate distances from origin for coloring
        distances = np.linalg.norm(o2_positions, axis=1)
        
        scatter = ax.scatter(o2_positions[:, 0], o2_positions[:, 1], o2_positions[:, 2],
                           c=distances, cmap='viridis', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Mark origin (oscillatory hole)
        ax.scatter([0], [0], [0], c='red', s=300, marker='*',
                  edgecolors='black', linewidth=2, label='Hole Center', zorder=10)
        
        # Styling
        ax.set_xlabel('X (Å)', fontsize=8)
        ax.set_ylabel('Y (Å)', fontsize=8)
        ax.set_zlabel('Z (Å)', fontsize=8)
        ax.set_title('A. O₂ Configuration (3D)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.view_init(elev=20, azim=45)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.1)
        cbar.set_label('Distance from center (Å)', fontsize=7)
        
        # Add info text
        ax.text2D(0.02, 0.98, f'N = {len(o2_positions)} O₂', 
                 transform=ax.transAxes, fontsize=7, va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        return ax
    
    def panel_b_distances(self, ax, o2_positions):
        """Panel B: Distance distribution from origin"""
        
        distances = np.linalg.norm(o2_positions, axis=1)
        
        # Histogram
        ax.hist(distances, bins=20, color='#3498db', alpha=0.7, 
               edgecolor='black', linewidth=1)
        
        # Statistics
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_dist:.2f} Å')
        ax.axvline(mean_dist - std_dist, color='orange', linestyle=':', linewidth=1.5,
                  alpha=0.7)
        ax.axvline(mean_dist + std_dist, color='orange', linestyle=':', linewidth=1.5,
                  alpha=0.7, label=f'±1σ: {std_dist:.2f} Å')
        
        # Styling
        ax.set_xlabel('Distance from Origin (Å)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Count', fontsize=9, fontweight='bold')
        ax.set_title('B. Radial Distance Distribution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def panel_c_pairwise_matrix(self, ax, o2_positions):
        """Panel C: Pairwise distance matrix"""
        
        # Calculate pairwise distances
        dist_matrix = squareform(pdist(o2_positions))
        
        # Plot heatmap
        im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto', alpha=0.8)
        
        # Styling
        ax.set_xlabel('O₂ Index', fontsize=9, fontweight='bold')
        ax.set_ylabel('O₂ Index', fontsize=9, fontweight='bold')
        ax.set_title('C. Pairwise Distance Matrix', fontsize=10, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance (Å)', fontsize=7)
        
        # Add statistics
        mean_pairwise = dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean()
        ax.text(0.02, 0.98, f'Mean pairwise:\n{mean_pairwise:.2f} Å', 
               transform=ax.transAxes, fontsize=7, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        return ax
    
    def panel_d_xy_projection(self, ax, o2_positions):
        """Panel D: XY projection"""
        
        distances = np.linalg.norm(o2_positions, axis=1)
        
        scatter = ax.scatter(o2_positions[:, 0], o2_positions[:, 1],
                           c=distances, cmap='viridis', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Mark origin
        ax.scatter([0], [0], c='red', s=200, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)
        
        # Styling
        ax.set_xlabel('X (Å)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=9, fontweight='bold')
        ax.set_title('D. XY Projection', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def panel_e_xz_projection(self, ax, o2_positions):
        """Panel E: XZ projection"""
        
        distances = np.linalg.norm(o2_positions, axis=1)
        
        scatter = ax.scatter(o2_positions[:, 0], o2_positions[:, 2],
                           c=distances, cmap='viridis', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Mark origin
        ax.scatter([0], [0], c='red', s=200, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)
        
        # Styling
        ax.set_xlabel('X (Å)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Z (Å)', fontsize=9, fontweight='bold')
        ax.set_title('E. XZ Projection', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def panel_f_yz_projection(self, ax, o2_positions):
        """Panel F: YZ projection"""
        
        distances = np.linalg.norm(o2_positions, axis=1)
        
        scatter = ax.scatter(o2_positions[:, 1], o2_positions[:, 2],
                           c=distances, cmap='viridis', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Mark origin
        ax.scatter([0], [0], c='red', s=200, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)
        
        # Styling
        ax.set_xlabel('Y (Å)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Z (Å)', fontsize=9, fontweight='bold')
        ax.set_title('F. YZ Projection', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def panel_g_radial_distribution(self, ax, o2_positions):
        """Panel G: Radial distribution function"""
        
        # Calculate all pairwise distances
        pairwise_dists = pdist(o2_positions)
        
        # Create histogram (radial distribution function)
        bins = np.linspace(0, pairwise_dists.max(), 50)
        counts, edges = np.histogram(pairwise_dists, bins=bins)
        
        # Normalize by shell volume (4πr²)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        shell_volumes = 4 * np.pi * bin_centers**2 * (edges[1] - edges[0])
        rdf = counts / shell_volumes
        rdf = rdf / rdf.max()  # Normalize to 1
        
        # Plot
        ax.plot(bin_centers, rdf, 'b-', linewidth=2)
        ax.fill_between(bin_centers, 0, rdf, alpha=0.3, color='blue')
        
        # Styling
        ax.set_xlabel('Distance (Å)', fontsize=9, fontweight='bold')
        ax.set_ylabel('g(r) (normalized)', fontsize=9, fontweight='bold')
        ax.set_title('G. Radial Distribution Function', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rdf, height=0.3)
        if len(peaks) > 0:
            ax.plot(bin_centers[peaks], rdf[peaks], 'ro', markersize=8,
                   label=f'{len(peaks)} peaks')
            ax.legend(fontsize=7)
        
        return ax
    
    def panel_h_pca(self, ax, o2_positions):
        """Panel H: PCA analysis"""
        
        # PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(o2_positions)
        
        # Color by distance from origin
        distances = np.linalg.norm(o2_positions, axis=1)
        
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=distances, cmap='viridis', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Styling
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                     fontsize=9, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                     fontsize=9, fontweight='bold')
        ax.set_title('H. PCA (2D Projection)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance (Å)', fontsize=7)
        
        return ax
    
    def panel_i_validation(self, ax):
        """Panel I: Validation metrics"""
        
        ax.axis('off')
        
        # Get validation data
        if len(self.validations) > 0:
            first_json = list(self.validations.keys())[0]
            val_data = self.validations[first_json]
            
            # Format as text
            y_pos = 0.95
            ax.text(0.5, y_pos, 'Validation Metrics', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            
            y_pos -= 0.1
            
            # Extract and display metrics
            if 'n_thoughts_captured' in val_data:
                ax.text(0.1, y_pos, f"Thoughts captured: {val_data['n_thoughts_captured']}", 
                       ha='left', va='top', fontsize=8, transform=ax.transAxes)
                y_pos -= 0.08
            
            if 'similarity' in val_data:
                sim = val_data['similarity']
                ax.text(0.1, y_pos, f"Similarity analysis:", 
                       ha='left', va='top', fontsize=8, fontweight='bold',
                       transform=ax.transAxes)
                y_pos -= 0.08
                
                for key, value in sim.items():
                    if isinstance(value, (int, float)):
                        ax.text(0.15, y_pos, f"{key}: {value:.4f}", 
                               ha='left', va='top', fontsize=7,
                               transform=ax.transAxes, family='monospace')
                        y_pos -= 0.07
            
            # Add JSON preview
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Full validation data:", 
                   ha='left', va='top', fontsize=7, fontweight='bold',
                   transform=ax.transAxes)
            y_pos -= 0.06
            
            json_str = json.dumps(val_data, indent=2)
            lines = json_str.split('\n')[:10]  # First 10 lines
            for line in lines:
                ax.text(0.1, y_pos, line[:50], 
                       ha='left', va='top', fontsize=5,
                       transform=ax.transAxes, family='monospace')
                y_pos -= 0.04
        else:
            ax.text(0.5, 0.5, 'No validation data available', 
                   ha='center', va='center', fontsize=10,
                   transform=ax.transAxes, style='italic')
        
        return ax
    
    def compare_libraries(self, output='thought_library_comparison'):
        """Compare multiple thought libraries"""
        
        if len(self.thoughts) < 2:
            print("⚠ Need at least 2 libraries to compare")
            return
        
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Extract all O2 positions
        all_o2 = []
        labels = []
        
        for npz_file, thought_data in self.thoughts.items():
            o2_key = [k for k in thought_data.keys() if 'o2_positions' in k.lower()][0]
            o2_positions = thought_data[o2_key]
            all_o2.append(o2_positions)
            labels.append(Path(npz_file).stem)
        
        # Panel A: Overlaid 3D configurations
        ax_a = fig.add_subplot(gs[0, 0], projection='3d')
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, (o2, label) in enumerate(zip(all_o2, labels)):
            ax_a.scatter(o2[:, 0], o2[:, 1], o2[:, 2],
                        c=colors[i % len(colors)], s=30, alpha=0.5,
                        label=label)
        ax_a.set_title('A. Overlaid Configurations', fontsize=10, fontweight='bold')
        ax_a.legend(fontsize=6)
        
        # Panel B: Distance distributions
        ax_b = fig.add_subplot(gs[0, 1])
        for i, (o2, label) in enumerate(zip(all_o2, labels)):
            distances = np.linalg.norm(o2, axis=1)
            ax_b.hist(distances, bins=20, alpha=0.5, label=label,
                     color=colors[i % len(colors)])
        ax_b.set_xlabel('Distance (Å)', fontsize=9, fontweight='bold')
        ax_b.set_ylabel('Count', fontsize=9, fontweight='bold')
        ax_b.set_title('B. Distance Distributions', fontsize=10, fontweight='bold')
        ax_b.legend(fontsize=7)
        ax_b.grid(True, alpha=0.3)
        
        # Panel C: Statistics comparison
        ax_c = fig.add_subplot(gs[0, 2])
        stats_names = ['Mean dist', 'Std dist', 'Radius gyration']
        x = np.arange(len(stats_names))
        width = 0.8 / len(all_o2)
        
        for i, (o2, label) in enumerate(zip(all_o2, labels)):
            distances = np.linalg.norm(o2, axis=1)
            com = o2.mean(axis=0)
            rg = np.sqrt(np.mean(np.sum((o2 - com)**2, axis=1)))
            
            stats = [distances.mean(), distances.std(), rg]
            ax_c.bar(x + i*width, stats, width, label=label,
                    color=colors[i % len(colors)], alpha=0.7)
        
        ax_c.set_xticks(x + width * (len(all_o2)-1) / 2)
        ax_c.set_xticklabels(stats_names, fontsize=7)
        ax_c.set_ylabel('Value (Å)', fontsize=9, fontweight='bold')
        ax_c.set_title('C. Statistics Comparison', fontsize=10, fontweight='bold')
        ax_c.legend(fontsize=6)
        ax_c.grid(True, alpha=0.3, axis='y')
        
        # Panel D: Similarity matrix between libraries
        ax_d = fig.add_subplot(gs[1, 0])
        n_libs = len(all_o2)
        similarity_matrix = np.zeros((n_libs, n_libs))
        
        for i in range(n_libs):
            for j in range(n_libs):
                # Calculate similarity (inverse of mean distance difference)
                dist_i = np.linalg.norm(all_o2[i], axis=1)
                dist_j = np.linalg.norm(all_o2[j], axis=1)
                similarity_matrix[i, j] = 1 / (1 + np.abs(dist_i.mean() - dist_j.mean()))
        
        im = ax_d.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax_d.set_xticks(range(n_libs))
        ax_d.set_yticks(range(n_libs))
        ax_d.set_xticklabels([f'L{i+1}' for i in range(n_libs)], fontsize=7)
        ax_d.set_yticklabels([f'L{i+1}' for i in range(n_libs)], fontsize=7)
        ax_d.set_title('D. Library Similarity', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
        
        # Panel E: PCA of all configurations
        ax_e = fig.add_subplot(gs[1, 1])
        all_points = np.vstack(all_o2)
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_points)
        
        start_idx = 0
        for i, (o2, label) in enumerate(zip(all_o2, labels)):
            end_idx = start_idx + len(o2)
            ax_e.scatter(all_2d[start_idx:end_idx, 0], 
                        all_2d[start_idx:end_idx, 1],
                        c=colors[i % len(colors)], s=30, alpha=0.6,
                        label=label)
            start_idx = end_idx
        
        ax_e.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                       fontsize=9, fontweight='bold')
        ax_e.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                       fontsize=9, fontweight='bold')
        ax_e.set_title('E. Combined PCA', fontsize=10, fontweight='bold')
        ax_e.legend(fontsize=6)
        ax_e.grid(True, alpha=0.3)
        
        # Panel F: Validation comparison
        ax_f = fig.add_subplot(gs[1, 2])
        ax_f.axis('off')
        
        y_pos = 0.95
        ax_f.text(0.5, y_pos, 'Validation Comparison', 
                 ha='center', va='top', fontsize=10, fontweight='bold',
                 transform=ax_f.transAxes)
        y_pos -= 0.15
        
        for json_file, val_data in self.validations.items():
            label = Path(json_file).stem
            ax_f.text(0.1, y_pos, f"{label}:", 
                     ha='left', va='top', fontsize=8, fontweight='bold',
                     transform=ax_f.transAxes)
            y_pos -= 0.08
            
            if 'n_thoughts_captured' in val_data:
                ax_f.text(0.15, y_pos, f"Thoughts: {val_data['n_thoughts_captured']}", 
                         ha='left', va='top', fontsize=7,
                         transform=ax_f.transAxes)
                y_pos -= 0.08
        
        # Overall title
        fig.suptitle('Thought Library Comparison', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        plt.savefig(f'{output}.pdf', bbox_inches='tight')
        plt.savefig(f'{output}.png', bbox_inches='tight')
        print(f"\n✓ Saved: {output}.pdf/.png")
        
        return fig

# Run analysis
if __name__ == "__main__":
    
    # File paths
    npz_files = [
        'thought_library_20251026_051925.npz',
        'thought_library_20251026_052257.npz',
    ]
    
    json_files = [
        'validation_20251026_051925.json',
        'validation_20251026_052257.json',
    ]
    
    # Create inspector
    inspector = ThoughtLibraryInspector(npz_files, json_files)
    
    # Load all data
    thoughts, validations = inspector.load_all()
    
    # Analyze structure
    inspector.analyze_structure()
    
    # Analyze validations
    inspector.analyze_validations()
    
    # Create visualizations
    inspector.create_visualization('thought_library_analysis')
    
    # Compare libraries
    inspector.compare_libraries('thought_library_comparison')
    
    print("\n" + "="*70)
    print("✓✓✓ THOUGHT LIBRARY ANALYSIS COMPLETE ✓✓✓")
    print("="*70)
    print("\nGenerated files:")
    print("  - thought_library_analysis.pdf/.png")
    print("  - thought_library_comparison.pdf/.png")
    print("\n")
