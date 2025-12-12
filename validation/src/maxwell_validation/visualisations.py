"""
Publication-quality visualization for Maxwell's Demon validation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")


class MaxwellDemonVisualizer:
    """Create publication-quality figures"""
    
    def __init__(self, results):
        """
        Initialize visualizer with experimental results
        
        Args:
            results: Dictionary of results from experiments
        """
        self.results = results
        self.fig_size = (20, 14)
    
    def create_publication_figure(self, save_path='maxwell_demon_resolution'):
        """
        Create complete 7-panel publication figure
        
        Args:
            save_path: Base path for saving (without extension)
        """
        print("\nCreating publication figure...")
        
        fig = plt.figure(figsize=self.fig_size)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        # Panel A: Temperature Independence
        ax_a = fig.add_subplot(gs[0, :2])
        self._plot_temperature_independence(ax_a)
        
        # Panel B: Kinetic Independence
        ax_b = fig.add_subplot(gs[0, 2])
        self._plot_kinetic_independence(ax_b)
        
        # Panel C: Distance Inequivalence
        ax_c = fig.add_subplot(gs[1, 0])
        self._plot_distance_inequivalence(ax_c)
        
        # Panel D: Temperature Emergence
        ax_d = fig.add_subplot(gs[1, 1])
        self._plot_temperature_emergence(ax_d)
        
        # Panel E: Sorting Entropy
        ax_e = fig.add_subplot(gs[1, 2])
        self._plot_sorting_entropy(ax_e)
        
        # Panel F: Velocity Blindness
        ax_f = fig.add_subplot(gs[2, 0])
        self._plot_velocity_blindness(ax_f)
        
        # Panel G: Complementarity
        ax_g = fig.add_subplot(gs[2, 1:])
        self._plot_complementarity(ax_g)
        
        # Overall title
        fig.suptitle(
            'Resolution of Maxwell\'s Demon: Seven-Fold Dissolution\n' +
            'Phase-Lock Network Topology and Categorical Completion',
            fontsize=18, fontweight='bold', y=0.98
        )
        
        # Save
        plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        
        print(f"✓ Saved: {save_path}.pdf")
        print(f"✓ Saved: {save_path}.png")
        
        return fig
    
    def _plot_temperature_independence(self, ax):
        """Panel A: Temperature independence"""
        df = self.results['exp1']['dataframe']
        
        ax2 = ax.twinx()
        
        # Network edges (constant)
        line1 = ax.plot(df['temperature'], df['n_edges'], 
                       'o-', linewidth=2.5, markersize=10, 
                       color='#2E86AB', label='Network Edges')
        
        # Kinetic energy (scales with T)
        line2 = ax2.plot(df['temperature'], df['kinetic_energy'],
                        's-', linewidth=2.5, markersize=10,
                        color='#A23B72', label='Kinetic Energy')
        
        ax.set_xlabel('Temperature', fontsize=13, fontweight='bold')
        ax.set_ylabel('Network Edges (constant)', fontsize=12, 
                     color='#2E86AB', fontweight='bold')
        ax2.set_ylabel('Kinetic Energy (∝ T)', fontsize=12,
                      color='#A23B72', fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='#2E86AB', labelsize=11)
        ax2.tick_params(axis='y', labelcolor='#A23B72', labelsize=11)
        ax.tick_params(axis='x', labelsize=11)
        
        ax.set_title('A. Temperature Independence: ∂G/∂T = 0',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)
    
    def _plot_kinetic_independence(self, ax):
        """Panel B: Kinetic independence"""
        df = self.results['exp2']['dataframe']
        corr = self.results['exp2']['correlations']['KE_vs_edges']
        
        # Sample for clarity
        sample_size = min(500, len(df))
        idx = np.random.choice(len(df), sample_size, replace=False)
        
        ax.scatter(df['kinetic_energy'].iloc[idx], df['n_edges'].iloc[idx],
                  alpha=0.4, s=30, color='#F18F01')
        
        ax.set_xlabel('Kinetic Energy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Network Edges', fontsize=12, fontweight='bold')
        ax.set_title(f'B. Kinetic Independence\n∂G/∂E_kin = 0\n(r = {corr:.4f})',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
    
    def _plot_distance_inequivalence(self, ax):
        """Panel C: Distance inequivalence"""
        spatial = self.results['exp3']['spatial']
        categorical = self.results['exp3']['categorical']
        corr = self.results['exp3']['correlations']['spatial_vs_categorical']
        
        # Sample for clarity
        sample_size = min(1000, len(spatial))
        idx = np.random.choice(len(spatial), sample_size, replace=False)
        
        ax.scatter(spatial[idx], categorical[idx],
                  alpha=0.3, s=20, color='#6A994E')
        
        ax.set_xlabel('Spatial Distance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Categorical Distance', fontsize=12, fontweight='bold')
        ax.set_title(f'C. Distance Inequivalence\n(r = {corr:.3f})',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
    
    def _plot_temperature_emergence(self, ax):
        """Panel D: Temperature emergence"""
        global_T = self.results['exp4']['global_temp']
        cluster_temps = self.results['exp4']['cluster_temps']
        
        ax.hist(cluster_temps, bins=20, alpha=0.7, 
               edgecolor='black', color='#BC4B51', linewidth=1.5)
        ax.axvline(global_T, color='#2E86AB', linestyle='--', 
                  linewidth=3, label=f'Global T = {global_T:.2f}')
        
        ax.set_xlabel('Cluster Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('D. Temperature Emergence\nfrom Cluster Statistics',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(labelsize=11)
    
    def _plot_sorting_entropy(self, ax):
        """Panel E: Sorting increases entropy"""
        entropy = self.results['exp5']['entropy']
        
        steps = np.arange(len(entropy))
        ax.plot(steps, entropy, linewidth=2.5, color='#A23B72')
        ax.fill_between(steps, entropy.min(), entropy, alpha=0.3, color='#A23B72')
        
        delta_S = entropy[-1] - entropy[0]
        
        ax.set_xlabel('Sorting Attempts', fontsize=12, fontweight='bold')
        ax.set_ylabel('Entropy (network density)', fontsize=12, fontweight='bold')
        ax.set_title(f'E. "Sorting" Increases Entropy\nΔS = +{delta_S:.3f}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Annotate increase
        ax.annotate('', xy=(len(entropy)-1, entropy[-1]), 
                   xytext=(len(entropy)-1, entropy[0]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(len(entropy)*0.85, (entropy[0] + entropy[-1])/2,
               f'ΔS > 0', fontsize=11, color='red', fontweight='bold')
    
    def _plot_velocity_blindness(self, ax):
        """Panel F: Velocity blindness"""
        df = self.results['exp6']['dataframe']
        
        ax.scatter(df['temperature'], df['mean_vel_diff'],
                  alpha=0.5, s=40, color='#F18F01')
        
        match_pct = (df['paths_match'].sum() / len(df)) * 100
        
        ax.set_xlabel('Temperature (varies)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Velocity Difference', fontsize=12, fontweight='bold')
        ax.set_title(f'F. Velocity-Blind Completion\n' +
                    f'Categorical paths identical: {match_pct:.0f}%',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Add text box
        textstr = 'Categorical paths\nindependent of\nvelocity distribution'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', bbox=props)
    
    def _plot_complementarity(self, ax):
        """Panel G: Information complementarity"""
        ax.axis('off')
        
        # Create text explanation
        text = (
            'Information Complementarity: Two Conjugate Faces\n\n'
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n'
            'KINETIC FACE                    CATEGORICAL FACE\n'
            '(Observable)                    (Observable)\n\n'
            '• Molecular velocities          • Phase-lock network\n'
            '• Kinetic energy                • Network topology\n'
            '• Temperature                   • Categorical distances\n'
            '• Speed distributions           • Cluster structure\n\n'
            'Hidden: Network topology        Hidden: Velocities\n\n'
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n'
            'COMPLEMENTARITY (like ammeter/voltmeter)\n\n'
            '• Cannot observe both faces simultaneously\n'
            '• Conjugate observables of same phenomenon\n'
            '• Measurement incompatibility\n\n'
            'Maxwell observed ONLY kinetic face\n'
            '→ Categorical dynamics were HIDDEN\n'
            '→ "Demon" = projection of hidden face onto observable face\n'
            '→ Not an agent, but a SHADOW\n\n'
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        )
        
        ax.text(0.5, 0.5, text, ha='center', va='center',
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#E8F4F8', 
                        alpha=0.9, edgecolor='#2E86AB', linewidth=2))
        
        ax.set_title('G. Information Complementarity',
                    fontsize=14, fontweight='bold', pad=15)


def create_supplementary_figures(results, save_prefix='supplementary'):
    """Create additional supplementary figures"""
    print("\nCreating supplementary figures...")
    
    # Supplementary Figure 1: 3D distance comparison
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    spatial = results['exp3']['spatial']
    kinetic = results['exp3']['kinetic']
    categorical = results['exp3']['categorical']
    
    # Sample for clarity
    sample_size = min(500, len(spatial))
    idx = np.random.choice(len(spatial), sample_size, replace=False)
    
    scatter = ax1.scatter(spatial[idx], kinetic[idx], categorical[idx],
                         c=categorical[idx], cmap='viridis', alpha=0.6, s=30)
    
    ax1.set_xlabel('Spatial Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kinetic Distance', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Categorical Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Three Inequivalent Distance Metrics\n(3D Visualization)',
                 fontsize=14, fontweight='bold', pad=20)
    
    fig1.colorbar(scatter, ax=ax1, label='Categorical Distance', shrink=0.5)
    
    plt.savefig(f'{save_prefix}_3d_distances.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_3d_distances.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_prefix}_3d_distances.pdf/png")
    
    plt.close(fig1)
    
    # Supplementary Figure 2: Network visualization
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # This would require networkx for proper visualization
    # Placeholder for now
    for ax in axes:
        ax.text(0.5, 0.5, 'Network visualization\n(requires networkx)',
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    fig2.suptitle('Phase-Lock Network Topology',
                 fontsize=16, fontweight='bold')
    
    plt.savefig(f'{save_prefix}_network.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_network.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_prefix}_network.pdf/png")
    
    plt.close(fig2)


if __name__ == "__main__":
    print("Visualization module - import and use with experimental results")
