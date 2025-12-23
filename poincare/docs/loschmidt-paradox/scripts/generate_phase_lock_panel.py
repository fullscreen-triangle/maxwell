#!/usr/bin/env python3
"""
Generate Panel L-2: Phase-Lock Network Evolution
Shows network densification during mixing and residual edges after separation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def create_separated_network(n_per_container=10, seed=42):
    """Create two disconnected networks (separated gases)."""
    np.random.seed(seed)
    G = nx.Graph()
    
    # Container A nodes
    for i in range(n_per_container):
        G.add_node(i, container='A', pos=(np.random.uniform(0, 1), np.random.uniform(0, 1)))
    
    # Container B nodes
    for i in range(n_per_container, 2*n_per_container):
        G.add_node(i, container='B', pos=(np.random.uniform(1.5, 2.5), np.random.uniform(0, 1)))
    
    # Edges within containers only
    for i in range(n_per_container):
        for j in range(i+1, n_per_container):
            if np.random.random() < 0.3:
                G.add_edge(i, j, weight=np.random.uniform(0.5, 1.0))
    
    for i in range(n_per_container, 2*n_per_container):
        for j in range(i+1, 2*n_per_container):
            if np.random.random() < 0.3:
                G.add_edge(i, j, weight=np.random.uniform(0.5, 1.0))
    
    return G

def create_mixed_network(n_total=20, seed=43):
    """Create a connected network (mixed gases)."""
    np.random.seed(seed)
    G = nx.Graph()
    
    # All nodes in combined space
    for i in range(n_total):
        container = 'A' if i < n_total//2 else 'B'
        G.add_node(i, container=container, pos=(np.random.uniform(0, 2.5), np.random.uniform(0, 1)))
    
    # Edges including cross-container
    for i in range(n_total):
        for j in range(i+1, n_total):
            if np.random.random() < 0.25:
                G.add_edge(i, j, weight=np.random.uniform(0.5, 1.0))
    
    return G

def create_reseparated_network(n_per_container=10, seed=44):
    """Create separated network with residual cross-edges."""
    np.random.seed(seed)
    G = nx.Graph()
    
    # Container A nodes
    for i in range(n_per_container):
        G.add_node(i, container='A', pos=(np.random.uniform(0, 1), np.random.uniform(0, 1)))
    
    # Container B nodes
    for i in range(n_per_container, 2*n_per_container):
        G.add_node(i, container='B', pos=(np.random.uniform(1.5, 2.5), np.random.uniform(0, 1)))
    
    # Edges within containers
    for i in range(n_per_container):
        for j in range(i+1, n_per_container):
            if np.random.random() < 0.3:
                G.add_edge(i, j, weight=np.random.uniform(0.5, 1.0), type='internal')
    
    for i in range(n_per_container, 2*n_per_container):
        for j in range(i+1, 2*n_per_container):
            if np.random.random() < 0.3:
                G.add_edge(i, j, weight=np.random.uniform(0.5, 1.0), type='internal')
    
    # Residual cross-container edges (from mixing)
    for i in range(n_per_container):
        for j in range(n_per_container, 2*n_per_container):
            if np.random.random() < 0.08:  # Fewer but still present
                G.add_edge(i, j, weight=np.random.uniform(0.3, 0.6), type='residual')
    
    return G

def draw_network(ax, G, title, show_residual=False):
    """Draw network on axis."""
    pos = nx.get_node_attributes(G, 'pos')
    containers = nx.get_node_attributes(G, 'container')
    
    # Node colors
    node_colors = ['royalblue' if containers[n] == 'A' else 'tomato' for n in G.nodes()]
    
    # Draw edges
    if show_residual:
        internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'residual']
        residual_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'residual']
        
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges, ax=ax, 
                               edge_color='gray', alpha=0.5, width=1)
        nx.draw_networkx_edges(G, pos, edgelist=residual_edges, ax=ax, 
                               edge_color='red', alpha=0.8, width=2, style='dashed')
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5, width=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=200, edgecolors='black', linewidths=0.5)
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.axis('off')
    
    return len(G.edges())

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    # Panel A: Separated Network
    ax1 = fig.add_subplot(gs[0, 0])
    G_sep = create_separated_network()
    n_edges_sep = draw_network(ax1, G_sep, '(A) Initial: Separated Networks')
    ax1.text(1.25, -0.15, f'|E| = {n_edges_sep} edges\nTwo disconnected components', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel B: Mixed Network
    ax2 = fig.add_subplot(gs[0, 1])
    G_mix = create_mixed_network()
    n_edges_mix = draw_network(ax2, G_mix, '(B) Mixed: Connected Network')
    ax2.text(1.25, -0.15, f'|E| = {n_edges_mix} edges\nSingle connected component', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Panel C: Re-separated Network with residual edges
    ax3 = fig.add_subplot(gs[1, 0])
    G_resep = create_reseparated_network()
    n_edges_resep = draw_network(ax3, G_resep, '(C) Re-separated: Residual Edges Persist', show_residual=True)
    n_residual = len([e for e in G_resep.edges(data=True) if e[2].get('type') == 'residual'])
    ax3.text(1.25, -0.15, f'|E| = {n_edges_resep} edges\n{n_residual} residual cross-edges (red dashed)', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    # Panel D: Edge count evolution
    ax4 = fig.add_subplot(gs[1, 1])
    
    stages = ['Initial\n(separated)', 'Mixed', 'Re-separated']
    edge_counts = [n_edges_sep, n_edges_mix, n_edges_resep]
    colors = ['royalblue', 'green', 'darkorange']
    
    bars = ax4.bar(stages, edge_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, edge_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('Number of Edges |E|', fontsize=11)
    ax4.set_title('(D) Edge Count Evolution', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, max(edge_counts) * 1.2)
    
    # Add inequality annotation
    ax4.annotate('', xy=(2, edge_counts[2]), xytext=(0, edge_counts[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(1, max(edge_counts) * 1.05, r'$|E_{final}| > |E_{initial}|$', 
             ha='center', fontsize=12, color='red', fontweight='bold')
    
    # Key insight box
    ax4.text(1, -max(edge_counts)*0.15, 
             'More edges → more constraints → higher entropy\nResidual edges = categorical memory of mixing',
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Panel L-2: Phase-Lock Network Densification and Residual Correlations', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_phase_lock_network.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_phase_lock_network.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_phase_lock_network.png'}")
    print(f"Saved: {output_dir / 'panel_phase_lock_network.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

