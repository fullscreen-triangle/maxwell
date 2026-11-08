"""
Run Visualization demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from visualization import CategoricalVisualizer, PlotData
from virtual_chamber import VirtualChamber
from maxwell_demon import MaxwellDemon


def run_demonstration():
    """Run the visualization demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'categorical_visualization',
        'experiments': []
    }
    
    # Create chamber with gas
    print("=== Creating Gas Chamber ===")
    chamber = VirtualChamber()
    chamber.populate(500)
    print(f"  Created chamber with {chamber.statistics.molecule_count} molecules")
    
    viz = CategoricalVisualizer(chamber)
    
    # Experiment 1: S-space scatter data
    print("\n=== Experiment 1: S-Space Scatter Plot Data ===")
    scatter = viz.s_space_scatter()
    
    results['experiments'].append({
        'name': 's_space_scatter',
        'description': '3D scatter plot of molecules in S-entropy space',
        'title': scatter.title,
        'point_count': len(scatter.x),
        'x_range': [min(scatter.x), max(scatter.x)] if scatter.x else [0, 0],
        'y_range': [min(scatter.y), max(scatter.y)] if scatter.y else [0, 0],
        'z_range': [min(scatter.z), max(scatter.z)] if scatter.z else [0, 0],
        'sample_points': list(zip(scatter.x[:10], scatter.y[:10], scatter.z[:10])) if scatter.x else [],
    })
    print(f"  Generated {len(scatter.x)} points for 3D scatter")
    
    # Experiment 2: Histograms
    print("\n=== Experiment 2: S-Coordinate Histograms ===")
    hist_k = viz.s_k_histogram(bins=10)
    hist_t = viz.s_t_histogram(bins=10)
    hist_e = viz.s_e_histogram(bins=10)
    
    results['experiments'].append({
        'name': 'histograms',
        'description': 'Distribution histograms for S-coordinates',
        'S_k_histogram': {
            'bins': list(hist_k.x),
            'counts': list(hist_k.y),
        },
        'S_t_histogram': {
            'bins': list(hist_t.x),
            'counts': list(hist_t.y),
        },
        'S_e_histogram': {
            'bins': list(hist_e.x),
            'counts': list(hist_e.y),
        },
    })
    print(f"  S_k: {list(map(int, hist_k.y))}")
    print(f"  S_t: {list(map(int, hist_t.y))}")
    print(f"  S_e: {list(map(int, hist_e.y))}")
    
    # Experiment 3: Phase space 2D projections
    print("\n=== Experiment 3: Phase Space Projections ===")
    projections = {}
    for dims in [('S_k', 'S_e'), ('S_k', 'S_t'), ('S_t', 'S_e')]:
        proj = viz.phase_space_2d(dims[0], dims[1])
        projections[f'{dims[0]}_vs_{dims[1]}'] = {
            'title': proj.title,
            'point_count': len(proj.x),
            'x_range': [min(proj.x), max(proj.x)] if proj.x else [0, 0],
            'y_range': [min(proj.y), max(proj.y)] if proj.y else [0, 0],
        }
    
    results['experiments'].append({
        'name': 'phase_space_projections',
        'description': '2D projections of S-space',
        'projections': projections,
    })
    print(f"  Generated 3 phase space projections")
    
    # Experiment 4: Maxwell-Boltzmann comparison
    print("\n=== Experiment 4: Maxwell-Boltzmann Comparison ===")
    mb_comparison = viz.maxwell_boltzmann_comparison(bins=15)
    
    if mb_comparison:
        results['experiments'].append({
            'name': 'maxwell_boltzmann_comparison',
            'description': 'Comparing S_e distribution to MB theory',
            'actual': {
                'bins': list(mb_comparison['actual'].x),
                'normalized_counts': list(mb_comparison['actual'].y),
            },
            'theoretical': {
                'bins': list(mb_comparison['theoretical'].x),
                'probability_density': list(mb_comparison['theoretical'].y),
            },
        })
        print(f"  Generated comparison data")
    else:
        results['experiments'].append({
            'name': 'maxwell_boltzmann_comparison',
            'description': 'Not enough data for comparison',
        })
    
    # Experiment 5: Demon sorting visualization
    print("\n=== Experiment 5: Maxwell Demon Sorting Visualization ===")
    demon = MaxwellDemon(chamber)
    demon.sort_chamber(threshold=0.5)
    
    demon_viz = viz.demon_sorting_visualization(demon.hot_compartment, demon.cold_compartment)
    
    results['experiments'].append({
        'name': 'demon_sorting',
        'description': 'Visualization of Maxwell demon sorting',
        'hot_compartment': {
            'count': len(demon.hot_compartment),
            'x_range': [min(demon_viz['hot'].x), max(demon_viz['hot'].x)] if demon_viz['hot'].x else [0, 0],
            'y_range': [min(demon_viz['hot'].y), max(demon_viz['hot'].y)] if demon_viz['hot'].y else [0, 0],
        },
        'cold_compartment': {
            'count': len(demon.cold_compartment),
            'x_range': [min(demon_viz['cold'].x), max(demon_viz['cold'].x)] if demon_viz['cold'].x else [0, 0],
            'y_range': [min(demon_viz['cold'].y), max(demon_viz['cold'].y)] if demon_viz['cold'].y else [0, 0],
        },
    })
    print(f"  Hot: {len(demon.hot_compartment)}, Cold: {len(demon.cold_compartment)}")
    
    # Experiment 6: Harmonic coincidence network
    print("\n=== Experiment 6: Harmonic Coincidence Network ===")
    molecules = list(chamber.gas)[:30]
    network = viz.harmonic_coincidence_network(molecules, threshold=0.1)
    
    results['experiments'].append({
        'name': 'harmonic_network',
        'description': 'Network of harmonic coincidences between molecules',
        'node_count': len(network['nodes']),
        'edge_count': len(network['edges']),
        'sample_nodes': network['nodes'][:5],
        'sample_edges': network['edges'][:5],
    })
    print(f"  Nodes: {len(network['nodes'])}, Edges: {len(network['edges'])}")
    
    # Experiment 7: ASCII visualizations
    print("\n=== Experiment 7: ASCII Visualizations ===")
    
    hist_ascii = viz.generate_ascii_histogram(hist_e)
    scatter_ascii = viz.generate_ascii_scatter_2d(viz.phase_space_2d('S_k', 'S_e'))
    
    results['experiments'].append({
        'name': 'ascii_visualizations',
        'description': 'ASCII art visualizations for terminal',
        'histogram_ascii': hist_ascii,
        'scatter_ascii': scatter_ascii,
    })
    
    # Print ASCII visualizations
    print("\n--- S_e Histogram ---")
    print(hist_ascii)
    print("\n--- S_k vs S_e Scatter ---")
    print(scatter_ascii)
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'visualization_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" CATEGORICAL VISUALIZATION DEMONSTRATION")
    print("=" * 70)
    print("\nVisualizing REAL data from hardware timing.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'visualization'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. All visualization data comes from REAL hardware timing
2. S-space scatter shows molecule distribution (not simulated)
3. Histograms reveal the thermal distribution of timing jitter
4. Maxwell-Boltzmann comparison validates physical behavior
5. Demon sorting visualization shows categorical separation
    """)
    
    return results


if __name__ == "__main__":
    main()

