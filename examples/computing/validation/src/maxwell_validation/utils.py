"""
Utility Functions for Maxwell's Demon Validation Experiments
=============================================================

Core utilities for:
- Molecular configuration generation
- Velocity assignment (Maxwell-Boltzmann)
- Phase-lock network construction
- Network property calculation
- Distance metric computation
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
from scipy.sparse.csgraph import connected_components
from typing import Tuple, Dict, List, Any


def shortest_path(network, directed=False):
    """Wrapper for scipy shortest path"""
    return scipy_shortest_path(network, directed=directed)


def generate_configuration(n_molecules: int = 100, 
                          box_size: float = 10.0, 
                          seed: int = None) -> np.ndarray:
    """
    Generate random molecular configuration
    
    Args:
        n_molecules: Number of molecules
        box_size: Size of simulation box
        seed: Random seed for reproducibility
    
    Returns:
        positions: (n_molecules, 3) array of positions
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, box_size, (n_molecules, 3))


def assign_velocities(positions: np.ndarray, 
                      temperature: float, 
                      mass: float = 1.0, 
                      k_B: float = 1.0, 
                      seed: int = None) -> np.ndarray:
    """
    Assign Maxwell-Boltzmann velocities at given temperature
    
    Args:
        positions: Molecular positions
        temperature: System temperature
        mass: Molecular mass
        k_B: Boltzmann constant
        seed: Random seed
    
    Returns:
        velocities: (n_molecules, 3) array of velocities
    """
    if seed is not None:
        np.random.seed(seed)
    n_molecules = len(positions)
    sigma = np.sqrt(k_B * temperature / mass)
    return np.random.normal(0, sigma, (n_molecules, 3))


def build_phase_lock_network(positions: np.ndarray, 
                             cutoff: float = 2.0, 
                             interaction_type: str = 'vdw') -> np.ndarray:
    """
    Build phase-lock network from spatial configuration
    
    The phase-lock network is determined by spatial proximity,
    NOT by molecular velocities. This is a key insight.
    
    Args:
        positions: Molecular positions
        cutoff: Interaction cutoff distance
        interaction_type: 'vdw' (r^-6) or 'dipole' (r^-3)
    
    Returns:
        adjacency: Adjacency matrix of phase-lock network
    """
    distances = squareform(pdist(positions))
    
    if interaction_type == 'vdw':
        # Van der Waals: U ~ r^-6
        # Network edge if interaction significant
        adjacency = (distances < cutoff) & (distances > 0)
    elif interaction_type == 'dipole':
        # Dipole: U ~ r^-3
        adjacency = (distances < cutoff * 1.5) & (distances > 0)
    else:
        adjacency = (distances < cutoff) & (distances > 0)
    
    return adjacency.astype(float)


def calculate_network_properties(adjacency: np.ndarray) -> Dict[str, float]:
    """
    Calculate topological properties of phase-lock network
    
    Args:
        adjacency: Network adjacency matrix
    
    Returns:
        props: Dictionary of network properties
    """
    n_nodes = len(adjacency)
    degree = adjacency.sum(axis=1)
    n_edges = adjacency.sum() / 2
    
    # Clustering coefficient
    clustering = []
    for i in range(n_nodes):
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) < 2:
            clustering.append(0)
        else:
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            actual = adjacency[np.ix_(neighbors, neighbors)].sum() / 2
            clustering.append(actual / possible if possible > 0 else 0)
    
    # Average path length
    dist_matrix = scipy_shortest_path(adjacency, directed=False)
    finite_dists = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)]
    avg_path_length = finite_dists.mean() if len(finite_dists) > 0 else 0
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'degree_mean': float(degree.mean()),
        'degree_std': float(degree.std()),
        'degree_max': float(degree.max()),
        'clustering_mean': float(np.mean(clustering)),
        'clustering_std': float(np.std(clustering)),
        'avg_path_length': float(avg_path_length),
        'density': float(n_edges / (n_nodes * (n_nodes - 1) / 2)) if n_nodes > 1 else 0
    }


def calculate_kinetic_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    """Calculate total kinetic energy"""
    return float(0.5 * mass * (velocities**2).sum())


def calculate_temperature(velocities: np.ndarray, 
                          mass: float = 1.0, 
                          k_B: float = 1.0) -> float:
    """Calculate temperature from kinetic energy"""
    n_molecules = len(velocities)
    KE = calculate_kinetic_energy(velocities, mass)
    # Equipartition: <KE> = (3/2) N k_B T
    return float(KE / (1.5 * n_molecules * k_B))


def calculate_entropy(network_props: Dict[str, float]) -> float:
    """
    Calculate topological entropy from network properties
    
    Entropy increases with network densification.
    This is the categorical entropy, not spatial entropy.
    """
    return float(network_props['n_edges'] / network_props['n_nodes'])


def identify_clusters(network: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Identify connected components (clusters) in phase-lock network
    
    Args:
        network: Adjacency matrix
    
    Returns:
        n_clusters: Number of clusters
        labels: Cluster label for each node
    """
    n_clusters, labels = connected_components(network, directed=False)
    return n_clusters, labels


def calculate_three_distances(positions: np.ndarray, 
                              velocities: np.ndarray, 
                              network: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate three types of distance metrics
    
    This demonstrates the key insight: three distances are INEQUIVALENT.
    
    Args:
        positions: Molecular positions
        velocities: Molecular velocities
        network: Phase-lock network adjacency
    
    Returns:
        spatial_dist: Spatial distance matrix (Euclidean)
        kinetic_dist: Kinetic distance matrix (velocity difference)
        categorical_dist: Categorical (network) distance matrix (shortest path)
    """
    n_molecules = len(positions)
    
    # Spatial distance (Euclidean in position space)
    spatial_dist = squareform(pdist(positions))
    
    # Kinetic distance (Euclidean in velocity space)
    kinetic_dist = squareform(pdist(velocities))
    
    # Categorical distance (shortest path in network)
    categorical_dist = scipy_shortest_path(network, directed=False)
    categorical_dist[np.isinf(categorical_dist)] = n_molecules  # Replace inf
    
    return spatial_dist, kinetic_dist, categorical_dist


def print_header(title: str, width: int = 70) -> None:
    """Print formatted section header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_results(results_dict: Dict[str, Any], indent: int = 2) -> None:
    """Print formatted results dictionary"""
    spaces = " " * indent
    for key, value in results_dict.items():
        if isinstance(value, float):
            print(f"{spaces}{key}: {value:.6f}")
        elif isinstance(value, dict):
            print(f"{spaces}{key}:")
            print_results(value, indent + 2)
        else:
            print(f"{spaces}{key}: {value}")


def validate_configuration(positions: np.ndarray, 
                           velocities: np.ndarray,
                           expected_temperature: float,
                           tolerance: float = 0.2) -> Dict[str, bool]:
    """
    Validate that a configuration matches expectations
    
    Args:
        positions: Molecular positions
        velocities: Molecular velocities  
        expected_temperature: Expected temperature
        tolerance: Relative tolerance for temperature check
    
    Returns:
        Dictionary of validation results
    """
    T_actual = calculate_temperature(velocities)
    
    return {
        'positions_valid': len(positions) > 0 and not np.any(np.isnan(positions)),
        'velocities_valid': len(velocities) > 0 and not np.any(np.isnan(velocities)),
        'temperature_valid': abs(T_actual - expected_temperature) / expected_temperature < tolerance,
        'shapes_match': positions.shape == velocities.shape,
    }
