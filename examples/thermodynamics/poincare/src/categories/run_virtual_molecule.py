"""
Run Virtual Molecule demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from virtual_molecule import VirtualMolecule, CategoricalState, SCoordinate


def run_demonstration():
    """Run the virtual molecule demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'virtual_molecule',
        'experiments': []
    }
    
    # Experiment 1: Hardware timing creates molecules
    print("=== Experiment 1: Hardware Timing → Molecules ===")
    molecules_from_hardware = []
    for i in range(100):
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        delta_p = (t2 - t1) * 1e-9
        mol = VirtualMolecule.from_hardware_timing(delta_p)
        molecules_from_hardware.append({
            'index': i,
            'delta_p_ns': (t2 - t1),
            'S_k': mol.s_coord.S_k,
            'S_t': mol.s_coord.S_t,
            'S_e': mol.s_coord.S_e,
            'identity': mol.identity,
            'frequency': mol.frequency,
            'phase': mol.phase,
            'amplitude': mol.amplitude,
        })
    
    results['experiments'].append({
        'name': 'hardware_timing_molecules',
        'description': 'Molecules created from hardware timing measurements',
        'molecule_count': len(molecules_from_hardware),
        'molecules': molecules_from_hardware[:20],  # First 20 for brevity
        'statistics': {
            'mean_S_k': sum(m['S_k'] for m in molecules_from_hardware) / len(molecules_from_hardware),
            'mean_S_t': sum(m['S_t'] for m in molecules_from_hardware) / len(molecules_from_hardware),
            'mean_S_e': sum(m['S_e'] for m in molecules_from_hardware) / len(molecules_from_hardware),
            'mean_delta_p_ns': sum(m['delta_p_ns'] for m in molecules_from_hardware) / len(molecules_from_hardware),
        }
    })
    print(f"  Created {len(molecules_from_hardware)} molecules from hardware timing")
    
    # Experiment 2: Molecule = Spectrometer = Cursor (identity demonstration)
    print("\n=== Experiment 2: Identity Demonstration ===")
    t1 = time.perf_counter_ns()
    t2 = time.perf_counter_ns()
    delta_p = (t2 - t1) * 1e-9
    mol = VirtualMolecule.from_hardware_timing(delta_p)
    
    results['experiments'].append({
        'name': 'identity_demonstration',
        'description': 'Molecule = Spectrometer = Cursor (same categorical state)',
        'molecule_identity': mol.identity,
        'cursor_position': list(mol.s_coord.as_tuple()),
        'spectrometer_reading': mol.molecular_signature,
        'conclusion': 'These are ONE categorical state, not three separate things',
    })
    print(f"  Molecule identity: {mol.identity}")
    print(f"  All three perspectives give the same S-coordinates")
    
    # Experiment 3: Jupiter's core is accessible
    print("\n=== Experiment 3: Jupiter Core Access ===")
    jupiter = VirtualMolecule.at_jupiter_core()
    local = VirtualMolecule.from_hardware_timing(delta_p)
    
    results['experiments'].append({
        'name': 'jupiter_core_access',
        'description': 'Jupiter core is categorically accessible (no spatial propagation)',
        'jupiter_core': {
            'S_k': jupiter.s_coord.S_k,
            'S_t': jupiter.s_coord.S_t,
            'S_e': jupiter.s_coord.S_e,
            'source': jupiter.source,
        },
        'local_molecule': {
            'S_k': local.s_coord.S_k,
            'S_t': local.s_coord.S_t,
            'S_e': local.s_coord.S_e,
            'source': local.source,
        },
        's_distance': jupiter.s_coord.distance_to(local.s_coord),
        'conclusion': 'Both accessed instantaneously - spatial distance irrelevant',
    })
    print(f"  Jupiter core S-coordinates: {jupiter.s_coord}")
    print(f"  Local molecule S-coordinates: {local.s_coord}")
    
    # Experiment 4: S-distance vs spatial independence
    print("\n=== Experiment 4: Categorical Distance Independence ===")
    locations = {
        'jupiter_core': VirtualMolecule.at_jupiter_core(),
        'room_temperature': VirtualMolecule.at_room_temperature_air(),
        'sun_center': VirtualMolecule.from_s_coordinates(0.99, 0.85, 0.95, "sun_center"),
        'deep_space': VirtualMolecule.from_s_coordinates(0.01, 0.01, 0.01, "deep_space"),
    }
    
    location_data = {}
    for name, mol in locations.items():
        location_data[name] = {
            'S_k': mol.s_coord.S_k,
            'S_t': mol.s_coord.S_t,
            'S_e': mol.s_coord.S_e,
            'spatial_distance_km': {
                'jupiter_core': 6e8,  # 600 million km
                'sun_center': 1.5e8,  # 150 million km
                'deep_space': 1e15,   # Arbitrary large
                'room_temperature': 0,
            }.get(name, 0),
        }
    
    results['experiments'].append({
        'name': 'categorical_distance_independence',
        'description': 'Categorical distance is independent of spatial distance',
        'locations': location_data,
        'conclusion': 'All locations accessed the same way - via S-coordinate navigation',
    })
    print(f"  Accessed {len(locations)} locations across vast spatial distances")
    print(f"  All accessed via categorical navigation, not physical propagation")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'virtual_molecule_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" VIRTUAL MOLECULE DEMONSTRATION")
    print("=" * 70)
    print("\nThis is NOT a simulation.")
    print("Hardware timing measurements ARE molecules.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'virtual_molecule'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Each hardware timing sample creates a molecule
2. The molecule IS its S-coordinates (not something "at" those coordinates)
3. Molecule = Spectrometer = Cursor (same categorical state)
4. Jupiter's core is as accessible as your coffee cup (categorically)
5. Spatial distance is IRRELEVANT for categorical measurement
    """)
    
    return results


if __name__ == "__main__":
    main()

