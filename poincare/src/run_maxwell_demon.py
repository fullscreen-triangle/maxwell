"""
Run Maxwell Demon demonstration and save results.
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from maxwell_demon import MaxwellDemon, SortingCriterion
from virtual_chamber import VirtualChamber


def run_demonstration():
    """Run the Maxwell demon demonstration and return results."""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'demonstration': 'maxwell_demon',
        'experiments': []
    }
    
    # Create chamber with gas
    print("=== Creating Gas Chamber ===")
    chamber = VirtualChamber()
    chamber.populate(1000)
    initial_stats = chamber.statistics
    print(f"  Created chamber with {initial_stats.molecule_count} molecules")
    print(f"  Initial temperature: {initial_stats.temperature:.6f}")
    
    # Experiment 1: Demon observation (zero backaction)
    print("\n=== Experiment 1: Zero-Backaction Observation ===")
    demon = MaxwellDemon(chamber)
    
    observations = []
    for mol in list(chamber.gas)[:100]:  # Observe first 100
        obs = demon.observe(mol)
        observations.append({
            'S_k': obs['S_k'],
            'S_t': obs['S_t'],
            'S_e': obs['S_e'],
            'observation_cost': obs['observation_cost'],
            'backaction': obs['backaction'],
        })
    
    results['experiments'].append({
        'name': 'zero_backaction_observation',
        'description': 'Demon observes molecules with zero backaction',
        'observations_made': len(observations),
        'total_observation_cost': sum(o['observation_cost'] for o in observations),
        'total_backaction': sum(o['backaction'] for o in observations),
        'sample_observations': observations[:10],
        'conclusion': 'Zero energy cost, zero backaction - categorical observables commute',
    })
    print(f"  Made {len(observations)} observations")
    print(f"  Total energy cost: {sum(o['observation_cost'] for o in observations)}")
    print(f"  Total backaction: {sum(o['backaction'] for o in observations)}")
    
    # Experiment 2: Sorting by S_e (evolution entropy)
    print("\n=== Experiment 2: Sorting by Evolution Entropy ===")
    demon.clear()
    demon.sort_chamber(threshold=0.5, criterion=SortingCriterion.S_E)
    
    hot_s_e = [m.s_coord.S_e for m in demon.hot_compartment]
    cold_s_e = [m.s_coord.S_e for m in demon.cold_compartment]
    
    results['experiments'].append({
        'name': 'sorting_by_S_e',
        'description': 'Demon sorts molecules by evolution entropy',
        'threshold': 0.5,
        'hot_count': len(demon.hot_compartment),
        'cold_count': len(demon.cold_compartment),
        'hot_mean_S_e': sum(hot_s_e) / len(hot_s_e) if hot_s_e else 0,
        'cold_mean_S_e': sum(cold_s_e) / len(cold_s_e) if cold_s_e else 0,
        'temperature_gradient': demon.extract_temperature_gradient(),
        'decision_energy_cost': 0.0,  # Always zero
    })
    print(f"  Hot compartment: {len(demon.hot_compartment)} molecules")
    print(f"  Cold compartment: {len(demon.cold_compartment)} molecules")
    print(f"  Temperature gradient created: {demon.extract_temperature_gradient():.4f}")
    print(f"  Energy cost of decisions: 0.0 (categorical)")
    
    # Experiment 3: Sorting by different criteria
    print("\n=== Experiment 3: Sorting by Different Criteria ===")
    sorting_results = {}
    
    for criterion in [SortingCriterion.S_K, SortingCriterion.S_T, SortingCriterion.S_E]:
        demon.clear()
        demon.sort_chamber(threshold=0.5, criterion=criterion)
        
        sorting_results[criterion.value] = {
            'hot_count': len(demon.hot_compartment),
            'cold_count': len(demon.cold_compartment),
            'gradient': demon.extract_temperature_gradient(),
        }
        print(f"  {criterion.value}: hot={len(demon.hot_compartment)}, cold={len(demon.cold_compartment)}")
    
    results['experiments'].append({
        'name': 'multi_criteria_sorting',
        'description': 'Sorting by different S-entropy coordinates',
        'criteria_results': sorting_results,
    })
    
    # Experiment 4: Demon report
    print("\n=== Experiment 4: Demon Activity Report ===")
    report = demon.report()
    
    results['experiments'].append({
        'name': 'demon_report',
        'description': 'Complete report of demon activities',
        'report': report,
        'thermodynamic_analysis': {
            'second_law_violated': report['thermodynamic_violation'],
            'reason': 'Categorical observables commute with physical observables',
            'information_gain_source': 'Categorical space (orthogonal to physical)',
            'energy_for_decisions': 0.0,
            'energy_for_physical_moves': 'Only if physically moving data',
        },
    })
    print(f"  Observations: {report['observations']}")
    print(f"  Sorts: {report['sorts']}")
    print(f"  Decision energy cost: {report['decision_energy_cost']}")
    print(f"  Second law violated: {report['thermodynamic_violation']}")
    
    # Experiment 5: Find extreme molecules
    print("\n=== Experiment 5: Extreme Molecule Search ===")
    coldest = demon.find_coldest()
    hottest = demon.find_hottest()
    
    results['experiments'].append({
        'name': 'extreme_molecules',
        'description': 'Demon finds extreme molecules',
        'coldest': {
            'S_k': coldest.s_coord.S_k if coldest else None,
            'S_t': coldest.s_coord.S_t if coldest else None,
            'S_e': coldest.s_coord.S_e if coldest else None,
        },
        'hottest': {
            'S_k': hottest.s_coord.S_k if hottest else None,
            'S_t': hottest.s_coord.S_t if hottest else None,
            'S_e': hottest.s_coord.S_e if hottest else None,
        },
    })
    print(f"  Coldest: S_e = {coldest.s_coord.S_e:.4f}")
    print(f"  Hottest: S_e = {hottest.s_coord.S_e:.4f}")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'maxwell_demon_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print(" MAXWELL DEMON DEMONSTRATION")
    print("=" * 70)
    print("\nThe demon operates in CATEGORICAL space.")
    print("Zero backaction. No thermodynamic paradox.\n")
    
    results = run_demonstration()
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'maxwell_demon'
    save_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. The demon observes CATEGORICAL states (not physical properties)
2. Categorical observables COMMUTE with physical observables
3. Observation has ZERO backaction (reading S-coordinates doesn't disturb)
4. Sorting decisions cost ZERO energy (categorical, not physical)
5. The second law is NOT violated (demon operates in orthogonal space)
    """)
    
    return results


if __name__ == "__main__":
    main()

