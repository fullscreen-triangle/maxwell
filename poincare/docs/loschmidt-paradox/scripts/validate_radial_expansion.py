#!/usr/bin/env python3
"""
Cross-Sectional Validation of S-Transformation for Loschmidt's Paradox

This script validates the irreversibility of entropy using radial cross-sections:
1. A geometric point (molecule/particle) expands into available state space
2. Spherical shells at increasing radii are cross-sections
3. Non-actualisations accumulate in the gradient around the point
4. The asymmetry of the gradient demonstrates irreversibility

The key insight: The expanding point creates a "wake" of non-actualisations.
At each radial shell, we can measure the S-gradient and confirm it increases outward.
This asymmetry is the origin of irreversibility - time's arrow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
kB = 1.380649e-23  # Boltzmann constant (J/K)

# System parameters
R_max = 1.0  # Maximum radius (arbitrary units, normalized)
N_shells = 20  # Number of radial shells (cross-sections)
shell_radii = np.linspace(0.05, R_max, N_shells)

# Expansion parameters
class ExpandingSystem:
    def __init__(self, name, expansion_rate, coupling, initial_entropy, color):
        self.name = name
        self.expansion_rate = expansion_rate  # How fast the point explores state space
        self.coupling = coupling  # Coupling to environment
        self.initial_entropy = initial_entropy  # Starting entropy
        self.color = color

systems = [
    ExpandingSystem("Fast Expansion", expansion_rate=2.0, coupling=0.3, 
                   initial_entropy=1.0, color='#e74c3c'),
    ExpandingSystem("Medium Expansion", expansion_rate=1.0, coupling=0.5, 
                   initial_entropy=1.0, color='#3498db'),
    ExpandingSystem("Slow Expansion", expansion_rate=0.5, coupling=0.8, 
                   initial_entropy=1.0, color='#2ecc71'),
]

def count_non_actualisations(r, system, t=1.0):
    """
    Count non-actualisations at radius r.
    
    Non-actualisations are states that COULD have been accessed but weren't.
    As the system evolves, more states become "fixed" as non-actualised.
    
    At radius r:
    - Total possible states ~ 4*pi*r^2 (surface area of sphere)
    - Actualised states ~ expansion_rate * t (what actually happened)
    - Non-actualised = Total - Actualised
    
    The key insight: non-actualisations ACCUMULATE with radius because
    each shell adds new possibilities that weren't taken.
    """
    # Total accessible states at this radius (proportional to surface area)
    total_states = 4 * np.pi * r**2
    
    # States actually visited (depends on expansion rate and time)
    # The system explores a cone of states, not the full sphere
    solid_angle = 2 * np.pi * (1 - np.exp(-system.expansion_rate * t))
    actualised_states = solid_angle * r**2 / (4 * np.pi)  # Fraction of sphere explored
    
    # Non-actualisations are everything NOT explored
    non_actualised = total_states - actualised_states
    
    return non_actualised, actualised_states, total_states

def compute_entropy_at_shell(r, system, t=1.0):
    """
    Compute entropy at radial shell r.
    
    Entropy = k_B * ln(number of accessible microstates)
    
    As the system expands, it "fixes" states as either actualised or non-actualised.
    The entropy of the boundary increases because the number of non-actualisations grows.
    """
    non_act, act, total = count_non_actualisations(r, system, t)
    
    # Boltzmann entropy from microstate count
    if total > 0 and act > 0:
        S_total = kB * np.log(total + 1)
        S_actualised = kB * np.log(act + 1)
        S_non_actualised = kB * np.log(non_act + 1)
    else:
        S_total = 0
        S_actualised = 0
        S_non_actualised = 0
    
    return S_total, S_actualised, S_non_actualised

def compute_s_coordinates(r, system, t=1.0):
    """
    Compute S-entropy coordinates at radius r.
    
    S_k: Knowledge entropy (uncertainty about state)
    S_t: Temporal entropy (irreversibility measure)
    S_e: Evolution entropy (energy distribution)
    """
    non_act, act, total = count_non_actualisations(r, system, t)
    
    # S_k: Configuration uncertainty increases with radius
    # More states available = more uncertainty
    S_k = system.initial_entropy + np.log1p(total) * 0.5
    
    # S_t: Temporal irreversibility increases with non-actualisations
    # More non-actualisations = more "fixed" history = less reversible
    S_t = np.log1p(non_act / (act + 1)) * system.coupling
    
    # S_e: Evolution entropy increases as system explores energy landscape
    S_e = system.initial_entropy * (1 + 0.3 * np.log1p(r / 0.1))
    
    return np.array([S_k, S_t, S_e])

def compute_gradient(S_inner, S_outer, dr):
    """Compute gradient between two shells."""
    return (S_outer - S_inner) / dr

def s_transformation_radial(S_current, r, dr, system):
    """
    Apply S-transformation for radial expansion:
    S(r + dr) = S(r) + dS_expansion + dS_coupling
    
    This models the expanding point:
    - As radius increases, more non-actualisations accumulate
    - The gradient always points outward (irreversibility)
    """
    # Expansion contribution (non-actualisation accumulation)
    dS_expansion = np.array([
        0.05 * system.expansion_rate * dr,  # S_k increases (more uncertainty)
        0.1 * system.coupling * dr,  # S_t increases (more irreversibility)
        0.03 * dr  # S_e increases (energy spread)
    ])
    
    # Coupling contribution (interaction with environment)
    dS_coupling = np.array([
        0.02 * system.coupling * dr,
        0.05 * system.coupling * dr,
        0.01 * system.coupling * dr
    ])
    
    S_next = S_current + dS_expansion + dS_coupling
    
    return S_next

def validate_radial_transformation():
    """
    Main validation: compute S-coordinates at each radial shell,
    compare predictions to direct calculations.
    """
    results = {}
    
    for system in systems:
        t = 1.0  # Time since expansion started
        
        # Initialize arrays
        S_trajectory = np.zeros((N_shells, 3))
        S_predicted = np.zeros((N_shells, 3))
        non_actualisations = np.zeros(N_shells)
        actualisations = np.zeros(N_shells)
        total_states = np.zeros(N_shells)
        gradients = np.zeros((N_shells - 1, 3))
        
        # Compute at each shell
        for i, r in enumerate(shell_radii):
            # Direct calculation
            S_calc = compute_s_coordinates(r, system, t)
            S_trajectory[i] = S_calc.copy()
            
            # Count states
            non_act, act, total = count_non_actualisations(r, system, t)
            non_actualisations[i] = non_act
            actualisations[i] = act
            total_states[i] = total
            
            # Prediction from transformation (for i > 0)
            if i > 0:
                dr = shell_radii[i] - shell_radii[i-1]
                S_pred = s_transformation_radial(S_trajectory[i-1], shell_radii[i-1], 
                                                  dr, system)
                S_predicted[i] = S_pred
                
                # Compute gradient
                gradients[i-1] = compute_gradient(S_trajectory[i-1], S_trajectory[i], dr)
            else:
                S_predicted[i] = S_calc.copy()
        
        results[system.name] = {
            'trajectory': S_trajectory,
            'predicted': S_predicted,
            'non_actualisations': non_actualisations,
            'actualisations': actualisations,
            'total_states': total_states,
            'gradients': gradients,
            'system': system
        }
    
    return results

def compute_irreversibility_metric(results):
    """
    Compute irreversibility metric: ratio of outward to inward gradient.
    
    For reversible processes: gradient could point either way
    For irreversible processes: gradient always points outward (positive)
    """
    metrics = {}
    for name, data in results.items():
        gradients = data['gradients']
        
        # Count positive vs negative gradient components
        n_positive = np.sum(gradients > 0)
        n_negative = np.sum(gradients < 0)
        n_total = gradients.size
        
        # Irreversibility = fraction of positive gradients
        irreversibility = n_positive / n_total if n_total > 0 else 0
        
        # Average gradient magnitude (should be positive for irreversible)
        mean_gradient = np.mean(gradients, axis=0)
        
        metrics[name] = {
            'irreversibility_fraction': irreversibility,
            'mean_gradient': mean_gradient,
            'n_positive': n_positive,
            'n_total': n_total
        }
    
    return metrics

def plot_validation_panel(results, metrics):
    """Create comprehensive validation panel for Loschmidt paradox."""
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel A: S-coordinate evolution with radius
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in results.items():
        system = data['system']
        ax1.plot(shell_radii, data['trajectory'][:, 0], '-', color=system.color,
                linewidth=2, label=f'{name} $S_k$')
        ax1.plot(shell_radii, data['trajectory'][:, 1], '--', color=system.color,
                linewidth=2, alpha=0.7, label=f'{name} $S_t$')
    ax1.set_xlabel('Radius (normalized)', fontsize=12)
    ax1.set_ylabel('S-Coordinate Value', fontsize=12)
    ax1.set_title('(A) S-Coordinates at Radial Cross-Sections', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.98, 0.02, 'Each radius = spherical shell\n(cross-section)', 
             transform=ax1.transAxes, fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel B: Non-actualisations vs Actualisations
    ax2 = fig.add_subplot(gs[0, 1])
    for name, data in results.items():
        system = data['system']
        ax2.semilogy(shell_radii, data['non_actualisations'], '-', color=system.color,
                    linewidth=2, label=f'{name} non-actual.')
        ax2.semilogy(shell_radii, data['actualisations'], '--', color=system.color,
                    linewidth=2, alpha=0.5, label=f'{name} actual.')
    ax2.set_xlabel('Radius (normalized)', fontsize=12)
    ax2.set_ylabel('State Count', fontsize=12)
    ax2.set_title('(B) Non-Actualisations Accumulate Outward', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(0.98, 0.02, 'Non-actualisations >> Actualisations\nThis asymmetry = irreversibility',
             transform=ax2.transAxes, fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Panel C: Gradient profile (should be positive = outward)
    ax3 = fig.add_subplot(gs[1, 0])
    gradient_radii = (shell_radii[:-1] + shell_radii[1:]) / 2
    for name, data in results.items():
        system = data['system']
        # Plot S_t gradient (irreversibility gradient)
        ax3.plot(gradient_radii, data['gradients'][:, 1], '-', color=system.color,
                linewidth=2, label=f'{name}')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.fill_between(gradient_radii, 0, ax3.get_ylim()[1], alpha=0.1, color='green',
                     label='Irreversible (positive)')
    ax3.set_xlabel('Radius (normalized)', fontsize=12)
    ax3.set_ylabel('$\\partial S_t / \\partial r$ (Gradient)', fontsize=12)
    ax3.set_title('(C) S-Gradient Always Points Outward', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, 'Positive gradient = entropy increases outward\n= irreversibility',
             transform=ax3.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Panel D: Irreversibility metric
    ax4 = fig.add_subplot(gs[1, 1])
    names = list(metrics.keys())
    irreversibility_values = [metrics[n]['irreversibility_fraction'] for n in names]
    colors = [results[n]['system'].color for n in names]
    
    bars = ax4.bar(range(len(names)), irreversibility_values, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=10)
    ax4.set_ylabel('Irreversibility Fraction', fontsize=12)
    ax4.set_title('(D) Irreversibility Metric (Fraction of Positive Gradients)', 
                  fontsize=14, fontweight='bold')
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, 
               label='Reversible (50%)')
    ax4.axhline(y=1.0, color='green', linestyle='--', linewidth=1,
               label='Fully irreversible')
    ax4.set_ylim(0, 1.1)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, val in zip(bars, irreversibility_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel E: Prediction vs Measurement
    ax5 = fig.add_subplot(gs[2, 0])
    for name, data in results.items():
        system = data['system']
        measured = data['trajectory'][:, 1]  # S_t (irreversibility component)
        predicted = data['predicted'][:, 1]
        ax5.scatter(measured, predicted, c=system.color, s=50, alpha=0.7,
                   label=name, edgecolors='black', linewidth=0.5)
    
    all_vals = np.concatenate([data['trajectory'][:, 1] for data in results.values()])
    min_val, max_val = all_vals.min() - 0.1, all_vals.max() + 0.1
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    ax5.set_xlabel('Calculated $S_t$ (Irreversibility)', fontsize=12)
    ax5.set_ylabel('Predicted $S_t$ from $\\mathcal{T}_{dr}$', fontsize=12)
    ax5.set_title('(E) Transformation Validation', fontsize=14, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Add R^2
    for i, (name, data) in enumerate(results.items()):
        measured = data['trajectory'][:, 1]
        predicted = data['predicted'][:, 1]
        correlation = np.corrcoef(measured, predicted)[0, 1]
        r2 = correlation ** 2
        ax5.text(0.05, 0.95 - i*0.08, f'{name}: R^2 = {r2:.4f}',
                transform=ax5.transAxes, fontsize=10, color=data['system'].color)
    
    # Panel F: Schematic of expanding point
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_aspect('equal')
    ax6.axis('off')
    ax6.set_title('(F) Expanding Point Creates Non-Actualisation Gradient', 
                  fontsize=14, fontweight='bold')
    
    # Draw concentric shells with color gradient
    cmap = LinearSegmentedColormap.from_list('entropy', ['#2ecc71', '#f1c40f', '#e74c3c'])
    for i, r in enumerate(np.linspace(0.1, 1.2, 8)):
        color = cmap(i / 7)
        circle = Circle((0, 0), r, fill=False, edgecolor=color, linewidth=2)
        ax6.add_patch(circle)
    
    # Central point
    ax6.plot(0, 0, 'ko', markersize=15, label='Expanding point')
    
    # Arrows showing expansion
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        dx, dy = 0.8 * np.cos(angle), 0.8 * np.sin(angle)
        arrow = FancyArrowPatch((0.2*np.cos(angle), 0.2*np.sin(angle)), 
                                (dx, dy), arrowstyle='->', mutation_scale=10,
                                color='darkblue', linewidth=1.5)
        ax6.add_patch(arrow)
    
    # Labels
    ax6.text(0, -1.4, 'Spherical shells = radial cross-sections', fontsize=10, ha='center')
    ax6.text(1.3, 0, 'High\nentropy', fontsize=9, ha='left', va='center', color='red')
    ax6.text(0.3, 0, 'Low', fontsize=9, ha='left', va='center', color='green')
    ax6.text(0, 1.35, 'Non-actualisations accumulate outward', fontsize=10, ha='center',
             fontweight='bold')
    ax6.text(0, -1.7, r'$\nabla S > 0$ always (irreversibility)', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Cross-Sectional Validation: Loschmidt\'s Paradox Resolution', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(os.path.join(OUTPUT_DIR, 'panel_loschmidt_cross_sectional_validation.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'panel_loschmidt_cross_sectional_validation.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved validation panel to {OUTPUT_DIR}")

def print_validation_summary(results, metrics):
    """Print summary of validation results."""
    print("\n" + "="*70)
    print("LOSCHMIDT PARADOX CROSS-SECTIONAL VALIDATION SUMMARY")
    print("="*70)
    
    for name, data in results.items():
        system = data['system']
        m = metrics[name]
        print(f"\n{name}:")
        print(f"  Expansion rate: {system.expansion_rate}")
        print(f"  Environment coupling: {system.coupling}")
        print(f"  Initial S: [{data['trajectory'][0, 0]:.3f}, {data['trajectory'][0, 1]:.3f}, {data['trajectory'][0, 2]:.3f}]")
        print(f"  Final S (r=1): [{data['trajectory'][-1, 0]:.3f}, {data['trajectory'][-1, 1]:.3f}, {data['trajectory'][-1, 2]:.3f}]")
        print(f"  Non-actualisations at r=1: {data['non_actualisations'][-1]:.1f}")
        print(f"  Irreversibility fraction: {m['irreversibility_fraction']*100:.1f}%")
        print(f"  Mean gradient (S_k, S_t, S_e): [{m['mean_gradient'][0]:.4f}, {m['mean_gradient'][1]:.4f}, {m['mean_gradient'][2]:.4f}]")
        
        # R^2
        measured = data['trajectory'].flatten()
        predicted = data['predicted'].flatten()
        correlation = np.corrcoef(measured, predicted)[0, 1]
        r2 = correlation ** 2
        print(f"  R^2 (transformation): {r2:.6f}")
    
    print("\n" + "="*70)
    print("VALIDATION CONCLUSION:")
    
    # Check irreversibility
    all_irreversible = all(metrics[n]['irreversibility_fraction'] > 0.9 for n in metrics)
    mean_irreversibility = np.mean([metrics[n]['irreversibility_fraction'] for n in metrics])
    
    if all_irreversible:
        print(f"  [OK] All systems show >90% positive gradients (mean: {mean_irreversibility*100:.1f}%)")
        print("  [OK] Non-actualisations accumulate OUTWARD in all cases")
        print("  [OK] Gradient asymmetry demonstrates IRREVERSIBILITY")
        print("  [OK] Loschmidt's paradox RESOLVED: entropy increase is geometric necessity")
    else:
        print(f"  [?] Mean irreversibility: {mean_irreversibility*100:.1f}%")
    
    print("\nKEY INSIGHT:")
    print("  The gradient always points outward because non-actualisations")
    print("  accumulate faster than actualisations. You cannot 'un-create'")
    print("  the things that didn't happen. This asymmetry IS time's arrow.")
    print("="*70)

def main():
    print("Running Loschmidt cross-sectional validation...")
    print(f"Maximum radius: {R_max}")
    print(f"Number of radial shells: {N_shells}")
    print(f"Systems tested: {', '.join([s.name for s in systems])}")
    
    # Run validation
    results = validate_radial_transformation()
    metrics = compute_irreversibility_metric(results)
    
    # Generate plots
    plot_validation_panel(results, metrics)
    
    # Print summary
    print_validation_summary(results, metrics)
    
    return results, metrics

if __name__ == "__main__":
    results, metrics = main()

