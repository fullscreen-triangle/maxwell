"""
Le Chatelier's Principle Through Categorical Entropy
Demonstrates that chemical equilibrium occurs when entropy production rates
from forward and reverse reactions balance - a direct consequence of
symmetric entropy increase in both "containers" (reactants/products).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from pathlib import Path
import json


# Style configuration
COLORS = {
    'reactants': '#3498DB',    # Blue for reactants (Container A)
    'products': '#E74C3C',     # Red for products (Container B)
    'forward': '#27AE60',      # Green for forward reaction
    'reverse': '#9B59B6',      # Purple for reverse reaction
    'equilibrium': '#F39C12',  # Orange for equilibrium
    'entropy': '#1ABC9C',      # Teal for entropy
    'primary': '#2C3E50',
    'background': '#FAFAFA'
}


def setup_panel_style():
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['primary'],
        'axes.linewidth': 1.0,
    })


def simulate_reaction_equilibrium():
    """
    Simulate approach to equilibrium through symmetric entropy production.
    """
    # Initial conditions
    N_A_init = 100  # Initial reactant molecules
    N_B_init = 0    # Initial product molecules
    
    # Rate constants (forward slightly favored initially)
    k_forward = 0.1
    k_reverse = 0.08
    
    # Equilibrium constant
    K_eq = k_forward / k_reverse
    
    # Time evolution
    dt = 0.1
    t_max = 50
    times = np.arange(0, t_max, dt)
    
    N_A = np.zeros(len(times))
    N_B = np.zeros(len(times))
    S_A = np.zeros(len(times))  # Entropy of reactant side
    S_B = np.zeros(len(times))  # Entropy of product side
    dS_forward = np.zeros(len(times))  # Entropy production rate (forward)
    dS_reverse = np.zeros(len(times))  # Entropy production rate (reverse)
    
    N_A[0] = N_A_init
    N_B[0] = N_B_init
    
    # Base entropy (from phase-lock network)
    k_B = 1.38e-23
    
    for i in range(1, len(times)):
        # Current populations
        n_a = N_A[i-1]
        n_b = N_B[i-1]
        
        # Reaction rates
        rate_f = k_forward * n_a
        rate_r = k_reverse * n_b
        
        # Net change
        d_n = (rate_f - rate_r) * dt
        
        N_A[i] = max(0, n_a - d_n)
        N_B[i] = max(0, n_b + d_n)
        
        # Entropy from phase-lock network (proportional to N^2 edges)
        # Using the insight: entropy ~ edge count ~ N^2
        S_A[i] = k_B * N_A[i] * (N_A[i] - 1) * 0.3
        S_B[i] = k_B * N_B[i] * (N_B[i] - 1) * 0.3
        
        # Entropy production rates
        # Each forward reaction: both sides gain entropy
        # Each reverse reaction: both sides gain entropy
        # The KEY insight: entropy production is SYMMETRIC
        
        dS_forward[i] = rate_f * k_B * 2  # Forward increases both
        dS_reverse[i] = rate_r * k_B * 2  # Reverse increases both
    
    # Equilibrium point (where rates balance)
    eq_idx = np.argmin(np.abs(N_A - N_B * K_eq))
    
    return {
        'times': times,
        'N_A': N_A,
        'N_B': N_B,
        'S_A': S_A,
        'S_B': S_B,
        'dS_forward': dS_forward,
        'dS_reverse': dS_reverse,
        'K_eq': K_eq,
        'eq_time': times[eq_idx],
        'eq_N_A': N_A[eq_idx],
        'eq_N_B': N_B[eq_idx]
    }


def generate_le_chatelier_panel(output_dir: str = "figures"):
    """Generate Le Chatelier's principle panel."""
    setup_panel_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Get simulation data
    data = simulate_reaction_equilibrium()
    
    # Panel A: The Two-Container Model
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Container A (Reactants)
    rect_a = FancyBboxPatch((0.05, 0.3), 0.35, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor='white',
                             edgecolor=COLORS['reactants'],
                             linewidth=2)
    ax1.add_patch(rect_a)
    ax1.text(0.225, 0.75, 'REACTANTS\n(Container A)', ha='center', 
            fontsize=9, fontweight='bold', color=COLORS['reactants'])
    
    # Draw molecules in A
    np.random.seed(42)
    for _ in range(15):
        x = 0.05 + 0.35 * np.random.random()
        y = 0.3 + 0.4 * np.random.random()
        circle = Circle((x, y), 0.02, color=COLORS['reactants'], alpha=0.6)
        ax1.add_patch(circle)
    
    # Container B (Products)
    rect_b = FancyBboxPatch((0.6, 0.3), 0.35, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor='white',
                             edgecolor=COLORS['products'],
                             linewidth=2)
    ax1.add_patch(rect_b)
    ax1.text(0.775, 0.75, 'PRODUCTS\n(Container B)', ha='center',
            fontsize=9, fontweight='bold', color=COLORS['products'])
    
    # Draw molecules in B
    for _ in range(8):
        x = 0.6 + 0.35 * np.random.random()
        y = 0.3 + 0.4 * np.random.random()
        circle = Circle((x, y), 0.02, color=COLORS['products'], alpha=0.6)
        ax1.add_patch(circle)
    
    # Forward/reverse arrows
    ax1.annotate('', xy=(0.55, 0.55), xytext=(0.45, 0.55),
                arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=2))
    ax1.text(0.5, 0.58, 'Forward', ha='center', fontsize=7, color=COLORS['forward'])
    
    ax1.annotate('', xy=(0.45, 0.45), xytext=(0.55, 0.45),
                arrowprops=dict(arrowstyle='->', color=COLORS['reverse'], lw=2))
    ax1.text(0.5, 0.38, 'Reverse', ha='center', fontsize=7, color=COLORS['reverse'])
    
    ax1.text(0.5, 0.15, 'A ⇌ B', ha='center', fontsize=14, fontweight='bold')
    ax1.set_title('A. Reaction as Two Containers', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Symmetric Entropy Increase (Forward)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Forward reaction visualization
    ax2.text(0.5, 0.9, 'FORWARD: A → B', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['forward'])
    
    # Container A loses
    rect_a2 = FancyBboxPatch((0.1, 0.4), 0.3, 0.35,
                              boxstyle="round,pad=0.02",
                              facecolor='#E8F8F5',
                              edgecolor=COLORS['reactants'],
                              linewidth=2)
    ax2.add_patch(rect_a2)
    ax2.text(0.25, 0.78, 'A loses molecule', ha='center', fontsize=8)
    ax2.text(0.25, 0.32, 'ΔS_A > 0', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['forward'])
    ax2.text(0.25, 0.25, '(categorical\ncompletion)', ha='center', fontsize=7)
    
    # Container B gains
    rect_b2 = FancyBboxPatch((0.6, 0.4), 0.3, 0.35,
                              boxstyle="round,pad=0.02",
                              facecolor='#FDEDEC',
                              edgecolor=COLORS['products'],
                              linewidth=2)
    ax2.add_patch(rect_b2)
    ax2.text(0.75, 0.78, 'B gains molecule', ha='center', fontsize=8)
    ax2.text(0.75, 0.32, 'ΔS_B > 0', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['forward'])
    ax2.text(0.75, 0.25, '(mixing\ndensification)', ha='center', fontsize=7)
    
    # Arrow
    ax2.annotate('', xy=(0.55, 0.57), xytext=(0.45, 0.57),
                arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=3))
    
    ax2.text(0.5, 0.1, 'BOTH increase entropy!', ha='center', fontsize=9,
            fontweight='bold', color=COLORS['forward'],
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax2.set_title('B. Forward Reaction: Both ΔS > 0', fontweight='bold')
    ax2.axis('off')
    
    # Panel C: Symmetric Entropy Increase (Reverse)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    ax3.text(0.5, 0.9, 'REVERSE: B → A', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['reverse'])
    
    # Container B loses
    rect_b3 = FancyBboxPatch((0.6, 0.4), 0.3, 0.35,
                              boxstyle="round,pad=0.02",
                              facecolor='#F5EEF8',
                              edgecolor=COLORS['products'],
                              linewidth=2)
    ax3.add_patch(rect_b3)
    ax3.text(0.75, 0.78, 'B loses molecule', ha='center', fontsize=8)
    ax3.text(0.75, 0.32, 'ΔS_B > 0', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['reverse'])
    ax3.text(0.75, 0.25, '(categorical\ncompletion)', ha='center', fontsize=7)
    
    # Container A gains
    rect_a3 = FancyBboxPatch((0.1, 0.4), 0.3, 0.35,
                              boxstyle="round,pad=0.02",
                              facecolor='#EBF5FB',
                              edgecolor=COLORS['reactants'],
                              linewidth=2)
    ax3.add_patch(rect_a3)
    ax3.text(0.25, 0.78, 'A gains molecule', ha='center', fontsize=8)
    ax3.text(0.25, 0.32, 'ΔS_A > 0', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['reverse'])
    ax3.text(0.25, 0.25, '(mixing\ndensification)', ha='center', fontsize=7)
    
    # Arrow
    ax3.annotate('', xy=(0.45, 0.57), xytext=(0.55, 0.57),
                arrowprops=dict(arrowstyle='->', color=COLORS['reverse'], lw=3))
    
    ax3.text(0.5, 0.1, 'BOTH increase entropy!', ha='center', fontsize=9,
            fontweight='bold', color=COLORS['reverse'],
            bbox=dict(boxstyle='round', facecolor='#F5EEF8'))
    
    ax3.set_title('C. Reverse Reaction: Both ΔS > 0', fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Population dynamics
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.plot(data['times'], data['N_A'], '-', color=COLORS['reactants'], lw=2, label='[A] Reactants')
    ax4.plot(data['times'], data['N_B'], '-', color=COLORS['products'], lw=2, label='[B] Products')
    ax4.axvline(x=data['eq_time'], color=COLORS['equilibrium'], linestyle='--', lw=1.5, label='Equilibrium')
    
    ax4.fill_betweenx([0, 100], data['eq_time'] - 2, data['eq_time'] + 2, 
                       alpha=0.2, color=COLORS['equilibrium'])
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Number of Molecules')
    ax4.set_title('D. Approach to Equilibrium', fontweight='bold')
    ax4.legend(loc='right')
    ax4.set_ylim(0, 110)
    
    # Panel E: Entropy production rates
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Smooth the data for visualization
    window = 10
    dS_f_smooth = np.convolve(data['dS_forward'], np.ones(window)/window, mode='same')
    dS_r_smooth = np.convolve(data['dS_reverse'], np.ones(window)/window, mode='same')
    
    ax5.plot(data['times'], dS_f_smooth * 1e23, '-', color=COLORS['forward'], lw=2, 
            label='Forward rate')
    ax5.plot(data['times'], dS_r_smooth * 1e23, '-', color=COLORS['reverse'], lw=2,
            label='Reverse rate')
    ax5.axvline(x=data['eq_time'], color=COLORS['equilibrium'], linestyle='--', lw=1.5)
    
    # Mark equilibrium intersection
    eq_idx = np.argmin(np.abs(data['times'] - data['eq_time']))
    ax5.scatter([data['eq_time']], [dS_f_smooth[eq_idx] * 1e23], 
               s=100, c=COLORS['equilibrium'], zorder=5, marker='*')
    ax5.text(data['eq_time'] + 2, dS_f_smooth[eq_idx] * 1e23, 
            'EQUILIBRIUM\n(rates equal!)', fontsize=8, color=COLORS['equilibrium'])
    
    ax5.set_xlabel('Time')
    ax5.set_ylabel('dS/dt (×10⁻²³ J/K·s)')
    ax5.set_title('E. Entropy Production Rates', fontweight='bold')
    ax5.legend(loc='upper right')
    
    # Panel F: The Key Insight
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    
    # Balance visualization
    ax6.text(0.5, 0.9, 'EQUILIBRIUM CONDITION', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['equilibrium'])
    
    # Equation
    ax6.text(0.5, 0.7, r'$\frac{dS_{forward}}{dt} = \frac{dS_{reverse}}{dt}$',
            ha='center', fontsize=14, fontweight='bold')
    
    # Balance beam
    ax6.plot([0.2, 0.8], [0.45, 0.45], 'k-', lw=3)
    ax6.plot([0.5, 0.5], [0.35, 0.45], 'k-', lw=3)
    
    # Left side (forward)
    ax6.add_patch(FancyBboxPatch((0.15, 0.48), 0.2, 0.12,
                                  boxstyle="round", facecolor=COLORS['forward'], alpha=0.7))
    ax6.text(0.25, 0.54, 'Forward\nΔS', ha='center', fontsize=8, color='white', fontweight='bold')
    
    # Right side (reverse)
    ax6.add_patch(FancyBboxPatch((0.65, 0.48), 0.2, 0.12,
                                  boxstyle="round", facecolor=COLORS['reverse'], alpha=0.7))
    ax6.text(0.75, 0.54, 'Reverse\nΔS', ha='center', fontsize=8, color='white', fontweight='bold')
    
    ax6.text(0.5, 0.2, 'At equilibrium:\nEntropy production\nrates BALANCE', 
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor=COLORS['equilibrium']))
    
    ax6.set_title('F. The Balance Point', fontweight='bold')
    ax6.axis('off')
    
    # Panel G: Le Chatelier Response
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    
    ax7.text(0.5, 0.92, 'LE CHATELIER: Add Reactants', ha='center', fontsize=10, fontweight='bold')
    
    # Before perturbation
    ax7.add_patch(FancyBboxPatch((0.05, 0.5), 0.4, 0.3,
                                  boxstyle="round", facecolor='#E8F8F5', edgecolor=COLORS['reactants']))
    ax7.text(0.25, 0.8, 'More A molecules', ha='center', fontsize=8)
    ax7.text(0.25, 0.55, '↑ Forward rate', ha='center', fontsize=9, 
            fontweight='bold', color=COLORS['forward'])
    
    # After response
    ax7.add_patch(FancyBboxPatch((0.55, 0.5), 0.4, 0.3,
                                  boxstyle="round", facecolor='#FDEDEC', edgecolor=COLORS['products']))
    ax7.text(0.75, 0.8, 'System shifts →', ha='center', fontsize=8)
    ax7.text(0.75, 0.55, 'More B forms', ha='center', fontsize=9,
            fontweight='bold', color=COLORS['products'])
    
    ax7.annotate('', xy=(0.52, 0.65), xytext=(0.48, 0.65),
                arrowprops=dict(arrowstyle='->', color=COLORS['forward'], lw=3))
    
    ax7.text(0.5, 0.25, 'Response: Forward entropy production\ntemporarily exceeds reverse\n→ System shifts to restore balance',
            ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['primary']))
    
    ax7.set_title('G. Perturbation Response', fontweight='bold')
    ax7.axis('off')
    
    # Panel H: Connection to K_eq
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Show K_eq as ratio of entropy production at equilibrium
    Q_values = np.linspace(0.1, 3, 100)
    K = data['K_eq']
    
    # Direction of shift based on Q vs K
    ax8.axhline(y=0, color='black', lw=1)
    ax8.axvline(x=K, color=COLORS['equilibrium'], lw=2, linestyle='--', label=f'K = {K:.2f}')
    
    # Forward favored region
    ax8.fill_between([0.1, K], [-1, -1], [1, 1], alpha=0.2, color=COLORS['forward'])
    ax8.text(K/2, 0.7, 'Q < K\nForward\nfavored', ha='center', fontsize=8, color=COLORS['forward'])
    
    # Reverse favored region
    ax8.fill_between([K, 3], [-1, -1], [1, 1], alpha=0.2, color=COLORS['reverse'])
    ax8.text((K + 3)/2, 0.7, 'Q > K\nReverse\nfavored', ha='center', fontsize=8, color=COLORS['reverse'])
    
    ax8.scatter([K], [0], s=150, c=COLORS['equilibrium'], marker='*', zorder=5)
    ax8.text(K, -0.5, 'Q = K\nEQUILIBRIUM', ha='center', fontsize=9, 
            fontweight='bold', color=COLORS['equilibrium'])
    
    ax8.set_xlabel('Reaction Quotient Q = [B]/[A]')
    ax8.set_ylabel('Net Entropy Flow')
    ax8.set_xlim(0, 3)
    ax8.set_ylim(-1, 1)
    ax8.set_title('H. Equilibrium Constant K', fontweight='bold')
    ax8.legend(loc='upper right')
    
    # Panel I: Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    
    summary_text = """LE CHATELIER THROUGH
CATEGORICAL ENTROPY

1. Every reaction (forward OR 
   reverse) increases entropy
   in BOTH "containers"

2. Equilibrium = balance point
   where entropy production
   rates are equal

3. Perturbation breaks balance
   → system shifts to restore
   entropy rate equality

4. K_eq = ratio where
   dS_forward/dt = dS_reverse/dt

This unifies:
• Maxwell's Demon resolution
• Gibbs paradox resolution  
• Le Chatelier's principle"""
    
    ax9.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=9,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor=COLORS['primary']))
    
    ax9.set_title('I. Unified Framework', fontweight='bold')
    ax9.axis('off')
    
    plt.suptitle("Le Chatelier's Principle: Equilibrium as Balanced Entropy Production",
                fontsize=14, fontweight='bold', y=0.98)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "le_chatelier_entropy_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Save results
    results_path = Path(output_dir).parent / "results" / "le_chatelier_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'Le Chatelier via Categorical Entropy',
            'key_insight': 'Equilibrium is where forward and reverse entropy production rates balance',
            'K_eq': data['K_eq'],
            'eq_time': float(data['eq_time']),
            'eq_N_A': float(data['eq_N_A']),
            'eq_N_B': float(data['eq_N_B']),
            'unified_framework': [
                'Maxwell Demon resolution',
                'Gibbs paradox resolution', 
                'Le Chatelier principle'
            ]
        }, f, indent=2)
    print(f"Saved: {results_path}")
    
    return data


def main():
    """Generate Le Chatelier panel."""
    output_dir = "figures"
    
    print("=" * 60)
    print("LE CHATELIER'S PRINCIPLE THROUGH CATEGORICAL ENTROPY")
    print("=" * 60)
    
    data = generate_le_chatelier_panel(output_dir)
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
At equilibrium:
    dS_forward/dt = dS_reverse/dt

Both directions INCREASE entropy (symmetric increase).
Equilibrium is where these rates BALANCE.

This unifies:
1. Maxwell's Demon → door opening increases entropy both ways
2. Gibbs Paradox → mixing/separation increases entropy both ways
3. Le Chatelier → forward/reverse reactions increase entropy both ways
   → equilibrium is the balance point
""")


if __name__ == "__main__":
    main()

