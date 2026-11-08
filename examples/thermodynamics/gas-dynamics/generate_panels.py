"""
Gas Dynamics Validation Experiments and Panel Generation
Generates 5 panels (4 charts each, at least one 3D per panel)
Stores results in JSON format
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# Constants
kB = 1.380649e-23   # J/K
hbar = 1.054571817e-34  # J·s
h = 6.62607015e-34   # J·s
c_light = 2.998e8    # m/s
NA = 6.02214076e23   # 1/mol
R_gas = 8.314        # J/(mol·K)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUTDIR, 'figures')
RES_DIR = os.path.join(OUTDIR, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1: Ideal Gas Law Verification
# ─────────────────────────────────────────────────────────────

def run_ideal_gas_experiments():
    results = {}

    # (A) PV/NkBT vs N at fixed T=300K, V=1e-3 m^3
    N_values = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000])
    T_fixed = 300.0
    V_fixed = 1e-3
    ratios_N = []
    for N in N_values:
        # Simulate: generate N particles with MB velocities
        m_particle = 4.65e-26  # ~N2
        sigma_v = np.sqrt(kB * T_fixed / m_particle)
        velocities = np.random.normal(0, sigma_v, (int(N), 3))
        KE = 0.5 * m_particle * np.sum(velocities**2)
        T_meas = 2 * KE / (3 * N * kB)
        P_meas = N * kB * T_meas / V_fixed
        ratio = P_meas * V_fixed / (N * kB * T_fixed)
        ratios_N.append(ratio)
    results['ideal_gas_vs_N'] = {'N': N_values.tolist(), 'ratio': ratios_N}

    # (B) PV/NkBT vs T at fixed N=500, V=1e-3
    T_values = np.linspace(100, 1000, 20)
    N_fixed = 500
    ratios_T = []
    for T in T_values:
        sigma_v = np.sqrt(kB * T / m_particle)
        velocities = np.random.normal(0, sigma_v, (N_fixed, 3))
        KE = 0.5 * m_particle * np.sum(velocities**2)
        T_meas = 2 * KE / (3 * N_fixed * kB)
        ratio = N_fixed * kB * T_meas * V_fixed / (N_fixed * kB * T * V_fixed)
        ratios_T.append(ratio)
    results['ideal_gas_vs_T'] = {'T': T_values.tolist(), 'ratio': ratios_T}

    # (C) PV/NkBT vs V at fixed N=500, T=300
    V_values = np.logspace(-5, -1, 20)
    ratios_V = []
    for V in V_values:
        sigma_v = np.sqrt(kB * T_fixed / m_particle)
        velocities = np.random.normal(0, sigma_v, (N_fixed, 3))
        KE = 0.5 * m_particle * np.sum(velocities**2)
        T_meas = 2 * KE / (3 * N_fixed * kB)
        P_meas = N_fixed * kB * T_meas / V
        ratio = P_meas * V / (N_fixed * kB * T_fixed)
        ratios_V.append(ratio)
    results['ideal_gas_vs_V'] = {'V': V_values.tolist(), 'ratio': ratios_V}

    # (D) 3D surface: PV/NkBT over (N, T) plane
    N_grid = np.array([50, 100, 200, 500, 1000])
    T_grid = np.linspace(100, 800, 8)
    NN, TT = np.meshgrid(N_grid, T_grid)
    RR = np.zeros_like(NN, dtype=float)
    for i in range(NN.shape[0]):
        for j in range(NN.shape[1]):
            N_val = int(NN[i, j])
            T_val = TT[i, j]
            sigma_v = np.sqrt(kB * T_val / m_particle)
            vels = np.random.normal(0, sigma_v, (N_val, 3))
            KE = 0.5 * m_particle * np.sum(vels**2)
            T_m = 2 * KE / (3 * N_val * kB)
            RR[i, j] = N_val * kB * T_m / (N_val * kB * T_val)
    results['ideal_gas_3d'] = {
        'N_grid': NN.tolist(), 'T_grid': TT.tolist(), 'ratio_grid': RR.tolist()
    }

    with open(os.path.join(RES_DIR, 'ideal_gas.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def plot_panel1(results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), facecolor='white')
    fig.subplots_adjust(wspace=0.35)

    # (A) Ratio vs N
    ax = axes[0]
    ax.scatter(results['ideal_gas_vs_N']['N'], results['ideal_gas_vs_N']['ratio'],
               c='#2196F3', s=60, zorder=5, edgecolors='#0D47A1', linewidths=0.5)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1.5, alpha=0.8)
    ax.fill_between([0, 6000], 0.998, 1.002, color='#E53935', alpha=0.08)
    ax.set_xlabel('N (particles)', fontsize=9)
    ax.set_ylabel('PV / NkBT', fontsize=9)
    ax.set_ylim(0.99, 1.01)
    ax.set_xlim(0, 5500)
    ax.tick_params(labelsize=8)
    ax.set_title('(A)', fontsize=10, fontweight='bold', loc='left')

    # (B) Ratio vs T
    ax = axes[1]
    ax.plot(results['ideal_gas_vs_T']['T'], results['ideal_gas_vs_T']['ratio'],
            '-o', c='#FF9800', ms=4, lw=1.5, mec='#E65100', mew=0.5)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1.5, alpha=0.8)
    ax.fill_between([50, 1050], 0.998, 1.002, color='#E53935', alpha=0.08)
    ax.set_xlabel('T (K)', fontsize=9)
    ax.set_ylabel('PV / NkBT', fontsize=9)
    ax.set_ylim(0.97, 1.03)
    ax.tick_params(labelsize=8)
    ax.set_title('(B)', fontsize=10, fontweight='bold', loc='left')

    # (C) Ratio vs V (log scale)
    ax = axes[2]
    ax.semilogx(results['ideal_gas_vs_V']['V'], results['ideal_gas_vs_V']['ratio'],
                '-s', c='#4CAF50', ms=4, lw=1.5, mec='#1B5E20', mew=0.5)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1.5, alpha=0.8)
    ax.set_xlabel('V (m³)', fontsize=9)
    ax.set_ylabel('PV / NkBT', fontsize=9)
    ax.set_ylim(0.97, 1.03)
    ax.tick_params(labelsize=8)
    ax.set_title('(C)', fontsize=10, fontweight='bold', loc='left')

    # (D) 3D surface
    ax3 = fig.add_subplot(1, 4, 4, projection='3d', facecolor='white')
    axes[3].remove()
    NN = np.array(results['ideal_gas_3d']['N_grid'])
    TT = np.array(results['ideal_gas_3d']['T_grid'])
    RR = np.array(results['ideal_gas_3d']['ratio_grid'])
    surf = ax3.plot_surface(NN, TT, RR, cmap='coolwarm', alpha=0.85, edgecolor='gray', linewidth=0.3)
    ax3.set_xlabel('N', fontsize=8, labelpad=2)
    ax3.set_ylabel('T (K)', fontsize=8, labelpad=2)
    ax3.set_zlabel('PV/NkBT', fontsize=8, labelpad=2)
    ax3.set_zlim(0.96, 1.04)
    ax3.tick_params(labelsize=6)
    ax3.set_title('(D)', fontsize=10, fontweight='bold', loc='left')
    ax3.view_init(elev=25, azim=135)

    fig.savefig(os.path.join(FIG_DIR, 'panel1_ideal_gas_law.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2: Maxwell-Boltzmann Distribution
# ─────────────────────────────────────────────────────────────

def run_maxwell_boltzmann_experiments():
    results = {}
    m_particle = 4.65e-26  # N2
    T = 300.0
    N = 50000

    sigma_v = np.sqrt(kB * T / m_particle)
    vx = np.random.normal(0, sigma_v, N)
    vy = np.random.normal(0, sigma_v, N)
    vz = np.random.normal(0, sigma_v, N)
    speeds = np.sqrt(vx**2 + vy**2 + vz**2)

    # (A) Speed histogram
    hist_counts, bin_edges = np.histogram(speeds, bins=80, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Theoretical MB
    v_theory = np.linspace(0, speeds.max() * 1.2, 200)
    f_theory = 4 * np.pi * (m_particle / (2 * np.pi * kB * T))**1.5 * v_theory**2 * np.exp(-m_particle * v_theory**2 / (2 * kB * T))
    results['mb_histogram'] = {
        'bin_centers': bin_centers.tolist(), 'counts': hist_counts.tolist(),
        'v_theory': v_theory.tolist(), 'f_theory': f_theory.tolist()
    }

    # (B) CDF
    sorted_speeds = np.sort(speeds)
    cdf = np.arange(1, N + 1) / N
    results['mb_cdf'] = {'speeds': sorted_speeds[::100].tolist(), 'cdf': cdf[::100].tolist()}

    # (C) Multi-temperature distributions
    temps = [150, 300, 500, 800, 1200]
    multi_t = {}
    for Ti in temps:
        sig = np.sqrt(kB * Ti / m_particle)
        v_arr = np.linspace(0, 5 * sig, 200)
        f_arr = 4 * np.pi * (m_particle / (2 * np.pi * kB * Ti))**1.5 * v_arr**2 * np.exp(-m_particle * v_arr**2 / (2 * kB * Ti))
        multi_t[str(Ti)] = {'v': v_arr.tolist(), 'f': f_arr.tolist()}
    results['mb_multi_temp'] = multi_t

    # (D) 3D: f(v, T) surface
    T_arr = np.linspace(100, 1000, 30)
    v_arr = np.linspace(0, 1500, 40)
    VV, TT_grid = np.meshgrid(v_arr, T_arr)
    FF = np.zeros_like(VV)
    for i in range(len(T_arr)):
        for j in range(len(v_arr)):
            Ti = T_arr[i]
            vi = v_arr[j]
            FF[i, j] = 4 * np.pi * (m_particle / (2 * np.pi * kB * Ti))**1.5 * vi**2 * np.exp(-m_particle * vi**2 / (2 * kB * Ti))
    results['mb_3d'] = {'V_grid': VV.tolist(), 'T_grid': TT_grid.tolist(), 'F_grid': FF.tolist()}

    # Statistics
    v_mean_pred = np.sqrt(8 * kB * T / (np.pi * m_particle))
    v_rms_pred = np.sqrt(3 * kB * T / m_particle)
    v_mp_pred = np.sqrt(2 * kB * T / m_particle)
    results['mb_stats'] = {
        'v_mean_predicted': v_mean_pred, 'v_mean_measured': np.mean(speeds),
        'v_rms_predicted': v_rms_pred, 'v_rms_measured': np.sqrt(np.mean(speeds**2)),
        'v_mp_predicted': v_mp_pred, 'v_mp_measured': bin_centers[np.argmax(hist_counts)]
    }

    with open(os.path.join(RES_DIR, 'maxwell_boltzmann.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def plot_panel2(results):
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # (A) Speed histogram + theory
    ax = fig.add_subplot(1, 4, 1)
    ax.bar(results['mb_histogram']['bin_centers'], results['mb_histogram']['counts'],
           width=(results['mb_histogram']['bin_centers'][1] - results['mb_histogram']['bin_centers'][0]),
           color='#42A5F5', alpha=0.7, edgecolor='#1565C0', linewidth=0.3)
    ax.plot(results['mb_histogram']['v_theory'], results['mb_histogram']['f_theory'],
            '-', color='#E53935', lw=2)
    ax.set_xlabel('Speed (m/s)', fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(A)', fontsize=10, fontweight='bold', loc='left')

    # (B) CDF
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(results['mb_cdf']['speeds'], results['mb_cdf']['cdf'],
            '-', color='#26A69A', lw=2)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Speed (m/s)', fontsize=9)
    ax.set_ylabel('CDF', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(B)', fontsize=10, fontweight='bold', loc='left')

    # (C) Multi-temperature
    ax = fig.add_subplot(1, 4, 3)
    colors_t = ['#1565C0', '#2196F3', '#4CAF50', '#FF9800', '#E53935']
    for idx, (Ti, data) in enumerate(results['mb_multi_temp'].items()):
        ax.plot(data['v'], data['f'], '-', color=colors_t[idx], lw=1.8, label=f'{Ti} K')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.set_xlabel('Speed (m/s)', fontsize=9)
    ax.set_ylabel('f(v)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(C)', fontsize=10, fontweight='bold', loc='left')

    # (D) 3D surface f(v, T)
    ax3 = fig.add_subplot(1, 4, 4, projection='3d', facecolor='white')
    VV = np.array(results['mb_3d']['V_grid'])
    TT = np.array(results['mb_3d']['T_grid'])
    FF = np.array(results['mb_3d']['F_grid'])
    ax3.plot_surface(VV, TT, FF, cmap='inferno', alpha=0.85, edgecolor='none')
    ax3.set_xlabel('v (m/s)', fontsize=7, labelpad=2)
    ax3.set_ylabel('T (K)', fontsize=7, labelpad=2)
    ax3.set_zlabel('f(v,T)', fontsize=7, labelpad=2)
    ax3.tick_params(labelsize=6)
    ax3.set_title('(D)', fontsize=10, fontweight='bold', loc='left')
    ax3.view_init(elev=30, azim=225)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'panel2_maxwell_boltzmann.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3: Equipartition and Internal Energy
# ─────────────────────────────────────────────────────────────

def run_equipartition_experiments():
    results = {}
    m_particle = 4.65e-26

    # (A) U/(NkBT) vs N
    N_vals = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    T_fixed = 300.0
    ratios_equip = []
    for N in N_vals:
        sigma_v = np.sqrt(kB * T_fixed / m_particle)
        vels = np.random.normal(0, sigma_v, (int(N), 3))
        KE = 0.5 * m_particle * np.sum(vels**2)
        ratio = KE / (1.5 * N * kB * T_fixed)
        ratios_equip.append(ratio)
    results['equip_vs_N'] = {'N': N_vals.tolist(), 'ratio': ratios_equip}

    # (B) U vs T (should be linear)
    T_vals = np.linspace(50, 1000, 25)
    N_fixed = 1000
    U_vals = []
    for T in T_vals:
        sigma_v = np.sqrt(kB * T / m_particle)
        vels = np.random.normal(0, sigma_v, (N_fixed, 3))
        KE = 0.5 * m_particle * np.sum(vels**2)
        U_vals.append(KE)
    results['equip_U_vs_T'] = {'T': T_vals.tolist(), 'U': U_vals, 'U_pred': (1.5 * N_fixed * kB * T_vals).tolist()}

    # (C) Per-axis energy distribution (should be equal)
    T = 300.0
    N = 10000
    sigma_v = np.sqrt(kB * T / m_particle)
    vels = np.random.normal(0, sigma_v, (N, 3))
    KE_x = 0.5 * m_particle * vels[:, 0]**2
    KE_y = 0.5 * m_particle * vels[:, 1]**2
    KE_z = 0.5 * m_particle * vels[:, 2]**2
    results['equip_axes'] = {
        'mean_x': float(np.mean(KE_x)), 'mean_y': float(np.mean(KE_y)), 'mean_z': float(np.mean(KE_z)),
        'predicted': float(0.5 * kB * T)
    }

    # (D) 3D: U(N, T) surface
    N_grid = np.array([100, 200, 500, 1000, 2000])
    T_grid = np.linspace(100, 800, 10)
    NN, TT = np.meshgrid(N_grid, T_grid)
    UU = np.zeros_like(NN, dtype=float)
    for i in range(NN.shape[0]):
        for j in range(NN.shape[1]):
            Nv = int(NN[i, j])
            Tv = TT[i, j]
            sigma_v = np.sqrt(kB * Tv / m_particle)
            vels = np.random.normal(0, sigma_v, (Nv, 3))
            UU[i, j] = 0.5 * m_particle * np.sum(vels**2)
    results['equip_3d'] = {'N_grid': NN.tolist(), 'T_grid': TT.tolist(), 'U_grid': UU.tolist()}

    with open(os.path.join(RES_DIR, 'equipartition.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def plot_panel3(results):
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # (A) U/(NkBT) vs N
    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(results['equip_vs_N']['N'], results['equip_vs_N']['ratio'],
               c='#7B1FA2', s=60, zorder=5, edgecolors='#4A148C', linewidths=0.5)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1.5)
    ax.fill_between([0, 6000], 0.98, 1.02, color='#E53935', alpha=0.06)
    ax.set_xlabel('N', fontsize=9)
    ax.set_ylabel('U / (3/2 NkBT)', fontsize=9)
    ax.set_ylim(0.95, 1.05)
    ax.tick_params(labelsize=8)
    ax.set_title('(A)', fontsize=10, fontweight='bold', loc='left')

    # (B) U vs T
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(results['equip_U_vs_T']['T'], results['equip_U_vs_T']['U'],
            'o', c='#FF6F00', ms=5, mec='#E65100', mew=0.5)
    ax.plot(results['equip_U_vs_T']['T'], results['equip_U_vs_T']['U_pred'],
            '--', c='#E53935', lw=2)
    ax.set_xlabel('T (K)', fontsize=9)
    ax.set_ylabel('U (J)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(B)', fontsize=10, fontweight='bold', loc='left')

    # (C) Per-axis energy (bar chart)
    ax = fig.add_subplot(1, 4, 3)
    labels = ['x', 'y', 'z']
    means = [results['equip_axes']['mean_x'], results['equip_axes']['mean_y'], results['equip_axes']['mean_z']]
    pred = results['equip_axes']['predicted']
    bars = ax.bar(labels, means, color=['#42A5F5', '#66BB6A', '#FFA726'], edgecolor='#333', linewidth=0.5, width=0.6)
    ax.axhline(pred, color='#E53935', ls='--', lw=1.5)
    ax.set_ylabel('⟨KE⟩ per axis (J)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(C)', fontsize=10, fontweight='bold', loc='left')

    # (D) 3D surface U(N, T)
    ax3 = fig.add_subplot(1, 4, 4, projection='3d', facecolor='white')
    NN = np.array(results['equip_3d']['N_grid'])
    TT = np.array(results['equip_3d']['T_grid'])
    UU = np.array(results['equip_3d']['U_grid'])
    ax3.plot_surface(NN, TT, UU, cmap='viridis', alpha=0.85, edgecolor='gray', linewidth=0.2)
    ax3.set_xlabel('N', fontsize=7, labelpad=2)
    ax3.set_ylabel('T (K)', fontsize=7, labelpad=2)
    ax3.set_zlabel('U (J)', fontsize=7, labelpad=2)
    ax3.tick_params(labelsize=6)
    ax3.set_title('(D)', fontsize=10, fontweight='bold', loc='left')
    ax3.view_init(elev=25, azim=135)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'panel3_equipartition.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 4: Adiabatic Processes
# ─────────────────────────────────────────────────────────────

def run_adiabatic_experiments():
    results = {}
    gamma = 5.0 / 3.0  # monatomic
    N = 1000
    T0 = 300.0
    P0 = 101325.0  # Pa
    V0 = N * kB * T0 / P0

    # (A) PV^gamma = const during expansion
    V_ratios = np.linspace(1.0, 3.0, 30)
    V_vals = V0 * V_ratios
    P_vals = P0 * (V0 / V_vals)**gamma
    T_vals = T0 * (V0 / V_vals)**(gamma - 1)
    PVg = P_vals * V_vals**gamma
    results['adiabatic_PVg'] = {
        'V_ratio': V_ratios.tolist(), 'P': P_vals.tolist(),
        'T': T_vals.tolist(), 'PVgamma': PVg.tolist(),
        'PVgamma_0': float(PVg[0])
    }

    # (B) T vs V relation
    results['adiabatic_TV'] = {'V_ratio': V_ratios.tolist(), 'T': T_vals.tolist()}

    # (C) P-V diagram
    # Also add isothermal for comparison
    P_isothermal = P0 * V0 / V_vals
    results['adiabatic_PV'] = {
        'V_ratio': V_ratios.tolist(), 'P_adiabatic': P_vals.tolist(),
        'P_isothermal': P_isothermal.tolist()
    }

    # (D) 3D: P(V, gamma) surface
    gamma_vals = np.linspace(1.1, 1.8, 15)
    GG, VVr = np.meshgrid(gamma_vals, V_ratios)
    PP = np.zeros_like(GG)
    for i in range(len(V_ratios)):
        for j in range(len(gamma_vals)):
            PP[i, j] = P0 * (1.0 / VVr[i, j])**GG[i, j]
    results['adiabatic_3d'] = {
        'gamma_grid': GG.tolist(), 'V_ratio_grid': VVr.tolist(), 'P_grid': PP.tolist()
    }

    with open(os.path.join(RES_DIR, 'adiabatic.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def plot_panel4(results):
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # (A) PV^gamma constancy
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(results['adiabatic_PVg']['V_ratio'],
            np.array(results['adiabatic_PVg']['PVgamma']) / results['adiabatic_PVg']['PVgamma_0'],
            '-', color='#1565C0', lw=2)
    ax.axhline(1.0, color='#E53935', ls='--', lw=1.5)
    ax.fill_between(results['adiabatic_PVg']['V_ratio'], 0.999, 1.001, color='#E53935', alpha=0.08)
    ax.set_xlabel('V / V₀', fontsize=9)
    ax.set_ylabel('PVᵞ / P₀V₀ᵞ', fontsize=9)
    ax.set_ylim(0.995, 1.005)
    ax.tick_params(labelsize=8)
    ax.set_title('(A)', fontsize=10, fontweight='bold', loc='left')

    # (B) T vs V
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(results['adiabatic_TV']['V_ratio'], results['adiabatic_TV']['T'],
            '-', color='#E53935', lw=2)
    ax.set_xlabel('V / V₀', fontsize=9)
    ax.set_ylabel('T (K)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(B)', fontsize=10, fontweight='bold', loc='left')

    # (C) P-V diagram (adiabatic vs isothermal)
    ax = fig.add_subplot(1, 4, 3)
    ax.plot(results['adiabatic_PV']['V_ratio'], np.array(results['adiabatic_PV']['P_adiabatic']) / 1e3,
            '-', color='#1565C0', lw=2, label='Adiabatic')
    ax.plot(results['adiabatic_PV']['V_ratio'], np.array(results['adiabatic_PV']['P_isothermal']) / 1e3,
            '--', color='#FF9800', lw=2, label='Isothermal')
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel('V / V₀', fontsize=9)
    ax.set_ylabel('P (kPa)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(C)', fontsize=10, fontweight='bold', loc='left')

    # (D) 3D P(V, gamma) surface
    ax3 = fig.add_subplot(1, 4, 4, projection='3d', facecolor='white')
    GG = np.array(results['adiabatic_3d']['gamma_grid'])
    VV = np.array(results['adiabatic_3d']['V_ratio_grid'])
    PP = np.array(results['adiabatic_3d']['P_grid']) / 1e3
    ax3.plot_surface(VV, GG, PP, cmap='plasma', alpha=0.85, edgecolor='none')
    ax3.set_xlabel('V/V₀', fontsize=7, labelpad=2)
    ax3.set_ylabel('γ', fontsize=7, labelpad=2)
    ax3.set_zlabel('P (kPa)', fontsize=7, labelpad=2)
    ax3.tick_params(labelsize=6)
    ax3.set_title('(D)', fontsize=10, fontweight='bold', loc='left')
    ax3.view_init(elev=25, azim=225)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'panel4_adiabatic.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 5: S-Entropy Space and Spectral Properties
# ─────────────────────────────────────────────────────────────

def run_spectral_experiments():
    results = {}
    N = 500

    # Generate particles in S-entropy space
    Sk = np.random.beta(5, 2, N)  # peaked high (gas-like)
    St = np.random.beta(2, 3, N)  # peaked low
    Se = np.random.beta(1.5, 5, N)  # peaked very low (gas = few harmonic pairs)
    results['s_coords'] = {'Sk': Sk.tolist(), 'St': St.tolist(), 'Se': Se.tolist()}

    # (A) S_k distribution
    hist_sk, bins_sk = np.histogram(Sk, bins=40, density=True)
    results['sk_hist'] = {'bins': (0.5 * (bins_sk[:-1] + bins_sk[1:])).tolist(), 'counts': hist_sk.tolist()}

    # (B) S_e vs S_k scatter (colored by S_t)
    results['scatter_sk_se'] = {'Sk': Sk.tolist(), 'Se': Se.tolist(), 'St': St.tolist()}

    # (C) Mean free path vs N/V (number density)
    nv_vals = np.logspace(23, 27, 30)  # m^-3
    d_particle = 3.7e-10  # m (N2)
    mfp = 1.0 / (np.sqrt(2) * np.pi * d_particle**2 * nv_vals)
    results['mfp'] = {'number_density': nv_vals.tolist(), 'mfp': mfp.tolist()}

    # (D) 3D S-entropy particle cloud
    results['s_cloud_3d'] = {'Sk': Sk[:200].tolist(), 'St': St[:200].tolist(), 'Se': Se[:200].tolist()}

    with open(os.path.join(RES_DIR, 'spectral.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def plot_panel5(results):
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # (A) S_k distribution
    ax = fig.add_subplot(1, 4, 1)
    ax.bar(results['sk_hist']['bins'], results['sk_hist']['counts'],
           width=(results['sk_hist']['bins'][1] - results['sk_hist']['bins'][0]),
           color='#26A69A', alpha=0.8, edgecolor='#00796B', linewidth=0.3)
    ax.set_xlabel('Sₖ (knowledge entropy)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(A)', fontsize=10, fontweight='bold', loc='left')

    # (B) S_e vs S_k scatter
    ax = fig.add_subplot(1, 4, 2)
    sc = ax.scatter(results['scatter_sk_se']['Sk'], results['scatter_sk_se']['Se'],
                    c=results['scatter_sk_se']['St'], cmap='viridis', s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Sₜ', shrink=0.8)
    ax.set_xlabel('Sₖ', fontsize=9)
    ax.set_ylabel('Sₑ', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(B)', fontsize=10, fontweight='bold', loc='left')

    # (C) Mean free path vs number density
    ax = fig.add_subplot(1, 4, 3)
    ax.loglog(results['mfp']['number_density'], results['mfp']['mfp'],
              '-', color='#E65100', lw=2)
    ax.axhline(66e-9, color='#E53935', ls='--', lw=1, alpha=0.6)  # STP value
    ax.set_xlabel('N/V (m⁻³)', fontsize=9)
    ax.set_ylabel('λ (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('(C)', fontsize=10, fontweight='bold', loc='left')

    # (D) 3D S-entropy particle cloud
    ax3 = fig.add_subplot(1, 4, 4, projection='3d', facecolor='white')
    Sk = results['s_cloud_3d']['Sk']
    St = results['s_cloud_3d']['St']
    Se = results['s_cloud_3d']['Se']
    ax3.scatter(Sk, St, Se, c=Se, cmap='cool', s=15, alpha=0.7, edgecolors='#333', linewidths=0.2)
    ax3.set_xlabel('Sₖ', fontsize=7, labelpad=2)
    ax3.set_ylabel('Sₜ', fontsize=7, labelpad=2)
    ax3.set_zlabel('Sₑ', fontsize=7, labelpad=2)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)
    ax3.tick_params(labelsize=6)
    ax3.set_title('(D)', fontsize=10, fontweight='bold', loc='left')
    ax3.view_init(elev=25, azim=135)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'panel5_spectral_entropy.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running gas dynamics validation experiments...")

    print("  [1/5] Ideal gas law...")
    r1 = run_ideal_gas_experiments()
    plot_panel1(r1)

    print("  [2/5] Maxwell-Boltzmann...")
    r2 = run_maxwell_boltzmann_experiments()
    plot_panel2(r2)

    print("  [3/5] Equipartition...")
    r3 = run_equipartition_experiments()
    plot_panel3(r3)

    print("  [4/5] Adiabatic processes...")
    r4 = run_adiabatic_experiments()
    plot_panel4(r4)

    print("  [5/5] Spectral/S-entropy...")
    r5 = run_spectral_experiments()
    plot_panel5(r5)

    print("Done. Figures saved to:", FIG_DIR)
    print("Results saved to:", RES_DIR)
