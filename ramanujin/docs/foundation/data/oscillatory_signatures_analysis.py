"""
Oscillatory Signatures Analysis
2-panel visualization of oscillatory hole signatures
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

if __name__ == "__main__":
    # Load data
    print("Loading oscillatory_signatures.json...")
    with open('oscillatory_signatures.json', 'r') as f:
        data = json.load(f)

    print(f"Timestamp: {data['timestamp']}")
    print(f"Keys: {list(data.keys())}")

    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================================
    # PANEL A: Signature Overview
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis('off')

    y_pos = 0.95
    ax_a.text(0.5, y_pos, 'Oscillatory Signature Data', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax_a.transAxes)
    y_pos -= 0.15

    # Display data structure
    for key, value in data.items():
        if key != 'timestamp':
            ax_a.text(0.1, y_pos, f'{key}:', 
                    ha='left', va='top', fontsize=10, fontweight='bold',
                    transform=ax_a.transAxes)
            y_pos -= 0.08
            
            if isinstance(value, (int, float)):
                ax_a.text(0.15, y_pos, f'{value}', 
                        ha='left', va='top', fontsize=9,
                        transform=ax_a.transAxes, family='monospace')
                y_pos -= 0.08
            elif isinstance(value, list):
                ax_a.text(0.15, y_pos, f'List with {len(value)} elements', 
                        ha='left', va='top', fontsize=9,
                        transform=ax_a.transAxes, family='monospace')
                y_pos -= 0.08
            elif isinstance(value, dict):
                ax_a.text(0.15, y_pos, f'Dict with keys: {list(value.keys())[:3]}...', 
                        ha='left', va='top', fontsize=9,
                        transform=ax_a.transAxes, family='monospace')
                y_pos -= 0.08

    ax_a.set_title('A. Signature Data Structure', fontsize=12, fontweight='bold')

    # ============================================================================
    # PANEL B: Timestamp Info
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis('off')

    y_pos = 0.95
    ax_b.text(0.5, y_pos, 'Detection Metadata', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax_b.transAxes)
    y_pos -= 0.15

    # Parse timestamp
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(data['timestamp'])
        
        ax_b.text(0.1, y_pos, 'Timestamp:', 
                ha='left', va='top', fontsize=10, fontweight='bold',
                transform=ax_b.transAxes)
        y_pos -= 0.08
        
        ax_b.text(0.15, y_pos, f'{dt.strftime("%Y-%m-%d")}', 
                ha='left', va='top', fontsize=9,
                transform=ax_b.transAxes, family='monospace')
        y_pos -= 0.06
        
        ax_b.text(0.15, y_pos, f'{dt.strftime("%H:%M:%S.%f")}', 
                ha='left', va='top', fontsize=9,
                transform=ax_b.transAxes, family='monospace')
        y_pos -= 0.12
        
    except Exception as e:
        ax_b.text(0.15, y_pos, f'Parse error: {e}', 
                ha='left', va='top', fontsize=8,
                transform=ax_b.transAxes)
        y_pos -= 0.12

    # Data summary
    ax_b.text(0.1, y_pos, 'Data Summary:', 
            ha='left', va='top', fontsize=10, fontweight='bold',
            transform=ax_b.transAxes)
    y_pos -= 0.08

    n_keys = len([k for k in data.keys() if k != 'timestamp'])
    ax_b.text(0.15, y_pos, f'Number of fields: {n_keys}', 
            ha='left', va='top', fontsize=9,
            transform=ax_b.transAxes)
    y_pos -= 0.08

    ax_b.text(0.15, y_pos, f'File: oscillatory_signatures.json', 
            ha='left', va='top', fontsize=9,
            transform=ax_b.transAxes)

    ax_b.set_title('B. Detection Metadata', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle('Oscillatory Signatures Analysis', 
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig('oscillatory_signatures_analysis.pdf', bbox_inches='tight')
    plt.savefig('oscillatory_signatures_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: oscillatory_signatures_analysis.pdf/.png")

    plt.show()
