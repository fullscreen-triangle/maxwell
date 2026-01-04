"""
Sentropy Coordinates Analysis
2-panel visualization of spatial entropy coordinates
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
    print("Loading sentropy_coordinates.json...")
    with open('sentropy_coordinates.json', 'r') as f:
        data = json.load(f)

    print(f"Timestamp: {data['timestamp']}")
    print(f"Keys: {list(data.keys())}")

    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================================
    # PANEL A: Sentropy Data Overview
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis('off')

    y_pos = 0.95
    ax_a.text(0.5, y_pos, 'Sentropy Coordinate Data', 
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
                
                # Show first few elements if numeric
                if len(value) > 0 and isinstance(value[0], (int, float)):
                    preview = str(value[:5])[1:-1]
                    if len(value) > 5:
                        preview += ', ...'
                    ax_a.text(0.2, y_pos, preview, 
                            ha='left', va='top', fontsize=7,
                            transform=ax_a.transAxes, family='monospace',
                            style='italic')
                    y_pos -= 0.06
                    
            elif isinstance(value, dict):
                ax_a.text(0.15, y_pos, f'Dict with {len(value)} keys', 
                        ha='left', va='top', fontsize=9,
                        transform=ax_a.transAxes, family='monospace')
                y_pos -= 0.08

    ax_a.set_title('A. Data Structure', fontsize=12, fontweight='bold')

    # ============================================================================
    # PANEL B: Information Geometry
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis('off')

    y_pos = 0.95
    ax_b.text(0.5, y_pos, 'Information Geometry', 
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax_b.transAxes)
    y_pos -= 0.15

    # Conceptual diagram
    ax_b.text(0.1, y_pos, 'Sentropy coordinates represent:', 
            ha='left', va='top', fontsize=10, fontweight='bold',
            transform=ax_b.transAxes)
    y_pos -= 0.10

    concepts = [
        '• Spatial entropy distribution',
        '• Information geometry of O₂ states',
        '• Coordinate system for thought space',
        '• Mapping from physical → phenomenal',
    ]

    for concept in concepts:
        ax_b.text(0.15, y_pos, concept, 
                ha='left', va='top', fontsize=9,
                transform=ax_b.transAxes)
        y_pos -= 0.08

    y_pos -= 0.05

    # Timestamp info
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(data['timestamp'])
        ax_b.text(0.1, y_pos, f'Captured: {dt.strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='left', va='top', fontsize=8,
                transform=ax_b.transAxes, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    except:
        pass

    ax_b.set_title('B. Conceptual Framework', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle('Sentropy Coordinates Analysis', 
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig('sentropy_coordinates_analysis.pdf', bbox_inches='tight')
    plt.savefig('sentropy_coordinates_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: sentropy_coordinates_analysis.pdf/.png")

    plt.show()
