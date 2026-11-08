#!/usr/bin/env python3
"""
Generate panel chart for Conservation and Equilibrium from the Ball Game.

This visualization demonstrates:
1. Ball conservation (total constant)
2. The "meaningless victory" - scoring all balls halts play
3. Why reactions can't go to 100% completion
4. Dynamic equilibrium as "everybody plays, nobody wins"
5. How conservation forces reversal
6. Le Chatelier from the ball game
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
from pathlib import Path
import json

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'figure.facecolor': '#0a0a12',
    'axes.facecolor': '#12121a',
    'axes.edgecolor': '#3a3a4a',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'grid.color': '#2a2a3a',
    'grid.alpha': 0.3
})

# Color scheme
COLORS = {
    'team_a': '#4ecdc4',      # Cyan-teal for Team A
    'team_b': '#ff6b6b',      # Coral for Team B
    'partition': '#8b8b8b',   # Gray for partition
    'aperture': '#ffd93d',    # Yellow for apertures
    'equilibrium': '#95e1a3', # Green for equilibrium
    'warning': '#ff9f43',     # Orange for warning states
    'accent': '#a78bfa',      # Purple accent
    'background': '#1a1a2e',
    'text': '#e0e0e0'
}


def draw_ball(ax, x, y, color, size=0.08, alpha=1.0):
    """Draw a ball/molecule."""
    ball = Circle((x, y), size, facecolor=color, edgecolor='white', 
                  linewidth=1.5, alpha=alpha, zorder=5)
    ax.add_patch(ball)


def draw_game_state(ax, n_a, n_b, title, highlight=None):
    """Draw a game state showing ball distribution."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    
    # Draw containers
    left_box = FancyBboxPatch((-1.1, -0.6), 0.95, 1.2, 
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['team_a'], alpha=0.15,
                               edgecolor=COLORS['team_a'], linewidth=2)
    right_box = FancyBboxPatch((0.15, -0.6), 0.95, 1.2,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=COLORS['team_b'], alpha=0.15,
                                edgecolor=COLORS['team_b'], linewidth=2)
    ax.add_patch(left_box)
    ax.add_patch(right_box)
    
    # Draw partition with apertures
    partition = Rectangle((-0.05, -0.6), 0.1, 1.2, facecolor=COLORS['partition'], 
                         edgecolor='white', linewidth=1, zorder=3)
    ax.add_patch(partition)
    
    # Draw apertures
    for y in [-0.3, 0, 0.3]:
        aperture = Circle((0, y), 0.08, facecolor=COLORS['aperture'], 
                          edgecolor='white', linewidth=1, zorder=4)
        ax.add_patch(aperture)
    
    # Draw balls for Team A
    np.random.seed(42)
    for i in range(n_a):
        x = np.random.uniform(-0.95, -0.25)
        y = np.random.uniform(-0.45, 0.45)
        draw_ball(ax, x, y, COLORS['team_a'], alpha=0.9)
    
    # Draw balls for Team B
    np.random.seed(123)
    for i in range(n_b):
        x = np.random.uniform(0.25, 0.95)
        y = np.random.uniform(-0.45, 0.45)
        draw_ball(ax, x, y, COLORS['team_b'], alpha=0.9)
    
    # Labels
    ax.text(-0.6, -0.75, f'Team A: {n_a}', ha='center', fontsize=9, 
            color=COLORS['team_a'], fontweight='bold')
    ax.text(0.6, -0.75, f'Team B: {n_b}', ha='center', fontsize=9, 
            color=COLORS['team_b'], fontweight='bold')
    
    # Highlight if needed
    if highlight == 'a_empty':
        ax.text(-0.6, 0.05, 'EMPTY', fontsize=18, ha='center', va='center',
                color=COLORS['warning'], alpha=0.9, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    elif highlight == 'equilibrium':
        ax.text(0, 0.7, '= =', fontsize=20, ha='center', va='center',
                color=COLORS['equilibrium'], fontweight='bold')


def panel_conservation(ax):
    """Panel A: Ball conservation principle."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Conservation Law', fontsize=11, fontweight='bold', pad=10)
    
    # Equation box
    eq_box = FancyBboxPatch((0.5, 6), 9, 2.5,
                             boxstyle="round,pad=0.1,rounding_size=0.3",
                             facecolor='#1a1a2e', alpha=0.8,
                             edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(eq_box)
    
    ax.text(5, 7.5, r'$n_A(t) + n_B(t) = N$', fontsize=16, ha='center', 
            va='center', color='white', fontweight='bold')
    ax.text(5, 6.5, 'Total balls = constant', fontsize=10, ha='center',
            color=COLORS['text'], style='italic')
    
    # Visual representation
    # Time 1: 5-3
    ax.text(1.5, 4.5, 't₁', fontsize=10, ha='center', color=COLORS['text'])
    for i in range(5):
        draw_ball(ax, 0.8 + i*0.35, 3.5, COLORS['team_a'], size=0.2)
    for i in range(3):
        draw_ball(ax, 0.8 + i*0.35, 2.5, COLORS['team_b'], size=0.2)
    ax.text(3, 3, '=', fontsize=16, ha='center', color='white')
    ax.text(4, 3, '8', fontsize=14, ha='center', color=COLORS['equilibrium'], fontweight='bold')
    
    # Time 2: 3-5
    ax.text(6.5, 4.5, 't₂', fontsize=10, ha='center', color=COLORS['text'])
    for i in range(3):
        draw_ball(ax, 5.8 + i*0.35, 3.5, COLORS['team_a'], size=0.2)
    for i in range(5):
        draw_ball(ax, 5.8 + i*0.35, 2.5, COLORS['team_b'], size=0.2)
    ax.text(8, 3, '=', fontsize=16, ha='center', color='white')
    ax.text(9, 3, '8', fontsize=14, ha='center', color=COLORS['equilibrium'], fontweight='bold')
    
    # Arrow showing redistribution
    ax.annotate('', xy=(7, 4), xytext=(3.5, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.text(5.25, 4.3, 'redistributed', fontsize=8, ha='center', color=COLORS['accent'])
    
    # Summary text
    ax.text(5, 1, 'Distribution changes, total conserved', fontsize=9, 
            ha='center', color=COLORS['text'])


def panel_progression(ax):
    """Panel B: Progression to empty state."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('B. Team A Scores Repeatedly', fontsize=11, fontweight='bold', pad=10)
    
    states = [
        (4, 2, 'Start'),
        (2, 4, 'A scores 2'),
        (0, 6, 'A scores all'),
    ]
    
    y_positions = [7.5, 4.5, 1.5]
    
    for (n_a, n_b, label), y in zip(states, y_positions):
        # Mini game state
        # Team A side
        ax.add_patch(FancyBboxPatch((0.5, y-0.8), 3, 1.6,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     facecolor=COLORS['team_a'], alpha=0.15,
                     edgecolor=COLORS['team_a'], linewidth=1))
        # Team B side
        ax.add_patch(FancyBboxPatch((5.5, y-0.8), 3, 1.6,
                     boxstyle="round,pad=0.02,rounding_size=0.1",
                     facecolor=COLORS['team_b'], alpha=0.15,
                     edgecolor=COLORS['team_b'], linewidth=1))
        
        # Partition
        ax.add_patch(Rectangle((3.8, y-0.8), 0.4, 1.6, facecolor=COLORS['partition']))
        # Aperture
        ax.add_patch(Circle((4, y), 0.15, facecolor=COLORS['aperture']))
        
        # Balls
        np.random.seed(int(y*10))
        for i in range(n_a):
            bx = 1 + i*0.5
            by = y + np.random.uniform(-0.3, 0.3)
            draw_ball(ax, bx, by, COLORS['team_a'], size=0.15)
        
        for i in range(n_b):
            bx = 6 + i*0.4
            by = y + np.random.uniform(-0.3, 0.3)
            draw_ball(ax, bx, by, COLORS['team_b'], size=0.15)
        
        # Labels
        ax.text(2, y-1.1, f'A:{n_a}', fontsize=8, ha='center', color=COLORS['team_a'])
        ax.text(7, y-1.1, f'B:{n_b}', fontsize=8, ha='center', color=COLORS['team_b'])
        ax.text(9.3, y, label, fontsize=8, ha='left', va='center', color=COLORS['text'])
        
        # Warning for empty state
        if n_a == 0:
            ax.text(2, y, 'EMPTY!', fontsize=10, ha='center', va='center',
                    color=COLORS['warning'], fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(5, 5.5), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))


def panel_meaningless(ax):
    """Panel C: The meaningless victory."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('C. "Victory" = Cannot Play', fontsize=11, fontweight='bold', pad=10)
    
    # Draw the empty A / full B state
    draw_game_state(ax, 0, 6, '', highlight='a_empty')
    
    # Add "HALTED" message
    ax.text(0, 0.85, 'GAME HALTED', fontsize=12, ha='center', 
            color=COLORS['warning'], fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    
    # Explanation
    ax.text(-0.6, -0.9, 'No balls\n→ Can\'t shoot', fontsize=8, ha='center',
            color=COLORS['team_a'])
    ax.text(0.6, -0.9, 'Must score\nback', fontsize=8, ha='center',
            color=COLORS['team_b'])


def panel_forced_reversal(ax):
    """Panel D: Forced reversal."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('D. Forced Direction Reversal', fontsize=11, fontweight='bold', pad=10)
    
    # State: 0-6 with reverse arrow
    # Team A (empty)
    ax.add_patch(FancyBboxPatch((0.5, 3), 3.5, 3,
                 boxstyle="round,pad=0.05,rounding_size=0.2",
                 facecolor=COLORS['team_a'], alpha=0.15,
                 edgecolor=COLORS['team_a'], linewidth=2))
    
    # Team B (full)
    ax.add_patch(FancyBboxPatch((6, 3), 3.5, 3,
                 boxstyle="round,pad=0.05,rounding_size=0.2",
                 facecolor=COLORS['team_b'], alpha=0.15,
                 edgecolor=COLORS['team_b'], linewidth=2))
    
    # Partition
    ax.add_patch(Rectangle((4.3, 3), 0.4, 3, facecolor=COLORS['partition']))
    ax.add_patch(Circle((4.5, 4.5), 0.2, facecolor=COLORS['aperture']))
    
    # Balls only on B side
    np.random.seed(77)
    for i in range(6):
        bx = 6.5 + (i % 3) * 0.8
        by = 3.8 + (i // 3) * 1.2
        draw_ball(ax, bx, by, COLORS['team_b'], size=0.25)
    
    # Empty indicator on A side
    ax.text(2.25, 4.5, 'EMPTY', fontsize=16, ha='center', va='center',
            color=COLORS['warning'], alpha=0.8, fontweight='bold')
    
    # Reverse arrow
    ax.annotate('', xy=(3, 4.5), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['team_b'], 
                               lw=4, mutation_scale=20))
    ax.text(4.5, 5.5, 'REVERSE', fontsize=12, ha='center', 
            color=COLORS['team_b'], fontweight='bold')
    
    # Labels
    ax.text(2.25, 2.5, 'A: 0 (waiting)', fontsize=10, ha='center', color=COLORS['team_a'])
    ax.text(7.75, 2.5, 'B: 6 (must return)', fontsize=10, ha='center', color=COLORS['team_b'])
    
    # Equation
    ax.text(5, 8, r'Rate$_A$ = f($n_A$) = 0', fontsize=11, ha='center', 
            color=COLORS['text'], style='italic')
    ax.text(5, 7, 'Forward halts → Reverse proceeds', fontsize=10, 
            ha='center', color=COLORS['accent'])


def panel_equilibrium(ax):
    """Panel E: Dynamic equilibrium."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('E. Dynamic Equilibrium', fontsize=11, fontweight='bold', pad=10)
    
    draw_game_state(ax, 3, 3, '', highlight='equilibrium')
    
    # Add bidirectional arrows
    ax.annotate('', xy=(0.12, 0.15), xytext=(-0.12, 0.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['team_a'], lw=2))
    ax.annotate('', xy=(-0.12, -0.15), xytext=(0.12, -0.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['team_b'], lw=2))
    
    # Text
    ax.text(0, -0.95, '"Everybody plays, nobody wins"', fontsize=9, ha='center',
            color=COLORS['equilibrium'], style='italic')


def panel_dynamic_not_static(ax):
    """Panel F: Dynamic vs static equilibrium."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('F. Dynamic ≠ Static', fontsize=11, fontweight='bold', pad=10)
    
    # Static view (wrong)
    ax.add_patch(FancyBboxPatch((0.3, 5.5), 4.2, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#2a1a1a', alpha=0.6,
                 edgecolor=COLORS['team_b'], linewidth=2, linestyle='--'))
    ax.text(2.4, 8.5, 'STATIC VIEW [X]', fontsize=10, ha='center', 
            color=COLORS['team_b'], fontweight='bold')
    ax.text(2.4, 7.3, '"Reaction stops"', fontsize=9, ha='center', 
            color=COLORS['text'])
    ax.text(2.4, 6.3, '"Nothing happens"', fontsize=9, ha='center', 
            color=COLORS['text'])
    
    # Dynamic view (correct)
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 4.2, 3.5,
                 boxstyle="round,pad=0.1",
                 facecolor='#1a2a1a', alpha=0.6,
                 edgecolor=COLORS['equilibrium'], linewidth=2))
    ax.text(7.6, 8.5, 'DYNAMIC VIEW [OK]', fontsize=10, ha='center', 
            color=COLORS['equilibrium'], fontweight='bold')
    ax.text(7.6, 7.3, '"Both reactions continue"', fontsize=9, ha='center', 
            color=COLORS['text'])
    ax.text(7.6, 6.3, '"Rates are equal"', fontsize=9, ha='center', 
            color=COLORS['text'])
    
    # Visual: Two bidirectional arrows
    ax.annotate('', xy=(8.5, 4), xytext=(6.7, 4),
                arrowprops=dict(arrowstyle='<->', color=COLORS['equilibrium'], lw=3))
    ax.text(7.6, 3.2, r'Rate$_→$ = Rate$_←$', fontsize=10, ha='center',
            color=COLORS['equilibrium'])
    
    # Chemical analogy
    ax.text(5, 1.5, 'A ⇌ B at equilibrium:', fontsize=10, ha='center', color='white')
    ax.text(5, 0.7, 'Both directions proceed continuously', fontsize=9, 
            ha='center', color=COLORS['text'], style='italic')


def panel_completion_impossible(ax):
    """Panel G: Why 100% completion is impossible."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('G. Complete Conversion Impossible', fontsize=11, fontweight='bold', pad=10)
    
    # Show asymptotic approach
    x = np.linspace(0, 8, 100)
    # Forward rate
    fwd = 4 * np.exp(-x/2)
    # Reverse rate
    rev = 4 * (1 - np.exp(-x/2))
    
    # Plot rates
    ax.plot(x + 1, fwd * 0.7 + 2, color=COLORS['team_a'], lw=2.5, label='Forward')
    ax.plot(x + 1, rev * 0.7 + 2, color=COLORS['team_b'], lw=2.5, label='Reverse')
    
    # Intersection point
    idx = np.argmin(np.abs(fwd - rev))
    ax.scatter([x[idx] + 1], [fwd[idx] * 0.7 + 2], s=100, color=COLORS['equilibrium'], 
               zorder=5, edgecolor='white', linewidth=2)
    ax.text(x[idx] + 1, fwd[idx] * 0.7 + 2.5, 'Equilibrium', fontsize=8, 
            ha='center', color=COLORS['equilibrium'])
    
    # Zero forward rate at completion
    ax.axhline(y=2, color='white', linestyle=':', alpha=0.3)
    ax.text(9.2, 2, '0', fontsize=9, color=COLORS['text'])
    
    # Arrow to asymptote
    ax.annotate('Forward → 0\nas [A] → 0', xy=(8.5, 2.2), xytext=(7, 3.5),
                fontsize=8, color=COLORS['team_a'],
                arrowprops=dict(arrowstyle='->', color=COLORS['team_a']))
    
    # Legend
    ax.text(2, 5.8, '→ Forward', fontsize=9, color=COLORS['team_a'])
    ax.text(2, 5.2, '← Reverse', fontsize=9, color=COLORS['team_b'])
    
    ax.text(5, 8.5, '[A] → 0 means Forward → 0', fontsize=9, ha='center', 
            color=COLORS['warning'])


def panel_le_chatelier(ax):
    """Panel H: Le Chatelier from conservation."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('H. Le Chatelier from Conservation', fontsize=11, fontweight='bold', pad=10)
    
    # Three scenarios
    scenarios = [
        ('Add to A', '+', COLORS['team_a'], 5, 3, 'Shift -> B'),
        ('Remove from B', '-', COLORS['team_b'], 3, 5, 'Shift -> B'),
        ('At equilibrium', '=', COLORS['equilibrium'], 4, 4, 'Balanced'),
    ]
    
    for i, (label, symbol, color, na, nb, result) in enumerate(scenarios):
        y = 7 - i * 2.5
        
        # Small diagram
        ax.add_patch(FancyBboxPatch((0.5, y-0.6), 1.8, 1.2,
                     facecolor=COLORS['team_a'], alpha=0.15,
                     edgecolor=COLORS['team_a']))
        ax.add_patch(FancyBboxPatch((2.7, y-0.6), 1.8, 1.2,
                     facecolor=COLORS['team_b'], alpha=0.15,
                     edgecolor=COLORS['team_b']))
        
        # Symbol
        ax.text(1.4, y, symbol, fontsize=20, ha='center', va='center', color=color)
        ax.text(3.6, y, f'{nb}', fontsize=14, ha='center', va='center', color=COLORS['team_b'])
        
        # Label and result
        ax.text(5.5, y + 0.2, label, fontsize=9, va='center', color=color, fontweight='bold')
        ax.text(5.5, y - 0.3, result, fontsize=9, va='center', color=COLORS['text'])
        
        # Arrow for non-equilibrium
        if symbol != '=':
            ax.annotate('', xy=(8.5, y), xytext=(7.5, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Conservation note
    ax.text(5, 0.8, r'$n_A + n_B = N$ always', fontsize=10, ha='center',
            color=COLORS['accent'], style='italic')


def panel_summary(ax):
    """Panel I: Summary equation and insight."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('I. The Meaningless Victory Theorem', fontsize=11, fontweight='bold', pad=10)
    
    # Main theorem box
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 9, 3.5,
                 boxstyle="round,pad=0.1,rounding_size=0.3",
                 facecolor='#1a1a2e', alpha=0.9,
                 edgecolor=COLORS['accent'], linewidth=2))
    
    ax.text(5, 8.2, 'THEOREM', fontsize=12, ha='center', 
            color=COLORS['accent'], fontweight='bold')
    ax.text(5, 7.2, 'If Team A scores all balls:', fontsize=10, ha='center',
            color=COLORS['text'])
    ax.text(5, 6.2, r'$n_A = 0 \Rightarrow \mathrm{Rate}_A = 0$', fontsize=12, 
            ha='center', color='white')
    
    # Consequence
    ax.add_patch(FancyBboxPatch((0.5, 2), 9, 2.5,
                 boxstyle="round,pad=0.1,rounding_size=0.3",
                 facecolor='#1a2a1a', alpha=0.9,
                 edgecolor=COLORS['equilibrium'], linewidth=2))
    
    ax.text(5, 3.7, '"Victory" halts the game', fontsize=11, ha='center',
            color=COLORS['equilibrium'], fontweight='bold')
    ax.text(5, 2.7, 'Equilibrium = sustained play, not winning', fontsize=10, 
            ha='center', color=COLORS['text'])
    
    # Chemical translation
    ax.text(5, 0.8, 'Chemical: [A]→0 halts reaction → equilibrium required', 
            fontsize=9, ha='center', color=COLORS['text'], style='italic')


def main():
    """Generate the 9-panel conservation equilibrium visualization."""
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0a0a12')
    
    # Create 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Row 2
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Row 3
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Generate panels
    panel_conservation(ax1)
    panel_progression(ax2)
    panel_meaningless(ax3)
    panel_forced_reversal(ax4)
    panel_equilibrium(ax5)
    panel_dynamic_not_static(ax6)
    panel_completion_impossible(ax7)
    panel_le_chatelier(ax8)
    panel_summary(ax9)
    
    # Main title
    fig.suptitle('Conservation and the "Meaningless Victory"\nWhy Chemical Equilibrium Exists',
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'catalysis' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'conservation_equilibrium_panel.png'
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    # Save results summary
    results = {
        'panel_type': 'conservation_equilibrium',
        'title': 'Conservation and the Meaningless Victory',
        'key_insights': [
            'Ball conservation: n_A + n_B = N (constant)',
            'Scoring all balls halts the game (Rate_A = 0 when n_A = 0)',
            'Victory is self-defeating - cannot continue playing',
            'Forced reversal when one side depletes',
            'Dynamic equilibrium: both sides play continuously',
            'Complete conversion impossible - halts forward reaction',
            'Le Chatelier emerges from redistributing conserved quantity',
            'Equilibrium = sustained mutual play, not winning'
        ],
        'chemical_translation': {
            'balls': 'molecules/substrate',
            'scoring': 'reaction proceeding',
            'Team A empty': '[A] → 0, forward rate → 0',
            'equilibrium': 'Rate_forward = Rate_reverse',
            'conservation': 'mass conservation'
        },
        'theorems': [
            'Meaningless Victory Theorem',
            'Equilibrium from Conservation',
            'Le Chatelier from Redistribution'
        ]
    }
    
    results_path = output_dir / 'conservation_equilibrium_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    
    plt.close()


if __name__ == '__main__':
    main()

