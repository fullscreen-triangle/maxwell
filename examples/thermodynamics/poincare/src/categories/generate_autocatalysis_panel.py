"""
Autocatalytic Apertures Panel Chart
The Ball Game Thought Experiment - deriving autocatalysis from first principles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from pathlib import Path
import json

# Style configuration
COLORS = {
    'team_a': '#3498DB',        # Blue
    'team_b': '#E74C3C',        # Red
    'aperture': '#1ABC9C',      # Teal
    'blocked': '#95A5A6',       # Gray
    'open': '#27AE60',          # Green
    'ball': '#F39C12',          # Gold
    'score': '#27AE60',         # Green
    'overwhelmed': '#E74C3C',   # Red
    'cascade': '#9B59B6',       # Purple
    'primary': '#2C3E50',
    'partition': '#34495E',
}


def setup_style():
    """Configure matplotlib style."""
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
        'axes.linewidth': 1.0,
    })


def draw_ball(ax, x, y, color, label=''):
    """Draw a ball."""
    ax.add_patch(Circle((x, y), 0.25, color=color, ec='black', lw=1))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=7, 
               fontweight='bold', color='white')


def draw_partition(ax, x, n_holes, hole_positions):
    """Draw partition with holes."""
    # Main partition
    ax.add_patch(Rectangle((x - 0.15, 0), 0.3, 10, color=COLORS['partition']))
    
    # Holes (apertures)
    for y in hole_positions:
        ax.add_patch(Circle((x, y), 0.35, color=COLORS['aperture']))


def generate_autocatalysis_panel(output_dir: str):
    """Generate the autocatalytic ball game panel chart."""
    setup_style()
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    # =========================================================================
    # Panel A: The Ball Game Setup
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Teams
    ax1.add_patch(FancyBboxPatch((0.5, 1), 4, 8, boxstyle="round",
                                  facecolor='#EBF5FB', edgecolor=COLORS['team_a'], lw=2))
    ax1.text(2.5, 9.5, 'TEAM A', ha='center', fontsize=10, fontweight='bold',
            color=COLORS['team_a'])
    
    ax1.add_patch(FancyBboxPatch((5.5, 1), 4, 8, boxstyle="round",
                                  facecolor='#FDEDEC', edgecolor=COLORS['team_b'], lw=2))
    ax1.text(7.5, 9.5, 'TEAM B', ha='center', fontsize=10, fontweight='bold',
            color=COLORS['team_b'])
    
    # Partition with holes
    draw_partition(ax1, 5, 3, [3, 5, 7])
    
    # Balls for each team
    for i, y in enumerate([3, 5, 7]):
        draw_ball(ax1, 2, y, COLORS['team_a'])
        draw_ball(ax1, 8, y, COLORS['team_b'])
        # Arrows showing shots
        ax1.annotate('', xy=(4.5, y), xytext=(2.3, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['team_a']))
        ax1.annotate('', xy=(5.5, y), xytext=(7.7, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['team_b']))
    
    ax1.text(5, 0.3, 'Rules: Cannot hold balls\nMust shoot immediately',
            ha='center', fontsize=8, style='italic')
    
    ax1.set_title('A. The Ball Game Setup\n(Two teams, partition with apertures)', fontweight='bold')
    ax1.axis('off')
    
    # =========================================================================
    # Panel B: Mutual Blocking = Equilibrium
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Teams (faded)
    ax2.add_patch(FancyBboxPatch((0.5, 1), 4, 8, boxstyle="round",
                                  facecolor='#EBF5FB', alpha=0.3))
    ax2.add_patch(FancyBboxPatch((5.5, 1), 4, 8, boxstyle="round",
                                  facecolor='#FDEDEC', alpha=0.3))
    
    # Partition
    draw_partition(ax2, 5, 3, [3, 5, 7])
    
    # Balls colliding at apertures (blocked)
    for y in [3, 5, 7]:
        draw_ball(ax2, 4.5, y, COLORS['team_a'])
        draw_ball(ax2, 5.5, y, COLORS['team_b'])
        # Collision marker
        ax2.plot([4.8, 5.2], [y + 0.15, y - 0.15], '-', color=COLORS['blocked'], lw=2)
        ax2.plot([4.8, 5.2], [y - 0.15, y + 0.15], '-', color=COLORS['blocked'], lw=2)
    
    ax2.text(5, 9, 'EQUILIBRIUM', ha='center', fontsize=12, fontweight='bold',
            color=COLORS['blocked'])
    ax2.text(5, 8, 'Every ball is blocked', ha='center', fontsize=9)
    
    ax2.text(5, 0.5, 'Mutual blocking = Penultimate state\n(Both sides one step from scoring)',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='#FEF9E7'))
    
    ax2.set_title('B. Equilibrium as Mutual Blocking\n(Neither side can score)', fontweight='bold')
    ax2.axis('off')
    
    # =========================================================================
    # Panel C: Velocity Is Irrelevant
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Fast ball (blocked)
    ax3.add_patch(FancyBboxPatch((0.5, 5.5), 4, 4, boxstyle="round",
                                  facecolor=COLORS['team_a'], alpha=0.1))
    draw_ball(ax3, 2, 7.5, COLORS['team_a'])
    ax3.annotate('', xy=(4, 7.5), xytext=(2.3, 7.5),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['team_a']))
    ax3.text(3, 8.3, 'FAST', fontsize=9, fontweight='bold', color=COLORS['team_a'])
    # Blocked indicator
    ax3.add_patch(Circle((4.5, 7.5), 0.3, color=COLORS['blocked']))
    ax3.text(4.5, 7.5, '✗', ha='center', va='center', fontsize=12, color='white')
    ax3.text(6, 7.5, 'BLOCKED\n→ No score', ha='left', fontsize=8, color=COLORS['team_b'])
    
    # Slow ball (open)
    ax3.add_patch(FancyBboxPatch((0.5, 0.5), 4, 4, boxstyle="round",
                                  facecolor=COLORS['team_a'], alpha=0.1))
    draw_ball(ax3, 2, 2.5, COLORS['team_a'])
    ax3.annotate('', xy=(4, 2.5), xytext=(2.3, 2.5),
                arrowprops=dict(arrowstyle='->', lw=1, color=COLORS['team_a']))
    ax3.text(3, 3.3, 'slow', fontsize=8, color=COLORS['team_a'])
    # Open indicator
    ax3.add_patch(Circle((4.5, 2.5), 0.3, color=COLORS['open']))
    ax3.text(4.5, 2.5, '✓', ha='center', va='center', fontsize=12, color='white')
    ax3.text(6, 2.5, 'OPEN\n→ SCORES!', ha='left', fontsize=8, color=COLORS['score'],
            fontweight='bold')
    
    ax3.text(5, 5, 'Speed doesn\'t matter!\nOnly aperture availability',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax3.set_title('C. Velocity Independence\n(Configuration, not speed)', fontweight='bold')
    ax3.axis('off')
    
    # =========================================================================
    # Panel D: First Score - Team A Gets One Through
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # Teams
    ax4.add_patch(FancyBboxPatch((0.5, 1), 4, 8, boxstyle="round",
                                  facecolor='#EBF5FB', alpha=0.5))
    ax4.text(2.5, 9.3, 'Team A: 2 balls', ha='center', fontsize=9, color=COLORS['team_a'])
    
    ax4.add_patch(FancyBboxPatch((5.5, 1), 4, 8, boxstyle="round",
                                  facecolor='#FDEDEC', alpha=0.5))
    ax4.text(7.5, 9.3, 'Team B: 4 balls!', ha='center', fontsize=9, color=COLORS['team_b'],
            fontweight='bold')
    
    # Partition
    draw_partition(ax4, 5, 3, [3, 5, 7])
    
    # Team A balls (2 left)
    draw_ball(ax4, 2, 4, COLORS['team_a'])
    draw_ball(ax4, 2, 6, COLORS['team_a'])
    
    # Team B balls (4 now - overwhelmed!)
    draw_ball(ax4, 7, 3, COLORS['team_b'])
    draw_ball(ax4, 7, 5, COLORS['team_b'])
    draw_ball(ax4, 7, 7, COLORS['team_b'])
    draw_ball(ax4, 8.5, 5, COLORS['team_b'])  # Extra ball
    
    # Highlight overwhelmed player
    ax4.add_patch(Circle((7.75, 5), 1.2, color=COLORS['overwhelmed'], 
                         fill=False, lw=2, linestyle='--'))
    ax4.text(7.75, 3.5, 'One player\njuggling 2!', ha='center', fontsize=7,
            color=COLORS['overwhelmed'])
    
    # Score indicator
    ax4.annotate('', xy=(7, 5), xytext=(4.5, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['score'],
                               connectionstyle='arc3,rad=0.3'))
    ax4.text(5.5, 6.5, 'SCORED!', fontsize=10, color=COLORS['score'], fontweight='bold')
    
    ax4.set_title('D. First Score: Team B Overwhelmed\n(4 balls, 3 holes)', fontweight='bold')
    ax4.axis('off')
    
    # =========================================================================
    # Panel E: The Cascade Effect
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    
    # Cascade diagram
    stages = [
        (1, 8, '3v3', 'Full coverage', COLORS['blocked']),
        (3, 6, '2v4', 'One overwhelmed', COLORS['overwhelmed']),
        (5, 4, '1v5', 'Two overwhelmed', COLORS['cascade']),
        (7, 2, '0v6', 'All overwhelmed!', COLORS['score']),
    ]
    
    for x, y, ratio, desc, color in stages:
        ax5.add_patch(Circle((x, y), 0.6, color=color, alpha=0.7))
        ax5.text(x, y, ratio, ha='center', va='center', fontsize=9, 
                fontweight='bold', color='white')
        ax5.text(x + 1.2, y, desc, ha='left', fontsize=8, va='center')
    
    # Arrows between stages
    for i in range(len(stages) - 1):
        x1, y1 = stages[i][0], stages[i][1]
        x2, y2 = stages[i+1][0], stages[i+1][1]
        ax5.annotate('', xy=(x2 - 0.3, y2 + 0.5), xytext=(x1 + 0.3, y1 - 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    ax5.text(5, 9, 'AUTOCATALYTIC CASCADE', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['cascade'])
    ax5.text(5, 0.5, 'Each score makes next score EASIER',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax5.set_title('E. The Cascade Effect\n(Resistance decreases with each score)', fontweight='bold')
    ax5.axis('off')
    
    # =========================================================================
    # Panel F: Resistance vs Scores
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Resistance equation: R = k/(n+m) where m = scores
    n = 3  # initial balls
    k = 3  # holes
    m = np.arange(0, 10)
    R = k / (n + m)
    
    ax6.plot(m, R, 'o-', color=COLORS['cascade'], lw=2.5, markersize=8)
    ax6.fill_between(m, 0, R, alpha=0.2, color=COLORS['cascade'])
    
    ax6.set_xlabel('Scores by Team A')
    ax6.set_ylabel('Resistance R = k/(n+m)')
    ax6.set_ylim(0, 1.1)
    ax6.set_xlim(-0.5, 9.5)
    
    # Annotations
    ax6.annotate('Full blocking', (0, 1), textcoords="offset points",
                xytext=(10, 10), fontsize=8)
    ax6.annotate('Half resistance', (3, 0.5), textcoords="offset points",
                xytext=(10, 10), fontsize=8)
    ax6.annotate('Low resistance', (6, 0.33), textcoords="offset points",
                xytext=(10, -15), fontsize=8)
    
    ax6.axhline(y=0.5, color=COLORS['blocked'], linestyle='--', lw=1, alpha=0.5)
    
    ax6.set_title('F. Resistance Decreases\n(Positive feedback)', fontweight='bold')
    
    # =========================================================================
    # Panel G: "Seeing Behind the Wall"
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    
    # Wall
    ax7.add_patch(Rectangle((4.5, 0), 1, 10, color=COLORS['partition']))
    ax7.text(5, 5, 'WALL', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white', rotation=90)
    
    # Team A side (known)
    ax7.add_patch(FancyBboxPatch((0.5, 1), 3.5, 8, boxstyle="round",
                                  facecolor='#EBF5FB', alpha=0.5))
    ax7.text(2.25, 9.3, 'Known', ha='center', fontsize=9, color=COLORS['team_a'])
    
    # Team B side (unknown, then revealed)
    ax7.add_patch(FancyBboxPatch((6, 1), 3.5, 8, boxstyle="round",
                                  facecolor='#2C3E50', alpha=0.3))
    ax7.text(7.75, 9.3, '???', ha='center', fontsize=9, color=COLORS['partition'])
    
    # Ball going through
    draw_ball(ax7, 3, 5, COLORS['team_a'])
    ax7.annotate('', xy=(6.5, 5), xytext=(3.3, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['score']))
    
    # Now we "see"
    draw_ball(ax7, 7, 5, COLORS['team_a'])
    ax7.add_patch(Circle((7, 5), 1, color=COLORS['score'], fill=False, lw=2))
    
    ax7.text(7.75, 3, 'Now we have\na presence!', ha='center', fontsize=8,
            color=COLORS['score'], fontweight='bold')
    
    ax7.text(5, 0.5, 'Scoring = establishing categorical\nstructure on the other side',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5'))
    
    ax7.set_title('G. "Seeing Behind the Wall"\n(Product creates categorical presence)', fontweight='bold')
    ax7.axis('off')
    
    # =========================================================================
    # Panel H: Connection to Enzyme Kinetics
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Lag-exponential-saturation curve
    t = np.linspace(0, 10, 100)
    # Autocatalytic curve: y = K / (1 + e^(-r(t-t0)))
    K = 1
    r = 1
    t0 = 3
    y = K / (1 + np.exp(-r * (t - t0)))
    
    ax8.plot(t, y, '-', color=COLORS['cascade'], lw=2.5)
    ax8.fill_between(t, 0, y, alpha=0.2, color=COLORS['cascade'])
    
    # Phase labels
    ax8.axvspan(0, 2, alpha=0.1, color=COLORS['blocked'])
    ax8.axvspan(2, 5, alpha=0.1, color=COLORS['cascade'])
    ax8.axvspan(5, 10, alpha=0.1, color=COLORS['score'])
    
    ax8.text(1, 0.9, 'LAG\n(full blocking)', ha='center', fontsize=8)
    ax8.text(3.5, 0.5, 'EXPONENTIAL\n(cascade)', ha='center', fontsize=8,
            color=COLORS['cascade'])
    ax8.text(7.5, 0.9, 'SATURATION', ha='center', fontsize=8, color=COLORS['score'])
    
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Product concentration')
    ax8.set_ylim(0, 1.1)
    
    ax8.set_title('H. Autocatalytic Kinetics\n(Lag → Exponential → Saturation)', fontweight='bold')
    
    # =========================================================================
    # Panel I: Summary
    # =========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_xlim(0, 10)
    ax9.set_ylim(0, 10)
    
    summary_lines = [
        ('BALL GAME REVEALS:', '', 'black', True),
        ('', '', 'black', False),
        ('1. Velocity is IRRELEVANT', '', COLORS['score'], False),
        ('   (only aperture availability)', '', COLORS['primary'], False),
        ('', '', 'black', False),
        ('2. Catalysis is AUTOCATALYTIC', '', COLORS['cascade'], False),
        ('   (each score reduces resistance)', '', COLORS['primary'], False),
        ('', '', 'black', False),
        ('3. Time is NOT fundamental', '', COLORS['team_a'], False),
        ('   (categorical availability is)', '', COLORS['primary'], False),
        ('', '', 'black', False),
        ('4. Products CREATE categories', '', COLORS['team_b'], False),
        ('   ("seeing behind the wall")', '', COLORS['primary'], False),
    ]
    
    y_pos = 9
    for left, right, color, bold in summary_lines:
        weight = 'bold' if bold else 'normal'
        if left:
            ax9.text(0.5, y_pos, left, fontsize=9, color=color, fontweight=weight)
        y_pos -= 0.65
    
    ax9.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 1.5, boxstyle="round",
                                  facecolor='#E8F8F5', edgecolor=COLORS['score'], lw=2))
    ax9.text(5, 1, 'CATALYSIS = AUTOCATALYTIC APERTURES', ha='center',
            fontsize=11, fontweight='bold', color=COLORS['score'])
    
    ax9.set_title('I. Summary: Key Insights', fontweight='bold')
    ax9.axis('off')
    
    # Main title
    plt.suptitle('The Ball Game: Deriving Autocatalysis from First Principles',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "autocatalysis_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Save results
    results = {
        'experiment': 'Ball Game Thought Experiment',
        'key_findings': {
            'velocity_independence': True,
            'autocatalytic_cascade': True,
            'resistance_equation': 'R = k / (n + m)',
            'seeing_behind_wall': 'Product creates categorical presence',
            'kinetics': 'Lag → Exponential → Saturation'
        },
        'conclusion': 'Catalysis is inherently autocatalytic: each successful transit reduces resistance to subsequent transits, independent of velocity or time'
    }
    
    results_path = Path(output_dir).parent / "results" / "autocatalysis_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")


def main():
    """Generate autocatalysis panel."""
    output_dir = "../docs/catalysis/figures"
    
    print("=" * 60)
    print("GENERATING AUTOCATALYSIS PANEL")
    print("=" * 60)
    
    generate_autocatalysis_panel(output_dir)
    
    print("\n" + "=" * 60)
    print("THE BALL GAME THOUGHT EXPERIMENT")
    print("=" * 60)
    print("""
SETUP:
  - Two teams, partition with holes (apertures)
  - Cannot hold balls, must shoot immediately
  - Balls collide at apertures (blocking)

KEY INSIGHTS:

1. VELOCITY IS IRRELEVANT
   - Fast ball + blocked hole = NO SCORE
   - Slow ball + open hole = SCORES!
   - Only aperture availability matters

2. EQUILIBRIUM = MUTUAL BLOCKING
   - Every ball blocked by opposing ball
   - Both sides at penultimate state
   - Neither can complete

3. AUTOCATALYTIC CASCADE
   - Team A scores: Team B now has +1 ball
   - Team B overwhelmed: can't cover all holes
   - Resistance decreases: R = k/(n+m)
   - Next score is EASIER
   
4. "SEEING BEHIND THE WALL"
   - Scoring = establishing presence
   - Product creates categorical structure
   - Opens pathways for more products

RESULT:
   Catalysis is inherently AUTOCATALYTIC
   Each success reduces resistance to next success
   Independent of velocity, distance, or time
""")


if __name__ == "__main__":
    main()

