"""
Categorical Visualization: Seeing the S-Entropy Space
======================================================

Visualize the categorical gas, molecular distributions,
and thermodynamic properties.
"""

import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from .virtual_molecule import VirtualMolecule, SCoordinate
    from .virtual_chamber import VirtualChamber
except ImportError:
    from virtual_molecule import VirtualMolecule, SCoordinate
    from virtual_chamber import VirtualChamber


@dataclass
class PlotData:
    """Data structure for plotting."""
    x: List[float]
    y: List[float]
    z: Optional[List[float]] = None
    labels: Optional[List[str]] = None
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""


class CategoricalVisualizer:
    """
    Visualizer for categorical gas and S-space.

    Generates data structures that can be plotted with matplotlib
    or any other plotting library.
    """

    def __init__(self, chamber: Optional[VirtualChamber] = None):
        self.chamber = chamber

    def set_chamber(self, chamber: VirtualChamber) -> None:
        self.chamber = chamber

    def s_space_scatter(self) -> PlotData:
        """
        3D scatter plot of molecules in S-space.

        Each molecule is a point at (S_k, S_t, S_e).
        """
        if not self.chamber:
            return PlotData([], [], [], title="No chamber")

        x = [mol.s_coord.S_k for mol in self.chamber.gas]
        y = [mol.s_coord.S_t for mol in self.chamber.gas]
        z = [mol.s_coord.S_e for mol in self.chamber.gas]

        return PlotData(
            x=x, y=y, z=z,
            title="Molecules in S-Entropy Space",
            xlabel="S_k (Knowledge)",
            ylabel="S_t (Temporal)",
            zlabel="S_e (Evolution)"
        )

    def s_k_histogram(self, bins: int = 20) -> PlotData:
        """Histogram of S_k (knowledge entropy) values."""
        if not self.chamber:
            return PlotData([], [], title="No chamber")

        values = [mol.s_coord.S_k for mol in self.chamber.gas]
        hist, edges = self._histogram(values, bins)
        centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]

        return PlotData(
            x=centers, y=hist,
            title="Distribution of Knowledge Entropy (S_k)",
            xlabel="S_k",
            ylabel="Count"
        )

    def s_t_histogram(self, bins: int = 20) -> PlotData:
        """Histogram of S_t (temporal entropy) values."""
        if not self.chamber:
            return PlotData([], [], title="No chamber")

        values = [mol.s_coord.S_t for mol in self.chamber.gas]
        hist, edges = self._histogram(values, bins)
        centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]

        return PlotData(
            x=centers, y=hist,
            title="Distribution of Temporal Entropy (S_t)",
            xlabel="S_t",
            ylabel="Count"
        )

    def s_e_histogram(self, bins: int = 20) -> PlotData:
        """Histogram of S_e (evolution entropy) values."""
        if not self.chamber:
            return PlotData([], [], title="No chamber")

        values = [mol.s_coord.S_e for mol in self.chamber.gas]
        hist, edges = self._histogram(values, bins)
        centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]

        return PlotData(
            x=centers, y=hist,
            title="Distribution of Evolution Entropy (S_e)",
            xlabel="S_e",
            ylabel="Count"
        )

    def temperature_evolution(self, samples: int = 100,
                             interval: float = 0.01) -> PlotData:
        """
        Track temperature over time as new molecules are added.

        This shows that temperature (jitter variance) is a real,
        measurable property that evolves over time.
        """
        import time

        temperatures = []
        times = []

        start = time.perf_counter()
        for i in range(samples):
            self.chamber.sample()
            T = self.chamber.gas.temperature
            temperatures.append(T)
            times.append(time.perf_counter() - start)
            time.sleep(interval)

        return PlotData(
            x=times, y=temperatures,
            title="Temperature Evolution (Real Hardware Jitter)",
            xlabel="Time (s)",
            ylabel="Categorical Temperature"
        )

    def phase_space_2d(self, dim1: str = 'S_k', dim2: str = 'S_e') -> PlotData:
        """2D projection of S-space."""
        if not self.chamber:
            return PlotData([], [], title="No chamber")

        dim_map = {
            'S_k': lambda m: m.s_coord.S_k,
            'S_t': lambda m: m.s_coord.S_t,
            'S_e': lambda m: m.s_coord.S_e,
        }

        if dim1 not in dim_map or dim2 not in dim_map:
            return PlotData([], [], title="Invalid dimensions")

        x = [dim_map[dim1](mol) for mol in self.chamber.gas]
        y = [dim_map[dim2](mol) for mol in self.chamber.gas]

        return PlotData(
            x=x, y=y,
            title=f"Phase Space: {dim1} vs {dim2}",
            xlabel=dim1,
            ylabel=dim2
        )

    def maxwell_boltzmann_comparison(self, bins: int = 20) -> Dict[str, PlotData]:
        """
        Compare S_e distribution to Maxwell-Boltzmann.

        Hardware timing should approximately follow MB distribution.
        """
        if not self.chamber or len(self.chamber.gas) < 10:
            return {}

        # Actual distribution
        values = [mol.s_coord.S_e for mol in self.chamber.gas]
        hist, edges = self._histogram(values, bins)
        centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]

        actual = PlotData(
            x=centers, y=[h/max(hist) for h in hist],  # Normalize
            title="S_e Distribution vs Maxwell-Boltzmann",
            xlabel="S_e",
            ylabel="Normalized Count"
        )

        # Theoretical MB distribution
        T = self.chamber.gas.temperature
        if T > 0:
            mb = [math.sqrt(2/math.pi) * (x/T)**0.5 * math.exp(-x/(2*T))
                  for x in centers]
            # Normalize
            max_mb = max(mb) if mb else 1
            mb = [v/max_mb for v in mb]
        else:
            mb = [0] * len(centers)

        theoretical = PlotData(
            x=centers, y=mb,
            title="Maxwell-Boltzmann (theoretical)",
            xlabel="S_e",
            ylabel="Probability Density"
        )

        return {'actual': actual, 'theoretical': theoretical}

    def demon_sorting_visualization(self,
                                   hot: List[VirtualMolecule],
                                   cold: List[VirtualMolecule]) -> Dict[str, PlotData]:
        """
        Visualize the Maxwell demon's sorting.

        Shows how molecules are separated in S-space.
        """
        hot_data = PlotData(
            x=[m.s_coord.S_k for m in hot],
            y=[m.s_coord.S_e for m in hot],
            title="Hot Compartment",
            xlabel="S_k",
            ylabel="S_e"
        )

        cold_data = PlotData(
            x=[m.s_coord.S_k for m in cold],
            y=[m.s_coord.S_e for m in cold],
            title="Cold Compartment",
            xlabel="S_k",
            ylabel="S_e"
        )

        return {'hot': hot_data, 'cold': cold_data}

    def harmonic_coincidence_network(self,
                                    molecules: List[VirtualMolecule],
                                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Generate network of harmonic coincidences between molecules.

        Returns nodes and edges for network visualization.
        """
        nodes = []
        edges = []

        for i, mol1 in enumerate(molecules):
            nodes.append({
                'id': i,
                'S_k': mol1.s_coord.S_k,
                'S_t': mol1.s_coord.S_t,
                'S_e': mol1.s_coord.S_e,
                'freq': mol1.frequency
            })

            for j, mol2 in enumerate(molecules[i+1:], start=i+1):
                if mol1.frequency > 0 and mol2.frequency > 0:
                    ratio = mol1.frequency / mol2.frequency
                    # Check for harmonic relationship
                    for n in range(1, 10):
                        for m in range(1, 10):
                            if abs(ratio - n/m) < threshold:
                                edges.append({
                                    'source': i,
                                    'target': j,
                                    'harmonic': (n, m),
                                    'strength': 1.0 / (n + m)
                                })
                                break

        return {'nodes': nodes, 'edges': edges}

    def _histogram(self, values: List[float], bins: int
                  ) -> Tuple[List[int], List[float]]:
        """Simple histogram implementation."""
        if not values:
            return [], []

        min_val = min(values)
        max_val = max(values)

        if min_val == max_val:
            return [len(values)], [min_val, max_val]

        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins + 1)]
        hist = [0] * bins

        for v in values:
            idx = min(bins - 1, int((v - min_val) / bin_width))
            hist[idx] += 1

        return hist, edges

    def generate_ascii_histogram(self, data: PlotData, width: int = 50) -> str:
        """Generate ASCII art histogram for terminal display."""
        if not data.y:
            return "No data"

        max_val = max(data.y)
        if max_val == 0:
            return "All zeros"

        lines = [data.title, "=" * len(data.title), ""]

        for i, (x, y) in enumerate(zip(data.x, data.y)):
            bar_len = int((y / max_val) * width)
            bar = "█" * bar_len
            lines.append(f"{x:.3f} | {bar} ({y:.0f})")

        return "\n".join(lines)

    def generate_ascii_scatter_2d(self, data: PlotData,
                                  width: int = 60, height: int = 20) -> str:
        """Generate ASCII art 2D scatter plot."""
        if not data.x or not data.y:
            return "No data"

        lines = [data.title, "=" * len(data.title), ""]

        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        min_x, max_x = min(data.x), max(data.x)
        min_y, max_y = min(data.y), max(data.y)

        range_x = max_x - min_x or 1
        range_y = max_y - min_y or 1

        # Plot points
        for x, y in zip(data.x, data.y):
            col = int((x - min_x) / range_x * (width - 1))
            row = int((y - min_y) / range_y * (height - 1))
            row = height - 1 - row  # Flip y-axis
            grid[row][col] = '●'

        # Add frame
        lines.append(f"  {max_y:.2f} ┌" + "─" * width + "┐")
        for row in grid:
            lines.append("       │" + "".join(row) + "│")
        lines.append(f"  {min_y:.2f} └" + "─" * width + "┘")
        lines.append(f"       {min_x:.2f}" + " " * (width - 10) + f"{max_x:.2f}")
        lines.append(f"       {data.xlabel}")

        return "\n".join(lines)


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    from .virtual_chamber import VirtualChamber

    print("=== CATEGORICAL VISUALIZATION DEMONSTRATION ===\n")

    # Create chamber
    chamber = VirtualChamber()
    chamber.populate(500)

    viz = CategoricalVisualizer(chamber)

    # S_e histogram
    hist_data = viz.s_e_histogram(bins=10)
    print(viz.generate_ascii_histogram(hist_data))

    print("\n")

    # 2D scatter
    scatter_data = viz.phase_space_2d('S_k', 'S_e')
    print(viz.generate_ascii_scatter_2d(scatter_data))

    print("\n=== KEY INSIGHT ===")
    print("These visualizations show REAL data from hardware timing.")
    print("The distributions are not simulated - they emerge from")
    print("actual oscillator variations in your computer.")

    return viz


if __name__ == "__main__":
    demonstrate_visualization()

