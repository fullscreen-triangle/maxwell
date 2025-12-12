"""
Results Manager - Persistent Storage for All Experimental Results
=================================================================

Every experiment produces results that are:
1. Stored in JSON/CSV format for accessibility
2. Timestamped for reproducibility
3. Structured for publication

This ensures complete documentation of all validation steps.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(v) for v in obj)
    return obj


@dataclass
class ExperimentResult:
    """Container for a single experiment result"""
    experiment_id: str
    experiment_name: str
    timestamp: str
    hypothesis: str
    conclusion: str
    validated: bool
    data: Dict[str, Any]
    metrics: Dict[str, float]
    figures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return numpy_to_python(asdict(self))
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentResult':
        """Create from dictionary"""
        return cls(**d)


class ResultsManager:
    """
    Manages persistent storage of all experimental results.
    
    Directory structure:
    results/
    ├── experiments/           # Individual experiment results
    │   ├── exp_001_temperature_independence.json
    │   ├── exp_002_kinetic_independence.json
    │   └── ...
    ├── data/                  # Raw data files
    │   ├── exp_001_data.csv
    │   └── ...
    ├── figures/               # Generated figures
    │   ├── panel_arg1_temporal_triviality.png
    │   └── ...
    ├── summary/               # Summary reports
    │   └── validation_summary.json
    └── publication/           # Publication-ready outputs
        └── maxwell_resolution_figures.pdf
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.data_dir = self.base_dir / "data"
        self.figures_dir = self.base_dir / "figures"
        self.summary_dir = self.base_dir / "summary"
        self.publication_dir = self.base_dir / "publication"
        
        # Create directories
        for d in [self.experiments_dir, self.data_dir, self.figures_dir, 
                  self.summary_dir, self.publication_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Results registry
        self.results: Dict[str, ExperimentResult] = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"ResultsManager initialized: {self.base_dir}")
        print(f"Run ID: {self.run_id}")
    
    def save_experiment(self, result: ExperimentResult) -> str:
        """Save experiment result to disk"""
        self.results[result.experiment_id] = result
        
        # Save JSON
        filename = f"{result.experiment_id}_{self.run_id}.json"
        filepath = self.experiments_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def save_dataframe(self, df: pd.DataFrame, name: str) -> str:
        """Save DataFrame to CSV"""
        filename = f"{name}_{self.run_id}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def save_array(self, arr: np.ndarray, name: str) -> str:
        """Save numpy array"""
        filename = f"{name}_{self.run_id}.npy"
        filepath = self.data_dir / filename
        np.save(filepath, arr)
        print(f"  Saved: {filepath}")
        return str(filepath)
    
    def save_figure(self, fig, name: str, formats: List[str] = ['png', 'pdf']) -> List[str]:
        """Save matplotlib figure in multiple formats"""
        paths = []
        for fmt in formats:
            filename = f"{name}_{self.run_id}.{fmt}"
            filepath = self.figures_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            paths.append(str(filepath))
            print(f"  Saved: {filepath}")
        return paths
    
    def save_publication_figure(self, fig, name: str) -> List[str]:
        """Save publication-ready figure"""
        paths = []
        for fmt in ['png', 'pdf', 'svg']:
            filename = f"{name}.{fmt}"
            filepath = self.publication_dir / filename
            fig.savefig(filepath, dpi=600 if fmt == 'png' else 300, 
                       bbox_inches='tight', facecolor='white')
            paths.append(str(filepath))
            print(f"  Saved: {filepath}")
        return paths
    
    def generate_summary(self) -> Dict:
        """Generate summary of all results"""
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "all_validated": bool(all(r.validated for r in self.results.values())),
            "experiments": {
                exp_id: {
                    "name": r.experiment_name,
                    "validated": bool(r.validated),
                    "conclusion": r.conclusion
                }
                for exp_id, r in self.results.items()
            }
        }
        
        # Convert any remaining numpy types
        summary = numpy_to_python(summary)
        
        # Save summary
        filepath = self.summary_dir / f"validation_summary_{self.run_id}.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved: {filepath}")
        return summary
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment result from disk"""
        # Find most recent file for this experiment
        pattern = f"{experiment_id}_*.json"
        files = sorted(self.experiments_dir.glob(pattern), reverse=True)
        
        if not files:
            return None
        
        with open(files[0], 'r') as f:
            data = json.load(f)
        
        return ExperimentResult.from_dict(data)
    
    def list_experiments(self) -> List[str]:
        """List all saved experiments"""
        files = self.experiments_dir.glob("*.json")
        return [f.stem for f in files]
    
    def get_all_figures(self) -> List[str]:
        """List all generated figures"""
        return [str(f) for f in self.figures_dir.glob("*.*")]


# Global instance
_results_manager: Optional[ResultsManager] = None


def get_results_manager(base_dir: str = "results") -> ResultsManager:
    """Get or create global results manager"""
    global _results_manager
    if _results_manager is None:
        _results_manager = ResultsManager(base_dir)
    return _results_manager

