"""
Substrate Harvester: Extract S-Entropy Coordinates from Hardware Processes

This module harvests physical processes from computer hardware to determine
the current position in S-entropy space (S_k, S_t, S_e).

The S-entropy coordinates guide navigation toward solutions.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class SEntropyCoordinates:
    """S-Entropy coordinate triple"""
    S_k: float  # Knowledge entropy (what is unknown)
    S_t: float  # Temporal entropy (time-asymmetric evolution)
    S_e: float  # Evolution entropy (trajectory uncertainty)
    
    confidence_k: float = 1.0
    confidence_t: float = 1.0
    confidence_e: float = 1.0
    
    def magnitude(self) -> float:
        """Total S-entropy magnitude"""
        return np.sqrt(self.S_k**2 + self.S_t**2 + self.S_e**2)
    
    def normalized(self) -> 'SEntropyCoordinates':
        """Return normalized coordinates (unit sphere)"""
        mag = self.magnitude()
        if mag < 1e-10:
            return SEntropyCoordinates(0, 0, 0)
        return SEntropyCoordinates(
            self.S_k / mag,
            self.S_t / mag, 
            self.S_e / mag,
            self.confidence_k,
            self.confidence_t,
            self.confidence_e
        )


class SubstrateHarvester:
    """
    Harvests hardware processes to extract S-entropy coordinates.
    
    Sources:
    - CPU clock jitter -> Temporal entropy (S_t)
    - Memory/cache patterns -> Knowledge entropy (S_k)  
    - Thermal/noise floor -> Evolution entropy (S_e)
    """
    
    def __init__(self, 
                 temporal_scale: float = 1.0,
                 knowledge_scale: float = 1.0,
                 evolution_scale: float = 1.0):
        """
        Initialize the harvester with scaling factors.
        
        Args:
            temporal_scale: Scaling for temporal entropy
            knowledge_scale: Scaling for knowledge entropy
            evolution_scale: Scaling for evolution entropy
        """
        self.temporal_scale = temporal_scale
        self.knowledge_scale = knowledge_scale
        self.evolution_scale = evolution_scale
        
        # Calibration baselines
        self._calibrate()
    
    def _calibrate(self):
        """Establish baseline measurements for normalization"""
        # Sample timing jitter baseline
        jitters = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            t1 = time.perf_counter_ns()
            jitters.append(t1 - t0)
        
        self.jitter_baseline = np.mean(jitters)
        self.jitter_std = np.std(jitters) if np.std(jitters) > 0 else 1.0
        
        # System entropy baseline
        self.system_baseline = self._sample_system_entropy()
    
    def _sample_clock_jitter(self) -> float:
        """
        Sample CPU clock jitter as source of temporal entropy.
        
        Returns:
            Normalized jitter value
        """
        samples = []
        for _ in range(10):
            t0 = time.perf_counter_ns()
            # Small computation to induce measurable timing
            _ = sum(range(100))
            t1 = time.perf_counter_ns()
            samples.append(t1 - t0)
        
        jitter = np.std(samples)
        normalized = (jitter - self.jitter_baseline) / self.jitter_std
        return np.clip(normalized, 0, 1)
    
    def _sample_system_entropy(self) -> float:
        """
        Sample system state entropy from available sources.
        
        Returns:
            System entropy estimate [0, 1]
        """
        entropy_sources = []
        
        # CPU usage variability
        if HAS_PSUTIL:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.01)
                entropy_sources.append(cpu_percent / 100.0)
            except:
                pass
        
        # Time-based entropy (nanosecond fractional part)
        ns = time.perf_counter_ns()
        ns_entropy = (ns % 1000) / 1000.0
        entropy_sources.append(ns_entropy)
        
        # Process ID entropy (semi-random component)
        pid_entropy = (os.getpid() % 1000) / 1000.0
        entropy_sources.append(pid_entropy)
        
        if entropy_sources:
            return np.mean(entropy_sources)
        return 0.5  # Default middle entropy
    
    def _estimate_information_gaps(self, input_data: any) -> float:
        """
        Estimate knowledge entropy from input completeness.
        
        Args:
            input_data: The input to analyze
            
        Returns:
            Knowledge entropy estimate [0, 1]
        """
        if input_data is None:
            return 1.0  # Maximum uncertainty
        
        if isinstance(input_data, str):
            # Shorter inputs = more gaps = higher entropy
            length = len(input_data)
            # Sigmoid mapping: short = high entropy, long = lower entropy
            return 1.0 / (1.0 + np.exp((length - 100) / 50))
        
        if isinstance(input_data, dict):
            # Count None/missing values
            total = len(input_data)
            if total == 0:
                return 1.0
            missing = sum(1 for v in input_data.values() if v is None)
            return missing / total
        
        # Default: moderate entropy
        return 0.5
    
    def extract(self, 
                input_data: any = None,
                context: Optional[dict] = None) -> SEntropyCoordinates:
        """
        Extract S-entropy coordinates from hardware and input analysis.
        
        Args:
            input_data: Optional input to analyze for knowledge gaps
            context: Optional context for enhanced extraction
            
        Returns:
            SEntropyCoordinates with (S_k, S_t, S_e)
        """
        # Temporal entropy: from clock jitter
        S_t = self._sample_clock_jitter() * self.temporal_scale
        
        # Knowledge entropy: from input completeness
        S_k = self._estimate_information_gaps(input_data) * self.knowledge_scale
        
        # Evolution entropy: from system state
        S_e = self._sample_system_entropy() * self.evolution_scale
        
        # Confidence based on sample quality
        confidence_t = 0.8  # Clock jitter is reliable
        confidence_k = 0.7 if input_data is not None else 0.3
        confidence_e = 0.6  # System entropy is noisier
        
        return SEntropyCoordinates(
            S_k=S_k,
            S_t=S_t,
            S_e=S_e,
            confidence_k=confidence_k,
            confidence_t=confidence_t,
            confidence_e=confidence_e
        )
    
    def extract_trajectory(self, 
                          previous: SEntropyCoordinates,
                          current: SEntropyCoordinates) -> np.ndarray:
        """
        Compute trajectory vector from previous to current state.
        
        Args:
            previous: Previous S-entropy coordinates
            current: Current S-entropy coordinates
            
        Returns:
            Trajectory vector (direction of S-entropy change)
        """
        return np.array([
            current.S_k - previous.S_k,
            current.S_t - previous.S_t,
            current.S_e - previous.S_e
        ])


# Test if run directly
if __name__ == "__main__":
    harvester = SubstrateHarvester()
    
    # Test extraction
    coords = harvester.extract("Hello, this is a test message")
    print(f"S-Entropy Coordinates:")
    print(f"  S_k (knowledge): {coords.S_k:.4f} (conf: {coords.confidence_k:.2f})")
    print(f"  S_t (temporal):  {coords.S_t:.4f} (conf: {coords.confidence_t:.2f})")
    print(f"  S_e (evolution): {coords.S_e:.4f} (conf: {coords.confidence_e:.2f})")
    print(f"  Magnitude: {coords.magnitude():.4f}")

