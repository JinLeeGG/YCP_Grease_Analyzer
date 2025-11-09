"""
Optimized FTIR Peak Detection and Analysis System
==================================================

Production-ready spectroscopy analysis implementing best practices:
- Hybrid peak significance (absolute + relative + statistical thresholds)
- Spectral alignment via cross-correlation
- Critical region monitoring (oxidation, water, additives)
- Rule-based decision logic with tunable thresholds
- JSON + human-readable output

Performance: <1s per sample with optimized NumPy operations
Accuracy: Statistical significance (3σ), prominence-based detection
"""

import numpy as np
import pandas as pd
import json
from scipy import signal
from scipy.stats import pearsonr
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path


@dataclass
class Peak:
    """Peak data structure with all required metrics"""
    wavenumber: float
    height: float
    area: float
    prominence: float
    fwhm: float
    category: str  # 'major', 'minor', 'trace'
    annotation: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AnalysisConfig:
    """Tunable configuration parameters"""
    # Peak detection thresholds
    sigma_multiplier: float = 3.0  # Statistical significance (3σ)
    prominence_pct: float = 0.10  # 10% of global max
    major_height_pct: float = 0.10  # 10% of global max for major peaks
    
    # Peak matching
    match_tolerance: float = 10.0  # ±10 cm⁻¹ default
    match_tolerance_loose: float = 20.0  # ±20 cm⁻¹ for low resolution
    shift_notable: float = 5.0  # >5 cm⁻¹ is notable
    shift_significant: float = 10.0  # >10 cm⁻¹ is significant
    
    # Noise estimation
    quiet_window_cm: float = 50.0  # Window size for noise estimation
    quiet_region: Tuple[float, float] = (2200, 2400)  # Default quiet region
    
    # Critical regions (wavenumber ranges)
    oxidation_region: Tuple[float, float] = (1650, 1800)
    water_region: Tuple[float, float] = (3200, 3600)
    glycol_region: Tuple[float, float] = (1000, 1300)
    ch_region: Tuple[float, float] = (2850, 2950)
    
    # Decision thresholds
    oxidation_critical: float = 0.50  # >50% increase = critical
    oxidation_moderate: float = 0.30  # 30-50% = moderate
    oxidation_low: float = 0.10  # 10-30% = low
    water_threshold: float = 0.12  # Absorbance threshold for water
    
    # Performance
    smooth_window: int = 7  # Savitzky-Golay window
    smooth_poly: int = 3  # Polynomial order
    
    def to_dict(self):
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d['quiet_region'] = list(d['quiet_region'])
        d['oxidation_region'] = list(d['oxidation_region'])
        d['water_region'] = list(d['water_region'])
        d['glycol_region'] = list(d['glycol_region'])
        d['ch_region'] = list(d['ch_region'])
        return d
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        # Convert lists back to tuples
        if 'quiet_region' in config_dict:
            config_dict['quiet_region'] = tuple(config_dict['quiet_region'])
        if 'oxidation_region' in config_dict:
            config_dict['oxidation_region'] = tuple(config_dict['oxidation_region'])
        if 'water_region' in config_dict:
            config_dict['water_region'] = tuple(config_dict['water_region'])
        if 'glycol_region' in config_dict:
            config_dict['glycol_region'] = tuple(config_dict['glycol_region'])
        if 'ch_region' in config_dict:
            config_dict['ch_region'] = tuple(config_dict['ch_region'])
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class FTIRAnalyzer:
    """
    Main FTIR Analysis Engine
    
    Implements:
    1. QC layer (saturation, missing data, noise checks)
    2. Baseline correction and smoothing
    3. Peak detection with statistical significance
    4. Spectral alignment via cross-correlation
    5. Peak matching with shift detection
    6. Critical region analysis
    7. Decision logic for maintenance recommendations
    """
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.qc_flags = []
    
    # ============================================================================
    # 1. QC LAYER
    # ============================================================================
    
    def run_qc_checks(self, wavenumbers: np.ndarray, absorbance: np.ndarray) -> List[str]:
        """
        Quality Control checks before analysis
        
        Checks for:
        - Saturation (flat-topped peaks near detector limit)
        - Missing data regions (NaN or gaps)
        - High noise levels
        - Negative absorbance (inverted baselines)
        
        Returns: List of QC flag strings
        """
        flags = []
        
        # Check for saturation (absorbance near max, typically ~2.0-3.0)
        if np.max(absorbance) > 2.5:
            flags.append("saturated")
        
        # Check for missing data
        if np.any(np.isnan(absorbance)):
            flags.append("missing_data")
        
        # Check for negative absorbance
        if np.min(absorbance) < -0.1:
            flags.append("negative_peaks")
        
        # Check for high noise
        sigma, quiet_region = self._estimate_noise(wavenumbers, absorbance)
        if quiet_region is not None:
            signal_max = np.max(absorbance)
            snr = signal_max / sigma if sigma > 0 else float('inf')
            if snr < 10:  # Poor S/N ratio
                flags.append("high_noise")
        
        return flags
    
    # ============================================================================
    # 2. NOISE ESTIMATION
    # ============================================================================
    
    def _estimate_noise(self, wavenumbers: np.ndarray, absorbance: np.ndarray) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Estimate noise from quiet region of spectrum
        
        Returns: (sigma, quiet_region_indices)
        """
        # Try configured quiet region first
        mask = (wavenumbers >= self.config.quiet_region[0]) & (wavenumbers <= self.config.quiet_region[1])
        if np.sum(mask) > 10:
            quiet_data = absorbance[mask]
            sigma = np.std(quiet_data)
            indices = np.where(mask)[0]
            return sigma, (indices[0], indices[-1])
        
        # Fallback: use lowest variance window
        window_size = int(self.config.quiet_window_cm / np.mean(np.diff(wavenumbers)))
        window_size = max(10, min(window_size, len(absorbance) // 4))
        
        min_variance = float('inf')
        best_start = 0
        
        for i in range(len(absorbance) - window_size):
            variance = np.var(absorbance[i:i + window_size])
            if variance < min_variance:
                min_variance = variance
                best_start = i
        
        sigma = np.sqrt(min_variance)
        return sigma, (best_start, best_start + window_size)
    
    def estimate_noise_conservative(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                                   sample_wn: np.ndarray, sample_abs: np.ndarray) -> float:
        """
        Conservative noise estimate using both spectra
        """
        sigma_baseline, _ = self._estimate_noise(baseline_wn, baseline_abs)
        sigma_sample, _ = self._estimate_noise(sample_wn, sample_abs)
        return max(sigma_baseline, sigma_sample)
    
    # ============================================================================
    # 3. BASELINE CORRECTION AND SMOOTHING
    # ============================================================================
    
    def correct_baseline(self, absorbance: np.ndarray, lam: float = 1e5, p: float = 0.01) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction
        """
        L = len(absorbance)
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        
        for _ in range(10):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            z = np.linalg.solve(Z, w * absorbance)
            w = p * (absorbance > z) + (1 - p) * (absorbance < z)
        
        return absorbance - z
    
    def smooth_spectrum(self, absorbance: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay smoothing
        """
        if len(absorbance) < self.config.smooth_window:
            return absorbance
        
        return signal.savgol_filter(absorbance, self.config.smooth_window, self.config.smooth_poly)
    
    # ============================================================================
    # 4. SPECTRAL ALIGNMENT
    # ============================================================================
    
    def align_spectra(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                     sample_wn: np.ndarray, sample_abs: np.ndarray,
                     region: Tuple[float, float] = (1000, 1800)) -> Tuple[np.ndarray, float]:
        """
        Align sample to baseline using cross-correlation
        
        Returns: (aligned_sample_wn, shift_cm)
        """
        # Extract region for correlation
        mask_base = (baseline_wn >= region[0]) & (baseline_wn <= region[1])
        mask_samp = (sample_wn >= region[0]) & (sample_wn <= region[1])
        
        if np.sum(mask_base) < 10 or np.sum(mask_samp) < 10:
            return sample_wn, 0.0
        
        region_base = baseline_abs[mask_base]
        region_samp = sample_abs[mask_samp]
        
        # Cross-correlation
        correlation = signal.correlate(region_base, region_samp, mode='same')
        shift_idx = np.argmax(correlation) - len(region_base) // 2
        
        # Convert to wavenumber shift
        avg_spacing = np.mean(np.diff(sample_wn))
        shift_cm = shift_idx * avg_spacing
        
        # Apply shift
        aligned_wn = sample_wn - shift_cm
        
        return aligned_wn, shift_cm
    
    # ============================================================================
    # 5. PEAK DETECTION
    # ============================================================================
    
    def detect_peaks_hybrid(self, wavenumbers: np.ndarray, absorbance: np.ndarray,
                           sigma: float) -> List[Peak]:
        """
        Hybrid peak detection with statistical significance
        
        Classification:
        - MAJOR: height ≥ 3σ OR (height ≥ 10% max AND prominence ≥ 10% max)
        - MINOR: 1σ ≤ height < 3σ OR prominence 5-10%
        - TRACE: height < 1σ
        """
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            absorbance,
            prominence=sigma,
            width=2
        )
        
        if len(peak_indices) == 0:
            return []
        
        peaks = []
        global_max = np.max(absorbance)
        
        for i, idx in enumerate(peak_indices):
            height = absorbance[idx]
            prominence = properties['prominences'][i]
            
            # Calculate FWHM
            width_samples = properties['widths'][i] if 'widths' in properties else 1.0
            fwhm = width_samples * np.mean(np.diff(wavenumbers))
            
            # Estimate area (triangular approximation)
            area = 0.5 * prominence * fwhm
            
            # Classify peak
            if height >= self.config.sigma_multiplier * sigma or \
               (height >= self.config.major_height_pct * global_max and 
                prominence >= self.config.prominence_pct * global_max):
                category = 'major'
            elif height >= sigma or \
                 (prominence >= 0.05 * global_max and prominence < 0.10 * global_max):
                category = 'minor'
            else:
                category = 'trace'
            
            annotation = self._annotate_peak(wavenumbers[idx])
            
            peaks.append(Peak(
                wavenumber=float(wavenumbers[idx]),
                height=float(height),
                area=float(area),
                prominence=float(prominence),
                fwhm=float(fwhm),
                category=category,
                annotation=annotation
            ))
        
        return peaks
    
    def _annotate_peak(self, wavenumber: float) -> str:
        """Annotate peak with functional group"""
        if 1650 <= wavenumber <= 1800:
            return "C=O (carbonyl/oxidation)"
        elif 3200 <= wavenumber <= 3600:
            return "O-H (water/alcohol)"
        elif 2850 <= wavenumber <= 2950:
            return "C-H (aliphatic)"
        elif 1000 <= wavenumber <= 1300:
            return "C-O (ester/glycol)"
        else:
            return ""
    
    # ============================================================================
    # 6. PEAK MATCHING
    # ============================================================================
    
    def match_peaks(self, baseline_peaks: List[Peak], sample_peaks: List[Peak],
                   tolerance: float = None) -> Dict:
        """
        Match peaks between baseline and sample
        
        Returns: {
            'n_matched': int,
            'n_new': int,
            'n_missing': int,
            'matched': List[Dict],
            'new': List[Dict],
            'missing': List[Dict],
            'shifts': List[Dict]
        }
        """
        tolerance = tolerance or self.config.match_tolerance
        
        matched = []
        new_peaks = []
        missing_peaks = []
        shifts = []
        
        baseline_wns = np.array([p.wavenumber for p in baseline_peaks])
        
        # Find matches for each sample peak
        for sample_peak in sample_peaks:
            if len(baseline_wns) == 0:
                new_peaks.append(sample_peak.to_dict())
                continue
            
            diffs = np.abs(baseline_wns - sample_peak.wavenumber)
            min_idx = np.argmin(diffs)
            min_diff = diffs[min_idx]
            
            if min_diff <= tolerance:
                baseline_peak = baseline_peaks[min_idx]
                
                height_change = sample_peak.height - baseline_peak.height
                height_change_pct = (height_change / baseline_peak.height * 100) if baseline_peak.height > 0 else 0
                
                shift = sample_peak.wavenumber - baseline_peak.wavenumber
                
                match_info = {
                    'baseline_wn': baseline_peak.wavenumber,
                    'sample_wn': sample_peak.wavenumber,
                    'shift': shift,
                    'baseline_height': baseline_peak.height,
                    'sample_height': sample_peak.height,
                    'height_change_pct': height_change_pct,
                    'annotation': sample_peak.annotation
                }
                
                matched.append(match_info)
                
                if abs(shift) > self.config.shift_notable:
                    shifts.append(match_info)
                
                # Mark this baseline peak as matched
                baseline_wns[min_idx] = np.nan
            else:
                new_peaks.append(sample_peak.to_dict())
        
        # Remaining baseline peaks are missing
        for i, baseline_peak in enumerate(baseline_peaks):
            if not np.isnan(baseline_wns[i]):
                missing_peaks.append(baseline_peak.to_dict())
        
        return {
            'n_matched': len(matched),
            'n_new': len(new_peaks),
            'n_missing': len(missing_peaks),
            'matched': matched,
            'new': new_peaks,
            'missing': missing_peaks,
            'shifts': shifts
        }
    
    # ============================================================================
    # 7. CRITICAL REGION ANALYSIS
    # ============================================================================
    
    def analyze_region(self, wavenumbers: np.ndarray, absorbance: np.ndarray,
                      region: Tuple[float, float]) -> Dict:
        """
        Analyze a specific wavenumber region
        
        Returns: {
            'max': float,
            'mean': float,
            'area': float,
            'peak_count': int
        }
        """
        mask = (wavenumbers >= region[0]) & (wavenumbers <= region[1])
        region_abs = absorbance[mask]
        
        if len(region_abs) == 0:
            return {'max': 0.0, 'mean': 0.0, 'area': 0.0, 'peak_count': 0}
        
        return {
            'max': float(np.max(region_abs)),
            'mean': float(np.mean(region_abs)),
            'area': float(np.trapz(region_abs)),
            'peak_count': len(signal.find_peaks(region_abs, prominence=0.01)[0])
        }
    
    def analyze_critical_regions(self, wavenumbers: np.ndarray, absorbance: np.ndarray) -> Dict:
        """Analyze all critical regions"""
        return {
            'oxidation': self.analyze_region(wavenumbers, absorbance, self.config.oxidation_region),
            'water': self.analyze_region(wavenumbers, absorbance, self.config.water_region),
            'glycol': self.analyze_region(wavenumbers, absorbance, self.config.glycol_region),
            'ch': self.analyze_region(wavenumbers, absorbance, self.config.ch_region)
        }
    
    # ============================================================================
    # 8. DECISION LOGIC
    # ============================================================================
    
    def make_recommendation(self, baseline_regions: Dict, sample_regions: Dict,
                          peak_matches: Dict, qc_flags: List[str]) -> Dict:
        """
        Generate maintenance recommendation
        
        Returns: {
            'action': str,
            'reasoning': str,
            'confidence': float,
            'oxidation_increase_pct': float,
            'water_present': bool,
            'water_severity': str,
            'retest_interval': str
        }
        """
        # Calculate oxidation increase
        baseline_ox = baseline_regions['oxidation']['max']
        sample_ox = sample_regions['oxidation']['max']
        
        if baseline_ox > 0:
            ox_increase = (sample_ox - baseline_ox) / baseline_ox
        else:
            ox_increase = 0.0 if sample_ox < 0.01 else 1.0
        
        # Water contamination
        water_max = sample_regions['water']['max']
        water_present = water_max > self.config.water_threshold
        
        if water_max > 0.25:
            water_severity = 'severe'
        elif water_max > 0.15:
            water_severity = 'moderate'
        elif water_max > self.config.water_threshold:
            water_severity = 'low'
        else:
            water_severity = 'none'
        
        # Decision logic
        confidence = 0.90  # Base confidence
        
        # Reduce confidence for QC issues
        if 'high_noise' in qc_flags:
            confidence -= 0.10
        if 'saturated' in qc_flags:
            confidence -= 0.15
        
        # Decision rules
        if ox_increase > self.config.oxidation_critical and water_present:
            action = 'replace_grease'
            reasoning = f"Critical oxidation detected ({ox_increase*100:.0f}%) with water contamination"
            retest_interval = 'immediate'
            confidence = max(confidence, 0.95)
        elif ox_increase > self.config.oxidation_moderate:
            action = 'schedule_maintenance'
            reasoning = f"Moderate oxidation detected ({ox_increase*100:.0f}%)"
            retest_interval = '2_weeks'
            confidence = max(confidence, 0.85)
        elif ox_increase > self.config.oxidation_low:
            action = 'monitor_closely'
            reasoning = f"Low oxidation detected ({ox_increase*100:.0f}%)"
            retest_interval = '4_weeks'
            confidence = max(confidence, 0.75)
        else:
            action = 'normal'
            reasoning = "Minimal changes detected, grease condition good"
            retest_interval = '8_weeks'
        
        return {
            'action': action,
            'reasoning': reasoning,
            'confidence': float(confidence),
            'oxidation_increase_pct': float(ox_increase * 100),
            'water_present': bool(water_present),
            'water_severity': water_severity,
            'retest_interval': retest_interval
        }
    
    # ============================================================================
    # 9. COMPLETE ANALYSIS PIPELINE
    # ============================================================================
    
    def analyze_complete(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                        sample_wn: np.ndarray, sample_abs: np.ndarray,
                        baseline_name: str = "", sample_name: str = "") -> Dict:
        """
        Complete analysis pipeline
        
        Returns comprehensive JSON-structured results
        """
        start_time = time.time()
        
        # 1. QC Checks
        qc_baseline = self.run_qc_checks(baseline_wn, baseline_abs)
        qc_sample = self.run_qc_checks(sample_wn, sample_abs)
        qc_flags = list(set(qc_baseline + qc_sample))
        
        # 2. Noise estimation
        sigma = self.estimate_noise_conservative(baseline_wn, baseline_abs, sample_wn, sample_abs)
        
        # 3. Preprocessing
        baseline_abs_proc = self.smooth_spectrum(baseline_abs)
        sample_abs_proc = self.smooth_spectrum(sample_abs)
        
        # 4. Spectral alignment
        sample_wn_aligned, shift = self.align_spectra(
            baseline_wn, baseline_abs_proc,
            sample_wn, sample_abs_proc
        )
        
        # 5. Peak detection
        baseline_peaks = self.detect_peaks_hybrid(baseline_wn, baseline_abs_proc, sigma)
        sample_peaks = self.detect_peaks_hybrid(sample_wn_aligned, sample_abs_proc, sigma)
        
        # 6. Peak matching
        peak_matches = self.match_peaks(baseline_peaks, sample_peaks)
        
        # 7. Critical region analysis
        baseline_regions = self.analyze_critical_regions(baseline_wn, baseline_abs_proc)
        sample_regions = self.analyze_critical_regions(sample_wn_aligned, sample_abs_proc)
        
        # 8. Recommendation
        recommendation = self.make_recommendation(baseline_regions, sample_regions, peak_matches, qc_flags)
        
        # 9. Generate summary
        human_summary = self._generate_human_summary(
            sample_name, baseline_peaks, sample_peaks, peak_matches,
            baseline_regions, sample_regions, recommendation, qc_flags
        )
        
        elapsed = time.time() - start_time
        
        return {
            'metadata': {
                'baseline_name': baseline_name,
                'sample_name': sample_name,
                'analysis_time_seconds': elapsed,
                'spectral_shift_cm': shift
            },
            'qc_flags': qc_flags,
            'noise_sigma': sigma,
            'n_peaks_detected': {
                'baseline': len(baseline_peaks),
                'sample': len(sample_peaks)
            },
            'major_peaks': {
                'baseline': [p.to_dict() for p in baseline_peaks if p.category == 'major'],
                'sample': [p.to_dict() for p in sample_peaks if p.category == 'major']
            },
            'peak_matches': peak_matches,
            'critical_regions': {
                'baseline': baseline_regions,
                'sample': sample_regions
            },
            'recommendation': recommendation,
            'human_summary': human_summary
        }
    
    def _generate_human_summary(self, sample_name: str, baseline_peaks: List[Peak],
                               sample_peaks: List[Peak], peak_matches: Dict,
                               baseline_regions: Dict, sample_regions: Dict,
                               recommendation: Dict, qc_flags: List[str]) -> str:
        """Generate human-readable summary"""
        
        # Status emoji
        action = recommendation['action']
        if action == 'replace_grease':
            status = "🔴 CRITICAL"
        elif action == 'schedule_maintenance':
            status = "⚠️ CAUTION"
        elif action == 'monitor_closely':
            status = "🟡 MONITOR"
        else:
            status = "✅ NORMAL"
        
        summary = f"**Sample {sample_name}: {status}**\n\n"
        
        # Key findings
        ox_pct = recommendation['oxidation_increase_pct']
        n_new = peak_matches['n_new']
        n_shifted = len(peak_matches['shifts'])
        
        summary += f"Key findings: "
        findings = []
        
        if ox_pct > 10:
            if ox_pct > 50:
                findings.append(f"severe oxidation (+{ox_pct:.0f}%)")
            elif ox_pct > 30:
                findings.append(f"moderate oxidation (+{ox_pct:.0f}%)")
            else:
                findings.append(f"low oxidation (+{ox_pct:.0f}%)")
        
        if n_new > 0:
            findings.append(f"{n_new} new peaks detected")
        
        if n_shifted > 0:
            findings.append(f"{n_shifted} peaks shifted")
        
        if recommendation['water_severity'] != 'none':
            findings.append(f"{recommendation['water_severity']} water contamination")
        
        if not findings:
            findings.append("minimal changes")
        
        summary += ", ".join(findings) + ".\n"
        
        # Critical region details
        ox_max = sample_regions['oxidation']['max']
        if ox_max > 0.05:
            summary += f"Carbonyl region (1650-1800 cm⁻¹): {ox_max:.2f} A (oxidation indicator).\n"
        
        # Action
        summary += f"\n**Action**: {recommendation['reasoning']}. "
        
        if action == 'replace_grease':
            summary += "Replace grease immediately."
        elif action == 'schedule_maintenance':
            summary += "Schedule maintenance within 2-4 weeks."
        elif action == 'monitor_closely':
            summary += "Monitor and retest in 4 weeks."
        else:
            summary += "Continue normal operation, retest in 8 weeks."
        
        # QC warnings
        if qc_flags:
            summary += f"\n\n⚠️ QC Flags: {', '.join(qc_flags)}"
        
        return summary


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_csv_spectrum_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from DataFrame
    
    Expected columns: 'X' (wavenumber) and 'Y' (absorbance)
    
    Returns: (wavenumbers, absorbance)
    """
    if 'X' not in df.columns or 'Y' not in df.columns:
        raise ValueError("DataFrame must have 'X' and 'Y' columns")
    
    wavenumbers = df['X'].values
    absorbance = df['Y'].values
    
    # Sort by wavenumber (ascending)
    sort_idx = np.argsort(wavenumbers)
    return wavenumbers[sort_idx], absorbance[sort_idx]


def choose_baseline_auto(df_a: pd.DataFrame, df_b: pd.DataFrame, 
                         name_a: str, name_b: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Automatically choose baseline from two DataFrames
    
    Strategy: Choose DataFrame with lower carbonyl area (less oxidized)
    
    Returns: (baseline_df, sample_df, baseline_name, sample_name)
    """
    wn_a, abs_a = load_csv_spectrum_from_df(df_a)
    wn_b, abs_b = load_csv_spectrum_from_df(df_b)
    
    def carbonyl_area(wn, ab):
        mask = (wn >= 1650) & (wn <= 1800)
        if np.sum(mask) > 0:
            return np.trapz(ab[mask])
        return 0.0
    
    area_a = carbonyl_area(wn_a, abs_a)
    area_b = carbonyl_area(wn_b, abs_b)
    
    if area_a < area_b:
        return df_a, df_b, name_a, name_b
    else:
        return df_b, df_a, name_b, name_a
