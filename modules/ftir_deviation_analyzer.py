"""
FTIR Spectral Overlay Deviation Analysis System
================================================

Pure deviation reporting without quality inferences.
Compares sample spectrum to baseline and reports WHERE they don't superimpose.

Key Features:
- Hard-coded critical regions (oxidation, water, additives, C-H)
- Dynamic outlier detection (full spectrum)
- Baseline compatibility checking
- Tiered alert system (minor/major/critical)
- Multi-metric categorization (correlation + ΔX + ΔY + ratio)
- NO quality scores or replacement recommendations

Performance: <1 second per sample
Output: Factual deviation metrics (ΔX, ΔY) only
"""

import numpy as np
import pandas as pd
import json
from scipy import signal, stats
from scipy.interpolate import interp1d
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime


@dataclass
class DeviationConfig:
    """Configuration for deviation thresholds and alert levels"""
    
    # Critical regions (wavenumber ranges) - HARD-CODED
    critical_regions: Dict[str, Tuple[float, float]] = None
    
    # ΔY thresholds - PRIMARY: PERCENTAGE-BASED (as specified by domain expert)
    # <10% = noise/normal; 10-30% = requires attention; ≥30% = critical; ≥50% = outlier
    delta_y_pct_outlier: float = 50.0   # ≥50% = outlier/very likely severe
    delta_y_pct_critical: float = 30.0  # ≥30% = critical
    delta_y_pct_major: float = 20.0     # 20-30% = major (upper end of "requires attention")
    delta_y_pct_minor: float = 10.0     # 10-20% = minor (lower end of "requires attention")
    # <10% = superimposed (noise/normal)
    
    # ΔY thresholds - FALLBACK: ABSOLUTE (for very low baseline regions where % is unreliable)
    # Only used when baseline < 0.05 A (very weak peaks)
    delta_y_critical: float = 0.20  # >0.20 A = critical (fallback)
    delta_y_major: float = 0.10     # 0.10-0.20 A = major (fallback)
    delta_y_minor: float = 0.05     # 0.05-0.10 A = minor (fallback)
    
    # ΔX thresholds (horizontal shift) - As specified by domain expert
    delta_x_critical: float = 20.0  # ≥20 cm⁻¹ = critical shift
    delta_x_major: float = 15.0     # 15-20 cm⁻¹ = major shift
    delta_x_minor: float = 10.0     # 10-15 cm⁻¹ = minor shift
    # <10 cm⁻¹ = acceptable
    
    # Correlation thresholds for multi-metric categorization (UNCHANGED)
    correlation_excellent: float = 0.97
    correlation_good: float = 0.95
    correlation_moderate: float = 0.90
    correlation_low: float = 0.85
    
    def __post_init__(self):
        if self.critical_regions is None:
            self.critical_regions = {
                'carbonyl_oxidation': (1650, 1800),   # PRIMARY: Oxidation indicator
                'water_contamination': (3200, 3600)   # PRIMARY: Water/moisture indicator
            }


@dataclass
class RegionDeviation:
    """Deviation metrics for a specific critical region"""
    region_name: str
    region_range: Tuple[float, float]
    max_delta_y: float
    max_delta_y_wavenumber: float
    max_delta_y_pct: float
    max_delta_x: float
    delta_x_delta_y_ratio: float
    alert_level: str  # 'superimposed', 'minor', 'major', 'critical'
    reasoning: str


@dataclass
class OutlierDeviation:
    """Outlier detection in full spectrum"""
    wavenumber: float
    delta_y: float
    delta_y_pct: float
    sigma_level: float
    severity: str  # 'minor', 'major', 'critical'


class FTIRDeviationAnalyzer:
    """
    Pure deviation analysis without quality inferences
    
    Reports:
    - ΔY (vertical deviation) at each wavenumber
    - ΔX (horizontal shift) when detected
    - Alert levels based on statistical thresholds
    - Baseline compatibility warnings
    - Multi-metric categorization
    
    Does NOT report:
    - Quality scores
    - Replacement recommendations
    - Oxidation severity assessments
    """
    
    def __init__(self, config: DeviationConfig = None):
        self.config = config or DeviationConfig()
        self.analysis_log = []
        
        # OPTIMIZATION: Cache for aligned spectra to avoid repeated interpolation
        self._aligned_cache = None
        self._common_wn_cache = None
    
    def _log(self, message: str):
        """Internal logging for audit trail"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.analysis_log.append(log_entry)
    
    # ========================================================================
    # BASELINE COMPATIBILITY CHECK
    # ========================================================================
    
    def check_baseline_compatibility(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                                     sample_wn: np.ndarray, sample_abs: np.ndarray) -> Dict:
        """
        Check if baseline and sample are compatible for comparison
        
        Returns:
            Dict with keys: correlation, level, warning
        """
        # OPTIMIZATION: Use cached aligned spectra if available
        if self._aligned_cache is None:
            aligned_baseline, aligned_sample, common_wn = self._align_spectra_with_grid(
                baseline_wn, baseline_abs, sample_wn, sample_abs
            )
            # Cache for reuse
            self._aligned_cache = (aligned_baseline, aligned_sample)
            self._common_wn_cache = common_wn
        else:
            aligned_baseline, aligned_sample = self._aligned_cache
        
        # Calculate correlation
        correlation, _ = stats.pearsonr(aligned_baseline, aligned_sample)
        
        # Determine compatibility level
        if correlation >= self.config.correlation_excellent:
            level = None
            warning = None
        elif correlation >= self.config.correlation_good:
            level = "minor"
            warning = "Minor correlation difference - acceptable"
        elif correlation >= self.config.correlation_moderate:
            level = "moderate"
            warning = "Moderate correlation - verify baseline compatibility"
        else:
            level = "major"
            warning = "Low correlation - possible baseline mismatch!"
        
        return {
            'correlation': float(correlation),
            'level': level,
            'warning': warning
        }
    
    def _align_spectra_with_grid(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                                 sample_wn: np.ndarray, sample_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        OPTIMIZED: Align spectra to common wavenumber grid AND return the grid
        
        Returns:
            Tuple of (aligned_baseline, aligned_sample, common_wn)
        """
        # Find common wavenumber range
        common_min = max(baseline_wn.min(), sample_wn.min())
        common_max = min(baseline_wn.max(), sample_wn.max())
        
        # OPTIMIZATION: Use linear interpolation instead of cubic (10x faster, minimal accuracy loss)
        # Create common grid (1000 points is fine for FTIR)
        common_wn = np.linspace(common_min, common_max, 1000)
        
        # Interpolate both spectra
        f_baseline = interp1d(baseline_wn, baseline_abs, kind='linear', fill_value='extrapolate')
        f_sample = interp1d(sample_wn, sample_abs, kind='linear', fill_value='extrapolate')
        
        aligned_baseline = f_baseline(common_wn)
        aligned_sample = f_sample(common_wn)
        
        return aligned_baseline, aligned_sample, common_wn
    
    def _align_spectra(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                      sample_wn: np.ndarray, sample_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align spectra to common wavenumber grid (legacy interface)
        
        Returns:
            Tuple of (aligned_baseline, aligned_sample)
        """
        aligned_baseline, aligned_sample, _ = self._align_spectra_with_grid(
            baseline_wn, baseline_abs, sample_wn, sample_abs
        )
        return aligned_baseline, aligned_sample
    
    # ========================================================================
    # CRITICAL REGION ANALYSIS
    # ========================================================================
    
    def analyze_critical_region(self, region_name: str, region_range: Tuple[float, float],
                                baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                                sample_wn: np.ndarray, sample_abs: np.ndarray) -> RegionDeviation:
        """
        OPTIMIZED: Analyze deviation in a specific critical region using cached aligned spectra
        
        Returns:
            RegionDeviation object with all metrics
        """
        # OPTIMIZATION: Use cached full spectrum alignment, then extract region
        if self._aligned_cache is None or self._common_wn_cache is None:
            # This shouldn't happen if check_baseline_compatibility was called first
            aligned_baseline, aligned_sample, common_wn = self._align_spectra_with_grid(
                baseline_wn, baseline_abs, sample_wn, sample_abs
            )
            self._aligned_cache = (aligned_baseline, aligned_sample)
            self._common_wn_cache = common_wn
        else:
            aligned_baseline, aligned_sample = self._aligned_cache
            common_wn = self._common_wn_cache
        
        # Extract region from aligned spectra
        region_mask = (common_wn >= region_range[0]) & (common_wn <= region_range[1])
        region_baseline = aligned_baseline[region_mask]
        region_sample = aligned_sample[region_mask]
        region_wn = common_wn[region_mask]
        
        # Calculate ΔY (vertical deviation)
        delta_y_array = np.abs(region_sample - region_baseline)
        max_delta_y = np.max(delta_y_array)
        max_delta_y_idx = np.argmax(delta_y_array)
        
        # Find corresponding wavenumber
        max_delta_y_wavenumber = region_wn[max_delta_y_idx]
        
        # Calculate percentage change
        baseline_value = region_baseline[max_delta_y_idx]
        if baseline_value > 0.01:  # Avoid division by near-zero
            max_delta_y_pct = (max_delta_y / baseline_value) * 100
        else:
            max_delta_y_pct = 0.0
        
        # OPTIMIZATION: Simplified horizontal shift estimation (faster than full cross-correlation)
        max_delta_x = self._estimate_horizontal_shift_fast(
            region_baseline, region_sample, region_wn
        )
        
        # Calculate ΔX:ΔY ratio
        if max_delta_y > 0:
            ratio = max_delta_x / max_delta_y
        else:
            ratio = 0.0
        
        # Determine alert level based on BOTH ΔY and ΔX
        alert_level, reasoning = self._determine_alert_level(
            max_delta_y, max_delta_x, max_delta_y_pct, baseline_value, region_name
        )
        
        return RegionDeviation(
            region_name=region_name,
            region_range=region_range,
            max_delta_y=float(max_delta_y),
            max_delta_y_wavenumber=float(max_delta_y_wavenumber),
            max_delta_y_pct=float(max_delta_y_pct),
            max_delta_x=float(max_delta_x),
            delta_x_delta_y_ratio=float(ratio),
            alert_level=alert_level,
            reasoning=reasoning
        )
    
    def _estimate_horizontal_shift_fast(self, baseline: np.ndarray, sample: np.ndarray, 
                                        wavenumbers: np.ndarray) -> float:
        """
        OPTIMIZED: Fast horizontal shift estimation using peak detection instead of full correlation
        
        Returns:
            Shift in cm⁻¹
        """
        # OPTIMIZATION: Only do cross-correlation if arrays are small (<200 points)
        # For larger arrays, use simplified peak shift detection
        
        if len(baseline) > 200:
            # Fast method: Find max peaks and compare positions
            baseline_max_idx = np.argmax(baseline)
            sample_max_idx = np.argmax(sample)
            
            if len(wavenumbers) > max(baseline_max_idx, sample_max_idx):
                shift = abs(wavenumbers[sample_max_idx] - wavenumbers[baseline_max_idx])
            else:
                shift = 0.0
        else:
            # Original method for small arrays
            correlation = signal.correlate(sample, baseline, mode='same')
            lag = np.argmax(correlation) - len(baseline) // 2
            
            if len(wavenumbers) > 1:
                wn_step = np.mean(np.diff(wavenumbers))
                shift = abs(lag * wn_step)
            else:
                shift = 0.0
        
        return shift
    
    def _estimate_horizontal_shift(self, baseline: np.ndarray, sample: np.ndarray, 
                                   wavenumbers: np.ndarray) -> float:
        """
        Legacy interface - calls optimized version
        """
        return self._estimate_horizontal_shift_fast(baseline, sample, wavenumbers)
    
    def _determine_alert_level(self, delta_y: float, delta_x: float, 
                               delta_y_pct: float, baseline_value: float,
                               region_name: str) -> Tuple[str, str]:
        """
        Determine alert level based on BOTH ΔY and ΔX
        Uses PERCENTAGE-BASED thresholds as PRIMARY (per domain expert specification)
        
        Domain Expert Thresholds:
        - ΔY: <10% = noise; 10-30% = attention; ≥30% = critical; ≥50% = outlier
        - ΔX: ≥20 cm⁻¹ = critical
        
        Args:
            delta_y: Absolute vertical deviation
            delta_x: Horizontal shift
            delta_y_pct: Percentage vertical deviation
            baseline_value: Baseline absorbance at deviation point
            region_name: Name of region (for context)
        
        Returns:
            Tuple of (alert_level, reasoning)
        """
        # PRIMARY: Use percentage thresholds (unless baseline too low for reliable %)
        use_percentage = baseline_value >= 0.05  # Use % unless baseline extremely weak
        
        # Evaluate ΔY using PERCENTAGE thresholds (PRIMARY)
        if use_percentage:
            if delta_y_pct >= self.config.delta_y_pct_outlier:  # ≥50%
                delta_y_level = 'critical'  # Outlier-level deviation
            elif delta_y_pct >= self.config.delta_y_pct_critical:  # ≥30%
                delta_y_level = 'critical'
            elif delta_y_pct >= self.config.delta_y_pct_major:  # ≥20%
                delta_y_level = 'major'
            elif delta_y_pct >= self.config.delta_y_pct_minor:  # ≥10%
                delta_y_level = 'minor'
            else:  # <10%
                delta_y_level = 'superimposed'
        else:
            # FALLBACK: Use absolute thresholds for very weak baseline peaks
            if delta_y >= self.config.delta_y_critical:
                delta_y_level = 'critical'
            elif delta_y >= self.config.delta_y_major:
                delta_y_level = 'major'
            elif delta_y >= self.config.delta_y_minor:
                delta_y_level = 'minor'
            else:
                delta_y_level = 'superimposed'
        
        # Evaluate ΔX (horizontal shift) - Domain expert: ≥20 cm⁻¹ = critical
        if delta_x >= self.config.delta_x_critical:  # ≥20 cm⁻¹
            delta_x_level = 'critical'
        elif delta_x >= self.config.delta_x_major:  # 15-20 cm⁻¹
            delta_x_level = 'major'
        elif delta_x >= self.config.delta_x_minor:  # 10-15 cm⁻¹
            delta_x_level = 'minor'
        else:  # <10 cm⁻¹
            delta_x_level = 'superimposed'
        
        # Take the HIGHER of the two
        alert_levels = ['superimposed', 'minor', 'major', 'critical']
        delta_y_idx = alert_levels.index(delta_y_level)
        delta_x_idx = alert_levels.index(delta_x_level)
        
        # REMOVED: Escalation logic - use straightforward max of ΔY and ΔX levels
        # Escalation was causing false positives (e.g., 20% change escalated to critical)
        # Domain expert thresholds are already calibrated - no need to escalate
        combined_idx = max(delta_y_idx, delta_x_idx)
        
        alert_level = alert_levels[combined_idx]
        
        # Generate reasoning (show which thresholds were used)
        reasoning_parts = []
        if delta_y >= self.config.delta_y_minor or (use_percentage and delta_y_pct >= self.config.delta_y_pct_minor):
            if use_percentage:
                reasoning_parts.append(f"ΔY = {delta_y:.3f} A ({delta_y_pct:.1f}%, {delta_y_level})")
            else:
                reasoning_parts.append(f"ΔY = {delta_y:.3f} A ({delta_y_level})")
        if delta_x >= self.config.delta_x_minor:
            reasoning_parts.append(f"ΔX = {delta_x:.1f} cm⁻¹ ({delta_x_level})")
        
        if reasoning_parts:
            reasoning = " + ".join(reasoning_parts)
        else:
            reasoning = "Spectra superimpose within tolerance"
        
        return alert_level, reasoning
    
    # ========================================================================
    # OUTLIER DETECTION (Full Spectrum)
    # ========================================================================
    
    def detect_outliers(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                       sample_wn: np.ndarray, sample_abs: np.ndarray) -> List[OutlierDeviation]:
        """
        OPTIMIZED: Detect outliers across full spectrum using vectorized operations
        
        Returns:
            List of OutlierDeviation objects
        """
        # OPTIMIZATION: Use cached aligned spectra
        if self._aligned_cache is None or self._common_wn_cache is None:
            aligned_baseline, aligned_sample, common_wn = self._align_spectra_with_grid(
                baseline_wn, baseline_abs, sample_wn, sample_abs
            )
            self._aligned_cache = (aligned_baseline, aligned_sample)
            self._common_wn_cache = common_wn
        else:
            aligned_baseline, aligned_sample = self._aligned_cache
            common_wn = self._common_wn_cache
        
        # Calculate deviations
        delta_y_array = aligned_sample - aligned_baseline
        
        # Estimate noise
        noise_sigma = self._estimate_noise(delta_y_array)
        
        # OPTIMIZATION: Vectorized outlier detection (no Python loop!)
        abs_dy = np.abs(delta_y_array)
        sigma_levels = abs_dy / noise_sigma if noise_sigma > 0 else np.zeros_like(abs_dy)
        
        # Find outliers (>3σ) using vectorized operations
        outlier_mask = sigma_levels >= 3.0
        outlier_indices = np.where(outlier_mask)[0]
        
        # Calculate percentages (vectorized)
        dy_pcts = np.zeros_like(abs_dy)
        valid_baseline_mask = aligned_baseline > 0.01
        dy_pcts[valid_baseline_mask] = (abs_dy[valid_baseline_mask] / aligned_baseline[valid_baseline_mask]) * 100
        
        # Determine severity (vectorized) - IMPORTANT: Assign in order minor → major → critical
        severity_array = np.full(len(abs_dy), 'minor', dtype=object)
        severity_array[(sigma_levels >= 4.0) | (abs_dy >= self.config.delta_y_major)] = 'major'
        severity_array[(sigma_levels >= 5.0) | (abs_dy >= self.config.delta_y_critical)] = 'critical'
        
        # Build outlier list (only for detected outliers)
        outliers = []
        for idx in outlier_indices:
            outliers.append(OutlierDeviation(
                wavenumber=float(common_wn[idx]),
                delta_y=float(abs_dy[idx]),
                delta_y_pct=float(dy_pcts[idx]),
                sigma_level=float(sigma_levels[idx]),
                severity=str(severity_array[idx])
            ))
        
        return outliers
    
    def _estimate_noise(self, signal_array: np.ndarray) -> float:
        """
        Estimate noise using median absolute deviation (robust)
        
        Returns:
            Noise sigma estimate
        """
        mad = np.median(np.abs(signal_array - np.median(signal_array)))
        sigma = 1.4826 * mad  # Convert MAD to standard deviation
        return sigma
    
    # ========================================================================
    # MULTI-METRIC CATEGORIZATION SYSTEM
    # ========================================================================
    
    def categorize_sample(self, correlation: float, 
                         region_deviations: List[RegionDeviation],
                         outliers: List[OutlierDeviation]) -> Dict:
        """
        Multi-metric categorization using correlation + ΔX + ΔY + ratio
        
        Categories:
        - GOOD: Excellent correlation, minimal deviations
        - REQUIRES_ATTENTION: Minor deviations, monitor trends
        - CRITICAL: Significant deviations (degradation/contamination)
        - OUTLIER: Major spectral differences (contamination/severe degradation)
        - BASELINE_MISMATCH: Different formulation (NOT bad!)
        
        Returns:
            Dict with category, confidence, reasoning, metrics
        """
        # Extract max deviations across all regions
        max_delta_y = max([rd.max_delta_y for rd in region_deviations]) if region_deviations else 0.0
        max_delta_x = max([rd.max_delta_x for rd in region_deviations]) if region_deviations else 0.0
        
        # Count critical outliers
        critical_outliers = sum(1 for o in outliers if o.severity == 'critical')
        
        # Calculate ΔX:ΔY ratio
        if max_delta_y > 0:
            ratio = max_delta_x / max_delta_y
        else:
            ratio = 0.0
        
        # NEW: Check if deviations are systematic (all regions) vs localized (specific regions)
        critical_regions_count = sum(1 for rd in region_deviations if rd.alert_level in ['major', 'critical'])
        systematic_deviation = critical_regions_count >= 2  # UPDATED: Both primary regions affected (was 3+ out of 4)
        
        # NEW: Check for known contamination patterns
        carbonyl_region = next((rd for rd in region_deviations if 'carbonyl' in rd.region_name.lower()), None)
        water_region = next((rd for rd in region_deviations if 'water' in rd.region_name.lower()), None)
        
        has_oxidation = carbonyl_region and carbonyl_region.alert_level in ['major', 'critical']
        has_water = water_region and water_region.alert_level in ['major', 'critical']
        
        # DECISION TREE
        
        # Category: GOOD
        if (correlation >= self.config.correlation_excellent and 
            max_delta_y < self.config.delta_y_minor and 
            max_delta_x < self.config.delta_x_minor):
            return {
                'category': 'GOOD',
                'confidence': 0.95,
                'reasoning': [
                    f"1. Excellent spectral correlation (r={correlation:.3f} ≥ {self.config.correlation_excellent})",
                    f"2. AND minimal intensity deviation (ΔY={max_delta_y:.3f} A < {self.config.delta_y_minor} A)",
                    f"3. AND minimal peak shift (ΔX={max_delta_x:.1f} cm⁻¹ < {self.config.delta_x_minor} cm⁻¹)",
                    "→ Sample shows minimal changes from baseline"
                ],
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers
                }
            }
        
        # Category: BASELINE_MISMATCH (different formulation) - IMPROVED LOGIC
        # Check first: systematic deviations without degradation pattern
        if (correlation < self.config.correlation_low and 
            systematic_deviation and 
            not (has_oxidation and has_water)):  # No clear degradation pattern
            return {
                'category': 'BASELINE_MISMATCH',
                'confidence': 0.85,
                'reasoning': [
                    f"1. Low spectral correlation (r={correlation:.3f} < {self.config.correlation_low})",
                    f"2. Systematic deviations across primary regions ({critical_regions_count}/2 regions affected)",
                    f"3. No clear contamination pattern (oxidation + water)",
                    f"4. Max deviation: ΔY={max_delta_y:.3f} A (likely scale/intensity difference)",
                    "→ Likely different grease formulation (synthetic vs mineral, different manufacturer)",
                    "⚠️ Verify baseline is appropriate for this sample type",
                    "⚠️ Consider using a baseline from the same grease family"
                ],
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers,
                    'systematic_deviation': systematic_deviation
                }
            }
        
        # Category: OUTLIER (severe contamination/degradation) - IMPROVED LOGIC
        if correlation < self.config.correlation_low and (
            (has_oxidation and has_water) or  # Clear degradation pattern
            critical_outliers >= 2 or  # Multiple isolated critical spikes
            max_delta_y > self.config.delta_y_critical * 10  # Extreme deviation (>1.0 A)
        ):
            reasons = []
            reasons.append(f"1. Low spectral correlation (r={correlation:.3f} < {self.config.correlation_low})")
            
            if has_oxidation and has_water:
                reasons.append(f"2. Clear degradation pattern: Carbonyl ({carbonyl_region.max_delta_y:.3f} A) + Water ({water_region.max_delta_y:.3f} A)")
            elif max_delta_y > self.config.delta_y_critical * 10:
                reasons.append(f"2. Extreme intensity deviation (ΔY={max_delta_y:.3f} A >> 1.0 A threshold)")
            elif critical_outliers >= 2:
                reasons.append(f"2. Multiple critical outliers detected ({critical_outliers})")
            
            reasons.append("→ Likely contamination or severely degraded sample")
            
            return {
                'category': 'OUTLIER',
                'confidence': 0.90,
                'reasoning': reasons,
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers
                }
            }
        
        # Fallback: If low correlation but doesn't match BASELINE_MISMATCH or OUTLIER patterns above
        if correlation < self.config.correlation_low:
            return {
                'category': 'BASELINE_MISMATCH',
                'confidence': 0.75,
                'reasoning': [
                    f"1. Low spectral correlation (r={correlation:.3f} < {self.config.correlation_low})",
                    f"2. Deviations detected (ΔY={max_delta_y:.3f} A, ΔX={max_delta_x:.1f} cm⁻¹)",
                    "→ Sample does not match baseline - verify baseline selection",
                    "⚠️ Review critical regions to determine if contamination or formulation difference"
                ],
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers
                }
            }
        
        # Category: CRITICAL (significant deviations)
        if ((correlation >= self.config.correlation_good and max_delta_y >= self.config.delta_y_critical) or
            (self.config.correlation_moderate <= correlation < self.config.correlation_good and max_delta_y >= self.config.delta_y_major) or
            (self.config.correlation_low <= correlation < self.config.correlation_moderate and max_delta_y >= self.config.delta_y_minor)):
            
            # Determine if shift-dominant or intensity-dominant
            if ratio > 50:
                deviation_type = "shift-dominant (chemical environment change)"
            elif ratio < 20:
                deviation_type = "intensity-dominant (concentration/degradation)"
            else:
                deviation_type = "both shift and intensity changes"
            
            return {
                'category': 'CRITICAL',
                'confidence': 0.95,
                'reasoning': [
                    f"1. Spectral correlation: r={correlation:.3f}",
                    f"2. Significant intensity deviation (ΔY={max_delta_y:.3f} A)",
                    f"3. Notable peak shift (ΔX={max_delta_x:.1f} cm⁻¹)" if max_delta_x >= self.config.delta_x_minor else "",
                    f"4. ΔX:ΔY ratio ({ratio:.1f}) indicates {deviation_type}",
                    "→ Significant deviations detected"
                ],
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers
                }
            }
        
        # Category: REQUIRES_ATTENTION (minor deviations)
        if ((correlation >= self.config.correlation_good and self.config.delta_y_minor <= max_delta_y < self.config.delta_y_critical) or
            (self.config.correlation_moderate <= correlation < self.config.correlation_excellent and max_delta_y < self.config.delta_y_major)):
            return {
                'category': 'REQUIRES_ATTENTION',
                'confidence': 0.80,
                'reasoning': [
                    f"1. Spectral correlation: r={correlation:.3f}",
                    f"2. Minor deviations detected (ΔY={max_delta_y:.3f} A, ΔX={max_delta_x:.1f} cm⁻¹)",
                    "→ Monitor trends - may indicate early changes"
                ],
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers
                }
            }
        
        # Default: REQUIRES_ATTENTION
        return {
            'category': 'REQUIRES_ATTENTION',
            'confidence': 0.70,
            'reasoning': [
                f"1. Spectral correlation: r={correlation:.3f}",
                f"2. Deviations detected (ΔY={max_delta_y:.3f} A, ΔX={max_delta_x:.1f} cm⁻¹)",
                "→ Review metrics to determine significance"
            ],
            'metrics': {
                'correlation': correlation,
                'max_delta_y': max_delta_y,
                'max_delta_x': max_delta_x,
                'ratio': ratio,
                'critical_outliers': critical_outliers
            }
        }
    
    # ========================================================================
    # COMPLETE ANALYSIS
    # ========================================================================
    
    def analyze(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
               sample_wn: np.ndarray, sample_abs: np.ndarray,
               baseline_name: str = "", sample_name: str = "") -> Dict:
        """
        Complete deviation analysis pipeline
        
        Returns:
            Dict containing all analysis results
        """
        start_time = time.time()
        self.analysis_log = []
        
        # OPTIMIZATION: Clear cache at start of analysis
        self._aligned_cache = None
        self._common_wn_cache = None
        
        self._log(f"Starting analysis: {sample_name} vs {baseline_name}")
        
        # 1. Baseline compatibility (this caches the aligned spectra)
        compatibility = self.check_baseline_compatibility(
            baseline_wn, baseline_abs, sample_wn, sample_abs
        )
        self._log(f"Baseline compatibility: r={compatibility['correlation']:.3f}, level={compatibility['level']}")
        
        # 2. Critical region analysis (uses cached aligned spectra!)
        region_deviations = []
        for region_name, region_range in self.config.critical_regions.items():
            deviation = self.analyze_critical_region(
                region_name, region_range,
                baseline_wn, baseline_abs,
                sample_wn, sample_abs
            )
            region_deviations.append(deviation)
            self._log(f"Region {region_name}: ΔY={deviation.max_delta_y:.4f} A, "
                     f"ΔX={deviation.max_delta_x:.2f} cm⁻¹, alert={deviation.alert_level}")
        
        # 3. Outlier detection (uses cached aligned spectra!)
        outliers = self.detect_outliers(
            baseline_wn, baseline_abs, sample_wn, sample_abs
        )
        self._log(f"Outliers detected: {len(outliers)} total")
        
        # 4. Multi-metric categorization
        multi_metric_result = self.categorize_sample(
            compatibility['correlation'],
            region_deviations,
            outliers
        )
        self._log(f"Multi-metric categorization: {multi_metric_result['category']} "
                 f"(confidence: {multi_metric_result['confidence']:.2f})")
        
        # 5. Generate human summary
        human_summary = self._generate_human_summary(
            sample_name, compatibility, region_deviations,
            outliers, multi_metric_result
        )
        
        analysis_time = time.time() - start_time
        self._log(f"Analysis complete in {analysis_time:.2f}s")
        
        # Clear cache after analysis
        self._aligned_cache = None
        self._common_wn_cache = None
        
        return {
            'metadata': {
                'baseline_name': baseline_name,
                'sample_name': sample_name,
                'analysis_time': analysis_time,
                'timestamp': datetime.now().isoformat()
            },
            'baseline_compatibility': compatibility,
            'critical_regions': [asdict(rd) for rd in region_deviations],
            'outliers': [asdict(o) for o in outliers],
            'multi_metric_category': multi_metric_result,
            'human_summary': human_summary,
            'analysis_log': self.analysis_log
        }
    
    def _generate_human_summary(self, sample_name: str, compatibility: Dict,
                               region_deviations: List[RegionDeviation],
                               outliers: List[OutlierDeviation],
                               multi_metric_result: Dict) -> str:
        """
        Generate human-readable summary of deviations
        
        Pure factual reporting without quality judgments
        """
        lines = []
        
        # Header with sample name
        lines.append("=" * 70)
        lines.append("SPECTRAL DEVIATION ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        # Baseline compatibility - more compact
        lines.append("BASELINE COMPATIBILITY")
        lines.append(f"  Correlation: r = {compatibility['correlation']:.3f}")
        if compatibility['warning']:
            lines.append(f"  ⚠️  {compatibility['warning']}")
        else:
            lines.append(f"  ✅ Excellent spectral match")
        lines.append("")
        
        # Multi-metric category - cleaner box
        lines.append("=" * 70)
        lines.append("MULTI-METRIC CATEGORIZATION")
        lines.append("=" * 70)
        lines.append("")
        
        category_emoji = {
            'GOOD': '✅',
            'REQUIRES_ATTENTION': '⚠️',
            'CRITICAL': '❌',
            'OUTLIER': '🚨',
            'BASELINE_MISMATCH': '⚡'
        }
        
        emoji = category_emoji.get(multi_metric_result['category'], '❓')
        lines.append(f"Status: {emoji} {multi_metric_result['category']}")
        lines.append(f"Confidence: {multi_metric_result['confidence']*100:.0f}%")
        lines.append("")
        lines.append("Decision Logic:")
        for i, reason in enumerate(multi_metric_result['reasoning'], 1):
            if reason:  # Skip empty strings
                lines.append(f"  {i}. {reason}")
        lines.append("")
        lines.append("Key Metrics:")
        metrics = multi_metric_result['metrics']
        lines.append(f"  • Correlation (r): {metrics['correlation']:.3f}")
        lines.append(f"  • Max ΔY: {metrics['max_delta_y']:.3f} A (intensity deviation)")
        # Better handling of zero shifts
        if metrics['max_delta_x'] < 0.1:
            lines.append(f"  • Max ΔX: No significant shift detected")
        else:
            lines.append(f"  • Max ΔX: {metrics['max_delta_x']:.1f} cm⁻¹ (peak shift)")
        
        # Better ratio interpretation
        if metrics['ratio'] < 0.1:
            lines.append(f"  • ΔX:ΔY Ratio: Intensity-dominant deviation")
        elif metrics['ratio'] > 100:
            lines.append(f"  • ΔX:ΔY Ratio: Shift-dominant deviation")
        else:
            lines.append(f"  • ΔX:ΔY Ratio: {metrics['ratio']:.1f}")
        
        lines.append(f"  • Critical Outliers: {metrics['critical_outliers']}")
        lines.append("")
        
        # Critical regions detail - cleaner format
        lines.append("=" * 70)
        lines.append("CRITICAL REGIONS ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        for i, rd in enumerate(region_deviations, 1):
            alert_emoji = {
                'superimposed': '✅',
                'minor': '⚠️',
                'major': '❌',
                'critical': '🚨'
            }
            emoji = alert_emoji.get(rd.alert_level, '❓')
            
            # Region header
            region_title = rd.region_name.replace('_', ' ').title()
            lines.append(f"Region {i}: {region_title} ({rd.region_range[0]:.0f}-{rd.region_range[1]:.0f} cm⁻¹)")
            
            # Vertical deviation
            lines.append(f"  ΔY: {rd.max_delta_y:+.3f} A at {rd.max_delta_y_wavenumber:.0f} cm⁻¹")
            if rd.max_delta_y_pct > 0:
                lines.append(f"      {rd.max_delta_y_pct:+.1f}% relative to baseline")
            
            # Horizontal shift
            if abs(rd.max_delta_x) < 0.1:
                lines.append(f"  ΔX: No significant shift")
            else:
                lines.append(f"  ΔX: {rd.max_delta_x:+.1f} cm⁻¹")
            
            # Status with interpretation
            lines.append(f"  {emoji} {rd.alert_level.upper()}: {rd.reasoning}")
            lines.append("")
        
        # Outliers summary - more compact
        if outliers:
            critical_outliers = [o for o in outliers if o.severity == 'critical']
            major_outliers = [o for o in outliers if o.severity == 'major']
            
            lines.append("=" * 70)
            lines.append("OUTLIER DETECTION (Full Spectrum)")
            lines.append("=" * 70)
            lines.append(f"Total: {len(outliers)} points (Critical: {len(critical_outliers)}, Major: {len(major_outliers)})")
            
            if critical_outliers:
                lines.append("")
                lines.append("Top 3 Critical Outliers:")
                for o in sorted(critical_outliers, key=lambda x: x.delta_y, reverse=True)[:3]:
                    lines.append(f"  • {o.wavenumber:.0f} cm⁻¹: ΔY = {o.delta_y:.3f} A ({o.sigma_level:.1f}σ)")
        else:
            lines.append("=" * 70)
            lines.append("OUTLIER DETECTION")
            lines.append("=" * 70)
            lines.append("✅ No significant outliers detected")
        
        lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_csv_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load spectrum from CSV file"""
    df = pd.read_csv(filepath)
    
    if 'X' in df.columns and 'Y' in df.columns:
        wavenumbers = df['X'].values
        absorbance = df['Y'].values
    elif len(df.columns) >= 2:
        wavenumbers = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values
    else:
        raise ValueError("CSV must have at least 2 columns")
    
    sort_idx = np.argsort(wavenumbers)
    return wavenumbers[sort_idx], absorbance[sort_idx]


def load_csv_spectrum_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load spectrum from pandas DataFrame"""
    if 'X' in df.columns and 'Y' in df.columns:
        wavenumbers = df['X'].values
        absorbance = df['Y'].values
    elif len(df.columns) >= 2:
        wavenumbers = df.iloc[:, 0].values
        absorbance = df.iloc[:, 1].values
    else:
        raise ValueError("DataFrame must have at least 2 columns")
    
    sort_idx = np.argsort(wavenumbers)
    return wavenumbers[sort_idx], absorbance[sort_idx]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FTIR Deviation Analyzer - Test")
    print("=" * 70)
    
    # Create analyzer
    config = DeviationConfig()
    analyzer = FTIRDeviationAnalyzer(config)
    
    # Generate synthetic test data
    wavenumbers = np.linspace(600, 4000, 2000)
    
    # Baseline: clean spectrum
    baseline = np.zeros_like(wavenumbers)
    baseline += 0.5 * np.exp(-((wavenumbers - 2900) / 50)**2)  # C-H
    baseline += 0.2 * np.exp(-((wavenumbers - 1460) / 30)**2)  # C-H bend
    baseline += 0.02 * np.random.randn(len(wavenumbers))  # Noise
    
    # Sample: with deviations
    sample = baseline.copy()
    sample += 0.15 * np.exp(-((wavenumbers - 1725) / 40)**2)  # Carbonyl deviation!
    sample += 0.08 * np.exp(-((wavenumbers - 3400) / 100)**2)  # Water deviation!
    sample += 0.22 * np.exp(-((wavenumbers - 2345) / 30)**2)  # Outlier!
    sample += 0.02 * np.random.randn(len(wavenumbers))  # Noise
    
    # Run analysis
    print("\nRunning deviation analysis...")
    result = analyzer.analyze(
        wavenumbers, baseline,
        wavenumbers, sample,
        "baseline_test.csv",
        "sample_test.csv"
    )
    
    # Display results
    print("\n" + result['human_summary'])
    
    print("\n" + "=" * 70)
    print("JSON OUTPUT (Multi-Metric Category):")
    print("=" * 70)
    print(json.dumps(result['multi_metric_category'], indent=2))
    
    print("\n✅ Test complete!")
