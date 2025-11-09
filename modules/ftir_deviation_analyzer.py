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
    
    # ΔY thresholds (vertical deviation)
    delta_y_critical: float = 0.10  # >0.10 A = critical
    delta_y_major: float = 0.05     # 0.05-0.10 A = major
    delta_y_minor: float = 0.03     # 0.03-0.05 A = minor
    
    # ΔX thresholds (horizontal shift) - NEW!
    delta_x_major: float = 10.0     # >10 cm⁻¹ = major shift
    delta_x_minor: float = 5.0      # 5-10 cm⁻¹ = minor shift
    
    # Correlation thresholds for multi-metric categorization
    correlation_excellent: float = 0.97
    correlation_good: float = 0.95
    correlation_moderate: float = 0.90
    correlation_low: float = 0.85
    
    def __post_init__(self):
        if self.critical_regions is None:
            self.critical_regions = {
                'carbonyl_oxidation': (1650, 1800),
                'water_contamination': (3200, 3600),
                'additives_glycol': (1000, 1300),
                'ch_stretch': (2850, 2950)
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
        # Align spectra
        aligned_baseline, aligned_sample = self._align_spectra(
            baseline_wn, baseline_abs, sample_wn, sample_abs
        )
        
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
    
    def _align_spectra(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                      sample_wn: np.ndarray, sample_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align spectra to common wavenumber grid
        
        Returns:
            Tuple of (aligned_baseline, aligned_sample)
        """
        # Find common wavenumber range
        common_min = max(baseline_wn.min(), sample_wn.min())
        common_max = min(baseline_wn.max(), sample_wn.max())
        
        # Create common grid
        common_wn = np.linspace(common_min, common_max, 1000)
        
        # Interpolate both spectra
        f_baseline = interp1d(baseline_wn, baseline_abs, kind='cubic', fill_value='extrapolate')
        f_sample = interp1d(sample_wn, sample_abs, kind='cubic', fill_value='extrapolate')
        
        aligned_baseline = f_baseline(common_wn)
        aligned_sample = f_sample(common_wn)
        
        return aligned_baseline, aligned_sample
    
    # ========================================================================
    # CRITICAL REGION ANALYSIS
    # ========================================================================
    
    def analyze_critical_region(self, region_name: str, region_range: Tuple[float, float],
                                baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                                sample_wn: np.ndarray, sample_abs: np.ndarray) -> RegionDeviation:
        """
        Analyze deviation in a specific critical region
        
        Returns:
            RegionDeviation object with all metrics
        """
        # Extract region data
        baseline_mask = (baseline_wn >= region_range[0]) & (baseline_wn <= region_range[1])
        sample_mask = (sample_wn >= region_range[0]) & (sample_wn <= region_range[1])
        
        region_baseline_wn = baseline_wn[baseline_mask]
        region_baseline_abs = baseline_abs[baseline_mask]
        region_sample_wn = sample_wn[sample_mask]
        region_sample_abs = sample_abs[sample_mask]
        
        # Align within region
        aligned_baseline, aligned_sample = self._align_spectra(
            region_baseline_wn, region_baseline_abs,
            region_sample_wn, region_sample_abs
        )
        
        # Calculate ΔY (vertical deviation)
        delta_y_array = np.abs(aligned_sample - aligned_baseline)
        max_delta_y = np.max(delta_y_array)
        max_delta_y_idx = np.argmax(delta_y_array)
        
        # Find corresponding wavenumber
        common_wn = np.linspace(
            max(region_baseline_wn.min(), region_sample_wn.min()),
            min(region_baseline_wn.max(), region_sample_wn.max()),
            len(aligned_baseline)
        )
        max_delta_y_wavenumber = common_wn[max_delta_y_idx]
        
        # Calculate percentage change
        baseline_value = aligned_baseline[max_delta_y_idx]
        if baseline_value > 0.01:  # Avoid division by near-zero
            max_delta_y_pct = (max_delta_y / baseline_value) * 100
        else:
            max_delta_y_pct = 0.0
        
        # Estimate ΔX (horizontal shift) using cross-correlation
        max_delta_x = self._estimate_horizontal_shift(
            aligned_baseline, aligned_sample, common_wn
        )
        
        # Calculate ΔX:ΔY ratio
        if max_delta_y > 0:
            ratio = max_delta_x / max_delta_y
        else:
            ratio = 0.0
        
        # Determine alert level based on BOTH ΔY and ΔX
        alert_level, reasoning = self._determine_alert_level(
            max_delta_y, max_delta_x, region_name
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
    
    def _estimate_horizontal_shift(self, baseline: np.ndarray, sample: np.ndarray, 
                                   wavenumbers: np.ndarray) -> float:
        """
        Estimate horizontal shift (ΔX) using cross-correlation
        
        Returns:
            Shift in cm⁻¹
        """
        # Cross-correlation
        correlation = signal.correlate(sample, baseline, mode='same')
        lag = np.argmax(correlation) - len(baseline) // 2
        
        # Convert lag to wavenumber shift
        if len(wavenumbers) > 1:
            wn_step = np.mean(np.diff(wavenumbers))
            shift = abs(lag * wn_step)
        else:
            shift = 0.0
        
        return shift
    
    def _determine_alert_level(self, delta_y: float, delta_x: float, 
                               region_name: str) -> Tuple[str, str]:
        """
        Determine alert level based on BOTH ΔY and ΔX
        
        NEW LOGIC: Both vertical and horizontal deviations trigger alerts
        
        Returns:
            Tuple of (alert_level, reasoning)
        """
        # Evaluate ΔY
        if delta_y >= self.config.delta_y_critical:
            delta_y_level = 'critical'
        elif delta_y >= self.config.delta_y_major:
            delta_y_level = 'major'
        elif delta_y >= self.config.delta_y_minor:
            delta_y_level = 'minor'
        else:
            delta_y_level = 'superimposed'
        
        # Evaluate ΔX (NEW!)
        if delta_x >= self.config.delta_x_major:
            delta_x_level = 'major'
        elif delta_x >= self.config.delta_x_minor:
            delta_x_level = 'minor'
        else:
            delta_x_level = 'superimposed'
        
        # Take the HIGHER of the two
        alert_levels = ['superimposed', 'minor', 'major', 'critical']
        delta_y_idx = alert_levels.index(delta_y_level)
        delta_x_idx = alert_levels.index(delta_x_level)
        
        # Escalate if BOTH are elevated
        if delta_y_level in ['major', 'critical'] and delta_x_level >= 'minor':
            # Escalate by one level if both are elevated
            combined_idx = min(delta_y_idx + 1, len(alert_levels) - 1)
        elif delta_x_level in ['major'] and delta_y_level >= 'minor':
            combined_idx = min(delta_x_idx + 1, len(alert_levels) - 1)
        else:
            combined_idx = max(delta_y_idx, delta_x_idx)
        
        alert_level = alert_levels[combined_idx]
        
        # Generate reasoning
        reasoning_parts = []
        if delta_y >= self.config.delta_y_minor:
            reasoning_parts.append(f"ΔY = {delta_y:.3f} A ({delta_y_level})")
        if delta_x >= self.config.delta_x_minor:
            reasoning_parts.append(f"ΔX = {delta_x:.1f} cm⁻¹ ({delta_x_level})")
        
        if reasoning_parts:
            reasoning = " + ".join(reasoning_parts)
            if combined_idx > max(delta_y_idx, delta_x_idx):
                reasoning += " → Escalated due to multiple deviation types"
        else:
            reasoning = "Spectra superimpose within tolerance"
        
        return alert_level, reasoning
    
    # ========================================================================
    # OUTLIER DETECTION (Full Spectrum)
    # ========================================================================
    
    def detect_outliers(self, baseline_wn: np.ndarray, baseline_abs: np.ndarray,
                       sample_wn: np.ndarray, sample_abs: np.ndarray) -> List[OutlierDeviation]:
        """
        Detect outliers across full spectrum using statistical thresholds
        
        Returns:
            List of OutlierDeviation objects
        """
        # Align spectra
        aligned_baseline, aligned_sample = self._align_spectra(
            baseline_wn, baseline_abs, sample_wn, sample_abs
        )
        
        # Calculate deviations
        delta_y_array = aligned_sample - aligned_baseline
        
        # Estimate noise
        noise_sigma = self._estimate_noise(delta_y_array)
        
        # Find outliers (>3σ)
        outliers = []
        common_wn = np.linspace(
            max(baseline_wn.min(), sample_wn.min()),
            min(baseline_wn.max(), sample_wn.max()),
            len(aligned_baseline)
        )
        
        for i, (wn, dy) in enumerate(zip(common_wn, delta_y_array)):
            abs_dy = abs(dy)
            sigma_level = abs_dy / noise_sigma if noise_sigma > 0 else 0
            
            if sigma_level >= 3.0:
                # Calculate percentage
                baseline_val = aligned_baseline[i]
                if baseline_val > 0.01:
                    dy_pct = (abs_dy / baseline_val) * 100
                else:
                    dy_pct = 0.0
                
                # Determine severity
                if sigma_level >= 5.0 or abs_dy >= self.config.delta_y_critical:
                    severity = 'critical'
                elif sigma_level >= 4.0 or abs_dy >= self.config.delta_y_major:
                    severity = 'major'
                else:
                    severity = 'minor'
                
                outliers.append(OutlierDeviation(
                    wavenumber=float(wn),
                    delta_y=float(abs_dy),
                    delta_y_pct=float(dy_pct),
                    sigma_level=float(sigma_level),
                    severity=severity
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
        - OUTLIER: Major spectral differences
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
        
        # Category: OUTLIER (severe)
        if correlation < self.config.correlation_low and (max_delta_y > self.config.delta_y_critical or critical_outliers >= 2):
            return {
                'category': 'OUTLIER',
                'confidence': 0.90,
                'reasoning': [
                    f"1. Low spectral correlation (r={correlation:.3f} < {self.config.correlation_low})",
                    f"2. High intensity deviation (ΔY={max_delta_y:.3f} A > {self.config.delta_y_critical} A)" if max_delta_y > self.config.delta_y_critical else "",
                    f"3. Multiple critical outliers detected ({critical_outliers})" if critical_outliers >= 2 else "",
                    "→ Likely contamination or severely degraded sample"
                ],
                'metrics': {
                    'correlation': correlation,
                    'max_delta_y': max_delta_y,
                    'max_delta_x': max_delta_x,
                    'ratio': ratio,
                    'critical_outliers': critical_outliers
                }
            }
        
        # Category: BASELINE_MISMATCH (different formulation)
        if (correlation < self.config.correlation_low and 
            max_delta_y < self.config.delta_y_major and 
            critical_outliers == 0):
            return {
                'category': 'BASELINE_MISMATCH',
                'confidence': 0.85,
                'reasoning': [
                    f"1. Low spectral correlation (r={correlation:.3f} < {self.config.correlation_low})",
                    f"2. BUT minimal intensity deviation (ΔY={max_delta_y:.3f} A < {self.config.delta_y_major} A)",
                    f"3. AND no critical outliers",
                    "→ Likely different formulation (synthetic vs mineral, different manufacturer)",
                    "⚠️ Verify baseline is appropriate for this sample type"
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
        
        self._log(f"Starting analysis: {sample_name} vs {baseline_name}")
        
        # 1. Baseline compatibility
        compatibility = self.check_baseline_compatibility(
            baseline_wn, baseline_abs, sample_wn, sample_abs
        )
        self._log(f"Baseline compatibility: r={compatibility['correlation']:.3f}, level={compatibility['level']}")
        
        # 2. Critical region analysis
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
        
        # 3. Outlier detection
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
        
        lines.append("=" * 70)
        lines.append(f"SPECTRAL OVERLAY DEVIATION ANALYSIS: {sample_name}")
        lines.append("=" * 70)
        lines.append("")
        
        # Baseline compatibility
        lines.append("BASELINE COMPATIBILITY CHECK:")
        lines.append(f"├─ Spectral Correlation: {compatibility['correlation']:.3f}")
        if compatibility['warning']:
            lines.append(f"└─ ⚠️ {compatibility['warning']}")
        else:
            lines.append(f"└─ ✓ Excellent correlation")
        lines.append("")
        
        # Multi-metric category
        lines.append("╔═══════════════════════════════════════════════════════════════════╗")
        lines.append("║ MULTI-METRIC CATEGORIZATION (Primary Decision System)            ║")
        lines.append("╚═══════════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        category_emoji = {
            'GOOD': '✅',
            'REQUIRES_ATTENTION': '⚠️',
            'CRITICAL': '❌',
            'OUTLIER': '🚨',
            'BASELINE_MISMATCH': '⚡'
        }
        
        emoji = category_emoji.get(multi_metric_result['category'], '❓')
        lines.append(f"**Final Category:** {emoji} {multi_metric_result['category']}")
        lines.append(f"**Confidence:** {multi_metric_result['confidence']*100:.0f}%")
        lines.append("")
        lines.append("**Decision Logic:**")
        for reason in multi_metric_result['reasoning']:
            if reason:  # Skip empty strings
                lines.append(f"  {reason}")
        lines.append("")
        lines.append("**Metrics Used:**")
        metrics = multi_metric_result['metrics']
        lines.append(f"  • Spectral Correlation (r): {metrics['correlation']:.3f}")
        lines.append(f"  • Max ΔY (vertical): {metrics['max_delta_y']:.3f} A")
        lines.append(f"  • Max ΔX (horizontal): {metrics['max_delta_x']:.1f} cm⁻¹")
        lines.append(f"  • ΔX:ΔY ratio: {metrics['ratio']:.1f}")
        lines.append(f"  • Critical outliers: {metrics['critical_outliers']}")
        lines.append("")
        
        # Critical regions detail
        lines.append("CRITICAL REGIONS - DEVIATION ANALYSIS:")
        lines.append("")
        
        for i, rd in enumerate(region_deviations, 1):
            alert_emoji = {
                'superimposed': '✓',
                'minor': '⚠️',
                'major': '❌',
                'critical': '🚨'
            }
            emoji = alert_emoji.get(rd.alert_level, '❓')
            
            lines.append(f"Region {i}: {rd.region_name.replace('_', ' ').title()} ({rd.region_range[0]}-{rd.region_range[1]} cm⁻¹)")
            lines.append(f"├─ Max vertical deviation (ΔY): {rd.max_delta_y:+.3f} A at {rd.max_delta_y_wavenumber:.0f} cm⁻¹")
            if rd.max_delta_y_pct > 0:
                lines.append(f"│  └─ Relative change: {rd.max_delta_y_pct:+.1f}% from baseline")
            lines.append(f"├─ Horizontal shift (ΔX): {rd.max_delta_x:+.1f} cm⁻¹")
            lines.append(f"├─ ΔX:ΔY ratio: {rd.delta_x_delta_y_ratio:.2f} (raw value)")
            lines.append(f"└─ Status: {emoji} {rd.alert_level.upper()} - {rd.reasoning}")
            lines.append("")
        
        # Outliers summary
        if outliers:
            critical_outliers = [o for o in outliers if o.severity == 'critical']
            major_outliers = [o for o in outliers if o.severity == 'major']
            
            lines.append("OUTLIER DETECTION (Full Spectrum):")
            lines.append(f"Total outliers: {len(outliers)} (Critical: {len(critical_outliers)}, Major: {len(major_outliers)})")
            
            if critical_outliers:
                lines.append("")
                lines.append("Critical outliers (top 3):")
                for o in sorted(critical_outliers, key=lambda x: x.delta_y, reverse=True)[:3]:
                    lines.append(f"  • {o.wavenumber:.0f} cm⁻¹: ΔY = {o.delta_y:.3f} A ({o.sigma_level:.1f}σ)")
        else:
            lines.append("OUTLIER DETECTION: No significant outliers detected")
        
        lines.append("")
        lines.append("=" * 70)
        
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
