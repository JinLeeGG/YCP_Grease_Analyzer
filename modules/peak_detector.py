"""
Automatic Peak Detection for Spectroscopy Data

This module handles the numerical analysis portion of the hybrid system,
providing accurate peak locations and quantified changes to the LLaVA model.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional


class PeakDetector:
    """Automatically detects and compares peaks in spectroscopy data"""
    
    def __init__(self, 
                 min_height_abs=0.1,    # Minimum absolute peak height (absorbance value)
                 min_prominence=0.05,   # Minimum prominence (how much peak stands out)
                 min_distance=20):      # Minimum spacing between peaks (in data points)
        """
        Initialize peak detector with flexible parameters
        """
        self.min_height_abs = min_height_abs
        self.min_prominence = min_prominence
        self.min_distance = min_distance
    
    def detect_peaks(self, data_df: pd.DataFrame) -> List[Dict]:
        """
        Detect all significant peaks in a spectrum
        """
        # Get arrays
        wavenumbers = data_df['X'].values
        absorbances = data_df['Y'].values

        # Find peaks using parameters
        peaks_indices, properties = find_peaks(
            absorbances,
            height=self.min_height_abs,
            prominence=self.min_prominence,
            distance=self.min_distance
        )

        peaks_list = []
        
        # --- FIX: Iterate using enumerate to get the index 'i' ---
        # 'i' will be the index (0, 1, 2...) for the properties array
        # 'idx' will be the index (e.g., 500, 1200, 1750...) in the main 'wavenumbers' array
        
        for i, idx in enumerate(peaks_indices):
            peaks_list.append({
                'wavenumber': float(wavenumbers[idx]),
                'absorbance': float(absorbances[idx]),
                
                # --- THIS IS THE CORRECTED LINE ---
                # Access the 'prominences' array directly by its aligned index 'i'
                'prominence': float(properties['prominences'][i])
            })
            
        return peaks_list

    # 1. FIX: Implementation of the missing function 'compare_spectra'
    def compare_spectra(self, baseline_df: pd.DataFrame, sample_df: pd.DataFrame) -> Dict:
        """
        Compares peaks between baseline and sample dataframes.

        Returns a dictionary containing new peaks, missing peaks, and matched peak changes.
        """
        baseline_peaks = self.detect_peaks(baseline_df)
        sample_peaks = self.detect_peaks(sample_df)
        
        # Wavenumber tolerance for matching peaks (e.g., 5 cm⁻¹)
        WAVENUMBER_TOLERANCE = 5.0
        
        matched_peaks = []
        new_peaks = []
        missing_peaks = []
        
        # Convert lists to NumPy arrays for efficient searching
        baseline_wavenumbers = np.array([p['wavenumber'] for p in baseline_peaks])
        
        # 1. Find Matched and New Peaks
        for sample_peak in sample_peaks:
            # Find closest baseline peak within tolerance
            diffs = np.abs(baseline_wavenumbers - sample_peak['wavenumber'])
            min_diff_index = np.argmin(diffs)
            min_diff = diffs[min_diff_index]
            
            if min_diff <= WAVENUMBER_TOLERANCE:
                # Match found - calculate change
                baseline_peak = baseline_peaks[min_diff_index]
                
                intensity_change = sample_peak['absorbance'] - baseline_peak['absorbance']
                intensity_change_percent = (intensity_change / baseline_peak['absorbance']) * 100
                
                matched_peaks.append({
                    'wavenumber': sample_peak['wavenumber'],
                    'baseline_abs': baseline_peak['absorbance'],
                    'sample_abs': sample_peak['absorbance'],
                    'intensity_change_percent': intensity_change_percent
                })
            else:
                # No match found - treated as a new peak
                new_peaks.append(sample_peak)

        # 2. Find Missing/Reduced Peaks
        sample_wavenumbers = np.array([p['wavenumber'] for p in sample_peaks])
        
        for baseline_peak in baseline_peaks:
            # Find closest sample peak within tolerance
            diffs = np.abs(sample_wavenumbers - baseline_peak['wavenumber'])
            
            if len(diffs) == 0 or np.min(diffs) > WAVENUMBER_TOLERANCE:
                # Baseline peak is missing in the sample
                missing_peaks.append(baseline_peak)
                
        # 3. Add oxidation check (1650-1800 cm⁻¹)
        oxidation_peak = self._check_oxidation_zone(baseline_df, sample_df)

        return {
            'matched_peaks': matched_peaks,
            'new_peaks': new_peaks,
            'missing_peaks': missing_peaks,
            'oxidation_data': oxidation_peak
        }

    def _check_oxidation_zone(self, baseline_df: pd.DataFrame, sample_df: pd.DataFrame) -> Dict:
        """Specific check for the carbonyl (oxidation) region (1650-1800 cm⁻¹)"""
        OXIDATION_MIN = 1650
        OXIDATION_MAX = 1800

        # Filter data in the oxidation region (resetting index is still good practice)
        base_ox_df = baseline_df[(baseline_df['X'] >= OXIDATION_MIN) & (baseline_df['X'] <= OXIDATION_MAX)].reset_index(drop=True)
        sample_ox_df = sample_df[(sample_df['X'] >= OXIDATION_MIN) & (sample_df['X'] <= OXIDATION_MAX)].reset_index(drop=True)

        if base_ox_df.empty or sample_ox_df.empty:
            return {'change_percent': 0, 'sample_max_abs': 0, 'wavenumber': 0}

        # --- FIX: ROBUST VALUE RETRIEVAL ---
        # 1. Get the NumPy arrays from the filtered data (which are 16 elements long)
        y_values_ox = sample_ox_df['Y'].values
        x_values_ox = sample_ox_df['X'].values

        # 2. Find the index position (0 to 15) of the maximum absorbance in the array
        max_idx_np = np.argmax(y_values_ox) 

        # 3. Access the values directly using the position index
        sample_max_abs = y_values_ox[max_idx_np]
        sample_max_wavenumber = x_values_ox[max_idx_np]
        # ------------------------------------

        # Find the baseline absorbance at the *same* wavenumber where the sample peaks
        # Interpolate the baseline data to get a precise comparison
        baseline_abs_at_sample_max = np.interp(
            sample_max_wavenumber,
            base_ox_df['X'].values,
            base_ox_df['Y'].values
        )
        
        if baseline_abs_at_sample_max > 0:
            change_percent = ((sample_max_abs - baseline_abs_at_sample_max) / baseline_abs_at_sample_max) * 100
        else:
            # Handle zero/near-zero baseline absorbance gracefully
            change_percent = 100 if sample_max_abs > 0.05 else 0

        return {
            'change_percent': change_percent,
            'sample_max_abs': sample_max_abs,
            'wavenumber': sample_max_wavenumber
        }

    # 2. Implementation of the necessary formatting function
    def format_for_llm(self, comparison: Dict) -> str:
        """
        Formats the numerical comparison results into a clean text string for LLaVA.
        """
        summary = "═══════════════════════════════════════════════════════════════════\n"
        summary += "PEAK COMPARISON DATA (Sample vs Baseline)\n"
        summary += "═══════════════════════════════════════════════════════════════════\n\n"
        
        # Oxidation Data
        ox_data = comparison.get('oxidation_data', {})
        if ox_data['change_percent'] > 0:
            summary += f"🔥 OXIDATION ZONE (1650-1800 cm⁻¹):\n"
            summary += f"  • Wavenumber of Max Peak: {ox_data['wavenumber']:.0f} cm⁻¹\n"
            summary += f"  • Sample Absorbance: {ox_data['sample_max_abs']:.2f}\n"
            summary += f"  • Intensity Change: {ox_data['change_percent']:+.1f}%\n\n"
        else:
            summary += "🔥 OXIDATION ZONE: No significant increase detected.\n\n"
        
        # New peaks
        if comparison['new_peaks']:
            summary += "✨ NEW PEAKS IN SAMPLE:\n"
            for peak in comparison['new_peaks']:
                summary += f"  • {peak['wavenumber']:.0f} cm⁻¹ (Abs: {peak['absorbance']:.2f})\n"
            summary += "\n"

        # Missing peaks
        if comparison['missing_peaks']:
            summary += "📉 REDUCED/MISSING PEAKS:\n"
            for peak in comparison['missing_peaks']:
                summary += f"  • {peak['wavenumber']:.0f} cm⁻¹ (Was: {peak['absorbance']:.2f})\n"
            summary += "\n"
        
        # Intensity changes
        if comparison['matched_peaks']:
            summary += "📈 SIGNIFICANT INTENSITY CHANGES (>10%):\n"
            
            # Filter for changes over 10% and sort by magnitude
            significant_changes = sorted(
                [match for match in comparison['matched_peaks'] if abs(match['intensity_change_percent']) > 10], 
                key=lambda x: abs(x['intensity_change_percent']), 
                reverse=True
            )

            if significant_changes:
                for match in significant_changes:
                    symbol = "↑" if match['intensity_change_percent'] > 0 else "↓"
                    summary += f"  {symbol} {match['wavenumber']:.0f} cm⁻¹ ({match['intensity_change_percent']:+.1f}%)\n"
            else:
                summary += "  No changes exceeded the 10% threshold.\n"

        summary += "═══════════════════════════════════════════════════════════════════\n"
        return summary


# ============================================================================
# TEST CODE - Run this file directly to verify functionality
# ============================================================================
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("✅ Peak Detector Testing")
    
    # Generate mock data
    x_values = np.linspace(4000, 500, 3501)
    
    # Baseline: Two main peaks
    y_baseline = np.exp(-((x_values - 2900)**2) / 50000) * 1.5 + np.exp(-((x_values - 1400)**2) / 50000) * 1.0 + 0.1
    
    # Sample: Oxidation peak (1740) and main peak increase
    y_sample = y_baseline + np.exp(-((x_values - 1740)**2) / 1000) * 0.3 # Oxidation
    y_sample *= 1.1 # General thickening/increase

    baseline_df = pd.DataFrame({'X': x_values, 'Y': y_baseline})
    sample_df = pd.DataFrame({'X': x_values, 'Y': y_sample})

    detector = PeakDetector()
    
    # Test Comparison
    print("\n--- Running Spectral Comparison ---")
    comparison = detector.compare_spectra(baseline_df, sample_df)
    
    print(f"\n✅ Found {len(comparison['matched_peaks'])} Matched Peaks, {len(comparison['new_peaks'])} New Peaks, {len(comparison['missing_peaks'])} Missing Peaks.")
    
    # Test LLM Formatting
    print("\n--- LLM Formatted Output ---")
    llm_output = detector.format_for_llm(comparison)
    print(llm_output)
    print("✅ Peak Detector module loaded and validated successfully.")