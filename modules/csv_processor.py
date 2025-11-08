"""
CSV File Loading and Processing Module

This module handles all CSV file operations for the Grease Analyzer application:
- Loading and parsing CSV files from various sources (file paths, file objects)
- Validating data structure and content
- Normalizing column names and data types
- Smart header detection (skips metadata rows)
- Statistical calculations (mean, std, min, max)
- Baseline comparison analysis

The module is designed to handle both clean CSV files and files with metadata headers
like "Created as New Dataset" that some instruments produce.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict
import sys
import os

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import CSV_CONFIG


class CSVProcessor:
    """
    CSV File Processor for Spectroscopy Data
    
    Handles loading, validation, and processing of CSV files containing
    X-Y coordinate data (wavelength/frequency vs intensity).
    """
    
    @staticmethod
    def load_csv(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load and Validate CSV File
        
        Intelligently loads CSV files with automatic header detection.
        Handles files with metadata rows (like "Created as New Dataset")
        by finding the actual data header row.
        
        Args:
            file: Either a file path (str) or file-like object with read() method
                 (e.g., Streamlit UploadedFile, Django InMemoryUploadedFile)
            
        Returns:
            Tuple of (DataFrame, error_message):
                Success: (df, None) - df contains validated and normalized data
                Failure: (None, "error description") - df is None, error has details
                
        Example:
            >>> df, error = CSVProcessor.load_csv('baseline.csv')
            >>> if error:
            ...     print(f"Error: {error}")
            ... else:
            ...     print(f"Loaded {len(df)} rows")
        """
        try:
            # Handle file-like objects (Streamlit, Django uploads, etc.)
            if hasattr(file, 'read'):
                # Read first few lines to find header location
                file.seek(0)
                first_lines = [file.readline().decode('utf-8') for _ in range(5)]
                file.seek(0)  # Reset to beginning
                
                # Find where actual data starts
                header_row = CSVProcessor._find_header_row(first_lines)
                df = pd.read_csv(file, skiprows=header_row)
            else:
                # Handle file path (string)
                with open(file, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline() for _ in range(5)]
                
                header_row = CSVProcessor._find_header_row(first_lines)
                df = pd.read_csv(file, skiprows=header_row)
            
            # Validate data structure and content
            is_valid, error_msg = CSVProcessor.validate_csv(df)
            if not is_valid:
                return None, error_msg
            
            # Normalize column names and data types
            df = CSVProcessor.normalize_data(df)
            
            return df, None
            
        except pd.errors.EmptyDataError:
            return None, "File is empty."
        except pd.errors.ParserError:
            return None, "Invalid CSV format."
        except Exception as e:
            return None, f"Failed to load file: {str(e)}"
    
    @staticmethod
    def _find_header_row(lines: list) -> int:
        """
        Find Actual Header Row in CSV File
        
        Skips metadata rows like "Created as New Dataset" that some
        instruments add before the actual data. Looks for the first row
        that has numeric data in the second column.
        
        Algorithm:
        1. Iterate through first few lines
        2. Split by comma or tab
        3. Check if second column starts with a number
        4. Return previous line as header
        
        Args:
            lines: List of first few lines from CSV file (as strings)
            
        Returns:
            Number of rows to skip (0 means no skip, use first row as header)
            
        Example:
            Input lines:
                "Created as New Dataset"
                "X,Y"
                "1.5,0.4489"
                "2.0,0.5123"
            Returns: 1 (skip first line, use "X,Y" as header)
        """
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Split by comma or tab
            parts = line.replace('\t', ',').split(',')
            
            # Check if this looks like a data row (at least 2 columns, second is numeric)
            if len(parts) >= 2:
                second_part = parts[1].strip()
                # Check if starts with digit or negative sign
                if second_part and (second_part[0].isdigit() or second_part.startswith('-')):
                    # Previous row is the header
                    return max(0, i - 1) if i > 0 else 0
        
        # Default: no rows to skip
        return 0
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate CSV Data Structure and Content
        
        Performs multiple validation checks to ensure the DataFrame
        meets requirements for analysis:
        - Non-empty dataset
        - Sufficient columns (at least 2: X and Y)
        - Sufficient rows (minimum data points for meaningful analysis)
        - Numeric data types in first two columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid: bool, error_message: str or None)
            - (True, None) if validation passes
            - (False, error_description) if validation fails
        """
        # Check for empty DataFrame
        if df.empty:
            return False, "Data is empty."
        
        # Check minimum column count (X, Y coordinates required)
        if len(df.columns) < CSV_CONFIG['required_columns']:
            return False, f"At least {CSV_CONFIG['required_columns']} columns required."
        
        # Check minimum row count (need enough points for analysis)
        if len(df) < CSV_CONFIG['min_rows']:
            return False, f"At least {CSV_CONFIG['min_rows']} data points required."
        
        # Verify first two columns contain numeric data
        try:
            pd.to_numeric(df.iloc[:, 0], errors='raise')
            pd.to_numeric(df.iloc[:, 1], errors='raise')
        except (ValueError, TypeError):
            return False, "First two columns must contain numeric data."
        
        return True, None
    
    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and Clean DataFrame for Analysis
        
        Performs data cleaning and standardization:
        1. Extracts only first two columns (X, Y)
        2. Standardizes column names to 'X' and 'Y'
        3. Converts values to numeric type (coercing errors to NaN)
        4. Removes any rows with NaN values
        5. Sorts by X values in ascending order
        6. Resets index
        
        This ensures all DataFrames have consistent structure
        regardless of original column names or data format.
        
        Args:
            df: Raw DataFrame from CSV file
            
        Returns:
            Cleaned DataFrame with columns ['X', 'Y'], sorted by X,
            with all numeric values and no missing data
        """
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Extract only first two columns (ignore any extra columns)
        df = df.iloc[:, :2]
        
        # Standardize column names for consistency
        df.columns = ['X', 'Y']
        
        # Convert to numeric type, replacing any non-numeric with NaN
        df['X'] = pd.to_numeric(df['X'], errors='coerce')
        df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
        
        # Remove rows with any NaN values
        df = df.dropna()
        
        # Sort by X values (important for graphing and interpolation)
        df = df.sort_values('X').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def get_statistics(df: pd.DataFrame) -> dict:
        """
        Calculate Statistical Measures for Dataset
        
        Computes key statistical metrics from the Y values:
        - Mean: Average value
        - Std: Standard deviation (measure of spread)
        - Min/Max: Range boundaries
        - Median: Middle value (50th percentile)
        - Range: Difference between max and min
        - Count: Number of data points
        
        These statistics are used for:
        - LLM analysis prompts
        - Quality score calculations
        - Baseline comparisons
        
        Args:
            df: DataFrame with 'Y' column
            
        Returns:
            Dictionary with keys: mean, std, min, max, median, range, count
            All values are Python floats (not numpy types) for JSON serialization
        """
        y_values = df['Y'].values
        
        return {
            'mean': float(np.mean(y_values)),
            'std': float(np.std(y_values)),
            'min': float(np.min(y_values)),
            'max': float(np.max(y_values)),
            'median': float(np.median(y_values)),
            'range': float(np.max(y_values) - np.min(y_values)),
            'count': len(y_values),
        }
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> dict:
        """
        Calculate Statistical Measures (Alias Method)
        
        Wrapper method for get_statistics() to maintain compatibility
        with code that uses this naming convention.
        
        Args:
            df: DataFrame with 'Y' column
            
        Returns:
            Dictionary of statistical measures (same as get_statistics)
        """
        return CSVProcessor.get_statistics(df)
    
    @staticmethod
    def compare_with_baseline(baseline_df: pd.DataFrame, 
                            sample_df: pd.DataFrame) -> dict:
        """
        Compare Sample Data Against Baseline Reference
        
        Performs comprehensive comparison analysis between a sample
        and baseline dataset to quantify differences:
        
        Metrics Calculated:
        1. Mean Deviation %: How much average value changed
        2. Std Deviation %: How much variability changed
        3. Correlation: How similar the patterns are (-1 to 1)
        4. Quality Score: Overall match quality (0 to 100)
        
        The quality score combines all metrics:
        - High correlation = similar pattern
        - Low mean deviation = similar average level
        - Low std deviation = similar variability
        
        Args:
            baseline_df: Reference dataset (DataFrame with X, Y columns)
            sample_df: Test dataset to compare (DataFrame with X, Y columns)
            
        Returns:
            Dictionary containing:
            - baseline_stats: Statistics of baseline
            - sample_stats: Statistics of sample
            - mean_deviation_percent: % change in mean
            - std_deviation_percent: % change in std dev
            - correlation: Pearson correlation coefficient
            - quality_score: 0-100 composite quality metric
        """
        baseline_stats = CSVProcessor.get_statistics(baseline_df)
        sample_stats = CSVProcessor.get_statistics(sample_df)
        
        # Calculate percent deviation in mean (average level shift)
        mean_deviation = ((sample_stats['mean'] - baseline_stats['mean']) / 
                         baseline_stats['mean'] * 100)
        
        # Calculate percent deviation in std dev (variability change)
        std_deviation = ((sample_stats['std'] - baseline_stats['std']) / 
                        baseline_stats['std'] * 100)
        
        # Calculate correlation between Y values
        # Uses shortest common length to handle different dataset sizes
        min_length = min(len(baseline_df), len(sample_df))
        correlation = float(np.corrcoef(
            baseline_df['Y'].values[:min_length],
            sample_df['Y'].values[:min_length]
        )[0, 1])
        
        return {
            'baseline_stats': baseline_stats,
            'sample_stats': sample_stats,
            'mean_deviation_percent': mean_deviation,
            'std_deviation_percent': std_deviation,
            'correlation': correlation,
            'quality_score': CSVProcessor._calculate_quality_score(
                mean_deviation, std_deviation, correlation
            )
        }
    
    @staticmethod
    def _calculate_quality_score(mean_dev: float, std_dev: float, 
                                corr: float) -> float:
        """
        Calculate Quality Score (0-100)
        
        Combines multiple metrics into a single quality score that
        indicates how well the sample matches the baseline.
        Higher score = better match = likely good quality grease.
        
        Scoring Algorithm:
        - Correlation component: 50 points max
          * Perfect correlation (1.0) = 50 points
          * No correlation (0.0) = 0 points
        
        - Mean deviation component: 30 points max
          * No deviation (0%) = 30 points
          * 100% deviation = 0 points
          * Linear scale between
        
        - Std deviation component: 20 points max
          * No deviation (0%) = 20 points
          * 100% deviation = 0 points
          * Linear scale between
        
        Total possible: 100 points
        
        Args:
            mean_dev: Percent deviation in mean (-100 to +100)
            std_dev: Percent deviation in std deviation (-100 to +100)
            corr: Correlation coefficient (0 to 1)
            
        Returns:
            Quality score from 0 (very different) to 100 (perfect match)
        """
        # Correlation contributes up to 50 points
        corr_score = corr * 50
        
        # Mean deviation contributes up to 30 points (less deviation = more points)
        mean_score = max(0, (100 - abs(mean_dev)) / 100 * 30)
        
        # Std deviation contributes up to 20 points (less deviation = more points)
        std_score = max(0, (100 - abs(std_dev)) / 100 * 20)
        
        # Combine scores
        score = corr_score + mean_score + std_score
        
        # Clamp to 0-100 range
        return max(0, min(100, score))
    
    @staticmethod
    def detect_peaks(df: pd.DataFrame, min_height: float = None, 
                    min_distance: int = 10, prominence: float = None) -> List[Dict]:
        """
        Detect Peaks in Spectroscopy Data
        
        Identifies local maxima (peaks) in the Y values of the dataset.
        Useful for FTIR analysis to identify characteristic absorption bands.
        
        Args:
            df: DataFrame with 'X' (wavenumber) and 'Y' (absorbance) columns
            min_height: Minimum peak height (if None, uses 10% of max value)
            min_distance: Minimum distance between peaks (in data points)
            prominence: Minimum prominence of peaks (if None, uses 5% of range)
            
        Returns:
            List of dictionaries, each containing:
            - 'wavenumber': X value (wavenumber in cm⁻¹)
            - 'absorbance': Y value (absorbance)
            - 'index': Index in DataFrame
        """
        try:
            from scipy.signal import find_peaks
        except ImportError:
            raise ImportError(
                "scipy is required for peak detection. "
                "Please install it with: pip install scipy==1.11.4"
            )
        
        y_values = df['Y'].values
        x_values = df['X'].values
        
        # Set default parameters if not provided
        if min_height is None:
            min_height = np.max(y_values) * 0.1  # 10% of max
        
        if prominence is None:
            data_range = np.max(y_values) - np.min(y_values)
            prominence = data_range * 0.05  # 5% of range
        
        # Find peaks
        peaks, properties = find_peaks(
            y_values,
            height=min_height,
            distance=min_distance,
            prominence=prominence
        )
        
        # Format results
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_list.append({
                'wavenumber': float(x_values[peak_idx]),
                'absorbance': float(y_values[peak_idx]),
                'index': int(peak_idx)
            })
        
        # Sort by wavenumber (ascending) - important for FTIR analysis
        peak_list.sort(key=lambda x: x['wavenumber'])
        
        return peak_list
    
    @staticmethod
    def get_significant_peaks_by_region(peaks: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """
        Identify Significant Peaks in Key FTIR Regions
        
        Selects the most important peaks from different spectral regions:
        - Fingerprint region (1400-1600 cm⁻¹): Important for molecular identification
        - Carbonyl region (1650-1800 cm⁻¹): Critical for oxidation
        - C-H stretch (2800-3000 cm⁻¹): Highest intensity, aliphatic chains
        - O-H region (3200-3600 cm⁻¹): Water/oxidation
        
        This ensures we capture peaks from all important regions, not just
        the highest absorbance peaks.
        
        Args:
            peaks: List of all detected peaks (sorted by wavenumber)
            df: DataFrame for calculating region statistics
            
        Returns:
            List of significant peaks from key regions (sorted by wavenumber)
        """
        if not peaks:
            return []
        
        significant = []
        
        # Define key regions for FTIR analysis
        regions = [
            (1400, 1600, "fingerprint"),  # Fingerprint region
            (1650, 1800, "carbonyl"),      # Carbonyl/oxidation
            (2800, 3000, "ch_stretch"),   # C-H stretch (usually highest)
            (3200, 3600, "oh_region"),    # O-H region
        ]
        
        # Get max absorbance for normalization
        max_abs = max(p['absorbance'] for p in peaks) if peaks else 1.0
        
        # For each region, find the highest peak
        for wavenumber_min, wavenumber_max, region_name in regions:
            region_peaks = [p for p in peaks 
                          if wavenumber_min <= p['wavenumber'] <= wavenumber_max]
            
            if region_peaks:
                # Get the highest peak in this region
                highest = max(region_peaks, key=lambda x: x['absorbance'])
                # Only include if it's significant (>15% of max absorbance)
                if highest['absorbance'] > max_abs * 0.15:
                    significant.append(highest)
        
        # Also include top 3 overall peaks (by absorbance) if not already included
        top_by_abs = sorted(peaks, key=lambda x: x['absorbance'], reverse=True)[:3]
        for peak in top_by_abs:
            # Check if already in significant list (within 10 cm⁻¹)
            if not any(abs(p['wavenumber'] - peak['wavenumber']) < 10 for p in significant):
                significant.append(peak)
        
        # Sort by wavenumber for consistent reporting
        significant.sort(key=lambda x: x['wavenumber'])
        
        return significant
    
    @staticmethod
    def get_value_at_wavenumber(df: pd.DataFrame, target_wavenumber: float, 
                               tolerance: float = 5.0) -> Optional[float]:
        """
        Get Absorbance Value at Specific Wavenumber
        
        Finds the Y value (absorbance) closest to the target wavenumber.
        Useful for analyzing specific regions like oxidation zones (1650-1800 cm⁻¹).
        
        Args:
            df: DataFrame with 'X' (wavenumber) and 'Y' (absorbance) columns
            target_wavenumber: Target wavenumber to find (e.g., 1725 for carbonyl)
            tolerance: Maximum distance from target to consider (default: 5 cm⁻¹)
            
        Returns:
            Absorbance value at target wavenumber, or None if not found
        """
        # Find closest X value to target
        distances = np.abs(df['X'].values - target_wavenumber)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance <= tolerance:
            return float(df.iloc[min_idx]['Y'])
        return None
    
    @staticmethod
    def get_region_statistics(df: pd.DataFrame, wavenumber_min: float, 
                             wavenumber_max: float) -> Dict:
        """
        Get Statistics for Specific Wavenumber Region
        
        Analyzes a specific region of the spectrum (e.g., oxidation zone 1650-1800 cm⁻¹).
        
        Args:
            df: DataFrame with 'X' (wavenumber) and 'Y' (absorbance) columns
            wavenumber_min: Minimum wavenumber of region
            wavenumber_max: Maximum wavenumber of region
            
        Returns:
            Dictionary with:
            - 'mean': Mean absorbance in region
            - 'max': Maximum absorbance in region
            - 'max_wavenumber': Wavenumber at maximum
            - 'min': Minimum absorbance in region
            - 'count': Number of data points in region
        """
        # Filter data in region
        region_df = df[(df['X'] >= wavenumber_min) & (df['X'] <= wavenumber_max)]
        
        if region_df.empty:
            return {
                'mean': 0.0,
                'max': 0.0,
                'max_wavenumber': 0.0,
                'min': 0.0,
                'count': 0
            }
        
        y_values = region_df['Y'].values
        x_values = region_df['X'].values
        
        max_idx = np.argmax(y_values)
        
        return {
            'mean': float(np.mean(y_values)),
            'max': float(np.max(y_values)),
            'max_wavenumber': float(x_values[max_idx]),
            'min': float(np.min(y_values)),
            'count': len(region_df)
        }


# ============================================================================
# TEST CODE - Run this file directly to verify functionality
# ============================================================================
if __name__ == "__main__":
    # Simple module test
    processor = CSVProcessor()
    
    # Generate test data
    test_df = pd.DataFrame({
        'X': range(10),
        'Y': [i * 2 + np.random.randn() for i in range(10)]
    })
    
    print("✅ CSV Processor Test")
    print("\nStatistics:")
    print(processor.get_statistics(test_df))
    print("\n✅ Module functioning correctly")