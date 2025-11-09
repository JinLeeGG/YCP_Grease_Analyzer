"""
Configuration Constants for Grease Analyzer Project

This module defines all application-wide configuration settings including:
- Graph visualization parameters (colors, line styles, sizes)
- CSV file validation rules and limits
- LLM (Large Language Model) settings for AI analysis
- PDF export configuration
- File format support and default paths

All settings are centralized here for easy maintenance and consistency.
"""

import os

# ============================================================================
# GRAPH CONFIGURATION
# ============================================================================
# Settings for matplotlib visualizations of baseline and sample data
# Controls appearance, styling, and layout of all generated graphs
GRAPH_CONFIG = {
    'figsize': (12, 7),              # Figure dimensions in inches (width, height)
    'dpi': 300,                       # Resolution for saved images (dots per inch)
    'baseline_color': 'green',        # Color for baseline data line
    'baseline_linewidth': 1.0,        # Thickness of baseline line
    'baseline_alpha': 0.9,            # Transparency of baseline line (0=transparent, 1=opaque)
    'sample_color': 'blue',           # Color for sample data line
    'sample_linewidth': 1.0,          # Thickness of sample line
    'sample_alpha': 0.85,             # Transparency of sample line
    'grid_alpha': 0.3,                # Transparency of grid lines
    'grid_linestyle': '--',           # Grid line style (dashed)
}

# ============================================================================
# CSV VALIDATION CONFIGURATION
# ============================================================================
# Rules for validating uploaded CSV files to ensure data quality
# Prevents processing of corrupted or incompatible files
CSV_CONFIG = {
    'max_file_size_mb': 50,           # Maximum allowed file size in megabytes
    'required_columns': 2,            # Minimum number of columns (X, Y coordinates)
    'min_rows': 10,                   # Minimum data points required for analysis
}

# ============================================================================
# LLM (LARGE LANGUAGE MODEL) CONFIGURATION
# ============================================================================
# Optimized settings for local Ollama LLM analysis
# Balanced for speed (3-5x improvement) while maintaining quality
# Uses parallel processing for analyzing multiple samples simultaneously
LLM_CONFIG = {
    'model': 'llava:7b-v1.6',         # Model: LLaVA 7B (optimized for vision/text)
    'timeout': 60,                    # Request timeout in seconds
    'temperature': 0.1,               # Lower = more consistent/deterministic responses
    'max_tokens': 1000,               # Maximum length of generated response
    'num_predict': 1000,              # Maximum tokens to generate per request
    'num_ctx': 2048,                  # Context window size
    'max_workers': 3,                 # Number of parallel threads for batch processing
    'use_llm_enhancement': True,      # Enable optional LLM enhancement (can be disabled for speed)
}

# ============================================================================
# FTIR ANALYZER CONFIGURATION (NEW)
# ============================================================================
# Settings for the optimized FTIRAnalyzer - production-ready numerical analysis
# Provides <1s analysis time with statistical rigor
FTIR_CONFIG = {
    # Peak detection thresholds
    'sigma_multiplier': 3.0,          # Statistical significance (3σ standard)
    'prominence_pct': 0.10,           # 10% of global max for prominence
    'major_height_pct': 0.10,         # 10% of global max for major peaks
    
    # Peak matching tolerance (wavenumber units: cm⁻¹)
    'match_tolerance': 10.0,          # ±10 cm⁻¹ default matching tolerance
    'match_tolerance_loose': 20.0,    # ±20 cm⁻¹ for low resolution instruments
    'shift_notable': 5.0,             # >5 cm⁻¹ shift is notable
    'shift_significant': 10.0,        # >10 cm⁻¹ shift is significant
    
    # Critical regions for grease analysis (wavenumber ranges in cm⁻¹)
    'oxidation_region': (1650, 1800), # Carbonyl/oxidation primary indicator
    'water_region': (3200, 3600),     # O-H stretch/water contamination
    'glycol_region': (1000, 1300),    # Additives/glycol contamination
    'ch_region': (2850, 2950),        # C-H stretch (internal reference)
    
    # Decision thresholds (percentage changes)
    'oxidation_critical': 0.50,       # >50% increase = critical, replace immediately
    'oxidation_moderate': 0.30,       # 30-50% increase = moderate, schedule maintenance
    'oxidation_low': 0.10,            # 10-30% increase = low, monitor closely
    'water_threshold': 0.12,          # Absorbance threshold for water detection
    
    # Preprocessing parameters
    'smooth_window': 7,               # Savitzky-Golay smoothing window
    'smooth_poly': 3,                 # Polynomial order for smoothing
    'noise_quiet_region': (2200, 2400),  # Quiet region for noise estimation
}

# ============================================================================
# ANALYSIS MODE CONFIGURATION (NEW)
# ============================================================================
# Control the analysis pipeline behavior
ANALYSIS_MODE = {
    'use_ftir_analyzer': True,        # Use optimized FTIRAnalyzer (recommended)
    'use_llm_enhancement': True,      # Enhance with LLM (optional, slower)
    'fallback_to_ftir': True,         # Always fallback to FTIR if LLM fails
    'parallel_processing': True,      # Enable parallel batch processing
}

# ============================================================================
# PDF EXPORT CONFIGURATION
# ============================================================================
# Settings for generating analysis reports in PDF format
# Defines page layout, metadata, and margins
PDF_CONFIG = {
    'page_size': 'letter',            # Paper size: 'letter' (US) or 'A4' (international)
    'title': 'Grease Analysis Report',    # Report title
    'subtitle': 'Schneider Prize 2025',   # Report subtitle
    'author': 'Your Team Name',           # Author metadata
    'margins': {                          # Page margins in points (1 inch = 72 points)
        'top': 72,
        'bottom': 72,
        'left': 72,
        'right': 72,
    }
}

# ============================================================================
# FILE FORMAT SUPPORT
# ============================================================================
# Supported image formats for graph exports
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg']

# ============================================================================
# EXPORT SETTINGS
# ============================================================================
# User-configurable export settings (can be changed at runtime)
EXPORT_SETTINGS = {
    'save_directory': '',             # No default - user must set this before saving
    'image_format': 'png',            # Default image format: 'png' or 'jpg'
}

# ============================================================================
# DEFAULT PATHS
# ============================================================================
# Default directory for saving analysis outputs (graphs, reports)
# User must configure this before saving graphs
DEFAULT_SAVE_PATH = ''