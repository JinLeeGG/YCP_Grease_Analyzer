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
    'baseline_linewidth': 2.5,        # Thickness of baseline line
    'baseline_alpha': 0.9,            # Transparency of baseline line (0=transparent, 1=opaque)
    'sample_color': 'blue',           # Color for sample data line
    'sample_linewidth': 1.8,          # Thickness of sample line
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
    'model': 'llama3.2:1b',           # Model: 1B parameters (faster than 3B, good quality)
    'timeout': 30,                    # Request timeout in seconds (reduced from 60)
    'temperature': 0.3,               # Lower = more consistent/deterministic responses
    'max_tokens': 250,                # Maximum length of generated response
    'num_predict': 250,               # Maximum tokens to generate per request
    'num_ctx': 2048,                  # Context window size (how much text model can process)
    'max_workers': 3,                 # Number of parallel threads for batch processing
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
# DEFAULT PATHS
# ============================================================================
# Default directory for saving analysis outputs (graphs, reports)
# Expands to user's Desktop folder
DEFAULT_SAVE_PATH = os.path.expanduser("~/Desktop/grease_analysis_output")