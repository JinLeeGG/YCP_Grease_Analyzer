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