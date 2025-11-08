"""
Graph Generation and Visualization Module

This module handles all graph creation for the Grease Analyzer application:

FEATURES:
- Overlay graphs comparing baseline and sample data
- Baseline-only graphs for reference
- Consistent styling across all visualizations
- High-resolution export (300 DPI)
- Matplotlib Figure objects for flexible display

GRAPH TYPES:
1. Overlay Graph: Shows baseline (green) and sample (blue) on same axes
2. Baseline Graph: Shows only reference data

The module uses configuration from utils.config for consistent styling
(colors, line widths, transparency, grid style).
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List
import os
import sys

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import GRAPH_CONFIG


class GraphGenerator:
    """
    Graph Generation Class
    
    Creates matplotlib visualizations of spectroscopy data with
    consistent styling and formatting.
    """
    
    def __init__(self):
        """
        Initialize graph generator with configuration
        
        Sets up matplotlib parameters including:
        - Font family (sans-serif for compatibility)
        - Unicode minus sign handling
        - Loads styling from GRAPH_CONFIG
        """
        self.config = GRAPH_CONFIG
        # Configure matplotlib fonts (Unicode support)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_overlay_graph(self, 
                           baseline_df: pd.DataFrame,
                           sample_df: pd.DataFrame,
                           baseline_name: str,
                           sample_name: str) -> plt.Figure:
        """
        Create Overlay Graph Comparing Baseline and Sample
        
        Generates a dual-line graph showing both baseline (reference) and
        sample data on the same axes for easy visual comparison.
        
        Visual Design:
        - Baseline: Green line (thicker, high opacity) - reference standard
        - Sample: Blue line (thinner, high opacity) - test data
        - Both lines clearly labeled in legend
        - Grid for easier value reading
        - Professional styling applied
        
        The overlay allows immediate visual identification of:
        - Overall pattern similarity
        - Amplitude differences
        - Phase shifts
        - Peak position changes
        
        Args:
            baseline_df: Baseline DataFrame with 'X' and 'Y' columns
            sample_df: Sample DataFrame with 'X' and 'Y' columns
            baseline_name: Name/filename of baseline (for legend)
            sample_name: Name/filename of sample (for legend)
            
        Returns:
            matplotlib.figure.Figure object (not saved to file yet)
            Caller can save, display, or further modify
        """
        # Create figure and axes with configured size
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        
        # Plot baseline (green, reference line)
        ax.plot(
            baseline_df['X'],
            baseline_df['Y'],
            color=self.config['baseline_color'],
            linewidth=self.config['baseline_linewidth'],
            alpha=self.config['baseline_alpha'],
            label=f'Baseline: {baseline_name}'
        )
        
        # Plot sample (blue, test line)
        ax.plot(
            sample_df['X'],
            sample_df['Y'],
            color=self.config['sample_color'],
            linewidth=self.config['sample_linewidth'],
            alpha=self.config['sample_alpha'],
            label=f'Sample: {sample_name}'
        )
        
        # Apply consistent styling (labels, grid, legend)
        self._style_graph(ax)
        
        return fig
    
    def create_baseline_only_graph(self,
                                   baseline_df: pd.DataFrame,
                                   baseline_name: str) -> plt.Figure:
        """
        Create Baseline-Only Reference Graph
        
        Generates a single-line graph showing only the baseline data.
        Useful for:
        - Initial baseline visualization
        - Reference documentation
        - Baseline quality verification
        
        Args:
            baseline_df: Baseline DataFrame with 'X' and 'Y' columns
            baseline_name: Name/filename of baseline (for title/legend)
            
        Returns:
            matplotlib.figure.Figure object
        """
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        
        ax.plot(
            baseline_df['X'],
            baseline_df['Y'],
            color=self.config['baseline_color'],
            linewidth=self.config['baseline_linewidth'],
            alpha=self.config['baseline_alpha'],
            label=f'Baseline: {baseline_name}'
        )
        
        self._style_graph(ax)
        
        return fig
    
    def _style_graph(self, ax):
        """
        Apply Consistent Graph Styling
        
        Applies professional styling to graph axes including:
        - Axis labels with bold font
        - Grid with dashed lines and subtle transparency
        - Legend with best auto-positioning
        - Tight layout to prevent label cutoff
        
        This ensures all graphs have consistent, publication-quality appearance.
        
        Args:
            ax: matplotlib Axes object to style
        """
        # Set axis labels with bold font for emphasis
        ax.set_xlabel('Wavelength / Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
        
        # Add grid for easier value reading
        ax.grid(
            True,
            alpha=self.config['grid_alpha'],
            linestyle=self.config['grid_linestyle']
        )
        
        # Add legend with automatic best positioning
        ax.legend(
            loc='upper right',
            fontsize=10,
            framealpha=0.9,  # Semi-transparent background
            edgecolor='gray'
        )
        
        # Use tight layout to prevent axis label cutoff
        plt.tight_layout()
    
    def save_graph(self,
                   fig: plt.Figure,
                   save_path: str,
                   filename: str,
                   format: str = 'png') -> bool:
        """
        Save Graph to File
        
        Exports matplotlib Figure to image file with high resolution.
        Creates directory if it doesn't exist.
        
        Args:
            fig: matplotlib Figure object to save
            save_path: Directory path where file will be saved
            filename: Filename without extension (e.g., "sample_01")
            format: Image format ('png', 'jpg', 'jpeg', 'pdf', 'svg')
                   Default: 'png' (recommended for quality)
            
        Returns:
            True if save successful, False if error occurred
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Construct full file path
            full_path = os.path.join(save_path, f"{filename}.{format}")
            
            # Save with high DPI and tight bounding box
            fig.savefig(
                full_path,
                dpi=self.config['dpi'],          # 300 DPI for print quality
                bbox_inches='tight',              # Remove whitespace margins
                format=format
            )
            
            # Release memory (important for batch operations)
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"❌ Graph save failed: {str(e)}")
            return False
    
    def batch_save_graphs(self,
                         baseline_df: pd.DataFrame,
                         samples_dict: dict,
                         baseline_name: str,
                         save_path: str,
                         format: str = 'png') -> List[str]:
        """
        Batch Generate and Save Graphs for All Samples
        
        Creates overlay graphs for multiple samples against the same baseline
        and saves them all to the specified directory. Useful for:
        - Exporting all analysis results at once
        - Creating report attachments
        - Batch documentation
        
        Args:
            baseline_df: Baseline DataFrame with 'X' and 'Y' columns
            samples_dict: Dictionary mapping sample names to DataFrames
                         Example: {'sample_01.csv': df1, 'sample_02.csv': df2}
            baseline_name: Name of baseline file (for graph labels)
            save_path: Directory where graphs will be saved
            format: Image format ('png', 'jpg', 'jpeg', 'pdf', 'svg')
            
        Returns:
            List of full file paths to saved graphs
            Empty list if all saves failed
        """
        saved_files = []
        
        for sample_name, sample_df in samples_dict.items():
            # Generate overlay graph
            fig = self.create_overlay_graph(
                baseline_df,
                sample_df,
                baseline_name,
                sample_name
            )
            
            # Create clean filename (remove .csv extension)
            clean_sample_name = sample_name.replace('.csv', '')
            filename = f"{clean_sample_name}_vs_baseline"
            
            # Save graph
            success = self.save_graph(fig, save_path, filename, format)
            
            if success:
                full_path = os.path.join(save_path, f"{filename}.{format}")
                saved_files.append(full_path)
        
        return saved_files


# ============================================================================
# TEST CODE - Run this file directly to verify functionality
# ============================================================================
if __name__ == "__main__":
    import numpy as np
    
    print("✅ Graph Generator Testing")
    
    # Test Datas
    baseline = pd.DataFrame({
        'X': np.linspace(0, 10, 100),
        'Y': np.sin(np.linspace(0, 10, 100))
    })
    
    sample = pd.DataFrame({
        'X': np.linspace(0, 10, 100),
        'Y': np.sin(np.linspace(0, 10, 100)) + 0.2
    })
    
    # Graph generating
    gen = GraphGenerator()
    fig = gen.create_overlay_graph(
        baseline, sample,
        "baseline_test.csv",
        "sample_test.csv"
    )
    
    print("✅ Graph generating Success")
    print("✅ Module Generating success")
    
    plt.close('all')