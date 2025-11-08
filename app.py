"""
Grease Analyzer - PyQt6 Desktop Application

MAIN APPLICATION FILE:
This is the entry point for the Grease Analyzer desktop application.
It provides a graphical user interface for analyzing grease spectroscopy data
by comparing sample measurements against baseline references.

KEY FEATURES:
- Load baseline (reference) and multiple sample CSV files
- Visualize data overlays with interactive graphs
- AI-powered analysis using local LLM (Ollama)
- Parallel processing for fast batch analysis
- Export graphs and generate reports

ARCHITECTURE:
- AnalysisWorker: QThread for non-blocking LLM analysis in background
- GreaseAnalyzerApp: Main window class managing UI and data flow
- Integration with: CSV processor, graph generator, LLM analyzer

USAGE:
Run this file directly to launch the application:
    python app_pyqt.py
"""

from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import project modules
from modules.csv_processor import CSVProcessor
from modules.graph_generator import GraphGenerator
from modules.llm_analyzer import LLMAnalyzer
from utils.config import LLM_CONFIG, EXPORT_SETTINGS, SUPPORTED_FORMATS


class AnalysisWorker(QThread):
    """
    Background Worker Thread for LLM Analysis
    
    This QThread subclass runs AI analysis in the background to prevent
    UI freezing during long-running LLM operations. It processes multiple
    samples in parallel and emits progress signals for UI updates.
    
    Signals:
        progress(int): Progress percentage (0-100)
        status(str): Current operation status message
        finished(dict): Analysis results when complete
        error(str): Error message if analysis fails
    """
    
    # Define Qt signals for thread-safe communication with main thread
    progress = pyqtSignal(int)      # Progress bar updates (0-100)
    status = pyqtSignal(str)        # Status message updates
    finished = pyqtSignal(dict)     # Success: returns analysis results
    error = pyqtSignal(str)         # Failure: returns error message
    
    def __init__(self, analyzer: LLMAnalyzer, samples_data: List[Dict]):
        """
        Initialize the worker thread
        
        Args:
            analyzer: LLMAnalyzer instance with Ollama connection
            samples_data: List of sample dictionaries containing:
                - baseline_stats: Statistical measures of baseline
                - sample_stats: Statistical measures of sample
                - comparison: Deviation metrics between baseline and sample
                - baseline_name: Name of baseline file
                - sample_name: Name of sample file
        """
        super().__init__()
        self.analyzer = analyzer
        self.samples_data = samples_data
        self._is_running = True
    
    def run(self):
        """
        Execute the analysis (called when thread starts)
        
        This method runs in a separate thread. It:
        1. Analyzes each sample using parallel processing
        2. Generates an executive summary of all results
        3. Emits progress updates throughout
        4. Returns results via finished signal or error via error signal
        """
        try:
            self.status.emit("🚀 Starting parallel analysis...")
            self.progress.emit(10)
            
            # Run parallel batch analysis on all samples
            results = self.analyzer.analyze_samples_batch(self.samples_data)
            
            self.progress.emit(80)
            self.status.emit("📝 Generating summary...")
            
            # Generate overall summary from individual results
            summary = self.analyzer.generate_summary(results)
            
            self.progress.emit(100)
            self.status.emit("✅ Analysis complete!")
            
            # Emit results to main thread
            self.finished.emit({
                'individual_results': results,
                'summary': summary
            })
            
        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")
    
    def stop(self):
        """Stop the worker thread gracefully"""
        self._is_running = False


class GreaseAnalyzerApp(QMainWindow):
    """
    Main Application Window for Grease Analyzer
    
    This class manages the entire application UI and coordinates between
    different modules (CSV processor, graph generator, LLM analyzer).
    
    WORKFLOW:
    1. User uploads baseline CSV file
    2. User uploads one or more sample CSV files
    3. Application displays overlay graphs comparing each sample to baseline
    4. User triggers AI analysis (runs in background thread)
    5. Results are displayed in summary panel
    6. User can export graphs or generate PDF reports
    
    DATA MANAGEMENT:
    - baseline_data: Reference dataset (DataFrame)
    - sample_data_list: List of sample datasets with metadata
    - analysis_results: AI-generated insights for each sample
    - current_graph_path: Path to currently displayed graph image
    """
    
    def __init__(self):
        """
        Initialize the main application window
        
        Loads UI from .ui file, initializes all modules,
        sets up data structures, and connects signals.
        """
        super().__init__()
        
        # Load UI layout from Qt Designer file
        ui_path = Path(__file__).parent / "Analyzer_main.ui"
        uic.loadUi(ui_path, self)
        
        # Initialize data storage structures
        self.baseline_data: Optional[pd.DataFrame] = None  # Reference dataset
        self.baseline_name: str = ""                        # Baseline filename
        self.sample_data_list: List[Dict] = []              # List of: {'name': str, 'data': DataFrame, 'stats': dict}
        self.current_sample_index: int = 0                  # Currently displayed sample
        self.analysis_results: Dict = {}                    # LLM analysis results
        self.current_graph_path: Optional[str] = None       # Path to displayed graph image
        self.current_figure = None                          # Current matplotlib figure for canvas
        
        # Export settings
        self.save_directory: str = EXPORT_SETTINGS['save_directory']  # Directory for saving graphs
        self.image_format: str = EXPORT_SETTINGS['image_format']      # Image format (png/jpg)
        
        # Initialize module instances
        self.csv_processor = CSVProcessor()          # CSV file loading and validation
        self.graph_generator = GraphGenerator()      # Matplotlib graph creation
        self.llm_analyzer = LLMAnalyzer()           # Local LLM (Ollama) for AI analysis
        
        # Background worker thread
        self.analysis_worker: Optional[AnalysisWorker] = None
        
        # Set up UI components and connect signals
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """
        Initialize UI component states
        
        Sets default values, disables buttons until data is loaded,
        and configures display areas for graphs and text.
        """
        self.setWindowTitle("Grease Analyzer - PyQt6 Edition")
        
        # Replace the QLabel "display" with matplotlib FigureCanvas for interactive graphs
        self.setup_matplotlib_canvas()
        
        # Set initial status messages
        self.status_inf.setText("STATUS: Ready to analyze data")
        self.aiSummaryText.setText(
            "No summary generated yet.\n"
            "1. Upload baseline data\n"
            "2. Upload sample data\n"
            "3. Click 'Generate Summary' to analyze"
        )
        
        # Initialize progress bars to 0%
        self.uploadProgress.setValue(0)
        self.aiProgress.setValue(0)
        
        # Disable buttons until baseline is loaded
        self.btn_current_filter.setEnabled(False)  # Upload samples button
        self.btn_invert.setEnabled(False)          # Generate analysis button
        self.btn_export_current.setEnabled(False)  # Export current graph button
        self.btn_export_all.setEnabled(False)      # Export all graphs button
        
        # Initialize sample dropdown menu
        self.comboBox.clear()
        self.comboBox.addItem("No samples loaded")
        
        # Display AI model configuration information
        model_name = LLM_CONFIG['model'].replace('llava:', 'LLaVA ').replace('-q4_K_M', ' Q4')
        self.aiModelInfo.setText(f"Model: {model_name} (Parallel: {LLM_CONFIG['max_workers']} workers)")
        
        # Update export info label
        self.update_export_info()
        
        # Make export info label clickable (opens directory settings)
        self.exportInfo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exportInfo.mousePressEvent = lambda event: self.change_save_directory()
        
        # Set default splitter sizes
        self.set_default_splitter_sizes()
    
    def setup_matplotlib_canvas(self):
        """
        Setup Interactive Matplotlib Canvas with Zoom/Pan Controls
        
        Replaces the QLabel "display" widget with a matplotlib FigureCanvas
        that provides built-in interactive features:
        - Zoom: Click and drag to create zoom rectangle
        - Pan: Right-click and drag to pan around
        - Navigation toolbar: Home, Back, Forward, Pan, Zoom, Save buttons
        """
        # Remove the old QLabel widget
        old_display = self.display
        display_layout = old_display.parent().layout()
        
        # Create matplotlib figure and canvas
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(8, 6), facecolor='#1a202c')  # Dark background
        self.canvas = FigureCanvas(self.figure)
        
        # Create navigation toolbar for zoom/pan controls
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Style the toolbar to match dark theme
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: rgb(38, 52, 66);
                border: 2px solid rgb(50, 68, 85);
                border-radius: 5px;
                spacing: 3px;
                padding: 3px;
            }
            QToolButton {
                background: rgb(38, 52, 66);
                border: 1px solid rgb(50, 68, 85);
                border-radius: 3px;
                color: rgb(220, 225, 230);
                padding: 5px;
            }
            QToolButton:hover {
                background: rgb(85, 105, 75);
                border: 1px solid rgb(100, 130, 80);
            }
            QToolButton:pressed {
                background: rgb(70, 90, 60);
            }
        """)
        
        # Remove old widget and add canvas with toolbar
        display_layout.removeWidget(old_display)
        old_display.deleteLater()
        
        # Add toolbar and canvas to the layout
        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(self.canvas)
        
        # Store reference for later use
        self.display = self.canvas
        
    def set_default_splitter_sizes(self):
        """
        Set Default Splitter Sizes for Initial Layout
        
        Configures the splitters to show:
        - Left panel (control panel): Minimized to ~320px
        - Right panel (visualization + AI): Maximized
        - Data visualization: Maximized vertically
        - AI Summary: Minimized to ~200px at bottom
        """
        # Get total window width and height
        total_width = self.width()
        total_height = self.height()
        
        # Horizontal splitter: Left panel small (320px), right panel gets remaining space
        left_panel_width = 320
        right_panel_width = total_width - left_panel_width - 4  # 4px for splitter handle
        self.mainSplitter.setSizes([left_panel_width, right_panel_width])
        
        # Vertical splitter: Visualization large, AI summary small (200px)
        ai_summary_height = 200
        visualization_height = total_height - ai_summary_height - 100  # Account for margins/menubar
        self.verticalSplitter.setSizes([visualization_height, ai_summary_height])
        
    def connect_signals(self):
        """
        Connect UI signals to slot methods
        
        Maps user interactions (button clicks, menu selections, combo box changes)
        to their corresponding handler methods. This is the Qt signal-slot pattern.
        """
        # Button click handlers
        self.btn_save.clicked.connect(self.upload_baseline)
        self.btn_current_filter.clicked.connect(self.upload_samples)
        self.btn_invert.clicked.connect(self.generate_analysis)
        
        # Export button handlers
        self.btn_export_current.clicked.connect(self.save_current_graph)
        self.btn_export_all.clicked.connect(self.save_all_graphs)
        
        # Dropdown selection handler
        self.comboBox.currentIndexChanged.connect(self.on_sample_changed)
        
        # Menu action handlers
        self.actionUpload_BaseLine.triggered.connect(self.upload_baseline)
        self.actionUpload_Samples.triggered.connect(self.upload_samples)
        self.actionSave_Current_Graph.triggered.connect(self.save_current_graph)
        self.actionSave_All_Sample_Graph.triggered.connect(self.save_all_graphs)
        self.actionChangeDirectory.triggered.connect(self.change_save_directory)
        self.actionExit.triggered.connect(self.close)
        self.actionDocumentation.triggered.connect(self.show_documentation)
        self.actionAbout.triggered.connect(self.show_about)
        
    def upload_baseline(self):
        """
        Upload and Load Baseline Reference Data
        
        Opens a file dialog for user to select a baseline CSV file.
        The baseline serves as the reference standard that all samples
        will be compared against. After loading:
        - Validates the CSV file format
        - Stores the data in self.baseline_data
        - Enables the sample upload button
        - Updates progress bar and status
        
        Returns:
            None. Updates UI state and self.baseline_data on success.
        """
        # Open file selection dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Baseline Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.uploadProgress.setValue(20)
            self.status_inf.setText("STATUS: Loading baseline data...")
            
            # Load CSV file using CSV processor module
            df, error = self.csv_processor.load_csv(file_path)
            
            if error:
                raise Exception(error)
            
            self.uploadProgress.setValue(60)
            
            # Store baseline data and filename
            self.baseline_data = df
            self.baseline_name = Path(file_path).name
            
            self.uploadProgress.setValue(100)
            self.status_inf.setText(f"STATUS: Baseline loaded - {self.baseline_name}")
            
            # Enable sample upload button now that baseline exists
            self.btn_current_filter.setEnabled(True)
            
            # If samples were already loaded, refresh the display with new baseline
            if self.sample_data_list:
                print(f"🔄 Refreshing graphs with new baseline: {self.baseline_name}")
                self.display_current_sample()
            
            # Show success message with data info
            QMessageBox.information(
                self,
                "Success",
                f"Baseline loaded successfully!\n{self.baseline_name}\n"
                f"Records: {len(df)}"
            )
            
        except Exception as e:
            # Handle errors: reset progress and show error dialog
            self.uploadProgress.setValue(0)
            self.status_inf.setText("STATUS: Error loading baseline")
            QMessageBox.critical(self, "Error", f"Failed to load baseline:\n{str(e)}")
    
    def upload_samples(self):
        """
        Upload and Load Sample Data Files (Multiple Selection Supported)
        
        Opens a file dialog allowing user to select one or more sample CSV files.
        Each sample is:
        - Validated and loaded
        - Compared against the baseline
        - Statistical analysis performed
        - Added to the sample list for display
        
        Requires baseline to be loaded first. Updates combo box and displays
        first sample automatically after loading.
        
        Returns:
            None. Updates self.sample_data_list and UI on success.
        """
        # Open multi-file selection dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Upload Sample Data (Multiple Selection)",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        # Ensure baseline exists before loading samples
        if self.baseline_data is None:
            QMessageBox.warning(self, "Warning", "Please upload baseline data first!")
            return
        
        try:
            total_files = len(file_paths)
            self.sample_data_list.clear()  # Clear previous samples
            
            # Process each selected file
            for i, file_path in enumerate(file_paths):
                # Update progress bar
                progress = int((i + 1) / total_files * 100)
                self.uploadProgress.setValue(progress)
                self.status_inf.setText(f"STATUS: Loading sample {i+1}/{total_files}...")
                
                # Load CSV file
                df, error = self.csv_processor.load_csv(file_path)
                
                if error:
                    # Skip files that fail to load
                    print(f"⚠️ Skipping {Path(file_path).name}: {error}")
                    continue
                
                # Calculate statistics for this sample
                stats = self.csv_processor.calculate_statistics(df)
                
                # Compare sample against baseline
                baseline_stats = self.csv_processor.calculate_statistics(self.baseline_data)
                comparison = self.csv_processor.compare_with_baseline(
                    self.baseline_data,
                    df
                )
                
                # Add sample to list with all metadata
                sample_name = Path(file_path).name
                self.sample_data_list.append({
                    'name': sample_name,
                    'data': df,
                    'stats': stats,
                    'comparison': comparison,
                    'baseline_stats': baseline_stats
                })
            
            self.uploadProgress.setValue(100)
            self.status_inf.setText(f"STATUS: {len(self.sample_data_list)} samples loaded")
            
            # Update combo box
            self.update_sample_combobox()
            
            # Display first sample automatically
            if self.sample_data_list:
                self.current_sample_index = 0
                # Force immediate graph display
                QApplication.processEvents()  # Process any pending events first
                self.display_current_sample()
                QApplication.processEvents()  # Ensure graph is rendered
                self.btn_invert.setEnabled(True)
                self.btn_export_current.setEnabled(True)  # Enable export buttons
                self.btn_export_all.setEnabled(True)
                print(f"✅ Auto-displayed first sample: {self.sample_data_list[0]['name']}")
            
            QMessageBox.information(
                self,
                "Success",
                f"{len(self.sample_data_list)} samples loaded successfully!"
            )
            
        except Exception as e:
            self.uploadProgress.setValue(0)
            self.status_inf.setText("STATUS: Error loading samples")
            QMessageBox.critical(self, "Error", f"Failed to load samples:\n{str(e)}")
    
    def update_sample_combobox(self):
        """
        Update Sample Dropdown Menu
        
        Populates the combo box with names of all loaded samples.
        Called after samples are uploaded.
        """
        self.comboBox.clear()
        
        for sample in self.sample_data_list:
            self.comboBox.addItem(sample['name'])
    
    def on_sample_changed(self, index: int):
        """
        Handle Sample Selection Change
        
        Called when user selects a different sample from dropdown.
        Displays the graph for the newly selected sample.
        
        Args:
            index: Index of selected sample in combo box
        """
        if 0 <= index < len(self.sample_data_list):
            self.current_sample_index = index
            self.display_current_sample()
    
    def display_current_sample(self):
        """
        Display Graph for Currently Selected Sample
        
        Generates an overlay graph comparing the selected sample against
        the baseline. Steps:
        1. Creates matplotlib figure with both datasets
        2. Saves figure to temporary file
        3. Loads as QPixmap and displays in UI
        4. Updates sample info label with statistics
        
        Uses temporary files for graph display to avoid memory issues
        with repeated matplotlib operations in Qt.
        """
        if not self.sample_data_list:
            print("⚠️ Cannot display: No samples loaded")
            return
        
        if self.baseline_data is None:
            print("⚠️ Cannot display: No baseline loaded")
            return
        
        try:
            sample = self.sample_data_list[self.current_sample_index]
            print(f"📊 Displaying graph for sample: {sample['name']}")
            
            # Update sample information label
            records = len(sample['data'])
            self.sampleInfo.setText(f"Records: {records} | Quality: {sample['comparison']['quality_score']:.1f}/100")
            
            # Generate overlay graph (returns matplotlib Figure object)
            fig = self.graph_generator.create_overlay_graph(
                self.baseline_data,
                sample['data'],
                self.baseline_name,
                sample['name']
            )
            
            print(f"✅ Graph generated successfully")
            
            # Store the figure for export functionality
            self.current_figure = fig
            
            # Display the graph directly on canvas (interactive with zoom/pan)
            self.update_graph_display(fig)
            print(f"✅ Graph displayed on canvas")
            self.status_inf.setText(f"STATUS: Displaying {sample['name']}")
            
        except Exception as e:
            print(f"❌ Display error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_inf.setText("STATUS: Error displaying graph")
            QMessageBox.critical(self, "Display Error", f"Failed to display graph:\n{str(e)}")
    
    def update_graph_display(self, fig=None):
        """
        Update Graph Display on Canvas
        
        Displays the matplotlib figure on the interactive canvas.
        The canvas provides built-in zoom and pan functionality.
        
        Args:
            fig: matplotlib Figure object to display (optional, uses current if None)
        """
        if fig is None:
            fig = self.current_figure
            
        if fig is None:
            return
            
        # Clear the current canvas
        self.figure.clear()
        
        # Copy the axes from the generated figure to our canvas figure
        for ax_src in fig.get_axes():
            ax_dest = self.figure.add_subplot(111)
            
            # Copy all lines and their properties
            for line in ax_src.get_lines():
                ax_dest.plot(line.get_xdata(), line.get_ydata(),
                           color=line.get_color(),
                           linewidth=line.get_linewidth(),
                           alpha=line.get_alpha(),
                           label=line.get_label())
            
            # Copy axis labels and title
            ax_dest.set_xlabel(ax_src.get_xlabel(), fontsize=12, color='black')
            ax_dest.set_ylabel(ax_src.get_ylabel(), fontsize=12, color='black')
            ax_dest.set_title(ax_src.get_title(), fontsize=14, fontweight='bold', color='black')
            
            # Copy legend
            if ax_src.get_legend():
                ax_dest.legend(loc='best', framealpha=0.9, fontsize=10)
            
            # Copy grid settings from source
            if len(ax_src.get_xgridlines()) > 0:
                # Grid exists in source, enable it with same style
                ax_dest.grid(True, alpha=0.3, linestyle='--')
            
            # Set axis limits
            ax_dest.set_xlim(ax_src.get_xlim())
            ax_dest.set_ylim(ax_src.get_ylim())
            
            # Style for white background
            ax_dest.set_facecolor('white')
            ax_dest.tick_params(colors='black', labelsize=10)
            for spine in ax_dest.spines.values():
                spine.set_edgecolor('black')
        
        # Update the canvas figure background
        self.figure.patch.set_facecolor('white')
        self.figure.tight_layout()
        
        # Refresh the canvas to show the new graph
        self.canvas.draw()
        
        # Close the original figure to free memory
        plt.close(fig)
    
    def generate_analysis(self):
        """
        Generate AI Analysis Using LLM (Parallel Processing)
        
        Triggers LLM-based analysis of all loaded samples. Creates a worker
        thread to run analysis in background without blocking UI.
        
        Process:
        1. Prepares sample data for LLM prompt
        2. Creates AnalysisWorker thread
        3. Connects progress/status signals
        4. Starts parallel batch analysis
        5. Results handled in on_analysis_finished callback
        
        The analysis runs 3 samples simultaneously (max_workers=3) for speed.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No samples to analyze!")
            return
        
        # Prepare analysis data for each sample
        samples_data = []
        for sample in self.sample_data_list:
            samples_data.append({
                'baseline_df': self.baseline_data,  # Pass DataFrame for peak analysis
                'sample_df': sample['data'],         # Pass DataFrame for peak analysis
                'baseline_stats': sample['baseline_stats'],
                'sample_stats': sample['stats'],
                'comparison': sample['comparison'],
                'baseline_name': self.baseline_name,
                'sample_name': sample['name']
            })
        
        # Start worker thread for background analysis
        self.aiProgress.setValue(0)
        self.aiSummaryText.setText("🔄 Analyzing... Please wait.")
        self.btn_invert.setEnabled(False)  # Disable button during analysis
        
        # Create and configure worker thread
        self.analysis_worker = AnalysisWorker(self.llm_analyzer, samples_data)
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.status.connect(self.on_analysis_status)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()
    
    def on_analysis_progress(self, value: int):
        """
        Update Analysis Progress Bar
        
        Callback for progress signal from worker thread.
        Updates the progress bar in UI thread-safely.
        
        Args:
            value: Progress percentage (0-100)
        """
        self.aiProgress.setValue(value)
    
    def on_analysis_status(self, message: str):
        """
        Update Analysis Status Message
        
        Callback for status signal from worker thread.
        Displays current operation status to user.
        
        Args:
            message: Status message string
        """
        self.status_inf.setText(f"STATUS: {message}")
    
    def on_analysis_finished(self, results: Dict):
        """
        Handle Completed Analysis Results
        
        Callback when worker thread finishes successfully.
        Formats and displays the AI-generated analysis including:
        - Executive summary (overall assessment)
        - Individual sample analyses
        - Statistical metrics for each sample
        
        Args:
            results: Dictionary containing:
                - 'summary': Overall summary text
                - 'individual_results': Dict of sample_name -> analysis_text
        """
        self.analysis_results = results
        
        # Format and display results
        summary_text = "🤖 AI Analysis Results\n"
        summary_text += "=" * 60 + "\n\n"
        summary_text += "📋 Executive Summary\n\n"
        summary_text += results['summary'] + "\n\n"
        summary_text += "=" * 60 + "\n\n"
        
        # Add individual sample analyses with metrics
        for sample_name, analysis in results['individual_results'].items():
            sample_info = next((s for s in self.sample_data_list if s['name'] == sample_name), None)
            if sample_info:
                summary_text += f"📊 {sample_name}\n\n"
                summary_text += analysis + "\n\n"
                summary_text += f"Quality Score: {sample_info['comparison']['quality_score']:.1f}/100\n"
                summary_text += f"Mean Deviation: {sample_info['comparison']['mean_deviation_percent']:+.1f}%\n"
                summary_text += f"Correlation: {sample_info['comparison']['correlation']:.3f}\n"
                summary_text += f"Std Dev Change: {sample_info['comparison']['std_deviation_percent']:+.1f}%\n"
                summary_text += "\n" + "-" * 60 + "\n\n"
        
        self.aiSummaryText.setText(summary_text)
        self.aiProgress.setValue(100)
        self.btn_invert.setEnabled(True)  # Re-enable analysis button
        
        QMessageBox.information(self, "Success", "AI analysis completed!")
    
    def on_analysis_error(self, error_msg: str):
        """
        Handle Analysis Error
        
        Callback when worker thread encounters an error.
        Displays error message and resets UI state.
        
        Args:
            error_msg: Error description string
        """
        self.aiSummaryText.setText(f"❌ Error: {error_msg}")
        self.aiProgress.setValue(0)
        self.btn_invert.setEnabled(True)  # Re-enable button
        QMessageBox.critical(self, "Error", error_msg)
    
    def save_current_graph(self):
        """
        Save Currently Displayed Graph
        
        Saves the currently displayed graph to the designated directory
        using the configured image format (PNG or JPG).
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No graph to save!")
            return
        
        # Check if save directory has been configured
        if not self.save_directory or self.save_directory == '':
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please set up the destination path first!\n\n"
                "Go to: Export → Change Save Directory"
            )
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        try:
            # Regenerate graph with high quality
            sample = self.sample_data_list[self.current_sample_index]
            fig = self.graph_generator.create_overlay_graph(
                self.baseline_data,
                sample['data'],
                self.baseline_name,
                sample['name']
            )
            
            # Prepare filename with configured format
            base_name = sample['name'].rsplit('.', 1)[0] if '.' in sample['name'] else sample['name']
            filename = f"{base_name}_vs_baseline.{self.image_format}"
            file_path = os.path.join(self.save_directory, filename)
            
            # Save with appropriate settings for the format
            if self.image_format == 'jpg':
                fig.savefig(file_path, dpi=300, bbox_inches='tight', format='jpg', quality=95)
            else:
                fig.savefig(file_path, dpi=300, bbox_inches='tight', format='png')
            
            plt.close(fig)
            QMessageBox.information(
                self, 
                "Success", 
                f"Graph saved as {self.image_format.upper()}:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph:\n{str(e)}")
    
    def save_all_graphs(self):
        """
        Save All Sample Overlay Graphs
        
        Generates and saves overlay graphs (sample vs baseline) for all loaded samples.
        Uses the pre-configured save directory and image format.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No graphs to save!")
            return
        
        if self.baseline_data is None:
            QMessageBox.warning(self, "Warning", "No baseline data loaded!")
            return
        
        # Check if save directory has been configured
        if not self.save_directory or self.save_directory == '':
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please set up the destination path first!\n\n"
                "Go to: Export → Change Save Directory"
            )
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        try:
            saved_count = 0
            
            # Save all sample overlay graphs
            for sample in self.sample_data_list:
                # Generate overlay graph for this sample
                fig = self.graph_generator.create_overlay_graph(
                    self.baseline_data,
                    sample['data'],
                    self.baseline_name,
                    sample['name']
                )
                
                # Prepare filename with configured format
                base_name = sample['name'].rsplit('.', 1)[0] if '.' in sample['name'] else sample['name']
                graph_path = os.path.join(self.save_directory, f"{base_name}_vs_baseline.{self.image_format}")
                
                # Save with appropriate settings for the format
                if self.image_format == 'jpg':
                    fig.savefig(graph_path, dpi=300, bbox_inches='tight', format='jpg', quality=95)
                else:
                    fig.savefig(graph_path, dpi=300, bbox_inches='tight', format='png')
                
                plt.close(fig)  # Release memory
                saved_count += 1
            
            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"{saved_count} overlay graphs saved as {self.image_format.upper()} to:\n{self.save_directory}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graphs:\n{str(e)}")
    
    def change_save_directory(self):
        """
        Change Default Save Directory and Image Format
        
        Opens a custom dialog allowing user to:
        1. Select a directory for saving graphs
        2. Choose image format (PNG or JPG)
        
        These settings are used by save_current_graph and save_all_graphs.
        The settings persist for the current session.
        """
        # Load the custom UI dialog
        dialog_path = Path(__file__).parent / "path.ui"
        dialog = uic.loadUi(dialog_path)
        
        # Set current values in the dialog - show "..." if not set
        if self.save_directory and self.save_directory != '':
            dialog.pathDisplay.setText(self.save_directory)
        else:
            dialog.pathDisplay.setText("...")
        
        # Set format combobox
        if self.image_format == 'png':
            dialog.formatComboBox.setCurrentIndex(0)
        else:
            dialog.formatComboBox.setCurrentIndex(1)
        
        # Connect browse button
        def browse_directory():
            # Start from user's home if no directory set
            start_dir = self.save_directory if self.save_directory else str(Path.home())
            directory = QFileDialog.getExistingDirectory(
                dialog,
                "Select Save Directory",
                start_dir
            )
            if directory:
                dialog.pathDisplay.setText(directory)
        
        dialog.browseButton.clicked.connect(browse_directory)
        
        # Connect OK button
        def apply_settings():
            # Get selected directory
            new_directory = dialog.pathDisplay.text()
            if not new_directory or new_directory == '...':
                QMessageBox.warning(dialog, "Warning", "Please select a directory!")
                return
            
            self.save_directory = new_directory
            
            # Get selected format from combobox
            if dialog.formatComboBox.currentIndex() == 0:
                self.image_format = 'png'
            else:
                self.image_format = 'jpg'
            
            # Create directory if it doesn't exist
            os.makedirs(self.save_directory, exist_ok=True)
            
            # Update global config
            EXPORT_SETTINGS['save_directory'] = self.save_directory
            EXPORT_SETTINGS['image_format'] = self.image_format
            
            # Close dialog
            dialog.accept()
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Settings Updated",
                f"Save settings updated:\n"
                f"Directory: {self.save_directory}\n"
                f"Format: {self.image_format.upper()}"
            )
            
            # Update export info label
            self.update_export_info()
        
        dialog.okButton.clicked.connect(apply_settings)
        dialog.cancelButton.clicked.connect(dialog.reject)
        
        # Show dialog
        dialog.exec()
    
    def update_export_info(self):
        """
        Update Export Info Label
        
        Updates the export info label in the sidebar to show current
        export format and whether a save directory has been configured.
        """
        if self.save_directory and self.save_directory != '':
            # Show format and that directory is configured
            self.exportInfo.setText(f"Format: {self.image_format.upper()} | Directory: Set ✓")
        else:
            # Prompt user to configure directory
            self.exportInfo.setText(f"Format: {self.image_format.upper()} | Set directory...")
    
    def show_documentation(self):
        """
        Show Documentation Dialog
        
        Displays a help dialog with basic usage instructions.
        """
        QMessageBox.information(
            self,
            "Documentation",
            "Grease Analyzer Documentation\n\n"
            "1. Upload baseline data (reference)\n"
            "2. Upload one or more sample files\n"
            "3. View graphs for each sample\n"
            "4. Generate AI analysis (parallel processing)\n"
            "5. Export results\n\n"
            "For more info, see docs/LLM_OPTIMIZATION.md"
        )
    
    def show_about(self):
        """
        Show About Dialog
        
        Displays application information including version,
        LLM model, and configuration details.
        """
        QMessageBox.about(
            self,
            "About Grease Analyzer",
            f"<h3>Grease Analyzer - PyQt6 Edition</h3>"
            f"<p><b>Version:</b> 1.0</p>"
            f"<p><b>LLM Model:</b> {LLM_CONFIG['model']}</p>"
            f"<p><b>Parallel Workers:</b> {LLM_CONFIG['max_workers']}</p>"
            f"<p>High-performance grease analysis with local LLM</p>"
            f"<p>© 2025 Your Team</p>"
        )
    
    def resizeEvent(self, event):
        """
        Handle Window Resize Event
        
        Called automatically when window is resized.
        Rescales the displayed graph to fit the new window size.
        
        Args:
            event: QResizeEvent object
        """
        super().resizeEvent(event)
        # Rescale graph if one is currently displayed
        if self.current_graph_path:
            self.update_graph_display()
    
    def closeEvent(self, event):
        """
        Handle Application Close Event
        
        Called automatically when user closes the window.
        Ensures background worker thread is stopped gracefully
        before application exits.
        
        Args:
            event: QCloseEvent object
        """
        # Stop worker thread if it's still running
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.analysis_worker.wait()  # Wait for thread to finish
        event.accept()  # Allow window to close



def main():
    """
    Main Application Entry Point
    
    Initializes the Qt application, creates the main window,
    and starts the event loop. Sets Fusion style for consistent
    cross-platform appearance.
    """
    app = QApplication(sys.argv)
    
    # Set Fusion style for modern appearance (optional)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = GreaseAnalyzerApp()
    window.show()
    
    # Start Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

