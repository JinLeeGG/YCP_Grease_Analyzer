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

# Import project modules
from modules.csv_processor import CSVProcessor
from modules.graph_generator import GraphGenerator
from modules.llm_analyzer import LLMAnalyzer
from utils.config import LLM_CONFIG


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
        
        # Configure the display QLabel for graph visualization
        # setScaledContents(False) means we manually handle scaling for better quality
        self.display.setScaledContents(False)
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
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
        
        # Initialize sample dropdown menu
        self.comboBox.clear()
        self.comboBox.addItem("No samples loaded")
        
        # Display AI model configuration information
        model_name = LLM_CONFIG['model'].replace('llava:', 'LLaVA ').replace('-q4_K_M', ' Q4')
        self.aiModelInfo.setText(f"Model: {model_name} (Parallel: {LLM_CONFIG['max_workers']} workers)")
        
        # Set default splitter sizes
        self.set_default_splitter_sizes()
        
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
        
        # Dropdown selection handler
        self.comboBox.currentIndexChanged.connect(self.on_sample_changed)
        
        # Menu action handlers
        self.actionUpload_BaseLine.triggered.connect(self.upload_baseline)
        self.actionUpload_Samples.triggered.connect(self.upload_samples)
        self.actionSave_Current_Graph.triggered.connect(self.save_current_graph)
        self.actionSave_All_Sample_Graph.triggered.connect(self.save_all_graphs)
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
            
            # Display first sample
            if self.sample_data_list:
                self.current_sample_index = 0
                self.display_current_sample()
                self.btn_invert.setEnabled(True)
            
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
            return
        
        try:
            sample = self.sample_data_list[self.current_sample_index]
            
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
            
            # Save to temporary file for display
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"grease_graph_{sample['name']}.png")
            
            fig.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Release memory
            
            # Store path for later use (resizing, saving)
            self.current_graph_path = temp_path
            
            # Display the graph in UI
            self.update_graph_display()
            self.status_inf.setText(f"STATUS: Displaying {sample['name']}")
            
        except Exception as e:
            print(f"⚠️ Display error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_inf.setText("STATUS: Error displaying graph")
    
    def update_graph_display(self):
        """
        Update Graph Display to Fit Current Window Size
        
        Loads the graph image and scales it to fit the display area
        while maintaining aspect ratio. Called when:
        - New sample is selected
        - Window is resized
        """
        if self.current_graph_path and os.path.exists(self.current_graph_path):
            pixmap = QPixmap(self.current_graph_path)
            # Scale to fit display area while preserving aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.display.setPixmap(scaled_pixmap)
    
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
        
        Opens a save file dialog and exports the currently displayed
        graph as a PNG image file.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No graph to save!")
            return
        
        # Open save dialog with default filename
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Graph",
            f"{self.sample_data_list[self.current_sample_index]['name']}.png",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                # Get pixmap from display and save
                pixmap = self.display.pixmap()
                if pixmap:
                    pixmap.save(file_path)
                    QMessageBox.information(self, "Success", f"Graph saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save graph:\n{str(e)}")
    
    def save_all_graphs(self):
        """
        Save All Sample Graphs
        
        Generates and saves overlay graphs for all loaded samples.
        User selects a directory, and all graphs are saved as PNG files
        with sample names as filenames.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No graphs to save!")
            return
        
        # Select directory for batch save
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Graphs")
        
        if directory:
            try:
                saved_count = 0
                for sample in self.sample_data_list:
                    # Generate graph for this sample
                    fig = self.graph_generator.create_overlay_graph(
                        self.baseline_data,
                        sample['data'],
                        self.baseline_name,
                        sample['name']
                    )
                    
                    # Save to file
                    graph_path = os.path.join(directory, f"{sample['name']}.png")
                    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # Release memory
                    saved_count += 1
                
                # Show success message
                QMessageBox.information(
                    self,
                    "Success",
                    f"{saved_count} graphs saved to:\n{directory}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save graphs:\n{str(e)}")
    
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

