"""
Grease Analyzer - PyQt6 Desktop Application with Optimized Numerical AI Analysis

MAIN APPLICATION FILE:
This is the entry point for the Grease Analyzer desktop application.
Uses an OPTIMIZED NUMERICAL approach: Fast FTIRDeviationAnalyzer (pure metrics).

KEY FEATURES:
- Load baseline (reference) and multiple sample CSV files
- Visualize data overlays with interactive graphs
- OPTIMIZED numerical AI-powered analysis: <1s per sample
- Export graphs and generate reports

PERFORMANCE:
- Core analysis: <1 second per sample (FTIRDeviationAnalyzer)
- 10-50x faster than LLM-only approach
- 100% reliable with pure numerical analysis

ARCHITECTURE:
- AnalysisWorker: QThread for non-blocking analysis
- GreaseAnalyzerApp: Main window class managing UI and data flow
- Integration with: CSV processor, graph generator, optimized numerical analyzer

Note: Image-based LLM analysis has been disabled for faster, more reliable processing.
"""

from PyQt6 import uic
from PyQt6.QtWidgets import (QMainWindow, QApplication, QFileDialog, QMessageBox, 
                              QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QCheckBox, QScrollArea, QGridLayout, QLabel,
                              QSizePolicy, QMenu)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QIcon
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Import project modules
from modules.csv_processor import CSVProcessor
from modules.graph_generator import GraphGenerator
from modules.llm_analyzer import LLMAnalyzer  # Now includes FTIRAnalyzer internally
from utils.config import LLM_CONFIG, EXPORT_SETTINGS, SUPPORTED_FORMATS


class ChatWorker(QThread):
    """
    Background Worker Thread for AI Chat Responses
    
    Handles real-time chat interactions with the local LLM without blocking the UI.
    """
    
    response_ready = pyqtSignal(str)  # Emits the AI's response
    error = pyqtSignal(str)           # Emits error message
    
    def __init__(self, analyzer: LLMAnalyzer, user_message: str, context: Dict):
        """
        Initialize chat worker
        
        Args:
            analyzer: LLMAnalyzer instance
            user_message: User's question/message
            context: Dictionary containing analysis context (baseline, samples, results)
        """
        super().__init__()
        self.analyzer = analyzer
        self.user_message = user_message
        self.context = context
    
    def run(self):
        """Execute chat query with LLM"""
        try:
            response = self.analyzer.chat_with_context(
                self.user_message,
                self.context
            )
            self.response_ready.emit(response)
        except Exception as e:
            self.error.emit(f"Chat error: {str(e)}")


class AnalysisWorker(QThread):
    """
    Background Worker Thread for Optimized Numerical Analysis

    Uses the production-ready FTIRDeviationAnalyzer for fast, accurate numerical 
    analysis (<1s per sample) without requiring image generation.
    """

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, analyzer: LLMAnalyzer, 
                 baseline_data: pd.DataFrame, baseline_name: str, 
                 sample_data_list: List[Dict], sample_names: List[str]):
        """
        Initialize optimized numerical analysis worker

        Args:
            analyzer: LLMAnalyzer instance (uses FTIRDeviationAnalyzer internally)
            baseline_data: Baseline DataFrame
            baseline_name: Baseline filename
            sample_data_list: List of dictionaries containing sample dataframes
            sample_names: List of sample filenames
        """
        super().__init__()
        self.analyzer = analyzer
        self.baseline_data = baseline_data
        self.baseline_name = baseline_name
        self.sample_data_list = sample_data_list
        self.sample_names = sample_names
        self._is_running = True

    def run(self):
        """
        Execute optimized numerical analysis on all samples
        
        Process:
        1. Fast numerical analysis with FTIRDeviationAnalyzer (<1s per sample)
        2. Generate executive summary
        
        Note: Image-based LLM enhancement has been disabled for faster processing.
        """
        try:
            self.status.emit("🔍 Starting optimized numerical analysis...")
            total_samples = len(self.sample_names)
            
            results = {'individual_results': {}, 'summary': ''}
            
            for i, sample_name in enumerate(self.sample_names):
                if not self._is_running:
                    return

                # Retrieve the full DataFrame for the current sample
                sample_info = next(s for s in self.sample_data_list if s['name'] == sample_name)
                sample_df = sample_info['data']
                
                self.status.emit(f"📊 [{i+1}/{total_samples}] Analyzing {sample_name}...")
                
                # Use the optimized analyze_sample method (no image required)
                # This runs FTIRDeviationAnalyzer (<1s)
                analysis_result = self.analyzer.analyze_sample(
                    self.baseline_data,
                    sample_df,
                    self.baseline_name,
                    sample_name,
                    image_path=None  # No image generation
                )
                
                # Store the human summary for backward compatibility
                results['individual_results'][sample_name] = analysis_result['human_summary']
                
                # Also store full structured results for advanced usage
                results[f'{sample_name}_full'] = analysis_result
                
                progress_value = int((i + 1) / total_samples * 90) # Leave 10% for summary
                self.progress.emit(progress_value)
                
                # Show timing info
                time_str = f"{analysis_result['analysis_time']:.2f}s"
                self.status.emit(f"✅ [{i+1}/{total_samples}] {sample_name} analyzed in {time_str}")

            self.progress.emit(90)
            self.status.emit("📝 Generating executive summary...")

            # Generate executive summary
            results['summary'] = self.analyzer.generate_summary(results['individual_results'])

            self.progress.emit(100)
            self.status.emit("✅ Analysis complete!")

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")

    def stop(self):
        """Stop the worker thread gracefully"""
        self._is_running = False


class GreaseAnalyzerApp(QMainWindow):
    """
    Main Application Window for Grease Analyzer with Visual AI
    """

    def __init__(self):
        super().__init__()
        
        # Load UI layout from Qt Designer file
        ui_path = Path(__file__).parent / "GUI" / "Analyzer_main.ui"
        uic.loadUi(ui_path, self)
        
        # Initialize data storage structures
        self.baseline_data: Optional[pd.DataFrame] = None  # Reference dataset
        self.baseline_name: str = ""                        # Baseline filename
        self.sample_data_list: List[Dict] = []              # List of: {'name': str, 'data': DataFrame, 'stats': dict}
        self.current_sample_index: int = 0                  # Currently displayed sample
        self.analysis_results: Dict = {}                    # LLM analysis results
        self.current_graph_path: Optional[str] = None       # Path to displayed graph image
        self.current_figure = None                          # Current matplotlib figure for canvas
        self.saved_graph_paths: List[str] = []              # List of saved graph paths for analysis
        
        # Multi-view comparison mode
        self.comparison_mode: str = "single"                # "single" or "grid"
        self.tab_widget: Optional[QTabWidget] = None        # Tab widget for sample navigation
        self.grid_widget: Optional[QWidget] = None          # Grid widget for multi-comparison
        self.sample_canvases: Dict[str, FigureCanvas] = {}  # Cache of canvases per sample
        
        # Export settings
        self.save_directory: str = EXPORT_SETTINGS['save_directory']  # Directory for saving graphs
        self.image_format: str = EXPORT_SETTINGS['image_format']      # Image format (png/jpg)
        
        # Initialize module instances
        self.csv_processor = CSVProcessor()          # CSV file loading and validation
        self.graph_generator = GraphGenerator()      # Matplotlib graph creation
        self.llm_analyzer = LLMAnalyzer(
            model=LLM_CONFIG.get('model', 'llava:7b-v1.6'),
            use_llm=LLM_CONFIG.get('use_llm_enhancement', True)
        )  # Local LLM (Ollama) for AI analysis - reads config
        
        # Background worker threads
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.chat_worker: Optional[ChatWorker] = None
        
        # Chat history storage
        self.chat_history: List[Dict[str, str]] = []  # List of {'role': 'user'/'assistant', 'content': str}

        # Set up UI
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """
        Initialize UI component states
        
        Sets default values, disables buttons until data is loaded,
        and configures display areas for graphs and text.
        """
        self.setWindowTitle("Grease Analyzer - PyQt6 Edition")
        
        # Set application icon/logo
        logo_path = Path(__file__).parent / "GUI" / "logo.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        
        # Replace the QLabel "display" with matplotlib visualization system (tabs + grid)
        self.setup_visualization_system()
        
        # Set initial status messages
        self.status_inf.setText("STATUS: Ready to analyze data")
        self.aiSummaryText.setText(
            "No analysis yet.\n"
            "1. Upload baseline data\n"
            "2. Upload sample data\n"
            "3. Click 'Execute Analysis' to analyze all samples\n"
            "4. Switch between samples to view their results"
        )
        

        self.uploadProgress.setValue(0)
        
        # Disable buttons until baseline is loaded
        self.btn_current_filter.setEnabled(False)  # Upload samples button
        self.btn_invert.setEnabled(False)          # Generate analysis button
        self.btn_export_current.setEnabled(False)  # Export current graph button
        self.btn_export_all.setEnabled(False)      # Export all graphs button
        self.btn_generate_pdf_report.setEnabled(False)  # Generate PDF report button
        
        # Initialize sample dropdown menu
        self.comboBox.clear()
        self.comboBox.addItem("No samples loaded")
        
        # Initialize mode menu items
        self.actionTabMode.setChecked(True)  # Tab mode is default
        self.actionGridMode.setChecked(False)
        self.actionGridMode.setEnabled(False)  # Disabled until samples are loaded
        
        # # Display AI model configuration information
        # model_name = LLM_CONFIG['model'].replace('llava:', 'LLaVA ')
        # self.aiModelInfo.setText(f"Model: {model_name}")
        
        # Update export info label
        self.update_export_info()
        
        # Make export info label clickable (opens directory settings)
        self.exportInfo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exportInfo.mousePressEvent = lambda event: self.change_save_directory()
        
        # Add context menu to Graph Analysis Report for PDF export
        self.aiSummaryText.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.aiSummaryText.customContextMenuRequested.connect(self.show_analysis_report_context_menu)
        
        # Set default splitter sizes
        self.set_default_splitter_sizes()
    
    def setup_visualization_system(self):
        """
        Setup Advanced Visualization System with Tabs and Grid Comparison
        
        Creates a flexible visualization system that supports:
        1. Tab Mode (default): One tab per sample for easy navigation
        2. Grid Mode: View multiple samples simultaneously in a grid layout
        
        Users can toggle between modes using the comparison mode button.
        """
        # Remove the old display widget
        old_display = self.display
        display_layout = old_display.parent().layout()
        
        # Create container widget for visualization modes
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.viz_layout.setContentsMargins(0, 0, 0, 0)
        self.viz_layout.setSpacing(5)
        
        # Sample selection checkboxes (shown in grid mode only)
        # Create main container with horizontal scrollable layout
        self.sample_checkboxes_container = QWidget()
        container_layout = QHBoxLayout(self.sample_checkboxes_container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        
        # Add label for checkboxes
        checkbox_label = QLabel("Select samples:")
        checkbox_label.setStyleSheet("color: rgb(220, 225, 230); font: 11pt 'Roboto'; padding-right: 5px;")
        container_layout.addWidget(checkbox_label)
        
        # Left scroll button
        self.scroll_left_btn = QPushButton("◀")
        self.scroll_left_btn.setFixedSize(30, 40)
        self.scroll_left_btn.setStyleSheet("""
            QPushButton {
                background: rgb(45, 60, 75);
                color: rgb(220, 225, 230);
                border: 1px solid rgb(50, 68, 85);
                border-radius: 5px;
                font: bold 12pt;
            }
            QPushButton:hover {
                background: rgb(70, 90, 110);
            }
            QPushButton:pressed {
                background: rgb(85, 105, 75);
            }
        """)
        self.scroll_left_btn.clicked.connect(self.scroll_checkboxes_left)
        container_layout.addWidget(self.scroll_left_btn)
        
        # Scrollable area for checkboxes
        self.checkboxes_scroll = QScrollArea()
        self.checkboxes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.checkboxes_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.checkboxes_scroll.setWidgetResizable(True)
        self.checkboxes_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid rgb(50, 68, 85);
                border-radius: 5px;
                background: rgb(32, 44, 56);
            }
        """)
        
        # Widget to hold checkboxes in horizontal layout
        self.sample_checkboxes_widget = QWidget()
        self.sample_checkboxes_layout = QHBoxLayout(self.sample_checkboxes_widget)
        self.sample_checkboxes_layout.setContentsMargins(5, 5, 5, 5)
        self.sample_checkboxes_layout.setSpacing(8)
        self.sample_checkboxes_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sample_checkboxes: List[QCheckBox] = []
        
        self.checkboxes_scroll.setWidget(self.sample_checkboxes_widget)
        container_layout.addWidget(self.checkboxes_scroll, 1)  # Stretch to fill space
        
        # Right scroll button
        self.scroll_right_btn = QPushButton("▶")
        self.scroll_right_btn.setFixedSize(30, 40)
        self.scroll_right_btn.setStyleSheet("""
            QPushButton {
                background: rgb(45, 60, 75);
                color: rgb(220, 225, 230);
                border: 1px solid rgb(50, 68, 85);
                border-radius: 5px;
                font: bold 12pt;
            }
            QPushButton:hover {
                background: rgb(70, 90, 110);
            }
            QPushButton:pressed {
                background: rgb(85, 105, 75);
            }
        """)
        self.scroll_right_btn.clicked.connect(self.scroll_checkboxes_right)
        container_layout.addWidget(self.scroll_right_btn)
        
        # Select All button
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setFixedSize(80, 40)
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background: rgb(60, 100, 80);
                color: rgb(220, 225, 230);
                border: 1px solid rgb(80, 120, 100);
                border-radius: 5px;
                font: 9pt "Roboto";
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgb(75, 115, 95);
            }
            QPushButton:pressed {
                background: rgb(85, 125, 105);
            }
        """)
        self.select_all_btn.clicked.connect(self.select_all_samples)
        container_layout.addWidget(self.select_all_btn)
        
        # Deselect All button
        self.deselect_all_btn = QPushButton("Clear All")
        self.deselect_all_btn.setFixedSize(80, 40)
        self.deselect_all_btn.setStyleSheet("""
            QPushButton {
                background: rgb(90, 60, 60);
                color: rgb(220, 225, 230);
                border: 1px solid rgb(110, 80, 80);
                border-radius: 5px;
                font: 9pt "Roboto";
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgb(105, 75, 75);
            }
            QPushButton:pressed {
                background: rgb(120, 90, 90);
            }
        """)
        self.deselect_all_btn.clicked.connect(self.deselect_all_samples)
        container_layout.addWidget(self.deselect_all_btn)
        
        self.sample_checkboxes_container.hide()  # Hidden initially
        self.sample_checkboxes_container.setStyleSheet("""
            QWidget {
                background: rgb(38, 52, 66);
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.sample_checkboxes_container.setMaximumHeight(60)  # Prevent vertical expansion
        self.viz_layout.addWidget(self.sample_checkboxes_container)
        
        # Create tab widget (default mode)
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid rgb(50, 68, 85);
                border-radius: 8px;
                background: rgb(32, 44, 56);
            }
            QTabBar::tab {
                background: rgb(38, 52, 66);
                color: rgb(220, 225, 230);
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font: 11pt "Roboto";
            }
            QTabBar::tab:selected {
                background: rgb(85, 105, 75);
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: rgb(70, 90, 110);
            }
        """)
        # Sync tab changes with combobox
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.viz_layout.addWidget(self.tab_widget)
        
        # Create grid widget (comparison mode) - hidden initially
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid rgb(50, 68, 85);
                border-radius: 8px;
                background: rgb(32, 44, 56);
            }
        """)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_scroll.setWidget(self.grid_widget)
        self.viz_layout.addWidget(self.grid_scroll)
        self.grid_scroll.hide()  # Hidden by default
        
        # Replace old display with new system
        display_layout.replaceWidget(old_display, self.viz_container)
        old_display.deleteLater()
        
        # Store reference for backward compatibility
        self.display = self.viz_container
    
    def setup_matplotlib_canvas(self):
        """
        Legacy method - now replaced by setup_visualization_system()
        Kept for compatibility but redirects to new system.
        """
        pass
    
    def switch_to_tab_mode(self):
        """Switch to Tab Mode - view one sample at a time"""
        if self.comparison_mode == "single":
            return  # Already in tab mode
        
        self.comparison_mode = "single"
        self.grid_scroll.hide()
        self.sample_checkboxes_container.hide()  # Hide container instead of widget
        self.tab_widget.show()
        
        # Update menu checkmarks
        self.actionTabMode.setChecked(True)
        self.actionGridMode.setChecked(False)
        
        self.status_inf.setText("STATUS: Switched to Tab View")
    
    def switch_to_grid_mode(self):
        """Switch to Grid Comparison Mode - view multiple samples side-by-side"""
        if not self.sample_data_list:
            QMessageBox.information(
                self, 
                "No Samples", 
                "Please upload samples first to use Grid Comparison Mode."
            )
            # Keep tab mode checked
            self.actionTabMode.setChecked(True)
            self.actionGridMode.setChecked(False)
            return
        
        if self.comparison_mode == "grid":
            return  # Already in grid mode
        
        self.comparison_mode = "grid"
        self.tab_widget.hide()
        self.sample_checkboxes_container.show()  # Show container instead of widget
        self.grid_scroll.show()
        self.update_grid_view()
        
        # Update menu checkmarks
        self.actionTabMode.setChecked(False)
        self.actionGridMode.setChecked(True)
        
        self.status_inf.setText("STATUS: Switched to Grid Comparison View")
    
    def toggle_comparison_mode(self):
        """
        Toggle between Single Tab Mode and Grid Comparison Mode
        
        Legacy method - now redirects to menu-based switching
        """
        if self.comparison_mode == "single":
            self.switch_to_grid_mode()
        else:
            self.switch_to_tab_mode()
    
    def update_sample_checkboxes(self):
        """
        Update the sample selection checkboxes for grid comparison mode
        
        Creates checkboxes for each loaded sample, allowing users to select
        which samples to display in the grid comparison view.
        """
        # Clear existing checkboxes
        for checkbox in self.sample_checkboxes:
            checkbox.deleteLater()
        self.sample_checkboxes.clear()
        
        # Create checkbox for each sample
        for sample in self.sample_data_list:
            checkbox = QCheckBox(sample['name'])
            checkbox.setChecked(False)  # All unchecked by default
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: rgb(220, 225, 230);
                    font: 9pt "Roboto";
                    spacing: 3px;
                    padding: 3px 8px;
                    background: rgb(45, 60, 75);
                    border-radius: 4px;
                }
                QCheckBox:hover {
                    background: rgb(60, 75, 90);
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:checked {
                    background: rgb(85, 105, 75);
                    border: 1px solid rgb(100, 130, 90);
                }
            """)
            checkbox.stateChanged.connect(self.update_grid_view)
            self.sample_checkboxes_layout.addWidget(checkbox)
            self.sample_checkboxes.append(checkbox)
        
        # No stretch needed - keep checkboxes in horizontal line
    
    def scroll_checkboxes_left(self):
        """Scroll the checkbox list to the left"""
        scrollbar = self.checkboxes_scroll.horizontalScrollBar()
        # Scroll by 150 pixels to the left
        scrollbar.setValue(scrollbar.value() - 150)
    
    def scroll_checkboxes_right(self):
        """Scroll the checkbox list to the right"""
        scrollbar = self.checkboxes_scroll.horizontalScrollBar()
        # Scroll by 150 pixels to the right
        scrollbar.setValue(scrollbar.value() + 150)
    
    def select_all_samples(self):
        """Select all sample checkboxes in grid comparison mode"""
        for checkbox in self.sample_checkboxes:
            checkbox.setChecked(True)
        self.status_inf.setText("STATUS: All samples selected")
    
    def deselect_all_samples(self):
        """Deselect all sample checkboxes in grid comparison mode"""
        for checkbox in self.sample_checkboxes:
            checkbox.setChecked(False)
        self.status_inf.setText("STATUS: All samples deselected")
    
    def update_grid_view(self):
        """
        Update Grid Comparison View with Selected Samples
        
        Displays selected samples in an optimal grid layout:
        - 1 sample: 1x1
        - 2 samples: 1x2 or 2x1
        - 3-4 samples: 2x2
        - 5-6 samples: 2x3
        - 7-9 samples: 3x3
        """
        # Clear existing grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get selected samples
        selected_samples = []
        for i, checkbox in enumerate(self.sample_checkboxes):
            if checkbox.isChecked() and i < len(self.sample_data_list):
                selected_samples.append(self.sample_data_list[i])
        
        if not selected_samples:
            # No samples selected - show message in center
            message_widget = QWidget()
            message_layout = QVBoxLayout(message_widget)
            message_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            label = QLabel("No samples selected.\nCheck at least one sample to view.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("""
                QLabel {
                    color: rgb(220, 225, 230); 
                    font: 16pt 'Roboto';
                    padding: 40px;
                }
            """)
            message_layout.addWidget(label)
            
            self.grid_layout.addWidget(message_widget, 0, 0, 1, 1)
            self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return
        
        # Determine grid size
        num_samples = len(selected_samples)
        if num_samples == 1:
            rows, cols = 1, 1
        elif num_samples == 2:
            rows, cols = 1, 2
        elif num_samples <= 4:
            rows, cols = 2, 2
        elif num_samples <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # Create grid of graphs
        for idx, sample in enumerate(selected_samples[:rows * cols]):
            row = idx // cols
            col = idx % cols
            
            # Create container for this graph
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            container_layout.setSpacing(5)
            
            # Sample name label
            name_label = QLabel(sample['name'])
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("""
                QLabel {
                    color: rgb(220, 225, 230);
                    font: 12pt "Roboto";
                    font-weight: bold;
                    background: rgb(45, 60, 75);
                    padding: 5px;
                    border-radius: 5px;
                }
            """)
            container_layout.addWidget(name_label)
            
            # Always create a fresh canvas for grid view (can't reuse from tabs)
            fig = self.graph_generator.create_overlay_graph(
                self.baseline_data,
                sample['data'],
                self.baseline_name,
                sample['name']
            )
            
            # Create new canvas for this grid cell
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(300, 250)
            
            # Allow canvas to expand and fill available space
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            container_layout.addWidget(canvas, 1)  # Stretch factor 1 to fill space
            
            # Add to grid
            self.grid_layout.addWidget(container, row, col)
    
    def create_sample_tab(self, sample_name: str, sample_data: pd.DataFrame):
        """
        Create a Tab for a Single Sample
        
        Creates a new tab containing:
        - Matplotlib canvas with interactive graph
        - Navigation toolbar for zoom/pan
        
        Args:
            sample_name: Name of the sample
            sample_data: DataFrame containing sample data
        """
        # Create tab widget
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        
        # Generate graph
        fig = self.graph_generator.create_overlay_graph(
            self.baseline_data,
            sample_data,
            self.baseline_name,
            sample_name
        )
        
        # Create canvas
        canvas = FigureCanvas(fig)
        
        # Create navigation toolbar
        toolbar = NavigationToolbar(canvas, tab)
        toolbar.setStyleSheet("""
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
            }
        """)
        
        tab_layout.addWidget(toolbar)
        tab_layout.addWidget(canvas)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, sample_name)
        
        # Cache the canvas and figure
        self.sample_canvases[sample_name] = canvas
        self.current_figure = fig
    
    def refresh_all_tabs(self):
        """
        Refresh All Sample Tabs
        
        Clears existing tabs and recreates them with current data.
        Used when baseline changes or samples are reloaded.
        """
        # Clear tabs
        self.tab_widget.clear()
        self.sample_canvases.clear()
        
        # Recreate tabs for all samples
        for sample in self.sample_data_list:
            self.create_sample_tab(sample['name'], sample['data'])
        
        # Enable/disable mode menu items based on sample count
        has_multiple_samples = len(self.sample_data_list) > 1
        self.actionGridMode.setEnabled(has_multiple_samples)
        
        if not has_multiple_samples and self.comparison_mode == "grid":
            # Switch back to tab mode if only one sample
            self.switch_to_tab_mode()
        
        # Update checkboxes
        self.update_sample_checkboxes()
    
    def on_tab_changed(self, index: int):
        """Handle tab selection change - sync with combobox and update analysis display"""
        if 0 <= index < len(self.sample_data_list):
            self.current_sample_index = index
            # Sync combobox with tab selection (without triggering its signal)
            self.comboBox.blockSignals(True)
            self.comboBox.setCurrentIndex(index)
            self.comboBox.blockSignals(False)
            # Update current figure for export
            if self.sample_data_list[index]['name'] in self.sample_canvases:
                canvas = self.sample_canvases[self.sample_data_list[index]['name']]
                self.current_figure = canvas.figure
            
            # Update the analysis display for the new sample
            # Don't reset if we have analysis results - just show the current sample's result
            if self.analysis_results and 'individual_results' in self.analysis_results:
                self.display_current_sample_analysis()
            else:
                # No analysis available yet, show prompt
                self.aiSummaryText.setText(
                    "No analysis for this sample yet.\n\n"
                    "Click 'Execute Analysis' to analyze all samples."
                )
        
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
        
        # Horizontal splitter for AI area: Summary and Chat panels equal size
        if hasattr(self, 'aiHorizontalSplitter'):
            ai_panel_width = right_panel_width // 2
            self.aiHorizontalSplitter.setSizes([ai_panel_width, ai_panel_width])
    
    def setup_chat_interface(self):
        """
        Initialize Chat Interface
        
        Sets up the chat display with welcome message and configures
        the input field for user interactions.
        """
        # Check if chat widgets exist in the UI
        if not hasattr(self, 'chatDisplay'):
            print("⚠️ Chat interface not available in UI")
            return
            
        welcome_msg = (
            "<div style='color: #b0b0b0; margin-top: 10px;'>"
            "I can help you understand your FTIR analysis results. "
            "Ask me about oxidation levels, contamination or peak changes"
            "</div>"
        )
        self.chatDisplay.setHtml(welcome_msg)
        self.chatDisplay.setReadOnly(True)
        
        # Initially disable chat until data is loaded
        if hasattr(self, 'chatInput'):
            self.chatInput.setEnabled(False)
            self.chatInput.setPlaceholderText("Load data and run analysis to enable chat...")
        if hasattr(self, 'btn_send_message'):
            self.btn_send_message.setEnabled(False)
    
    def enable_chat_interface(self):
        """Enable chat interface after analysis is complete"""
        if hasattr(self, 'chatInput'):
            self.chatInput.setEnabled(True)
            self.chatInput.setPlaceholderText("Ask questions about your data analysis...")
            self.chatInput.setFocus()
        if hasattr(self, 'btn_send_message'):
            self.btn_send_message.setEnabled(True)
    
    def send_chat_message(self):
        """
        Send User Message to AI Chat
        
        Handles user input, displays it in the chat, and sends it to
        the LLM for processing in a background thread.
        """
        if not hasattr(self, 'chatInput'):
            return
            
        user_message = self.chatInput.text().strip()
        
        if not user_message:
            return
        
        # Check if analysis has been run
        if not self.analysis_results:
            self.append_chat_message(
                "assistant",
                "Please run the 'Generate Summary' analysis first before asking questions."
            )
            return
        
        # Clear input field
        self.chatInput.clear()
        
        # Display user message
        self.append_chat_message("user", user_message)
        
        # Add to chat history
        self.chat_history.append({'role': 'user', 'content': user_message})
        
        # Disable input while processing
        self.chatInput.setEnabled(False)
        if hasattr(self, 'btn_send_message'):
            self.btn_send_message.setEnabled(False)
        
        # Show thinking indicator
        self.append_chat_message("assistant", "Thinking...")
        
        # Prepare context for LLM
        context = self.build_chat_context()
        
        # Start background worker
        self.chat_worker = ChatWorker(self.llm_analyzer, user_message, context)
        self.chat_worker.response_ready.connect(self.on_chat_response)
        self.chat_worker.error.connect(self.on_chat_error)
        self.chat_worker.start()
    
    def build_chat_context(self) -> Dict:
        """
        Build Context for AI Chat (Current Sample Only)
        
        Creates a dictionary containing relevant analysis data for the
        currently selected sample that the AI can reference when answering questions.
        
        Returns:
            Dictionary with baseline and current sample analysis results
        """
        current_sample = self.sample_data_list[self.current_sample_index] if self.sample_data_list else None
        current_sample_name = current_sample['name'] if current_sample else None
        
        context = {
            'baseline_name': self.baseline_name,
            'sample_count': len(self.sample_data_list),
            'sample_names': [s['name'] for s in self.sample_data_list],
            'current_sample': current_sample_name,
            'chat_history': self.chat_history[-5:]  # Last 5 exchanges for context
        }
        
        # Add analysis results for current sample only
        individual_results = self.analysis_results.get('individual_results', {})
        if current_sample_name and current_sample_name in individual_results:
            # Only include current sample's analysis
            context['individual_analyses'] = {current_sample_name: individual_results[current_sample_name]}
            context['analysis_summary'] = f"Analysis for {current_sample_name}"
        else:
            context['individual_analyses'] = {}
            context['analysis_summary'] = 'No analysis available for current sample'
        
        # Add sample statistics for current sample
        if current_sample:
            context['current_sample_stats'] = {
                'quality_score': current_sample['comparison']['quality_score'],
                'mean_deviation': current_sample['comparison']['mean_deviation_percent'],
                'correlation': current_sample['comparison']['correlation']
            }
        
        return context
    
    def append_chat_message(self, role: str, message: str):
        """
        Append Message to Chat Display
        
        Formats and displays messages in the chat interface with
        appropriate styling for user vs assistant messages.
        
        Args:
            role: 'user' or 'assistant'
            message: Message text to display
        """
        if not hasattr(self, 'chatDisplay'):
            return
            
        current_html = self.chatDisplay.toHtml()
        
        if role == "user":
            color = "#4fc3f7"
            label = "You"
        else:  # assistant
            color = "#81c784"
            label = "AI Assistant"
        
        # Format message with proper HTML escaping
        formatted_message = message.replace('\n', '<br>')
        
        new_message = f"""
        <div style='margin: 10px 0; padding: 10px; border-radius: 8px; border-left: 3px solid {color};'>
            <div style='color: {color}; font-weight: bold; margin-bottom: 5px;'>{label}:</div>
            <div style='color: #d0d0d0;'>{formatted_message}</div>
        </div>
        """
        
        # Append to existing HTML
        self.chatDisplay.setHtml(current_html + new_message)
        
        # Scroll to bottom
        scrollbar = self.chatDisplay.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_chat_response(self, response: str):
        """
        Handle AI Chat Response
        
        Called when the background worker completes processing.
        Removes the "Thinking..." message and displays the actual response.
        """
        if not hasattr(self, 'chatDisplay'):
            return
            
        # Remove "Thinking..." message
        html = self.chatDisplay.toHtml()
        if "Thinking..." in html:
            # Remove the last message (thinking indicator)
            last_div_start = html.rfind('<div style=\'margin: 10px 0;')
            if last_div_start > 0:
                html = html[:last_div_start]
                self.chatDisplay.setHtml(html)
        
        # Display actual response
        self.append_chat_message("assistant", response)
        
        # Add to chat history
        self.chat_history.append({'role': 'assistant', 'content': response})
        
        # Re-enable input
        if hasattr(self, 'chatInput'):
            self.chatInput.setEnabled(True)
            self.chatInput.setFocus()
        if hasattr(self, 'btn_send_message'):
            self.btn_send_message.setEnabled(True)
    
    def on_chat_error(self, error_msg: str):
        """Handle chat error"""
        if not hasattr(self, 'chatDisplay'):
            return
            
        # Remove "Thinking..." message
        html = self.chatDisplay.toHtml()
        if "Thinking..." in html:
            last_div_start = html.rfind('<div style=\'margin: 10px 0;')
            if last_div_start > 0:
                html = html[:last_div_start]
                self.chatDisplay.setHtml(html)
        
        # Display error
        self.append_chat_message("assistant", f"❌ {error_msg}")
        
        # Re-enable input
        if hasattr(self, 'chatInput'):
            self.chatInput.setEnabled(True)
            self.chatInput.setFocus()
        if hasattr(self, 'btn_send_message'):
            self.btn_send_message.setEnabled(True)
        self.chatInput.setEnabled(True)
        self.btn_send_message.setEnabled(True)
        
    def connect_signals(self):
        """Connect UI signals to handlers"""
        self.btn_save.clicked.connect(self.upload_baseline)
        self.btn_current_filter.clicked.connect(self.upload_samples)
        self.btn_invert.clicked.connect(self.generate_analysis)
        
        # Export button handlers
        self.btn_export_current.clicked.connect(self.save_current_graph)
        self.btn_export_all.clicked.connect(self.save_all_graphs)
        self.btn_generate_pdf_report.clicked.connect(self.export_graph_analysis_report_pdf)
        
        # Dropdown selection handler
        self.comboBox.currentIndexChanged.connect(self.on_sample_changed)

        # Menu actions
        self.actionUpload_BaseLine.triggered.connect(self.upload_baseline)
        self.actionUpload_Samples.triggered.connect(self.upload_samples)
        self.actionSave_Current_Graph.triggered.connect(self.save_current_graph)
        self.actionSave_All_Sample_Graph.triggered.connect(self.save_all_graphs)
        self.actionChangeDirectory.triggered.connect(self.change_save_directory)
        self.actionExit.triggered.connect(self.close)
        self.actionDocumentation.triggered.connect(self.show_documentation)
        self.actionAbout.triggered.connect(self.show_about)
        
        # Mode switching menu actions
        self.actionTabMode.triggered.connect(self.switch_to_tab_mode)
        self.actionGridMode.triggered.connect(self.switch_to_grid_mode)

    def upload_baseline(self):
        """Upload and load baseline reference data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Upload Baseline Data", "", "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.uploadProgress.setValue(20)
            self.status_inf.setText("STATUS: Loading baseline data...")

            df, error = self.csv_processor.load_csv(file_path)

            if error:
                raise Exception(error)

            self.uploadProgress.setValue(60)

            self.baseline_data = df
            self.baseline_name = Path(file_path).name

            self.uploadProgress.setValue(100)
            self.status_inf.setText(f"STATUS: Baseline loaded - {self.baseline_name}")

            self.btn_current_filter.setEnabled(True)
            
            # If samples were already loaded, refresh tabs with new baseline
            if self.sample_data_list:
                print(f"🔄 Refreshing tabs with new baseline: {self.baseline_name}")
                self.refresh_all_tabs()
            
            # Show success message with data info
            QMessageBox.information(
                self, "Success",
                f"Baseline loaded!\n{self.baseline_name}\nRecords: {len(df)}"
            )

        except Exception as e:
            self.uploadProgress.setValue(0)
            self.status_inf.setText("STATUS: Error loading baseline")
            QMessageBox.critical(self, "Error", f"Failed to load baseline:\n{str(e)}")

    def upload_samples(self):
        """Upload and load multiple sample data files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Upload Sample Data (Multiple Selection)", "", 
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_paths:
            return

        if self.baseline_data is None:
            QMessageBox.warning(self, "Warning", "Please upload baseline data first!")
            return

        try:
            total_files = len(file_paths)
            self.sample_data_list.clear()
            self.saved_graph_paths.clear() 

            for i, file_path in enumerate(file_paths):
                progress = int((i + 1) / total_files * 100)
                self.uploadProgress.setValue(progress)
                self.status_inf.setText(f"STATUS: Loading sample {i+1}/{total_files}...")

                df, error = self.csv_processor.load_csv(file_path)

                if error:
                    print(f"⚠️ Skipping {Path(file_path).name}: {error}")
                    continue

                stats = self.csv_processor.calculate_statistics(df)
                baseline_stats = self.csv_processor.calculate_statistics(self.baseline_data)
                comparison = self.csv_processor.compare_with_baseline(self.baseline_data, df)

                sample_name = Path(file_path).name
                self.sample_data_list.append({
                    'name': sample_name,
                    'data': df, # Store DataFrame for PeakDetector access
                    'stats': stats,
                    'comparison': comparison,
                    'baseline_stats': baseline_stats
                })

            self.uploadProgress.setValue(100)
            self.status_inf.setText(f"STATUS: {len(self.sample_data_list)} samples loaded")

            self.update_sample_combobox()
            
            # Display samples in tabs
            if self.sample_data_list:
                self.current_sample_index = 0
                # Reset analysis and chat for fresh start
                self.reset_analysis_and_chat()
                # Create tabs for all samples
                self.refresh_all_tabs()
                # Enable buttons
                QApplication.processEvents()  # Process any pending events first
                self.btn_invert.setEnabled(True)
                self.btn_export_current.setEnabled(True)  # Enable export buttons
                self.btn_export_all.setEnabled(True)
                print(f"✅ Created tabs for {len(self.sample_data_list)} samples")
            
            QMessageBox.information(
                self, "Success",
                f"{len(self.sample_data_list)} samples loaded successfully!"
            )

        except Exception as e:
            self.uploadProgress.setValue(0)
            self.status_inf.setText("STATUS: Error loading samples")
            QMessageBox.critical(self, "Error", f"Failed to load samples:\n{str(e)}")

    def update_sample_combobox(self):
        """Update sample dropdown menu"""
        self.comboBox.clear()
        for sample in self.sample_data_list:
            self.comboBox.addItem(sample['name'])

    def on_sample_changed(self, index: int):
        """Handle sample selection change - sync with tab selection and update analysis display"""
        if 0 <= index < len(self.sample_data_list):
            self.current_sample_index = index
            # Sync tab selection with combobox
            if self.tab_widget.count() > index:
                self.tab_widget.setCurrentIndex(index)
            
            # Update the analysis display for the new sample
            # Don't reset if we have analysis results - just show the current sample's result
            if self.analysis_results and 'individual_results' in self.analysis_results:
                self.display_current_sample_analysis()
            else:
                # No analysis available yet, show prompt
                self.aiSummaryText.setText(
                    "No analysis for this sample yet.\n\n"
                    "Click 'Execute Analysis' to analyze all samples."
                )

    def reset_analysis_and_chat(self):
        """
        Reset AI Analysis and Chat when New Samples are Uploaded
        
        Clears the analysis results and chat history when new samples are loaded,
        allowing users to start fresh.
        """
        # Clear analysis results
        self.analysis_results = {}
        
        # Clear chat history
        self.chat_history = []
        
        # Reset AI summary text
        self.aiSummaryText.setText(
            "No analysis yet.\n\n"
            "Click 'Execute Analysis' to analyze all samples."
        )
        
        # Reset chat display (if it exists)
        if hasattr(self, 'chatDisplay'):
            welcome_msg = (
                "<div style='color: #b0b0b0; margin-top: 10px;'>"
                "I can help you understand your FTIR analysis results. "
                "Run analysis first, then ask me about oxidation levels, contamination or peak changes."
                "</div>"
            )
            self.chatDisplay.setHtml(welcome_msg)
        
        print(f"🔄 Analysis and chat reset for new samples")

    def display_current_sample(self):
        """Display graph for currently selected sample"""
        if not self.sample_data_list:
            print("⚠️ Cannot display: No samples loaded")
            return
        
        if self.baseline_data is None:
            print("⚠️ Cannot display: No baseline loaded")
            return

        try:
            sample = self.sample_data_list[self.current_sample_index]
            print(f"📊 Displaying graph for sample: {sample['name']}")

            # Generate graph
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
    
    def display_current_sample_analysis(self):
        """
        Display the analysis results for the currently selected sample.
        
        This is called after analysis is complete or when user switches samples.
        """
        if not self.sample_data_list:
            return
        
        if not self.analysis_results or 'individual_results' not in self.analysis_results:
            # No analysis results available yet
            self.aiSummaryText.setText(
                "No analysis for this sample yet.\n\n"
                "Click 'Execute Analysis' to analyze all samples."
            )
            return
        
        # Get current sample info
        current_sample = self.sample_data_list[self.current_sample_index]
        sample_name = current_sample['name']
        
        # Build analysis display text for the current sample
        summary_text = "=" * 70 + "\n"
        summary_text += "GRAPH ANALYSIS RESULTS\n"
        summary_text += "=" * 70 + "\n\n"
        summary_text += f"BASELINE: {self.baseline_name}\n"
        summary_text += f"SAMPLE: {sample_name}\n\n"
        
        # Display analysis for the sample
        if sample_name in self.analysis_results['individual_results']:
            analysis = self.analysis_results['individual_results'][sample_name]
            summary_text += analysis + "\n\n"
            summary_text += "=" * 70 + "\n"
            summary_text += "📈 SAMPLE STATISTICS\n"
            summary_text += "=" * 70 + "\n"
            summary_text += f"Quality Score: {current_sample['comparison']['quality_score']:.1f}/100\n"
            summary_text += f"Mean Deviation: {current_sample['comparison']['mean_deviation_percent']:+.1f}%\n"
            summary_text += f"Correlation: {current_sample['comparison']['correlation']:.3f}\n"
        else:
            summary_text += "⚠️ No analysis results found for this sample.\n"

        self.aiSummaryText.setText(summary_text)
        print(f"✅ Displayed analysis for: {sample_name}")
    
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
        Generate AI Hybrid Analysis (Numerical + Visual) for ALL Samples
        
        Analyzes all uploaded samples in one batch, storing results for each.
        When user changes samples, the corresponding result is displayed.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No samples to analyze!")
            return
        
        if self.baseline_data is None:
            QMessageBox.warning(self, "Warning", "Please upload baseline data first!")
            return
        
        try:
            self.status_inf.setText("STATUS: Starting numerical analysis for all samples...")
            
            # Start worker thread for background analysis (ALL samples)
            # No image generation required - purely numerical analysis
            if hasattr(self, 'aiProgress'):
                self.aiProgress.setValue(0)
            self.aiSummaryText.setText(
                f"🔄 Analyzing all {len(self.sample_data_list)} samples... Please wait.\n"
                "This may take a moment depending on the number of samples."
            )
            self.btn_invert.setEnabled(False)  # Disable button during analysis
            
            # Create and configure worker thread for ALL samples
            self.analysis_worker = AnalysisWorker(
                self.llm_analyzer,
                self.baseline_data,
                self.baseline_name,
                self.sample_data_list,  # ALL samples
                [s['name'] for s in self.sample_data_list]  # ALL sample names
            )
            self.analysis_worker.progress.connect(self.on_analysis_progress)
            self.analysis_worker.status.connect(self.on_analysis_status)
            self.analysis_worker.finished.connect(self.on_analysis_finished)
            self.analysis_worker.error.connect(self.on_analysis_error)
            self.analysis_worker.start()
            
        except Exception as e:
            if hasattr(self, 'aiProgress'):
                self.aiProgress.setValue(0)
            self.status_inf.setText("STATUS: Analysis preparation failed")
            QMessageBox.critical(self, "Error", f"Failed to prepare analysis:\n{str(e)}")
            self.btn_invert.setEnabled(True)
    
    def on_analysis_progress(self, value: int):
        """Update analysis progress bar"""
        if hasattr(self, 'aiProgress'):
            self.aiProgress.setValue(value)

    def on_analysis_status(self, message: str):
        """Update analysis status message"""
        self.status_inf.setText(f"STATUS: {message}")

    def on_analysis_finished(self, results: Dict):
        """Handle completed visual analysis results for ALL samples"""
        self.analysis_results = results

        # Display the current sample's analysis
        self.display_current_sample_analysis()
        
        if hasattr(self, 'aiProgress'):
            self.aiProgress.setValue(100)
        self.btn_invert.setEnabled(True)
        
        # Enable PDF export button after analysis is complete
        self.btn_generate_pdf_report.setEnabled(True)
        
        # Enable chat interface now that analysis is complete
        self.enable_chat_interface()
        
        # Add welcome message to chat
        num_samples = len(self.sample_data_list)
        welcome_chat_msg = (
            f"Analysis complete for all {num_samples} sample(s)! "
            "I now have full context about your samples. "
            "Feel free to ask me questions about the results, specific peaks, "
            "oxidation levels, or recommendations. You can also switch between samples "
            "to see their individual analyses."
        )
        self.append_chat_message("assistant", welcome_chat_msg)

        QMessageBox.information(
            self, 
            "Success", 
            f"Analysis completed for all {num_samples} sample(s)!\n\n"
            "You can now switch between samples to view their individual results."
        )

    def on_analysis_error(self, error_msg: str):
        """Handle analysis error"""
        # 5. Update error message to reflect the new model tag
        self.aiSummaryText.setText(f"❌ Error: {error_msg}\n\nMake sure Ollama is running:\n  ollama serve\n\nAnd LLaVA model is downloaded:\n  ollama pull {self.llm_analyzer.model}")
        self.aiProgress.setValue(0)
        self.btn_invert.setEnabled(True)
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
            
            # Prepare filename with configured format (just use sample name)
            base_name = sample['name'].rsplit('.', 1)[0] if '.' in sample['name'] else sample['name']
            filename = f"{base_name}.{self.image_format}"
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
                
                # Prepare filename with configured format (just use sample name)
                base_name = sample['name'].rsplit('.', 1)[0] if '.' in sample['name'] else sample['name']
                graph_path = os.path.join(self.save_directory, f"{base_name}.{self.image_format}")
                
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
        dialog_path = Path(__file__).parent / "GUI" / "path.ui"
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
    
    def generate_pdf_report(self):
        """
        Generate Professional PDF Report
        
        Creates a comprehensive PDF report containing:
        - Cover page with metadata
        - Executive summary
        - Individual sample analyses with metrics
        - Embedded overlay graphs
        
        Requires analysis to be completed first.
        """
        if not self.analysis_results:
            QMessageBox.warning(
                self, 
                "Warning", 
                "No analysis results available!\n\n"
                "Please run 'Generate Analysis' first."
            )
            return
        
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No samples loaded!")
            return
        
        try:
            # Ask user for save location
            default_filename = f"FTIR_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PDF Report",
                os.path.join(self.save_directory, default_filename),
                "PDF Files (*.pdf)"
            )
            
            if not save_path:
                return  # User cancelled
            
            # Show progress
            self.status_inf.setText("STATUS: Generating PDF report...")
            QApplication.processEvents()  # Update UI
            
            # Prepare analysis data for PDF export
            individual_results = self.analysis_results.get('individual_results', {})
            summary = self.analysis_results.get('summary', 'Analysis completed.')
            
            # Collect graph paths - try to find saved graphs or use temp graphs
            graph_paths = []
            temp_dir = Path(self.save_directory) / "temp_analysis"
            
            for sample_name in individual_results.keys():
                # Look for existing graph
                clean_name = sample_name.replace('.csv', '')
                
                # Check temp directory first
                temp_graph = temp_dir / f"analysis_{clean_name}.png"
                if temp_graph.exists():
                    graph_paths.append(str(temp_graph))
                else:
                    # Try save directory
                    save_graph = Path(self.save_directory) / f"{clean_name}.{self.image_format}"
                    if save_graph.exists():
                        graph_paths.append(str(save_graph))
                    else:
                        # Generate graph on the fly
                        for sample in self.sample_data_list:
                            if sample['name'] == sample_name:
                                fig = self.graph_generator.create_overlay_graph(
                                    self.baseline_data,
                                    sample['data'],
                                    self.baseline_name,
                                    sample_name
                                )
                                temp_path = temp_dir / f"pdf_temp_{clean_name}.png"
                                temp_dir.mkdir(parents=True, exist_ok=True)
                                fig.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white')
                                plt.close(fig)
                                graph_paths.append(str(temp_path))
                                break
            
            # Generate PDF
            success = self.pdf_exporter.export_analysis_report(
                output_path=save_path,
                baseline_name=self.baseline_name,
                analyses=individual_results,
                executive_summary=summary,
                graph_paths=graph_paths if graph_paths else None
            )
            
            if success:
                self.status_inf.setText("STATUS: PDF report generated successfully!")
                QMessageBox.information(
                    self,
                    "Success",
                    f"PDF report saved successfully!\n\n"
                    f"Location: {save_path}\n\n"
                    f"Samples included: {len(individual_results)}"
                )
                
                # Ask if user wants to open the PDF
                reply = QMessageBox.question(
                    self,
                    "Open PDF?",
                    "Would you like to open the PDF report now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Open PDF with default system viewer
                    if sys.platform == 'win32':
                        os.startfile(save_path)
                    elif sys.platform == 'darwin':  # macOS
                        os.system(f'open "{save_path}"')
                    else:  # Linux
                        os.system(f'xdg-open "{save_path}"')
            else:
                self.status_inf.setText("STATUS: PDF generation failed!")
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to generate PDF report.\n\n"
                    "Make sure the 'reportlab' package is installed:\n"
                    "pip install reportlab"
                )
                
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                f"PDF export requires the 'reportlab' library.\n\n"
                f"Please install it using:\n"
                f"pip install reportlab\n\n"
                f"Error: {str(e)}"
            )
        except Exception as e:
            self.status_inf.setText("STATUS: PDF generation failed!")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to generate PDF report:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def show_analysis_report_context_menu(self, position):
        """Show context menu for Graph Analysis Report"""
        # Check if there's content to export
        report_text = self.aiSummaryText.toPlainText()
        if not report_text or report_text.strip() == "" or "No analysis" in report_text:
            return
        
        # Create context menu
        menu = QMenu(self)
        
        # Add export action
        export_action = menu.addAction("Export to PDF...")
        export_action.triggered.connect(self.export_graph_analysis_report_pdf)
        
        # Show menu at cursor position
        menu.exec(self.aiSummaryText.mapToGlobal(position))
    
    def export_graph_analysis_report_pdf(self):
        """Export Graph Analysis Report text to PDF"""
        # Check if reportlab is available
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "PDF export requires the 'reportlab' library.\n\n"
                "Please install it using:\n"
                "pip install reportlab"
            )
            return
        
        # Get the report text
        report_text = self.aiSummaryText.toPlainText()
        
        if not report_text or report_text.strip() == "" or "No analysis" in report_text:
            QMessageBox.warning(
                self,
                "Warning",
                "No graph analysis report available!\n\n"
                "Please run 'Generate Analysis' first."
            )
            return
        
        try:
            # Get current sample info for metadata
            sample_name = None
            baseline_name = self.baseline_name if hasattr(self, 'baseline_name') and self.baseline_name else None
            
            if self.sample_data_list and self.current_sample_index < len(self.sample_data_list):
                sample_name = self.sample_data_list[self.current_sample_index]['name']
            
            # Ask user for save location
            default_filename = f"Graph_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Graph Analysis Report as PDF",
                default_filename,
                "PDF Files (*.pdf)"
            )
            
            if not save_path:
                return  # User cancelled
            
            # Show progress
            self.status_inf.setText("STATUS: Exporting graph analysis report to PDF...")
            QApplication.processEvents()  # Update UI
            
            # Create PDF directly
            try:
                # Create PDF document
                doc = SimpleDocTemplate(
                    save_path,
                    pagesize=letter,
                    rightMargin=0.75*inch,
                    leftMargin=0.75*inch,
                    topMargin=0.75*inch,
                    bottomMargin=0.75*inch
                )
                
                # Build content
                story = []
                styles = getSampleStyleSheet()
                
                # Add custom title style
                title_style = ParagraphStyle(
                    name='CustomTitle',
                    parent=styles['Title'],
                    fontSize=24,
                    textColor=colors.HexColor('#1a5490'),
                    spaceAfter=20,
                    alignment=TA_CENTER,
                    fontName='Helvetica-Bold'
                )
                
                # Add title
                story.append(Paragraph("Graph Analysis Report", title_style))
                story.append(Spacer(1, 0.3*inch))
                
                # Add metadata if provided
                if baseline_name or sample_name:
                    current_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
                    metadata = []
                    
                    if baseline_name:
                        metadata.append(['Baseline:', baseline_name])
                    if sample_name:
                        metadata.append(['Sample:', sample_name])
                    metadata.append(['Report Generated:', current_date])
                    
                    table = Table(metadata, colWidths=[1.5*inch, 5*inch])
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c5aa0')),
                    ]))
                    
                    story.append(table)
                    story.append(Spacer(1, 0.3*inch))
                
                # Parse and format the report text - extract only clean text
                lines = report_text.split('\n')
                
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Skip separator lines (====)
                    if line.strip().startswith('='):
                        continue
                    
                    # Remove all emojis from the line
                    line_clean = line
                    emojis_to_remove = ['✅', '⚠️', '❌', '🚨', '⚡', '📈', '🔍', '⏱️', '📊', '🎯', '💡', '⚙️', '🔧', '✔️', '❗', '⭐']
                    for emoji in emojis_to_remove:
                        line_clean = line_clean.replace(emoji, '')
                    
                    # Replace superscript characters with regular notation
                    line_clean = line_clean.replace('⁻¹', '-1')
                    line_clean = line_clean.replace('⁰', '0').replace('¹', '1').replace('²', '2').replace('³', '3')
                    line_clean = line_clean.replace('⁴', '4').replace('⁵', '5').replace('⁶', '6').replace('⁷', '7')
                    line_clean = line_clean.replace('⁸', '8').replace('⁹', '9').replace('⁺', '+').replace('⁻', '-')
                    
                    # Clean line for HTML
                    line_clean = line_clean.strip()
                    if not line_clean:  # Skip if line becomes empty after emoji removal
                        continue
                    
                    line_clean = line_clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    # Detect headers (all caps or ends with colon)
                    if (line_clean.isupper() and len(line_clean) < 50) or line_clean.endswith(':'):
                        story.append(Spacer(1, 0.1*inch))
                        story.append(Paragraph(f"<b>{line_clean}</b>", styles['Heading2']))
                        story.append(Spacer(1, 0.05*inch))
                    else:
                        # Regular text - handle indentation
                        leading_spaces = len(line) - len(line.lstrip())
                        indent = leading_spaces * 5
                        
                        if indent > 0:
                            custom_style = ParagraphStyle(
                                'CustomIndent',
                                parent=styles['Normal'],
                                leftIndent=indent
                            )
                            story.append(Paragraph(line_clean, custom_style))
                        else:
                            story.append(Paragraph(line_clean, styles['Normal']))
                
                # Build PDF
                doc.build(story)
                success = True
                
            except Exception as e:
                success = False
                print(f"❌ PDF generation error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            if success:
                self.status_inf.setText("STATUS: Graph analysis report PDF exported successfully!")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Graph analysis report exported successfully!\n\n"
                    f"Location: {save_path}"
                )
                
                # Ask if user wants to open the PDF
                reply = QMessageBox.question(
                    self,
                    "Open PDF?",
                    "Would you like to open the PDF report now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Open PDF with default system viewer
                    if sys.platform == 'win32':
                        os.startfile(save_path)
                    elif sys.platform == 'darwin':  # macOS
                        os.system(f'open "{save_path}"')
                    else:  # Linux
                        os.system(f'xdg-open "{save_path}"')
            else:
                self.status_inf.setText("STATUS: PDF export failed!")
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to export graph analysis report to PDF."
                )
                
        except Exception as e:
            self.status_inf.setText("STATUS: PDF export failed!")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to export graph analysis report:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def show_documentation(self):
        """
        Save All Graphs and Reports Together
        
        Note: Only the currently analyzed sample will have AI analysis.
        Other samples will export graphs only.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No data to export!")
            return
        
        # Select directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            os.path.expanduser("~/Desktop")
        )
        
        if not directory:
            return
        
        try:
            exported_count = 0
            analyzed_sample = None
            
            # Check which sample has analysis
            if self.analysis_results and 'individual_results' in self.analysis_results:
                individual_results = self.analysis_results['individual_results']
                if individual_results:
                    analyzed_sample = list(individual_results.keys())[0]
            
            for sample in self.sample_data_list:
                sample_name = sample['name'].replace('.csv', '')
                
                # Save graph
                graph_filename = f"{sample_name}_graph.png"
                graph_path = os.path.join(directory, graph_filename)
                
                fig = self.graph_generator.create_overlay_graph(
                    self.baseline_data,
                    sample['data'],
                    self.baseline_name,
                    sample['name']
                )
                fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Save analysis report if available for this sample
                report_filename = f"{sample_name}_analysis.txt"
                report_path = os.path.join(directory, report_filename)
                
                if sample['name'] == analyzed_sample:
                    # This sample has AI analysis
                    analysis_text = self.analysis_results['individual_results'].get(
                        sample['name'],
                        "No analysis available."
                    )
                else:
                    # This sample doesn't have AI analysis yet
                    analysis_text = "⚠️ AI analysis not performed for this sample yet.\n\nTo analyze this sample:\n1. Select it from the dropdown\n2. Click 'Generate Analysis'"
                
                report_content = f"""
═══════════════════════════════════════════════════════════════════
GREASE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════

Sample: {sample['name']}
Baseline: {self.baseline_name}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════════
STATISTICAL METRICS
═══════════════════════════════════════════════════════════════════

Quality Score: {sample['comparison']['quality_score']:.1f}/100
Mean Deviation: {sample['comparison']['mean_deviation_percent']:+.1f}%
Correlation: {sample['comparison']['correlation']:.3f}
Std Dev Change: {sample['comparison']['std_deviation_percent']:+.1f}%

═══════════════════════════════════════════════════════════════════
AI VISUAL ANALYSIS (LLaVA Hybrid)
═══════════════════════════════════════════════════════════════════

{analysis_text}

═══════════════════════════════════════════════════════════════════
"""
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                exported_count += 1
            
            # Save executive summary
            summary_path = os.path.join(directory, "00_EXPORT_SUMMARY.txt")
            summary_content = f"""
═══════════════════════════════════════════════════════════════════
EXPORT SUMMARY
═══════════════════════════════════════════════════════════════════

Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Baseline: {self.baseline_name}
Total Samples Exported: {len(self.sample_data_list)}

NOTE: AI analysis is performed per sample. Only the sample that was
analyzed will have detailed AI analysis in its report.

Analyzed Sample: {analyzed_sample if analyzed_sample else 'None'}

To analyze other samples:
1. Select the sample from the dropdown menu
2. Click 'Generate Analysis'
3. Export again to include the new analysis

═══════════════════════════════════════════════════════════════════
EXPORTED FILES
═══════════════════════════════════════════════════════════════════

See individual *_analysis.txt files for reports (with AI analysis if available).
See individual *_graph.png files for visual spectroscopy data.

═══════════════════════════════════════════════════════════════════
"""
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            QMessageBox.information(
                self,
                "Batch Export Successful",
                f"Exported {exported_count} samples to:\n{directory}\n\n"
                f"Files per sample:\n"
                f"  • [sample]_graph.png\n"
                f"  • [sample]_analysis.txt\n\n"
                f"Plus:\n"
                f"  • 00_EXECUTIVE_SUMMARY.txt"
            )
            
            self.status_inf.setText(f"STATUS: Exported {exported_count} samples")
            
        except Exception as e: 
            QMessageBox.critical(self, "Error", f"Failed to batch export:\n{str(e)}")

    def show_documentation(self):
        """Show documentation dialog"""
        QMessageBox.information(
            self, "Documentation",
            "Grease Analyzer with Hybrid AI\n\n"
            "1. Upload baseline data (reference)\n"
            "2. Upload sample files\n"
            "3. Select a sample from dropdown\n"
            "4. Generate AI analysis for current sample\n"
            "5. Change samples to analyze each one\n"
            "6. Export results\n\n"
            "Analysis is per sample - chat resets when you change samples!"
        )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            f"<h3>Grease Analyzer - Hybrid AI</h3>"
            f"<p><b>AI Model:</b> {self.llm_analyzer.model}</p>"
            f"<p>Uses LLaVA vision model with numerical data injection for accuracy</p>"
            f"<p>© 2025 Schneider Prize Team</p>"
        )

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if self.current_graph_path:
            self.update_graph_display()

    def closeEvent(self, event):
        """Handle application close"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.analysis_worker.wait()
        
        if self.chat_worker and self.chat_worker.isRunning():
            self.chat_worker.wait()
        
        event.accept()


def main():
    """
    Main Application Entry Point
    
    Initializes the Qt application, creates the main window,
    and starts the event loop. Sets Fusion style for consistent
    cross-platform appearance.
    """
    # Set App User Model ID for Windows taskbar (must be before QApplication)
    import platform
    if platform.system() == 'Windows':
        try:
            import ctypes
            # Set a unique app ID so Windows doesn't group it with Python
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('YCP.GreaseAnalyzer.1.0')
        except:
            pass
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon (for taskbar and window)
    logo_path = Path(__file__).parent / "GUI" / "logo.png"
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))
    
    # Create and show main window
    window = GreaseAnalyzerApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()