"""
Grease Analyzer - PyQt6 Desktop Application with Optimized Hybrid AI Analysis

MAIN APPLICATION FILE:
This is the entry point for the Grease Analyzer desktop application.
Uses an OPTIMIZED HYBRID approach: Fast FTIRAnalyzer (numerical) + Optional LLaVA Enhancement.

KEY FEATURES:
- Load baseline (reference) and multiple sample CSV files
- Visualize data overlays with interactive graphs
- OPTIMIZED AI-powered analysis: <1s numerical + optional LLM enhancement
- Export graphs and generate reports

PERFORMANCE:
- Core analysis: <1 second per sample (FTIRAnalyzer)
- With LLM enhancement: 5-15 seconds per sample (optional)
- 10-50x faster than LLM-only approach
- 100% reliable with automatic fallback

ARCHITECTURE:
- AnalysisWorker: QThread for non-blocking analysis
- GreaseAnalyzerApp: Main window class managing UI and data flow
- Integration with: CSV processor, graph generator, optimized LLM analyzer
"""

from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QIcon
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
    Background Worker Thread for Optimized Hybrid Analysis

    Uses the production-ready FTIRAnalyzer for fast, accurate numerical analysis (<1s)
    with optional LLM enhancement for better natural language summaries (5-15s).
    """

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, analyzer: LLMAnalyzer, graph_paths: List[str], 
                 baseline_data: pd.DataFrame, baseline_name: str, 
                 sample_data_list: List[Dict], sample_names: List[str]):
        """
        Initialize optimized hybrid analysis worker

        Args:
            analyzer: LLMAnalyzer instance (now uses FTIRAnalyzer internally)
            graph_paths: List of paths to saved graph images
            baseline_data: Baseline DataFrame
            baseline_name: Baseline filename
            sample_data_list: List of dictionaries containing sample dataframes
            sample_names: List of sample filenames
        """
        super().__init__()
        self.analyzer = analyzer
        self.graph_paths = graph_paths
        self.baseline_data = baseline_data
        self.baseline_name = baseline_name
        self.sample_data_list = sample_data_list
        self.sample_names = sample_names
        self._is_running = True

    def run(self):
        """
        Execute optimized hybrid analysis on all samples
        
        Process:
        1. Fast numerical analysis with FTIRAnalyzer (<1s per sample)
        2. Optional LLM enhancement (5-15s per sample, if available)
        3. Generate executive summary
        """
        try:
            self.status.emit("🔍 Starting optimized hybrid analysis...")
            total_samples = len(self.graph_paths)
            
            results = {'individual_results': {}, 'summary': ''}
            
            for i, (graph_path, sample_name) in enumerate(zip(self.graph_paths, self.sample_names)):
                if not self._is_running:
                    return

                # Retrieve the full DataFrame for the current sample
                sample_info = next(s for s in self.sample_data_list if s['name'] == sample_name)
                sample_df = sample_info['data']
                
                self.status.emit(f"📊 [{i+1}/{total_samples}] Analyzing {sample_name}...")
                
                # Use the new optimized analyze_sample method
                # This runs FTIRAnalyzer first (<1s), then optionally enhances with LLM
                analysis_result = self.analyzer.analyze_sample(
                    self.baseline_data,
                    sample_df,
                    self.baseline_name,
                    sample_name,
                    graph_path
                )
                
                # Store the human summary for backward compatibility
                results['individual_results'][sample_name] = analysis_result['human_summary']
                
                # Also store full structured results for advanced usage
                results[f'{sample_name}_full'] = analysis_result
                
                progress_value = int((i + 1) / total_samples * 90) # Leave 10% for summary
                self.progress.emit(progress_value)
                
                # Show timing info
                time_str = f"{analysis_result['analysis_time']:.2f}s"
                enhancement_str = " (LLM enhanced)" if analysis_result['llm_enhanced'] else " (fast mode)"
                self.status.emit(f"✅ [{i+1}/{total_samples}] {sample_name} analyzed in {time_str}{enhancement_str}")

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
        
        # Export settings
        self.save_directory: str = EXPORT_SETTINGS['save_directory']  # Directory for saving graphs
        self.image_format: str = EXPORT_SETTINGS['image_format']      # Image format (png/jpg)
        
        # Initialize module instances
        self.csv_processor = CSVProcessor()          # CSV file loading and validation
        self.graph_generator = GraphGenerator()      # Matplotlib graph creation
        self.llm_analyzer = LLMAnalyzer()           # Local LLM (Ollama) for AI analysis
        
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
        
        # Replace the QLabel "display" with matplotlib FigureCanvas for interactive graphs
        self.setup_matplotlib_canvas()
        
        # Set initial status messages
        self.status_inf.setText("STATUS: Ready to analyze data")
        self.aiSummaryText.setText(
            "No analysis yet.\n"
            "1. Upload baseline data\n"
            "2. Upload sample data\n"
            "3. Select a sample from dropdown\n"
            "4. Click 'Generate Analysis' to analyze the current sample"
        )
        
        # Initialize chat interface
        self.setup_chat_interface()

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
        model_name = LLM_CONFIG['model'].replace('llava:', 'LLaVA ')
        self.aiModelInfo.setText(f"Model: {model_name}")
        
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
        welcome_msg = (
            "<div style='color: #b0b0b0; margin-top: 10px;'>"
            "I can help you understand your FTIR analysis results. "
            "Ask me about oxidation levels, contamination or peak changes"
            "</div>"
        )
        self.chatDisplay.setHtml(welcome_msg)
        self.chatDisplay.setReadOnly(True)
        
        # Initially disable chat until data is loaded
        self.chatInput.setEnabled(False)
        self.btn_send_message.setEnabled(False)
        self.chatInput.setPlaceholderText("Load data and run analysis to enable chat...")
    
    def enable_chat_interface(self):
        """Enable chat interface after analysis is complete"""
        self.chatInput.setEnabled(True)
        self.btn_send_message.setEnabled(True)
        self.chatInput.setPlaceholderText("Ask questions about your data analysis...")
        self.chatInput.setFocus()
    
    def send_chat_message(self):
        """
        Send User Message to AI Chat
        
        Handles user input, displays it in the chat, and sends it to
        the LLM for processing in a background thread.
        """
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
        self.chatInput.setEnabled(True)
        self.btn_send_message.setEnabled(True)
        self.chatInput.setFocus()
    
    def on_chat_error(self, error_msg: str):
        """Handle chat error"""
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
        
        # Dropdown selection handler
        self.comboBox.currentIndexChanged.connect(self.on_sample_changed)
        
        # Chat interface handlers
        self.btn_send_message.clicked.connect(self.send_chat_message)
        self.chatInput.returnPressed.connect(self.send_chat_message)

        self.actionUpload_BaseLine.triggered.connect(self.upload_baseline)
        self.actionUpload_Samples.triggered.connect(self.upload_samples)
        self.actionSave_Current_Graph.triggered.connect(self.save_current_graph)
        self.actionSave_All_Sample_Graph.triggered.connect(self.save_all_graphs)
        self.actionChangeDirectory.triggered.connect(self.change_save_directory)
        self.actionExit.triggered.connect(self.close)
        self.actionDocumentation.triggered.connect(self.show_documentation)
        self.actionAbout.triggered.connect(self.show_about)

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
            
            # If samples were already loaded, refresh the display with new baseline
            if self.sample_data_list:
                print(f"🔄 Refreshing graphs with new baseline: {self.baseline_name}")
                self.display_current_sample()
            
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
            
            # Display first sample automatically
            if self.sample_data_list:
                self.current_sample_index = 0
                # Reset analysis and chat for fresh start
                self.reset_analysis_and_chat()
                # Force immediate graph display
                QApplication.processEvents()  # Process any pending events first
                self.display_current_sample()
                QApplication.processEvents()  # Ensure graph is rendered
                self.btn_invert.setEnabled(True)
                self.btn_export_current.setEnabled(True)  # Enable export buttons
                self.btn_export_all.setEnabled(True)
                print(f"✅ Auto-displayed first sample: {self.sample_data_list[0]['name']}")
            
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
        """Handle sample selection change"""
        if 0 <= index < len(self.sample_data_list):
            self.current_sample_index = index
            self.display_current_sample()
            # Reset AI analysis and chat when changing samples
            self.reset_analysis_and_chat()

    def reset_analysis_and_chat(self):
        """
        Reset AI Analysis and Chat when Sample Changes
        
        Clears the analysis results and chat history for the new sample,
        allowing users to start fresh with each sample.
        """
        # Clear analysis results
        self.analysis_results = {}
        
        # Clear chat history
        self.chat_history = []
        
        # Reset AI summary text
        self.aiSummaryText.setText(
            "No analysis for this sample yet.\n\n"
            "Click 'Generate Analysis' to analyze the current sample."
        )
        
        # Reset chat display
        welcome_msg = (
            "<div style='color: #b0b0b0; margin-top: 10px;'>"
            "I can help you understand your FTIR analysis results. "
            "Run analysis first, then ask me about oxidation levels, contamination or peak changes."
            "</div>"
        )
        self.chatDisplay.setHtml(welcome_msg)
        
        # Disable chat until new analysis is run
        self.chatInput.setEnabled(False)
        self.btn_send_message.setEnabled(False)
        self.chatInput.setPlaceholderText("Run analysis first to enable chat...")
        
        # Reset AI progress bar
        self.aiProgress.setValue(0)
        
        print(f"🔄 Analysis and chat reset for new sample")

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
            
            # Update sample information label
            records = len(sample['data'])
            quality = sample['comparison']['quality_score']
            self.sampleInfo.setText(f"Records: {records} | Quality: {quality:.1f}/100")

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
        Generate AI Hybrid Analysis (Numerical + Visual) for Current Sample Only
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No samples to analyze!")
            return
        
        if self.baseline_data is None:
            QMessageBox.warning(self, "Warning", "Please upload baseline data first!")
            return
        
        # Get current sample only
        current_sample = self.sample_data_list[self.current_sample_index]
        
        # Save graph for current sample only
        temp_dir = Path(self.save_directory) / "temp_analysis"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.status_inf.setText("STATUS: Preparing graph for analysis...")
            
            # Generate graph for current sample
            fig = self.graph_generator.create_overlay_graph(
                self.baseline_data,
                current_sample['data'],
                self.baseline_name,
                current_sample['name']
            )
            
            # Save to temp file
            graph_filename = f"analysis_{current_sample['name'].replace('.csv', '')}.png"
            graph_path = temp_dir / graph_filename
            fig.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Start worker thread for background analysis (single sample)
            self.aiProgress.setValue(0)
            self.aiSummaryText.setText("🔄 Analyzing current sample... Please wait.")
            self.btn_invert.setEnabled(False)  # Disable button during analysis
            
            # Create and configure worker thread for single sample
            self.analysis_worker = AnalysisWorker(
                self.llm_analyzer,
                [str(graph_path)],  # Single graph path
                self.baseline_data,
                self.baseline_name,
                [current_sample],  # Single sample data
                [current_sample['name']]  # Single sample name
            )
            self.analysis_worker.progress.connect(self.on_analysis_progress)
            self.analysis_worker.status.connect(self.on_analysis_status)
            self.analysis_worker.finished.connect(self.on_analysis_finished)
            self.analysis_worker.error.connect(self.on_analysis_error)
            self.analysis_worker.start()
            
        except Exception as e:
            self.aiProgress.setValue(0)
            self.status_inf.setText("STATUS: Analysis preparation failed")
            QMessageBox.critical(self, "Error", f"Failed to prepare analysis:\n{str(e)}")
            self.btn_invert.setEnabled(True)
    
    def on_analysis_progress(self, value: int):
        """Update analysis progress bar"""
        self.aiProgress.setValue(value)

    def on_analysis_status(self, message: str):
        """Update analysis status message"""
        self.status_inf.setText(f"STATUS: {message}")

    def on_analysis_finished(self, results: Dict):
        """Handle completed visual analysis results for current sample"""
        self.analysis_results = results

        # Get current sample info
        current_sample = self.sample_data_list[self.current_sample_index]
        sample_name = current_sample['name']
        
        # Build analysis display text for single sample
        summary_text = "🤖 Hybrid AI Analysis Results\n"
        summary_text += "=" * 70 + "\n\n"
        summary_text += f"� Sample: {sample_name}\n\n"
        summary_text += "=" * 70 + "\n\n"
        
        # Display analysis for the sample
        if sample_name in results['individual_results']:
            analysis = results['individual_results'][sample_name]
            summary_text += analysis + "\n\n"
            summary_text += "=" * 70 + "\n\n"
            summary_text += "📈 Sample Statistics:\n\n"
            summary_text += f"Quality Score: {current_sample['comparison']['quality_score']:.1f}/100\n"
            summary_text += f"Mean Deviation: {current_sample['comparison']['mean_deviation_percent']:+.1f}%\n"
            summary_text += f"Correlation: {current_sample['comparison']['correlation']:.3f}\n"
        else:
            summary_text += "⚠️ No analysis results found for this sample.\n"

        self.aiSummaryText.setText(summary_text)
        self.aiProgress.setValue(100)
        self.btn_invert.setEnabled(True)
        
        # Enable chat interface now that analysis is complete
        self.enable_chat_interface()
        
        # Add welcome message to chat
        welcome_chat_msg = (
            f"Analysis complete for {sample_name}! "
            "I now have full context about this sample. "
            "Feel free to ask me questions about the results, specific peaks, "
            "oxidation levels, or recommendations."
        )
        self.append_chat_message("assistant", welcome_chat_msg)

        QMessageBox.information(self, "Success", f"Analysis completed for {sample_name}!")

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