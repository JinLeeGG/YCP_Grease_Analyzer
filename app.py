"""
Grease Analyzer - PyQt6 Desktop Application with Visual AI Analysis

MAIN APPLICATION FILE:
This is the entry point for the Grease Analyzer desktop application.
Uses a HYBRID approach: Numerical Peak Detection + LLaVA Visual Analysis.

KEY FEATURES:
- Load baseline (reference) and multiple sample CSV files
- Visualize data overlays with interactive graphs
- AI-powered HYBRID analysis for speed and accuracy
- Export graphs and generate reports

ARCHITECTURE:
- AnalysisWorker: QThread for non-blocking HYBRID analysis
- GreaseAnalyzerApp: Main window class managing UI and data flow
- Integration with: CSV processor, graph generator, LLM analyzer (hybrid)
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
import tempfile
import time
from modules.peak_detector import PeakDetector # Ensure this module is implemented

# Import project modules
from modules.csv_processor import CSVProcessor
from modules.graph_generator import GraphGenerator
from modules.llm_analyzer import LLMAnalyzer
from utils.config import LLM_CONFIG


class AnalysisWorker(QThread):
    """
    Background Worker Thread for Hybrid LLaVA Analysis

    Runs numerical peak detection followed by AI hybrid analysis.
    """

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, analyzer: LLMAnalyzer, graph_paths: List[str], 
                 baseline_data: pd.DataFrame, baseline_name: str, 
                 sample_data_list: List[Dict], sample_names: List[str]):
        """
        Initialize hybrid analysis worker

        Args:
            analyzer: LLMAnalyzer instance
            graph_paths: List of paths to saved graph images
            baseline_data: Baseline DataFrame (for PeakDetector)
            baseline_name: Baseline filename
            sample_data_list: List of dictionaries containing sample dataframes
            sample_names: List of sample filenames
        """
        super().__init__()
        self.analyzer = analyzer
        self.peak_detector = PeakDetector() # Instantiate PeakDetector here
        self.graph_paths = graph_paths
        self.baseline_data = baseline_data
        self.baseline_name = baseline_name
        self.sample_data_list = sample_data_list
        self.sample_names = sample_names
        self._is_running = True

    def run(self):
        """
        Execute hybrid analysis on all graph images and numerical data
        """
        try:
            self.status.emit("🔍 Starting hybrid analysis (Numerical + LLaVA)...")
            total_samples = len(self.graph_paths)
            
            results = {'individual_results': {}, 'summary': ''}
            
            for i, (graph_path, sample_name) in enumerate(zip(self.graph_paths, self.sample_names)):
                if not self._is_running:
                    return

                # Retrieve the full DataFrame for the current sample
                sample_info = next(s for s in self.sample_data_list if s['name'] == sample_name)
                sample_df = sample_info['data']
                
                self.status.emit(f"📊 [{i+1}/{total_samples}] Running peak detection for {sample_name}...")
                
                # --- STEP 1: NUMERICAL PEAK ANALYSIS (FAST, ACCURATE) ---
                comparison_results = self.peak_detector.compare_spectra(self.baseline_data, sample_df)
                
                # IMPORTANT: Need to assume format_for_llm is implemented in PeakDetector
                peak_data_string = self.peak_detector.format_for_llm(comparison_results)
                
                self.status.emit(f"🤖 [{i+1}/{total_samples}] Running LLaVA vision analysis...")
                
                # --- STEP 2: LLaVA HYBRID ANALYSIS (INTERPRETATION) ---
                analysis = self.analyzer.analyze_ftir_hybrid( # Renamed function call
                    graph_path,
                    self.baseline_name,
                    sample_name,
                    peak_data_string # Pass the accurate numerical data
                )

                results['individual_results'][sample_name] = analysis
                
                progress_value = int((i + 1) / total_samples * 90) # Leave 10% for summary
                self.progress.emit(progress_value)

            self.progress.emit(90)
            self.status.emit("📝 Generating executive summary...")

            # Generate executive summary
            results['summary'] = self.analyzer.generate_summary(results['individual_results'])

            self.progress.emit(100)
            self.status.emit("✅ Hybrid analysis complete!")

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Hybrid analysis failed: {str(e)}")

    def stop(self):
        """Stop the worker thread gracefully"""
        self._is_running = False


class GreaseAnalyzerApp(QMainWindow):
    """
    Main Application Window for Grease Analyzer with Visual AI
    """

    def __init__(self):
        super().__init__()

        # Load UI layout
        ui_path = Path(__file__).parent / "Analyzer_main.ui"
        uic.loadUi(ui_path, self)

        # Initialize data storage
        self.baseline_data: Optional[pd.DataFrame] = None
        self.baseline_name: str = ""
        self.sample_data_list: List[Dict] = []
        self.current_sample_index: int = 0
        self.analysis_results: Dict = {}
        self.current_graph_path: Optional[str] = None
        self.saved_graph_paths: List[str] = [] 

        # Initialize modules
        self.csv_processor = CSVProcessor()
        self.graph_generator = GraphGenerator()
        # 4. SPEED FIX: Initialize LLMAnalyzer with the pulled model tag
        self.llm_analyzer = LLMAnalyzer(model="llava:7b-v1.6") 

        # Background worker
        self.analysis_worker: Optional[AnalysisWorker] = None

        # Set up UI
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize UI component states"""
        self.setWindowTitle("Grease Analyzer - Hybrid AI Edition")

        self.display.setScaledContents(False)
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_inf.setText("STATUS: Ready for hybrid analysis")
        self.aiSummaryText.setText(
            "No analysis yet.\n"
            "1. Upload baseline data\n"
            "2. Upload sample data\n"
            "3. Click 'Generate Analysis' for hybrid AI analysis"
        )

        self.uploadProgress.setValue(0)
        self.aiProgress.setValue(0)

        self.btn_current_filter.setEnabled(False)
        self.btn_invert.setEnabled(False)

        self.comboBox.clear()
        self.comboBox.addItem("No samples loaded")

        # Show that we're using the hybrid model
        self.aiModelInfo.setText(f"Model: {self.llm_analyzer.model} (Hybrid Analysis)")

    def connect_signals(self):
        """Connect UI signals to handlers"""
        self.btn_save.clicked.connect(self.upload_baseline)
        self.btn_current_filter.clicked.connect(self.upload_samples)
        self.btn_invert.clicked.connect(self.generate_analysis)

        self.comboBox.currentIndexChanged.connect(self.on_sample_changed)

        self.actionUpload_BaseLine.triggered.connect(self.upload_baseline)
        self.actionUpload_Samples.triggered.connect(self.upload_samples)
        
        # UPDATE THESE TWO LINES:
        self.actionSave_Current_Graph.triggered.connect(self.save_graph_with_report)
        self.actionSave_All_Sample_Graph.triggered.connect(self.save_all_graphs_with_reports) 
        
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

            if self.sample_data_list:
                self.current_sample_index = 0
                self.display_current_sample()
                self.btn_invert.setEnabled(True)

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

    def display_current_sample(self):
        """Display graph for currently selected sample"""
        if not self.sample_data_list:
            return

        try:
            sample = self.sample_data_list[self.current_sample_index]

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

            # Save to temp directory
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"grease_graph_{sample['name']}.png")

            fig.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.current_graph_path = temp_path

            # Store path for analysis later
            if temp_path not in self.saved_graph_paths:
                self.saved_graph_paths.append(temp_path)

            self.update_graph_display()
            self.status_inf.setText(f"STATUS: Displaying {sample['name']}")

        except Exception as e:
            print(f"⚠️ Display error: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_graph_display(self):
        """Update graph display to fit window"""
        if self.current_graph_path and os.path.exists(self.current_graph_path):
            pixmap = QPixmap(self.current_graph_path)
            scaled_pixmap = pixmap.scaled(
                self.display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.display.setPixmap(scaled_pixmap)

    def generate_analysis(self):
        """
        Generate AI Hybrid Analysis (Numerical + Visual)
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No samples to analyze!")
            return

        if self.llm_analyzer.ollama_available is False:
             QMessageBox.critical(self, "Error", 
                                  "Ollama not available or model not found. "
                                  f"Please check your Ollama server and pull the {self.llm_analyzer.model} model.")
             return

        try:
            # Step 1: Generate and save ALL graphs to temp for analysis
            self.status_inf.setText("STATUS: Generating graphs for analysis...")
            self.aiProgress.setValue(5)
            
            temp_dir = tempfile.gettempdir()
            graph_paths = []
            sample_names = []

            for i, sample in enumerate(self.sample_data_list):
                # Generate graph
                fig = self.graph_generator.create_overlay_graph(
                    self.baseline_data,
                    sample['data'],
                    self.baseline_name,
                    sample['name']
                )

                # Save to temp file
                graph_path = os.path.join(temp_dir, f"analysis_{sample['name']}.png")
                fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                graph_paths.append(graph_path)
                sample_names.append(sample['name'])

            self.aiProgress.setValue(10)

            # Step 2: Start hybrid analysis worker
            self.aiSummaryText.setText("🔍 Analyzing graphs with Hybrid (Numerical + Vision) model...\nThis should take under 45 seconds per sample.")
            self.btn_invert.setEnabled(False)
            
            # 3. UPDATE: Pass DataFrames to the Worker Thread
            self.analysis_worker = AnalysisWorker(
                self.llm_analyzer,
                graph_paths,
                self.baseline_data,        # Pass Baseline Data
                self.baseline_name,
                self.sample_data_list,     # Pass ALL Sample Data (with DFs)
                sample_names
            )
            
            self.analysis_worker.progress.connect(self.on_analysis_progress)
            self.analysis_worker.status.connect(self.on_analysis_status)
            self.analysis_worker.finished.connect(self.on_analysis_finished)
            self.analysis_worker.error.connect(self.on_analysis_error)
            self.analysis_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare analysis:\n{str(e)}")
            self.btn_invert.setEnabled(True)

    def on_analysis_progress(self, value: int):
        """Update analysis progress bar"""
        self.aiProgress.setValue(value)

    def on_analysis_status(self, message: str):
        """Update analysis status message"""
        self.status_inf.setText(f"STATUS: {message}")

    def on_analysis_finished(self, results: Dict):
        """Handle completed visual analysis results"""
        self.analysis_results = results

        summary_text = "🤖 Hybrid AI Analysis Results\n"
        summary_text += "=" * 70 + "\n\n"
        summary_text += "📋 Executive Summary\n\n"
        summary_text += results['summary'] + "\n\n"
        summary_text += "=" * 70 + "\n\n"

        for sample_name, analysis in results['individual_results'].items():
            sample_info = next((s for s in self.sample_data_list if s['name'] == sample_name), None)
            if sample_info:
                summary_text += f"📊 {sample_name}\n\n"
                summary_text += analysis + "\n\n"
                summary_text += f"Quality Score: {sample_info['comparison']['quality_score']:.1f}/100\n"
                summary_text += f"Mean Deviation: {sample_info['comparison']['mean_deviation_percent']:+.1f}%\n"
                summary_text += f"Correlation: {sample_info['comparison']['correlation']:.3f}\n"
                summary_text += "\n" + "-" * 70 + "\n\n"

        self.aiSummaryText.setText(summary_text)
        self.aiProgress.setValue(100)
        self.btn_invert.setEnabled(True)

        QMessageBox.information(self, "Success", "Hybrid AI analysis completed!")

    def on_analysis_error(self, error_msg: str):
        """Handle analysis error"""
        # 5. Update error message to reflect the new model tag
        self.aiSummaryText.setText(f"❌ Error: {error_msg}\n\nMake sure Ollama is running:\n  ollama serve\n\nAnd LLaVA model is downloaded:\n  ollama pull {self.llm_analyzer.model}")
        self.aiProgress.setValue(0)
        self.btn_invert.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)

    def save_current_graph(self):
        """Save currently displayed graph"""
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No graph to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current Graph",
            f"{self.sample_data_list[self.current_sample_index]['name']}.png",
            "PNG Files (*.png);;All Files (*)"
        )

        if file_path:
            try:
                pixmap = self.display.pixmap()
                if pixmap:
                    pixmap.save(file_path)
                    QMessageBox.information(self, "Success", f"Graph saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def save_all_graphs(self):
        """Save all sample graphs"""
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No graphs to save!")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory")

        if directory:
            try:
                saved_count = 0
                for sample in self.sample_data_list:
                    fig = self.graph_generator.create_overlay_graph(
                        self.baseline_data,
                        sample['data'],
                        self.baseline_name,
                        sample['name']
                    )

                    graph_path = os.path.join(directory, f"{sample['name']}.png")
                    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1

                QMessageBox.information(
                    self, "Success",
                    f"{saved_count} graphs saved to:\n{directory}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def save_graph_with_report(self):
        """
        Save Both Graph and AI Analysis Report Together
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No data to export!")
            return
        
        if not self.analysis_results:
            QMessageBox.warning(
                self, "Warning", 
                "No AI analysis available!\n\nPlease run 'Generate Analysis' first."
            )
            return
        
        # Select directory for export
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Export Directory",
            os.path.expanduser("~/Desktop")
        )
        
        if not directory:
            return
        
        try:
            current_sample = self.sample_data_list[self.current_sample_index]
            sample_name = current_sample['name'].replace('.csv', '')
            
            # 1. Save the graph image
            graph_filename = f"{sample_name}_graph.png"
            graph_path = os.path.join(directory, graph_filename)
            
            # Generate fresh graph
            fig = self.graph_generator.create_overlay_graph(
                self.baseline_data,
                current_sample['data'],
                self.baseline_name,
                current_sample['name']
            )
            fig.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 2. Save the AI analysis report
            report_filename = f"{sample_name}_analysis.txt"
            report_path = os.path.join(directory, report_filename)
            
            # Get analysis for this specific sample
            analysis_text = self.analysis_results['individual_results'].get(
                current_sample['name'], 
                "No analysis available for this sample."
            )
            
            # Create formatted report
            report_content = f"""
═══════════════════════════════════════════════════════════════════
GREASE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════

Sample: {current_sample['name']}
Baseline: {self.baseline_name}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════════
STATISTICAL METRICS
═══════════════════════════════════════════════════════════════════

Quality Score: {current_sample['comparison']['quality_score']:.1f}/100
Mean Deviation: {current_sample['comparison']['mean_deviation_percent']:+.1f}%
Correlation: {current_sample['comparison']['correlation']:.3f}
Std Dev Change: {current_sample['comparison']['std_deviation_percent']:+.1f}%

Sample Statistics:
  - Mean: {current_sample['stats']['mean']:.2f}
  - Std Dev: {current_sample['stats']['std']:.2f}
  - Min: {current_sample['stats']['min']:.2f}
  - Max: {current_sample['stats']['max']:.2f}
  - Data Points: {current_sample['stats']['count']}

═══════════════════════════════════════════════════════════════════
AI VISUAL ANALYSIS (LLaVA Hybrid)
═══════════════════════════════════════════════════════════════════

{analysis_text}

═══════════════════════════════════════════════════════════════════
GRAPH
═══════════════════════════════════════════════════════════════════

Graph image saved as: {graph_filename}

═══════════════════════════════════════════════════════════════════
Generated by Grease Analyzer - Visual AI Edition
═══════════════════════════════════════════════════════════════════
"""
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Successful",
                f"Exported to:\n{directory}\n\n"
                f"Files:\n"
                f"  • {graph_filename}\n"
                f"  • {report_filename}"
            )
            
            self.status_inf.setText(f"STATUS: Exported {sample_name}")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to export:\n{str(e)}"
            )

    def save_all_graphs_with_reports(self):
        """
        Save All Graphs and Reports Together
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No data to export!")
            return
        
        if not self.analysis_results:
            QMessageBox.warning(
                self, "Warning",
                "No AI analysis available!\n\nPlease run 'Generate Analysis' first."
            )
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
                
                # Save analysis report
                report_filename = f"{sample_name}_analysis.txt"
                report_path = os.path.join(directory, report_filename)
                
                analysis_text = self.analysis_results['individual_results'].get(
                    sample['name'],
                    "No analysis available."
                )
                
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
            summary_path = os.path.join(directory, "00_EXECUTIVE_SUMMARY.txt")
            summary_content = f"""
═══════════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY - ALL SAMPLES
═══════════════════════════════════════════════════════════════════

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Baseline: {self.baseline_name}
Total Samples: {len(self.sample_data_list)}

{self.analysis_results.get('summary', 'No summary available.')}

═══════════════════════════════════════════════════════════════════
INDIVIDUAL REPORTS
═══════════════════════════════════════════════════════════════════

See individual *_analysis.txt files for detailed assessments.
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
            "3. View graphs\n"
            "4. Generate hybrid AI analysis (Numerical Data + LLaVA Vision)\n"
            "5. Export results\n\n"
            "The hybrid mode provides faster, more accurate results!"
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
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = GreaseAnalyzerApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()