"""
Grease Analyzer - PyQt6 Desktop Application with Visual AI Analysis

MAIN APPLICATION FILE:
This is the entry point for the Grease Analyzer desktop application.
Uses LLaVA vision model to analyze FTIR spectroscopy graphs visually.

KEY FEATURES:
- Load baseline (reference) and multiple sample CSV files
- Visualize data overlays with interactive graphs
- AI-powered VISUAL analysis using LLaVA (reads actual graphs)
- Export graphs and generate reports

ARCHITECTURE:
- AnalysisWorker: QThread for non-blocking LLaVA visual analysis
- GreaseAnalyzerApp: Main window class managing UI and data flow
- Integration with: CSV processor, graph generator, LLM analyzer (visual)
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

# Import project modules
from modules.csv_processor import CSVProcessor
from modules.graph_generator import GraphGenerator
from modules.llm_analyzer import LLMAnalyzer
from utils.config import LLM_CONFIG


class AnalysisWorker(QThread):
    """
    Background Worker Thread for Visual LLaVA Analysis

    Runs AI vision analysis in the background to prevent UI freezing.
    Processes graph images sequentially with LLaVA vision model.
    """

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, analyzer: LLMAnalyzer, graph_paths: List[str], 
                 baseline_name: str, sample_names: List[str]):
        """
        Initialize visual analysis worker

        Args:
            analyzer: LLMAnalyzer instance with LLaVA
            graph_paths: List of paths to saved graph images
            baseline_name: Baseline filename
            sample_names: List of sample filenames
        """
        super().__init__()
        self.analyzer = analyzer
        self.graph_paths = graph_paths
        self.baseline_name = baseline_name
        self.sample_names = sample_names
        self._is_running = True

    def run(self):
        """
        Execute visual analysis on all graph images
        """
        try:
            self.status.emit("🔍 Starting visual analysis with LLaVA...")
            self.progress.emit(10)

            # Analyze each graph image
            results = self.analyzer.analyze_samples_batch(
                self.graph_paths,
                self.baseline_name,
                self.sample_names
            )

            self.progress.emit(80)
            self.status.emit("📝 Generating summary...")

            # Generate executive summary
            summary = self.analyzer.generate_summary(results)

            self.progress.emit(100)
            self.status.emit("✅ Visual analysis complete!")

            self.finished.emit({
                'individual_results': results,
                'summary': summary
            })

        except Exception as e:
            self.error.emit(f"Visual analysis failed: {str(e)}")

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
        self.saved_graph_paths: List[str] = []  # NEW: Store paths to saved graphs

        # Initialize modules
        self.csv_processor = CSVProcessor()
        self.graph_generator = GraphGenerator()
        self.llm_analyzer = LLMAnalyzer(model="llava:7b")  # Visual model

        # Background worker
        self.analysis_worker: Optional[AnalysisWorker] = None

        # Set up UI
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize UI component states"""
        self.setWindowTitle("Grease Analyzer - Visual AI Edition")

        self.display.setScaledContents(False)
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_inf.setText("STATUS: Ready for visual analysis")
        self.aiSummaryText.setText(
            "No analysis yet.\n"
            "1. Upload baseline data\n"
            "2. Upload sample data\n"
            "3. Click 'Generate Analysis' for AI vision analysis"
        )

        self.uploadProgress.setValue(0)
        self.aiProgress.setValue(0)

        self.btn_current_filter.setEnabled(False)
        self.btn_invert.setEnabled(False)

        self.comboBox.clear()
        self.comboBox.addItem("No samples loaded")

        # Show that we're using visual analysis
        self.aiModelInfo.setText(f"Model: {self.llm_analyzer.model} (Visual Analysis)")

    def connect_signals(self):
        """Connect UI signals to handlers"""
        self.btn_save.clicked.connect(self.upload_baseline)
        self.btn_current_filter.clicked.connect(self.upload_samples)
        self.btn_invert.clicked.connect(self.generate_analysis)

        self.comboBox.currentIndexChanged.connect(self.on_sample_changed)

        self.actionUpload_BaseLine.triggered.connect(self.upload_baseline)
        self.actionUpload_Samples.triggered.connect(self.upload_samples)
        self.actionSave_Current_Graph.triggered.connect(self.save_current_graph)
        self.actionSave_All_Sample_Graph.triggered.connect(self.save_all_graphs)
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
            self.saved_graph_paths.clear()  # Clear old graph paths

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
                    'data': df,
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

            fig.savefig(temp_path, dpi=300, bbox_inches='tight')
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
        Generate AI Visual Analysis Using LLaVA

        NEW: This now generates and saves all graphs first,
        then passes image paths to LLaVA for visual analysis.
        """
        if not self.sample_data_list:
            QMessageBox.warning(self, "Warning", "No samples to analyze!")
            return

        try:
            # Step 1: Generate and save ALL graphs for analysis
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

            # Step 2: Start visual analysis worker
            self.aiSummaryText.setText("🔍 Analyzing graphs with LLaVA vision model...\nThis may take 30-60 seconds per sample.")
            self.btn_invert.setEnabled(False)

            self.analysis_worker = AnalysisWorker(
                self.llm_analyzer,
                graph_paths,
                self.baseline_name,
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

        summary_text = "🤖 Visual AI Analysis Results (LLaVA)\n"
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

        QMessageBox.information(self, "Success", "Visual AI analysis completed!")

    def on_analysis_error(self, error_msg: str):
        """Handle analysis error"""
        self.aiSummaryText.setText(f"❌ Error: {error_msg}\n\nMake sure Ollama is running:\n  ollama serve\n\nAnd LLaVA model is downloaded:\n  ollama pull llava:7b")
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

    def show_documentation(self):
        """Show documentation dialog"""
        QMessageBox.information(
            self, "Documentation",
            "Grease Analyzer with Visual AI\n\n"
            "1. Upload baseline data (reference)\n"
            "2. Upload sample files\n"
            "3. View graphs\n"
            "4. Generate visual AI analysis (LLaVA reads graphs)\n"
            "5. Export results\n\n"
            "LLaVA actually SEES the spectrum and reads peak positions!"
        )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            f"<h3>Grease Analyzer - Visual AI</h3>"
            f"<p><b>AI Model:</b> {self.llm_analyzer.model}</p>"
            f"<p>Uses LLaVA vision model to analyze FTIR graphs</p>"
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
