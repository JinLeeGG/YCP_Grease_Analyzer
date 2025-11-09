"""
PDF Report Export Module

Generates professional PDF reports from FTIR deviation analysis results.
Includes:
- Cover page with metadata
- Executive summary
- Individual sample analyses with multi-metric categorization
- Decision logic and metrics
- Embedded graphs (optional)
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                    PageBreak, Table, TableStyle, Image, Preformatted)
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
else:
    # Runtime imports
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                    PageBreak, Table, TableStyle, Image)
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Dict, List, Optional
import os


class PDFExporter:
    """
    Professional PDF Report Generator for FTIR Deviation Analysis
    
    Creates publication-quality PDF reports with:
    - Professional styling and layout
    - Color-coded category badges
    - Detailed metrics tables
    - Embedded graphs
    - Complete traceability
    """
    
    def __init__(self):
        """Initialize PDF exporter with custom styling"""
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for professional appearance"""
        # Main title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#4a6fa5'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=15,
            spaceBefore=25,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3a6fa0'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Code/monospace style
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10
        ))
    
    def export_analysis_report(self,
                              output_path: str,
                              baseline_name: str,
                              analyses: Dict[str, Dict],
                              executive_summary: str,
                              graph_paths: Optional[List[str]] = None) -> bool:
        """
        Export complete analysis report to PDF
        
        Args:
            output_path: Full path for output PDF file
            baseline_name: Name of baseline file
            analyses: Dictionary mapping sample names to analysis results
            executive_summary: Executive summary text
            graph_paths: Optional list of graph image paths (same order as analyses)
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Build content
            story = []
            
            # Cover page
            story.extend(self._create_cover_page(baseline_name, len(analyses)))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._create_executive_summary_section(executive_summary))
            story.append(PageBreak())
            
            # Individual analyses
            sample_names = list(analyses.keys())
            for idx, sample_name in enumerate(sample_names):
                analysis = analyses[sample_name]
                
                story.extend(self._create_sample_analysis_page(
                    sample_name, analysis, idx + 1, len(analyses)
                ))
                
                # Add graph if available
                if graph_paths and idx < len(graph_paths):
                    graph_path = graph_paths[idx]
                    if os.path.exists(graph_path):
                        story.extend(self._add_graph_section(graph_path))
                
                # Page break between samples (except last)
                if idx < len(analyses) - 1:
                    story.append(PageBreak())
            
            # Build PDF
            doc.build(story)
            
            print(f"✅ PDF report exported successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ PDF export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_cover_page(self, baseline_name: str, sample_count: int) -> List:
        """Create professional cover page"""
        content = []
        
        # Add space from top
        content.append(Spacer(1, 1.5*inch))
        
        # Main title
        content.append(Paragraph(
            "FTIR DEVIATION ANALYSIS",
            self.styles['CustomTitle']
        ))
        content.append(Paragraph(
            "Multi-Metric Statistical Report",
            self.styles['Subtitle']
        ))
        
        content.append(Spacer(1, 0.8*inch))
        
        # Metadata table
        current_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        metadata = [
            ['Report Generated:', current_date],
            ['', ''],
            ['Baseline File:', baseline_name],
            ['Total Samples Analyzed:', str(sample_count)],
            ['Analysis Method:', 'Multi-Metric Deviation Analysis'],
            ['', ''],
            ['Metrics Evaluated:', '4 Parameters (r, ΔY, ΔX, Ratio)'],
            ['Decision Categories:', '6 Classifications'],
        ]
        
        table = Table(metadata, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c5aa0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 1), (-1, 1), 1, colors.HexColor('#e0e0e0')),
            ('LINEBELOW', (0, 5), (-1, 5), 1, colors.HexColor('#e0e0e0')),
        ]))
        
        content.append(table)
        content.append(Spacer(1, inch))
        
        # Method description box
        method_text = """
        <b>Analysis Methodology:</b><br/><br/>
        This report employs a multi-parameter deviation analysis system that evaluates 
        spectral differences using four key metrics:<br/>
        <br/>
        • <b>Correlation (r)</b> - Spectral shape similarity<br/>
        • <b>Vertical Deviation (ΔY)</b> - Intensity changes in critical chemical regions<br/>
        • <b>Horizontal Shift (ΔX)</b> - Peak position changes (wavenumber shifts)<br/>
        • <b>ΔX:ΔY Ratio</b> - Deviation type characterization<br/>
        <br/>
        Each sample is categorized using transparent, statistically-based thresholds with 
        explained decision logic. This approach provides reliable, reproducible results 
        without quality inferences, allowing proper interpretation based on known baseline type.
        """
        
        content.append(Paragraph(method_text, self.styles['Normal']))
        
        return content
    
    def _create_executive_summary_section(self, summary_text: str) -> List:
        """Create executive summary page with proper formatting"""
        content = []
        
        content.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeading']))
        content.append(Spacer(1, 0.2*inch))
        
        # Split summary into sections for better formatting
        lines = summary_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                content.append(Spacer(1, 0.1*inch))
                continue
            
            # Detect section headers (lines with ":" or all caps)
            if line.endswith(':') or (line.isupper() and len(line) < 50):
                if len(line) > 50:  # Long separator lines
                    content.append(Paragraph('<hr width="100%"/>', self.styles['Normal']))
                else:
                    content.append(Paragraph(f"<b>{line}</b>", self.styles['SubsectionHeading']))
            else:
                # Regular text with proper HTML encoding
                line_html = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Re-enable HTML for specific symbols
                line_html = line_html.replace('✅', '✅').replace('⚠️', '⚠️')
                line_html = line_html.replace('❌', '❌').replace('🚨', '🚨').replace('⚡', '⚡')
                content.append(Paragraph(line_html, self.styles['Normal']))
        
        return content
    
    def _create_sample_analysis_page(self, sample_name: str, 
                                    analysis: Dict, idx: int, total: int) -> List:
        """Create detailed individual sample analysis page"""
        content = []
        
        # Sample header
        header_text = f"Sample {idx} of {total}: <b>{sample_name}</b>"
        content.append(Paragraph(header_text, self.styles['SectionHeading']))
        content.append(Spacer(1, 0.15*inch))
        
        # Check if analysis is a string (old format) or dict (new format)
        if isinstance(analysis, str):
            # Old format - just text analysis
            content.extend(self._create_category_badge('UNKNOWN', 0))
            content.append(Spacer(1, 0.2*inch))
            
            content.append(Paragraph("<b>Analysis Results:</b>", self.styles['SubsectionHeading']))
            content.append(Spacer(1, 0.1*inch))
            
            # Format the text properly for PDF
            analysis_html = analysis.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Preserve line breaks
            for line in analysis_html.split('\n'):
                line = line.strip()
                if line:
                    # Check for section headers (lines ending with : or all caps)
                    if line.endswith(':') or (line.isupper() and len(line) < 50):
                        content.append(Paragraph(f"<b>{line}</b>", self.styles['SubsectionHeading']))
                    else:
                        content.append(Paragraph(line, self.styles['Normal']))
                else:
                    content.append(Spacer(1, 0.05*inch))
            
            return content
        
        # New format with structured data (dictionary)
        category_data = analysis.get('multi_metric_category', {})
        category = category_data.get('category', 'UNKNOWN')
        confidence = category_data.get('confidence', 0)
        reasoning_steps = category_data.get('reasoning_steps', [])
        metrics = category_data.get('metrics_used', {})
        
        # Category badge with color coding
        content.extend(self._create_category_badge(category, confidence))
        content.append(Spacer(1, 0.2*inch))
        
        # Baseline compatibility
        baseline_compat = analysis.get('baseline_compatibility', {})
        if baseline_compat:
            content.append(Paragraph("<b>Baseline Compatibility:</b>", self.styles['SubsectionHeading']))
            
            correlation = baseline_compat.get('correlation', 0)
            warning_level = baseline_compat.get('warning_level', None)
            
            compat_data = [
                ['Spectral Correlation (r):', f"{correlation:.3f}"],
                ['Status:', warning_level if warning_level else '✓ Compatible']
            ]
            
            compat_table = Table(compat_data, colWidths=[2.5*inch, 4*inch])
            compat_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f8ff')),
            ]))
            
            content.append(compat_table)
            content.append(Spacer(1, 0.2*inch))
        
        # Decision logic
        if reasoning_steps:
            content.append(Paragraph("<b>Decision Logic:</b>", self.styles['SubsectionHeading']))
            
            for i, step in enumerate(reasoning_steps, 1):
                # Clean step text
                step_clean = step.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                content.append(Paragraph(
                    f"{i}. {step_clean}",
                    self.styles['Normal']
                ))
            content.append(Spacer(1, 0.2*inch))
        
        # Key metrics table
        if metrics:
            content.append(Paragraph("<b>Key Metrics:</b>", self.styles['SubsectionHeading']))
            
            metrics_data = [
                ['Metric', 'Value', 'Description'],
                ['Correlation (r)', 
                 f"{metrics.get('correlation', 0):.3f}", 
                 'Spectral shape similarity'],
                ['Max ΔY', 
                 f"{metrics.get('max_delta_y', 0):.4f} A", 
                 'Intensity deviation'],
                ['Max ΔX', 
                 f"{metrics.get('max_delta_x', 0):.2f} cm⁻¹", 
                 'Peak shift'],
                ['ΔX:ΔY Ratio', 
                 f"{metrics.get('max_ratio', 0):.2f}", 
                 'Deviation type'],
                ['Critical Outliers', 
                 str(metrics.get('critical_outliers', 0)), 
                 'Unexpected deviations'],
            ]
            
            metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.8*inch, 3*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6f2ff')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ]))
            
            content.append(metrics_table)
        
        return content
    
    def _create_category_badge(self, category: str, confidence: float) -> List:
        """Create color-coded category badge"""
        content = []
        
        # Category colors and symbols
        category_info = {
            'GOOD': (colors.HexColor('#28a745'), '✓', 'GOOD'),
            'REQUIRES_ATTENTION': (colors.HexColor('#ffc107'), '⚠', 'REQUIRES ATTENTION'),
            'CRITICAL': (colors.HexColor('#dc3545'), '✕', 'CRITICAL'),
            'OUTLIER': (colors.HexColor('#8b0000'), '!', 'OUTLIER'),
            'BASELINE_MISMATCH': (colors.HexColor('#007bff'), '≠', 'BASELINE MISMATCH'),
            'UNKNOWN': (colors.grey, '?', 'UNKNOWN')
        }
        
        color, symbol, display_text = category_info.get(
            category, 
            (colors.grey, '?', category)
        )
        
        # Create badge table
        confidence_text = f"Confidence: {confidence:.0%}" if confidence > 0 else "Analysis Complete"
        badge_data = [
            [f"{symbol} {display_text}"],
            [confidence_text]
        ]
        
        badge_table = Table(badge_data, colWidths=[6.5*inch])
        badge_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), color),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, 0), 16),
            ('FONTSIZE', (0, 1), (-1, 1), 11),
            ('TOPPADDING', (0, 0), (-1, 0), 15),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
        ]))
        
        content.append(badge_table)
        
        return content
    
    def _add_graph_section(self, graph_path: str) -> List:
        """Add graph image section to PDF"""
        content = []
        
        try:
            content.append(Spacer(1, 0.3*inch))
            content.append(Paragraph("<b>Spectral Overlay:</b>", self.styles['SubsectionHeading']))
            content.append(Spacer(1, 0.1*inch))
            
            # Add image (scaled to fit page width while maintaining aspect ratio)
            img = Image(graph_path)
            img._restrictSize(6.5*inch, 4*inch)  # Max dimensions
            content.append(img)
            
        except Exception as e:
            content.append(Paragraph(
                f"<i>Graph image not available: {str(e)}</i>",
                self.styles['Normal']
            ))
        
        return content


# ============================================================================
# TEST CODE
# ============================================================================
if __name__ == "__main__":
    print("✅ PDF Exporter Module Test")
    
    # Create test data
    test_analyses = {
        'sample_01.csv': {
            'multi_metric_category': {
                'category': 'CRITICAL',
                'confidence': 0.95,
                'reasoning_steps': [
                    'High spectral correlation (r=0.978)',
                    'BUT high intensity deviation (ΔY=0.125 A)',
                    'Indicates intensity-dominant deviation'
                ],
                'metrics_used': {
                    'correlation': 0.978,
                    'max_delta_y': 0.125,
                    'max_delta_x': 2.3,
                    'max_ratio': 18.5,
                    'critical_outliers': 0
                }
            },
            'baseline_compatibility': {
                'correlation': 0.978,
                'warning_level': None
            }
        }
    }
    
    test_summary = """
EXECUTIVE SUMMARY

Total Samples: 1
CRITICAL: 1 sample

Review individual analyses for details.
    """
    
    exporter = PDFExporter()
    success = exporter.export_analysis_report(
        output_path="test_report.pdf",
        baseline_name="baseline.csv",
        analyses=test_analyses,
        executive_summary=test_summary
    )
    
    if success:
        print("✅ Test PDF generated: test_report.pdf")
    else:
        print("❌ Test failed")
