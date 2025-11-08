"""
LLaVA-Based Hybrid FTIR Analysis Module

This module uses a hybrid approach:
1. Numerical Accuracy: Uses precise data from PeakDetector for peak locations and changes.
2. Visual Interpretation: Uses LLaVA to confirm visual patterns (oxidation, contamination, trend).

KEY FEATURES:
- High Accuracy: Peak detection is numerical, not visual.
- Dynamic Peaks: Reports all peaks found by PeakDetector.
- Oxidation Zone: Hardcoded for chemical accuracy (1650-1800 cm⁻¹).

PERFORMANCE:
- Target: Under 45 seconds (due to faster model and clearer tasking).
- Model: llava:7b-v1.6 (Faster, balanced version)
"""

import sys
import os

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import LLM_CONFIG
from typing import Optional, Dict, List
import time
import base64
from pathlib import Path


class LLMAnalyzer:
    """
    LLaVA Hybrid FTIR Analyzer
    
    Uses Ollama's LLaVA model to interpret a graph image AND provided
    numerical peak data simultaneously for a complete analysis.
    """
    
    # 1. SPEED FIX: Using the successfully pulled, faster model
    def __init__(self, model: str = "llava:7b-v1.6"):
        """
        Initialize LLaVA Analyzer
        
        Args:
            model: Ollama vision model name (default: llava:7b-v1.6 for speed)
        """
        self.model = model
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            print(f"✅ Ollama connected (Hybrid Model: {self.model})")
        else:
            print(f"⚠️ Ollama not available - will use fallback analysis")
    
    def _check_ollama(self) -> bool:
        """
        Check Ollama Availability
        """
        try:
            import ollama
            # Check if our vision model is available
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if not any(self.model in name for name in model_names):
                print(f"⚠️ Model {self.model} not found. Run: ollama pull {self.model}")
                return False
            
            return True
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {str(e)}")
            return False
    
    # 2. ACCURACY FIX: New signature for hybrid analysis
    def analyze_ftir_hybrid(self,
                            image_path: str,
                            baseline_name: str,
                            sample_name: str,
                            peak_analysis_data: str) -> str:
        """
        Analyze FTIR Graph using Hybrid Approach (Image + Numerical Data)
        
        LLaVA is asked to use the provided 'peak_analysis_data' for accurate numbers
        and the 'image_path' for visual confirmation of general trends (e.g., oxidation peak presence).
        
        Args:
            image_path: Path to the saved graph image (PNG/JPG)
            baseline_name: Name of baseline file
            sample_name: Name of sample file
            peak_analysis_data: Clean, formatted text string of accurate peak data from PeakDetector.
            
        Returns:
            Detailed analysis report based on hybrid inspection
        """
        if not self.ollama_available:
            return self._fallback_visual_analysis(image_path, baseline_name, sample_name)
        
        try:
            import ollama
            
            print(f"\n🔍 Analyzing {sample_name} with hybrid data/vision approach...")
            start_time = time.time()
            
            # Read and encode image
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create detailed hybrid analysis prompt
            prompt = self._create_hybrid_ftir_prompt(baseline_name, sample_name, peak_analysis_data)
            
            # Call LLaVA vision model
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]
                }],
                options={
                    'temperature': 0.1,      # Low temp for factual analysis
                    'num_predict': 1000,     # Allow detailed response
                    'num_ctx': 2048,         # Large context for image + prompt
                }
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Hybrid analysis complete in {elapsed:.1f} seconds")
            
            return response['message']['content']
            
        except Exception as e:
            print(f"⚠️ Hybrid analysis failed: {str(e)}")
            return self._fallback_visual_analysis(image_path, baseline_name, sample_name)
    
    # 3. ACCURACY FIX: New prompt function name and content
    def _create_hybrid_ftir_prompt(self, baseline_name: str, sample_name: str, peak_analysis_data: str) -> str:
        """Concise hybrid prompt for high-accuracy analysis"""
        
        prompt = f"""You are an expert FTIR spectroscopy analyst specializing in industrial grease condition monitoring. Your task is to interpret the provided **NUMERICAL PEAK DATA** and **visually confirm** key findings from the graph.

═══════════════════════════════════════════════════════════════════
NUMERICAL PEAK DATA (Accurate Input from PeakDetector):
═══════════════════════════════════════════════════════════════════
{peak_analysis_data} 
(INSTRUCTION: For all wavenumber values, peak counts, and percentage changes, STRICTLY USE the data provided above. The image is for context and visual trend confirmation only.)

═══════════════════════════════════════════════════════════════════
GRAPH INFORMATION
═══════════════════════════════════════════════════════════════════
- GREEN LINE: Baseline grease - {baseline_name}
- BLUE LINE: Used sample - {sample_name}

Provide a concise, professional assessment using the following structure:

1. **PEAK DATA SUMMARY (Dynamic Peaks)**:
    - New Peaks in Sample: [List all new peaks detected in the numerical data]
    - Major Intensity Increases (>10%): [List up to 3 peaks from the numerical data]
    - Major Intensity Decreases (<-10%): [List up to 3 peaks from the numerical data]

2. **OXIDATION ANALYSIS (1650-1800 cm⁻¹)**: 
    - Is there a clear visual increase or new peak on the BLUE line in the 1650-1800 cm⁻¹ region? [YES/NO]
    - If YES: What is the exact Wavenumber and % Change reported in the NUMERICAL PEAK DATA for the most significant peak in this range?
    - Severity: [NONE/LOW/MODERATE/HIGH/SEVERE]

3. **CONTAMINATION (Visual Check)**:
    - Water/Glycol Peak (Visual Check near 3300-3500 cm⁻¹): [Present/Absent]
    - Soot/Diesel (Visual Check near 2100-2400 cm⁻¹): [Present/Absent]

4. **Assessment & Action**:
    - Condition: [EXCELLENT/GOOD/FAIR/POOR]
    - Action: [CONTINUE/MONITOR/REPLACE]
    - Retest in: ___ weeks
    - Brief Reasoning (2-3 sentences explaining the decision based on the numerical data and visual confirmation).

Be precise. Do not guess wavenumber values from the graph."""

        return prompt
    
    def _fallback_visual_analysis(self,
                                  image_path: str,
                                  baseline_name: str,
                                  sample_name: str) -> str:
        """
        Fallback Analysis When LLaVA Unavailable
        """
        return f"""⚠️ HYBRID ANALYSIS UNAVAILABLE

Image: {image_path}
Baseline: {baseline_name}
Sample: {sample_name}

To enable AI-powered hybrid analysis of FTIR graphs:

1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. Start Ollama: ollama serve
3. Download model: ollama pull {self.model}
4. Re-run analysis

The hybrid analysis uses both accurate numerical data and visual confirmation.
For now, please manually inspect the generated graph image."""
    
    # NOTE: analyze_samples_batch method is removed as it is now orchestrated in app.py's AnalysisWorker.
    # The AnalysisWorker loop will handle the sequential calling of analyze_ftir_hybrid
    
    def generate_summary(self, all_analyses: Dict[str, str]) -> str:
        """
        Generate Executive Summary
        """
        total = len(all_analyses)
        
        # Count action levels from analyses
        immediate = sum(1 for a in all_analyses.values() if 'IMMEDIATE' in a.upper())
        poor = sum(1 for a in all_analyses.values() if 'POOR' in a.upper() or 'CRITICAL' in a.upper())
        
        summary = f"""
═══════════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════

Total Samples Analyzed: {total}

Status Overview:
- Samples requiring immediate action: {immediate}
- Samples in poor/critical condition: {poor}
- Samples acceptable: {total - poor}

Recommendation:
"""
        
        if immediate > 0:
            summary += f"⚠️ URGENT: {immediate} sample(s) need immediate replacement.\n"
        elif poor > 0:
            summary += f"⚠️ ATTENTION: {poor} sample(s) showing degradation - plan replacement.\n"
        else:
            summary += "✅ All samples within acceptable parameters. Continue normal monitoring.\n"
        
        summary += "\nReview individual sample analyses below for detailed assessments.\n"
        summary += "═══════════════════════════════════════════════════════════════════\n"
        
        return summary


# ============================================================================
# TEST CODE
# ============================================================================
if __name__ == "__main__":
    print("✅ LLM Analyzer (Hybrid) Test")
    
    # Use the specific model we pulled
    analyzer = LLMAnalyzer(model="llava:7b-v1.6")
    
    # Test with a sample image path
    test_image = "test_graph.png"
    
    if analyzer.ollama_available:
        print(f"\n✅ Ready for hybrid FTIR analysis with {analyzer.model}")
    else:
        print("\n⚠️ Ollama not available - install and pull model")
    
    print("\n✅ Module loaded successfully")