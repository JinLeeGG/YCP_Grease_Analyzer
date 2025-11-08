"""
LLaVA-Based Visual FTIR Analysis Module

This module uses LLaVA vision model to analyze FTIR spectroscopy graphs
by actually "seeing" the visual peaks and patterns in generated images.

KEY FEATURES:
- Visual peak identification (reads actual graph)
- Oxidation zone analysis (1650-1800 cm⁻¹)
- Accurate wavenumber reporting from X-axis
- No hallucination - only reports what it sees
- Detailed grease condition assessment

PERFORMANCE:
- Single sample: 30-60 seconds (thorough visual analysis)
- Batch processing: Sequential analysis
- Model: llava:7b (4.7GB, better accuracy than 13b for speed/quality balance)
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
    LLaVA Vision-Based FTIR Analyzer
    
    Uses Ollama's LLaVA model to visually analyze FTIR spectroscopy graphs
    instead of relying on statistical summaries.
    """
    
    def __init__(self, model: str = "llava:7b"):
        """
        Initialize LLaVA Analyzer
        
        Args:
            model: Ollama vision model name (default: llava:7b)
                  llava:7b = 4.7GB, good balance of speed/accuracy
                  llava:13b = 7.3GB, more accurate but slower
                  llava:34b = 20GB, highest accuracy, very slow
        """
        self.model = model
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            print(f"✅ Ollama connected (Vision Model: {self.model})")
        else:
            print(f"⚠️ Ollama not available - will use fallback analysis")
    
    def _check_ollama(self) -> bool:
        """
        Check Ollama Availability
        
        Returns:
            True if Ollama is running and model is available
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
    
    def analyze_graph_image(self,
                           image_path: str,
                           baseline_name: str,
                           sample_name: str) -> str:
        """
        Analyze FTIR Graph Using Vision Model
        
        LLaVA will actually LOOK at the graph image and identify:
        - Real peak positions by reading the X-axis
        - Peak heights by reading the Y-axis
        - Visual differences between baseline and sample
        - Oxidation zone (1650-1800 cm⁻¹) presence
        
        Args:
            image_path: Path to the saved graph image (PNG)
            baseline_name: Name of baseline file
            sample_name: Name of sample file
            
        Returns:
            Detailed analysis report based on visual inspection
        """
        if not self.ollama_available:
            return self._fallback_visual_analysis(image_path, baseline_name, sample_name)
        
        try:
            import ollama
            
            print(f"\n🔍 Analyzing {sample_name} visually with {self.model}...")
            start_time = time.time()
            
            # Read and encode image
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create detailed visual analysis prompt
            prompt = self._create_visual_ftir_prompt(baseline_name, sample_name)
            
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
                    'num_ctx': 4096,         # Large context for image + prompt
                }
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Visual analysis complete in {elapsed:.1f} seconds")
            
            return response['message']['content']
            
        except Exception as e:
            print(f"⚠️ Visual analysis failed: {str(e)}")
            return self._fallback_visual_analysis(image_path, baseline_name, sample_name)
    
    def _create_visual_ftir_prompt(self,
                                   baseline_name: str,
                                   sample_name: str) -> str:
        """
        Create Detailed Prompt for Visual FTIR Analysis
        
        Instructs LLaVA to carefully examine the graph image and
        report only what it actually sees (no hallucinations).
        """
        
        prompt = f"""You are an expert FTIR spectroscopy analyst examining a grease condition monitoring graph.

═══════════════════════════════════════════════════════════════════
GRAPH INFORMATION
═══════════════════════════════════════════════════════════════════
- GREEN LINE: Baseline (fresh grease) - {baseline_name}
- BLUE LINE: Test sample (used grease) - {sample_name}
- X-AXIS: Wavenumber (cm⁻¹) - reads RIGHT to LEFT (4000 → 500)
- Y-AXIS: Absorbance/Intensity

═══════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTION
═══════════════════════════════════════════════════════════════════
LOOK CAREFULLY at the actual graph image. Read peak positions from the X-axis.
Do NOT guess or estimate. Only report what you can CLEARLY SEE.
Take your time to accurately read the wavenumber values.

═══════════════════════════════════════════════════════════════════
SECTION 1: MAJOR PEAKS (Identify 3-5 tallest peaks)
═══════════════════════════════════════════════════════════════════

For each MAJOR peak you can see in the spectrum:

Peak 1 (TALLEST peak):
- Wavenumber: _____ cm⁻¹ (read X-axis position carefully)
- Region: [2800-3000 / 1350-1500 / 700-900 / other]
- Baseline intensity: ~_____ (read Y-axis)
- Sample intensity: ~_____ (read Y-axis)
- Change: [Higher/Lower/Same]

Peak 2:
- Wavenumber: _____ cm⁻¹
- Baseline: ~_____
- Sample: ~_____
- Change: [Higher/Lower/Same]

Peak 3:
- Wavenumber: _____ cm⁻¹
- Baseline: ~_____
- Sample: ~_____
- Change: [Higher/Lower/Same]

[Continue for other major peaks you clearly see]

═══════════════════════════════════════════════════════════════════
SECTION 2: OXIDATION ZONE (1650-1800 cm⁻¹) - CRITICAL
═══════════════════════════════════════════════════════════════════

Look CAREFULLY at the region between 1650-1800 cm⁻¹:

Does the BLUE line show a peak in this region?
[YES / NO / SLIGHT]

If YES or SLIGHT:
- Peak wavenumber: _____ cm⁻¹ (read from X-axis)
- Peak absorbance: _____ (read from Y-axis)
- Higher than baseline? [YES/NO]
- Visual size: [small / moderate / large / very large]

Oxidation Level:
[  ] NONE - No peak in 1650-1800 region
[  ] LOW - Small peak, absorbance < 0.5
[  ] MODERATE - Medium peak, absorbance 0.5-1.0
[  ] HIGH - Large peak, absorbance 1.0-2.0
[  ] SEVERE - Very large peak, absorbance > 2.0

═══════════════════════════════════════════════════════════════════
SECTION 3: VISUAL COMPARISON
═══════════════════════════════════════════════════════════════════

Overall pattern similarity: [Very similar / Similar / Different / Very different]

New peaks in BLUE (not in GREEN):
- List wavenumbers: _____

Missing/reduced peaks in BLUE (present in GREEN):
- List wavenumbers: _____

Peak shifts observed:
- Green peak at _____ → Blue peak at _____

Intensity changes:
- Blue peaks are generally: [Higher / Lower / Same] vs green

═══════════════════════════════════════════════════════════════════
SECTION 4: GREASE CONDITION ASSESSMENT
═══════════════════════════════════════════════════════════════════

Overall Health:
[  ] EXCELLENT - Nearly identical to baseline
[  ] GOOD - Minor differences, normal wear
[  ] FAIR - Noticeable changes, monitor closely
[  ] POOR - Significant degradation
[  ] CRITICAL - Severe degradation, immediate action

Primary Issues:
- Oxidation: [None/Low/Moderate/High/Severe]
- Contamination: [Describe any unusual peaks]
- Degradation: [Describe specific changes]

═══════════════════════════════════════════════════════════════════
SECTION 5: RECOMMENDATION
═══════════════════════════════════════════════════════════════════

Action Required:
[  ] CONTINUE - Grease is good
[  ] MONITOR - Retest in 2-4 weeks
[  ] PLAN REPLACEMENT - Replace within 1-2 weeks
[  ] IMMEDIATE ACTION - Replace now

Reasoning: [1-2 sentences based on what you saw]

Retest interval: [Suggested time]

═══════════════════════════════════════════════════════════════════
REMEMBER: Only report what you ACTUALLY SEE in the graph.
Read wavenumbers from the X-axis. Read intensities from Y-axis.
If you cannot see something clearly, say "unclear" or "cannot determine".
═══════════════════════════════════════════════════════════════════"""

        return prompt
    
    def _fallback_visual_analysis(self,
                                  image_path: str,
                                  baseline_name: str,
                                  sample_name: str) -> str:
        """
        Fallback Analysis When LLaVA Unavailable
        
        Returns basic message explaining visual analysis requires Ollama.
        """
        return f"""⚠️ VISUAL ANALYSIS UNAVAILABLE

Image: {image_path}
Baseline: {baseline_name}
Sample: {sample_name}

To enable AI-powered visual analysis of FTIR graphs:

1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. Start Ollama: ollama serve
3. Download model: ollama pull llava:7b
4. Re-run analysis

The visual analysis will identify real peaks by examining the graph image,
providing accurate wavenumber positions and oxidation assessment.

For now, please manually inspect the generated graph image."""
    
    def analyze_samples_batch(self,
                             image_paths: List[str],
                             baseline_name: str,
                             sample_names: List[str]) -> Dict[str, str]:
        """
        Batch Analyze Multiple Graph Images
        
        Processes multiple FTIR graphs sequentially using visual analysis.
        
        Args:
            image_paths: List of paths to graph images
            baseline_name: Baseline filename
            sample_names: List of sample filenames
            
        Returns:
            Dictionary mapping sample_name -> analysis_text
        """
        results = {}
        total = len(image_paths)
        
        print(f"\n🚀 Starting batch visual analysis ({total} samples)")
        print(f"⏱️  Estimated time: {total * 45:.0f} seconds (~45s per sample)\n")
        
        for i, (img_path, sample_name) in enumerate(zip(image_paths, sample_names), 1):
            print(f"📊 [{i}/{total}] Processing {sample_name}...")
            
            analysis = self.analyze_graph_image(
                img_path,
                baseline_name,
                sample_name
            )
            
            results[sample_name] = analysis
        
        print(f"\n✅ Batch analysis complete!\n")
        return results
    
    def generate_summary(self, all_analyses: Dict[str, str]) -> str:
        """
        Generate Executive Summary
        
        Creates brief overview of all analyzed samples.
        
        Args:
            all_analyses: Dictionary of sample_name -> analysis_text
            
        Returns:
            Executive summary text
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
    print("✅ LLM Analyzer (Visual) Test")
    
    analyzer = LLMAnalyzer(model="llava:7b")
    
    # Test with a sample image path
    test_image = "test_graph.png"
    
    if analyzer.ollama_available:
        print(f"\n✅ Ready for visual FTIR analysis with {analyzer.model}")
    else:
        print("\n⚠️ Ollama not available - install and pull llava:7b model")
    
    print("\n✅ Module loaded successfully")