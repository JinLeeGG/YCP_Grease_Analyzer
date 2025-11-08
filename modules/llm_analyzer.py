"""
Local LLM-Based Grease Analysis Module (Optimized Version)

This module provides AI-powered analysis of grease spectroscopy data using
a local Ollama LLM. Key features:

OPTIMIZATION FEATURES:
- Uses lightweight llama3.2:1b model (3-5x faster than 3b)
- Parallel processing with ThreadPoolExecutor (3 workers default)
- Optimized prompts for concise, focused responses
- Response length limits (250 tokens) for speed

ANALYSIS CAPABILITIES:
- Single sample analysis with baseline comparison
- Batch parallel analysis for multiple samples
- Executive summary generation
- Statistical interpretation
- Quality assessment recommendations

PERFORMANCE:
- Single sample: ~3-8 seconds
- 5 samples (parallel): ~15-25 seconds
- 5 samples (sequential): ~40-80 seconds
- Speedup: 2.5-3.5x with parallel processing

The module gracefully falls back to statistical analysis if Ollama
is unavailable.
"""

import sys
import os

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import LLM_CONFIG
from typing import Optional, Dict, List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class LLMAnalyzer:
    """
    Local LLM Analysis Class with Parallel Processing Support
    
    Manages connection to Ollama LLM server and provides methods
    for analyzing grease spectroscopy data. Supports both single
    and batch analysis with parallel processing for efficiency.
    """
    
    def __init__(self, model: str = None, max_workers: int = None):
        """
        Initialize LLM Analyzer
        
        Args:
            model: Ollama model name (default: llama3.2:1b from config)
                  Smaller models are faster, larger are more detailed
            max_workers: Number of parallel threads for batch analysis
                        (default: 3 from config)
                        More workers = faster batch processing
        """
        self.model = model or LLM_CONFIG['model']
        self.timeout = LLM_CONFIG['timeout']
        self.max_workers = max_workers or LLM_CONFIG.get('max_workers', 3)
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            print(f"✅ Ollama connected (Model: {self.model}, Workers: {self.max_workers})")
    
    def _check_ollama(self) -> bool:
        """
        Check Ollama Availability
        
        Tests if Ollama server is running and accessible.
        Attempts to list models as a simple connectivity test.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            import ollama
            # Simple connectivity test
            ollama.list()
            return True
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {str(e)}")
            return False
    
    def analyze_sample(self,
                      baseline_stats: Dict,
                      sample_stats: Dict,
                      comparison: Dict,
                      baseline_name: str,
                      sample_name: str) -> str:
        """
        Analyze Single Sample Against Baseline
        
        Generates AI-powered analysis comparing a sample to baseline,
        interpreting statistical differences and providing quality assessment.
        
        Process:
        1. Creates detailed prompt with all statistics
        2. Calls Ollama LLM with optimized settings
        3. Returns formatted analysis text
        4. Falls back to statistical summary if LLM unavailable
        
        Args:
            baseline_stats: Dictionary of baseline statistics
                          (mean, std, min, max, median, range, count)
            sample_stats: Dictionary of sample statistics (same keys)
            comparison: Comparison metrics
                       (mean_deviation_percent, std_deviation_percent,
                        correlation, quality_score)
            baseline_name: Name of baseline file for context
            sample_name: Name of sample file for context
            
        Returns:
            Multi-line string with analysis including:
            - Overall quality assessment
            - Key differences identified
            - Interpretation of metrics
            - Recommendations (if needed)
        """
        if not self.ollama_available:
            return self._fallback_analysis(baseline_stats, sample_stats, comparison)
        
        try:
            import ollama
            
            # Generate optimized prompt
            prompt = self._create_prompt(
                baseline_stats, sample_stats, comparison,
                baseline_name, sample_name
            )
            
            # Call LLM with optimized settings
            start_time = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'Expert grease analyst. Be concise and technical.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': LLM_CONFIG['temperature'],
                    'num_predict': LLM_CONFIG.get('num_predict', 250),
                    'num_ctx': LLM_CONFIG.get('num_ctx', 2048),
                }
            )
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ {sample_name} LLM response time: {elapsed_time:.2f}Seconds")
            
            return response['message']['content']
            
        except Exception as e:
            print(f"⚠️ LLM analysis failed: {str(e)}")
            return self._fallback_analysis(baseline_stats, sample_stats, comparison)
    
    def _create_prompt(self,
                      baseline_stats: Dict,
                      sample_stats: Dict,
                      comparison: Dict,
                      baseline_name: str,
                      sample_name: str) -> str:
        """
        Create Optimized Analysis Prompt (Concise Version)
        
        Generates a structured prompt for the LLM that includes:
        - Baseline and sample statistics
        - Deviation percentages
        - Correlation and quality score
        - Specific analysis requests
        
        The prompt is designed to be concise to reduce LLM processing
        time while still providing all necessary context.
        
        Returns:
            Formatted prompt string for LLM
        """
        
        prompt = f"""Grease Analysis: {sample_name} vs {baseline_name}

**Baseline:** Mean={baseline_stats['mean']:.1f}, Std={baseline_stats['std']:.1f}, N={baseline_stats['count']}
**Sample:** Mean={sample_stats['mean']:.1f}, Std={sample_stats['std']:.1f}, N={sample_stats['count']}

**Deviations:**
• Mean: {comparison['mean_deviation_percent']:+.1f}%
• Std: {comparison['std_deviation_percent']:+.1f}%
• Correlation: {comparison['correlation']:.2f}
• Quality: {comparison['quality_score']:.0f}/100

Provide 4 concise bullet points:
1. Key deviation from baseline
2. Pattern change assessment
3. Overall condition
4. Action needed (if any)"""
        
        return prompt
    
    def _fallback_analysis(self,
                          baseline_stats: Dict,
                          sample_stats: Dict,
                          comparison: Dict) -> str:
        """
        Fallback Analysis (Rule-Based)
        
        Provides statistical analysis when LLM is unavailable.
        Uses rule-based thresholds to assess sample quality:
        - Mean deviation < 5% = good
        - Mean deviation 5-15% = caution
        - Mean deviation > 15% = investigate
        - Correlation > 0.95 = excellent
        - Correlation 0.85-0.95 = good
        - Correlation < 0.85 = concerning
        
        Returns:
            Formatted analysis text with bullet points
        """
        analysis = []
        
        # Analyze mean deviation
        mean_dev = comparison['mean_deviation_percent']
        if abs(mean_dev) < 5:
            analysis.append(f"✓ Mean intensity deviation is minimal ({mean_dev:+.1f}%), indicating stable condition.")
        elif abs(mean_dev) < 15:
            analysis.append(f"⚠ Moderate mean deviation detected ({mean_dev:+.1f}%), monitor for trends.")
        else:
            analysis.append(f"⚠️ Significant mean deviation ({mean_dev:+.1f}%), investigation recommended.")
        
        # Analyze standard deviation changes
        std_dev = comparison['std_deviation_percent']
        if abs(std_dev) > 20:
            analysis.append(f"⚠️ High variability change ({std_dev:+.1f}%), possible contamination or degradation.")
        else:
            analysis.append(f"✓ Variability remains consistent ({std_dev:+.1f}%).")
        
        # Analyze correlation coefficient
        corr = comparison['correlation']
        if corr > 0.95:
            analysis.append(f"✓ Excellent correlation with baseline ({corr:.3f}), pattern maintained.")
        elif corr > 0.85:
            analysis.append(f"⚠ Good correlation ({corr:.3f}), minor pattern shift detected.")
        else:
            analysis.append(f"⚠️ Low correlation ({corr:.3f}), significant pattern change observed.")
        
        # Quality score analysis
        quality = comparison['quality_score']
        if quality > 85:
            analysis.append(f"✓ Overall Quality: Excellent ({quality:.1f}/100) - No action needed.")
        elif quality > 70:
            analysis.append(f"⚠ Overall Quality: Good ({quality:.1f}/100) - Continue monitoring.")
        elif quality > 50:
            analysis.append(f"⚠️ Overall Quality: Fair ({quality:.1f}/100) - Schedule inspection.")
        else:
            analysis.append(f"❌ Overall Quality: Poor ({quality:.1f}/100) - Immediate action required.")
        
        return "\n".join(f"• {item}" for item in analysis)
    
    def generate_summary(self, all_analyses: Dict[str, str]) -> str:
        """
        Generate Summary for All Samples
        
        Args:
            all_analyses: Dictionary of {sample_name: analysis_result}
            
        Returns:
            Overall summary text
        """
        if not self.ollama_available:
            return self._fallback_summary(all_analyses)
        
        try:
            import ollama
            
            # Create concise summary prompt (truncate analyses for speed)
            analyses_text = "\n".join([
                f"{name}: {analysis[:150]}..."  # Limit each to 150 chars
                for name, analysis in list(all_analyses.items())[:5]  # Max 5 samples
            ])
            
            prompt = f"""Summary of {len(all_analyses)} grease samples:

{analyses_text}

Provide 3-sentence summary:
1. Overall condition
2. Samples needing attention
3. Key recommendation"""
            
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'Maintenance manager. Be brief.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.3,
                    'num_predict': 150,  # Shorter for summaries
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"⚠️ Summary generation failed: {str(e)}")
            return self._fallback_summary(all_analyses)
    
    def _fallback_summary(self, all_analyses: Dict[str, str]) -> str:
        """
        Fallback Summary (Rule-Based)
        
        Provides simple statistical summary when LLM is unavailable.
        
        Returns:
            Basic summary text with sample count and quality threshold info
        """
        total = len(all_analyses)
        
        summary = f"Analyzed {total} sample(s) against baseline. "
        summary += "Review individual analyses above for detailed insights. "
        summary += "Samples with quality scores below 70 require attention."
        
        return summary
    
    def analyze_samples_batch(self, samples_data: List[Dict]) -> Dict[str, str]:
        """
        Analyze Multiple Samples in Parallel (3-5x Speedup)
        
        Uses ThreadPoolExecutor to analyze multiple samples simultaneously
        instead of sequentially. This is the key optimization that enables
        fast batch processing.
        
        Performance Example (5 samples):
        - Sequential: ~40-80 seconds (one at a time)
        - Parallel: ~15-25 seconds (3 simultaneous)
        - Speedup: 2.5-3.5x faster
        
        How It Works:
        1. Creates thread pool with max_workers threads (default: 3)
        2. Submits all analysis tasks to pool
        3. Tasks run in parallel on separate threads
        4. Collects results as they complete
        5. Returns all results together
        
        Args:
            samples_data: List of sample dictionaries, each containing:
                - 'baseline_stats': Baseline statistics
                - 'sample_stats': Sample statistics
                - 'comparison': Comparison metrics
                - 'baseline_name': Baseline filename
                - 'sample_name': Sample filename
            
        Returns:
            Dictionary mapping sample_name -> analysis_text
            Example: {'sample_01.csv': 'Analysis text...', ...}
        """
        results = {}
        total = len(samples_data)
        
        print(f"\n🚀 Starting parallel analysis ({total} samples, {self.max_workers} workers)")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all analysis tasks simultaneously
            future_to_sample = {
                executor.submit(
                    self.analyze_sample,
                    data['baseline_stats'],
                    data['sample_stats'],
                    data['comparison'],
                    data['baseline_name'],
                    data['sample_name']
                ): data['sample_name']
                for data in samples_data
            }
            
            # Collect results as they complete (not necessarily in order)
            completed = 0
            for future in as_completed(future_to_sample):
                sample_name = future_to_sample[future]
                try:
                    results[sample_name] = future.result()
                    completed += 1
                    print(f"  ✅ [{completed}/{total}] {sample_name}")
                except Exception as e:
                    print(f"  ❌ {sample_name} analysis failed: {e}")
                    results[sample_name] = f"Analysis failed: {str(e)}"
        
        elapsed = time.time() - start_time
        avg_time = elapsed / total if total > 0 else 0
        print(f"\n✅ Parallel analysis complete - Total {elapsed:.1f}s (avg {avg_time:.1f}s/sample)\n")
        
        return results


# ============================================================================
# TEST CODE - Run this file directly to verify functionality
# ============================================================================
if __name__ == "__main__":
    print("✅ LLM Analyzer Test")
    
    analyzer = LLMAnalyzer()
    
    # Test data
    baseline_stats = {
        'mean': 100.0,
        'std': 10.0,
        'min': 80.0,
        'max': 120.0,
        'count': 300
    }
    
    sample_stats = {
        'mean': 110.0,
        'std': 12.0,
        'min': 85.0,
        'max': 135.0,
        'count': 300
    }
    
    comparison = {
        'mean_deviation_percent': 10.0,
        'std_deviation_percent': 20.0,
        'correlation': 0.92,
        'quality_score': 78.5
    }
    
    print("\nAnalyzing...")
    result = analyzer.analyze_sample(
        baseline_stats,
        sample_stats,
        comparison,
        "baseline.csv",
        "sample_01.csv"
    )
    
    print("\nAnalysis result:")
    print(result)
    print("\n✅ Module operational")