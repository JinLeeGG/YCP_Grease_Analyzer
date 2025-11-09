"""
Quick performance benchmark for optimized deviation analyzer
"""
import time
import numpy as np
from modules.ftir_deviation_analyzer import FTIRDeviationAnalyzer, DeviationConfig

print("=" * 70)
print("PERFORMANCE BENCHMARK - Optimized Deviation Analyzer")
print("=" * 70)

# Generate synthetic test data (same as test)
wavenumbers = np.linspace(600, 4000, 2000)

# Baseline: clean spectrum
baseline = np.zeros_like(wavenumbers)
baseline += 0.5 * np.exp(-((wavenumbers - 2900) / 50)**2)  # C-H
baseline += 0.2 * np.exp(-((wavenumbers - 1460) / 30)**2)  # C-H bend
baseline += 0.02 * np.random.randn(len(wavenumbers))  # Noise

# Sample: with deviations
sample = baseline.copy()
sample += 0.15 * np.exp(-((wavenumbers - 1725) / 40)**2)  # Carbonyl deviation
sample += 0.08 * np.exp(-((wavenumbers - 3400) / 100)**2)  # Water deviation
sample += 0.22 * np.exp(-((wavenumbers - 2345) / 30)**2)  # Outlier
sample += 0.02 * np.random.randn(len(wavenumbers))  # Noise

# Create analyzer
config = DeviationConfig()
analyzer = FTIRDeviationAnalyzer(config)

# Run multiple times to get average
print("\nRunning 5 analyses to measure average performance...")
times = []

for i in range(5):
    start_time = time.time()
    result = analyzer.analyze(
        wavenumbers, baseline,
        wavenumbers, sample,
        "baseline_test.csv",
        f"sample_test_{i}.csv"
    )
    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f}s - Category: {result['multi_metric_category']['category']}")

avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)

print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)
print(f"Average time: {avg_time:.3f} seconds")
print(f"Min time:     {min_time:.3f} seconds")
print(f"Max time:     {max_time:.3f} seconds")
print("\n✅ Target: <1 second per sample")
if avg_time < 1.0:
    print(f"✅ SUCCESS: {avg_time:.3f}s is {'MUCH ' if avg_time < 0.5 else ''}faster than 1s target!")
elif avg_time < 5.0:
    print(f"⚠️  GOOD: {avg_time:.3f}s is acceptable (target <1s, much better than 97s!)")
else:
    print(f"⚠️  NEEDS WORK: {avg_time:.3f}s (target <1s)")

print(f"\nImprovement over 97s baseline: {97/avg_time:.1f}x faster")
print("=" * 70)
