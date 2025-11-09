"""
Test script to demonstrate improved BASELINE_MISMATCH vs OUTLIER categorization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.ftir_deviation_analyzer import FTIRDeviationAnalyzer, DeviationConfig, RegionDeviation, OutlierDeviation

def test_categorization():
    """Test the improved categorization logic"""
    
    config = DeviationConfig()
    analyzer = FTIRDeviationAnalyzer(config)
    
    print("=" * 80)
    print("IMPROVED CATEGORIZATION TEST")
    print("=" * 80)
    
    # Test Case 1: Different formulation (should be BASELINE_MISMATCH)
    print("\n" + "=" * 80)
    print("TEST CASE 1: Different Formulation (Systematic Deviation, No Degradation)")
    print("=" * 80)
    
    region_deviations_case1 = [
        RegionDeviation('carbonyl_oxidation', (1650, 1800), 0.04, 1740, 40.0, 2.0, 50.0, 'major', 'minor deviation'),
        RegionDeviation('water_contamination', (3200, 3600), 0.02, 3400, 20.0, 1.5, 75.0, 'minor', 'minor deviation'),
        RegionDeviation('additives_glycol', (1000, 1300), 0.08, 1150, 80.0, 3.0, 37.5, 'major', 'moderate deviation'),
        RegionDeviation('ch_stretch', (2850, 2950), 0.06, 2900, 60.0, 2.5, 41.7, 'major', 'moderate deviation')
    ]
    
    outliers_case1 = [
        OutlierDeviation(1200, 0.05, 50.0, 3.5, 'minor'),
        OutlierDeviation(2500, 0.04, 40.0, 3.2, 'minor')
    ]
    
    result1 = analyzer.categorize_sample(
        correlation=0.82,  # Low correlation
        region_deviations=region_deviations_case1,
        outliers=outliers_case1
    )
    
    print(f"\n📊 Category: {result1['category']}")
    print(f"📊 Confidence: {result1['confidence']*100:.0f}%")
    print(f"\n💡 Reasoning:")
    for reason in result1['reasoning']:
        if reason:  # Skip empty strings
            print(f"   {reason}")
    
    print(f"\n📈 Metrics:")
    print(f"   - Correlation: {result1['metrics']['correlation']:.3f}")
    print(f"   - Max ΔY: {result1['metrics']['max_delta_y']:.3f} A")
    print(f"   - Max ΔX: {result1['metrics']['max_delta_x']:.1f} cm⁻¹")
    print(f"   - Critical outliers: {result1['metrics']['critical_outliers']}")
    if 'systematic_deviation' in result1['metrics']:
        print(f"   - Systematic deviation: {result1['metrics']['systematic_deviation']}")
    
    # Test Case 2: Contamination/Degradation (should be OUTLIER)
    print("\n" + "=" * 80)
    print("TEST CASE 2: Contamination/Degradation (Oxidation + Water Pattern)")
    print("=" * 80)
    
    region_deviations_case2 = [
        RegionDeviation('carbonyl_oxidation', (1650, 1800), 0.25, 1740, 250.0, 5.0, 20.0, 'critical', 'significant oxidation'),
        RegionDeviation('water_contamination', (3200, 3600), 0.18, 3400, 180.0, 3.0, 16.7, 'critical', 'water contamination'),
        RegionDeviation('additives_glycol', (1000, 1300), 0.08, 1150, 80.0, 1.0, 12.5, 'minor', 'minor deviation'),
        RegionDeviation('ch_stretch', (2850, 2950), 0.06, 2900, 60.0, 0.5, 8.3, 'minor', 'minor deviation')
    ]
    
    outliers_case2 = [
        OutlierDeviation(1725, 0.22, 220.0, 7.5, 'critical'),
        OutlierDeviation(3450, 0.15, 150.0, 6.2, 'major')
    ]
    
    result2 = analyzer.categorize_sample(
        correlation=0.78,  # Low correlation
        region_deviations=region_deviations_case2,
        outliers=outliers_case2
    )
    
    print(f"\n📊 Category: {result2['category']}")
    print(f"📊 Confidence: {result2['confidence']*100:.0f}%")
    print(f"\n💡 Reasoning:")
    for reason in result2['reasoning']:
        if reason:  # Skip empty strings
            print(f"   {reason}")
    
    print(f"\n📈 Metrics:")
    print(f"   - Correlation: {result2['metrics']['correlation']:.3f}")
    print(f"   - Max ΔY: {result2['metrics']['max_delta_y']:.3f} A")
    print(f"   - Max ΔX: {result2['metrics']['max_delta_x']:.1f} cm⁻¹")
    print(f"   - Critical outliers: {result2['metrics']['critical_outliers']}")
    
    # Test Case 3: Extreme deviation (should be OUTLIER due to extreme ΔY)
    print("\n" + "=" * 80)
    print("TEST CASE 3: Extreme Deviation (ΔY >> 1.0 A)")
    print("=" * 80)
    
    region_deviations_case3 = [
        RegionDeviation('carbonyl_oxidation', (1650, 1800), 2.03, 1741, 535.0, 0.0, 0.0, 'critical', 'extreme deviation'),
        RegionDeviation('water_contamination', (3200, 3600), 0.69, 3320, 67.9, 0.0, 0.0, 'critical', 'high deviation'),
        RegionDeviation('additives_glycol', (1000, 1300), 3.69, 1027, 87.9, 131.2, 35.5, 'critical', 'extreme deviation'),
        RegionDeviation('ch_stretch', (2850, 2950), 2.14, 2902, 45.5, 0.0, 0.0, 'critical', 'extreme deviation')
    ]
    
    outliers_case3 = []  # No critical outliers detected (as per your real case)
    
    result3 = analyzer.categorize_sample(
        correlation=0.809,  # Low correlation (your actual value)
        region_deviations=region_deviations_case3,
        outliers=outliers_case3
    )
    
    print(f"\n📊 Category: {result3['category']}")
    print(f"📊 Confidence: {result3['confidence']*100:.0f}%")
    print(f"\n💡 Reasoning:")
    for reason in result3['reasoning']:
        if reason:  # Skip empty strings
            print(f"   {reason}")
    
    print(f"\n📈 Metrics:")
    print(f"   - Correlation: {result3['metrics']['correlation']:.3f}")
    print(f"   - Max ΔY: {result3['metrics']['max_delta_y']:.3f} A")
    print(f"   - Max ΔX: {result3['metrics']['max_delta_x']:.1f} cm⁻¹")
    print(f"   - Critical outliers: {result3['metrics']['critical_outliers']}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print("\nKEY IMPROVEMENTS:")
    print("  1. BASELINE_MISMATCH: Now detects systematic deviations without degradation")
    print("  2. OUTLIER: Now requires degradation pattern (oxidation+water) OR extreme deviation")
    print("  3. Better distinction between 'different formulation' vs 'contaminated sample'")
    print("=" * 80)

if __name__ == "__main__":
    test_categorization()
