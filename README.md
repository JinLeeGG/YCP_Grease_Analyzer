# YCP Grease Analyzer - FTIR Deviation Analysis System

## 🎯 Overview

Production-ready FTIR spectroscopy analysis tool for grease condition monitoring. Uses a **deviation-focused approach** that reports factual spectral differences (ΔX, ΔY) rather than quality scores.

### Key Features

✅ **Pure Deviation Analysis** - Reports WHERE spectra differ (ΔX, ΔY metrics)  
✅ **Multi-Metric Categorization** - Combines correlation + ΔX + ΔY + ratio  
✅ **Fast Performance** - <1 second per sample  
✅ **Baseline Compatibility** - Detects formulation mismatches  
✅ **AI Translation** - Optional LLM for user-friendly language (NOT quality judgments)  
✅ **PyQt6 GUI** - Interactive desktop application  
✅ **Batch Analysis** - Process multiple samples efficiently

---

## 🏗️ System Architecture

```
CSV Files → Deviation Analyzer → Multi-Metric Categorization → Optional LLM Translation
            (ΔX, ΔY, correlation)   (GOOD/CRITICAL/MISMATCH)    (User-friendly summary)
```

### Two Analysis Engines:

1. **FTIRDeviationAnalyzer (PRIMARY)**

   - Pure deviation metrics (ΔX, ΔY)
   - Multi-metric categorization
   - No quality judgments
   - Always reliable (<1s)

2. **FTIRPeakAnalyzer (LEGACY)**
   - Peak detection
   - Quality recommendations
   - Kept for backward compatibility

---

## 🚀 Quick Start

### Installation

```powershell
# Clone repository
git clone https://github.com/JinLeeGG/YCP_Grease_Analyzer.git
cd YCP_Grease_Analyzer

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```powershell
python app.py
```

### Basic Usage

1. **Load Baseline** - Click "Select Baseline CSV"
2. **Load Samples** - Click "Select Sample CSV(s)" (multiple files OK)
3. **View Results** - Analysis runs automatically
4. **Export** - Save graphs and reports

---

## 📊 Understanding Results

### Multi-Metric Categories

| Category                  | Meaning                | Action                    |
| ------------------------- | ---------------------- | ------------------------- |
| ✅ **GOOD**               | Minimal deviations     | Continue operation        |
| ⚠️ **REQUIRES_ATTENTION** | Minor deviations       | Monitor trends            |
| ❌ **CRITICAL**           | Significant deviations | Investigate cause         |
| 🚨 **OUTLIER**            | Major differences      | Check for contamination   |
| ⚡ **BASELINE_MISMATCH**  | Different formulation  | Verify baseline selection |

### Key Metrics

**ΔY (Vertical Deviation)**

- Intensity difference at same wavenumber
- Thresholds: <0.03A (OK), 0.05-0.10A (major), >0.10A (critical)

**ΔX (Horizontal Shift)**

- Peak position change
- Thresholds: <5cm⁻¹ (OK), 5-10cm⁻¹ (minor), >10cm⁻¹ (major)

**ΔX:ΔY Ratio**

- > 100: Shift-dominant (chemical change)
- <20: Intensity-dominant (degradation/concentration)
- 20-100: Both types of deviation

**Correlation (r)**

- ≥0.97: Excellent similarity
- 0.85-0.97: Good to moderate
- <0.85: Poor (possible mismatch)

---

## 📂 Project Structure

```
YCP_Grease_Analyzer/
├── app.py                          # Main PyQt6 GUI application
├── requirements.txt                # Python dependencies
├── modules/
│   ├── ftir_deviation_analyzer.py  # PRIMARY: Deviation analysis
│   ├── ftir_peak_analyzer.py       # LEGACY: Peak detection
│   ├── llm_analyzer.py             # LLM integration (optional)
│   ├── csv_processor.py            # CSV data handling
│   ├── graph_generator.py          # Matplotlib plotting
│   └── peak_detector.py            # Legacy peak detection
├── utils/
│   ├── config.py                   # Configuration settings
│   └── validators.py               # Data validation
├── GUI/
│   ├── Analyzer_main.ui            # Main window UI
│   └── path.ui                     # Path selection dialog
├── docs/
│   ├── FINAL_SYSTEM_SUMMARY.md     # Complete system documentation
│   └── (other documentation)
└── tests/
    ├── test_csv_loader.py
    └── test_llm.py
```

---

## 🔧 Configuration

### Adjust Deviation Thresholds

```python
# In modules/ftir_deviation_analyzer.py
config = DeviationConfig()

# Vertical deviation (ΔY)
config.delta_y_critical = 0.10  # A (absorbance units)
config.delta_y_major = 0.05
config.delta_y_minor = 0.03

# Horizontal shift (ΔX)
config.delta_x_major = 10.0     # cm⁻¹
config.delta_x_minor = 5.0

# Correlation thresholds
config.correlation_excellent = 0.97
config.correlation_good = 0.95
```

### LLM Configuration (Optional)

```python
# In utils/config.py
LLM_CONFIG = {
    "enabled": True,
    "model": "llava:7b-v1.6",  # Ollama model
    "timeout": 30
}
```

To use LLM features:

```powershell
# Install Ollama
winget install Ollama.Ollama

# Pull model
ollama pull llava:7b-v1.6
```

---

## 🎓 Critical Regions Monitored

| Region                  | Wavenumber (cm⁻¹) | Chemical Feature | Purpose                  |
| ----------------------- | ----------------- | ---------------- | ------------------------ |
| **Carbonyl Oxidation**  | 1650-1800         | C=O stretch      | Primary oxidation marker |
| **Water Contamination** | 3200-3600         | O-H stretch      | Water ingress            |
| **Additives/Glycol**    | 1000-1300         | C-O stretch      | Additive changes         |
| **C-H Stretch**         | 2850-2950         | C-H bonds        | Base oil reference       |

---

## 📈 Performance

| Metric        | Value | Notes                                      |
| ------------- | ----- | ------------------------------------------ |
| Analysis time | <1s   | Deviation analyzer                         |
| With LLM      | 5-15s | Optional enhancement                       |
| Accuracy      | High  | Statistical thresholds                     |
| Reliability   | 100%  | No external dependencies for core analysis |

---

## 🐛 Troubleshooting

### Common Issues

**Problem: Too many false positives**

- **Solution:** Increase deviation thresholds in `DeviationConfig`

**Problem: BASELINE_MISMATCH errors**

- **Solution:** Verify you're comparing the same grease type

**Problem: LLM not working**

- **Solution:** LLM is optional - system works without it
- Check Ollama installation: `ollama --version`

**Problem: Slow performance**

- **Solution:** Disable LLM for batch processing
- Check data file size (should be <10K points)

---

## 📚 Documentation

- **[FINAL_SYSTEM_SUMMARY.md](docs/FINAL_SYSTEM_SUMMARY.md)** - Complete system overview
- **[OPTIMIZATION_CHANGELOG.md](OPTIMIZATION_CHANGELOG.md)** - Implementation history
- **[QUICK_START_NEW_SYSTEM.md](QUICK_START_NEW_SYSTEM.md)** - Quick reference guide

---

## 🎯 Key Philosophy

> **"The deviation analyzer does the math; AI makes it human-friendly"**

- **System reports FACTS** (ΔX, ΔY, correlation)
- **AI translates to plain language** (optional)
- **You make decisions** based on facts + your expertise

---

## 🔄 Updates & Version History

### v2.0 - Deviation-Focused Implementation (Current)

- ✅ Pure deviation metrics (ΔX, ΔY)
- ✅ Multi-metric categorization
- ✅ ΔX now triggers alerts
- ✅ ΔX:ΔY ratio diagnostic
- ✅ Baseline mismatch detection

### v1.0 - Peak Detection System (Legacy)

- Peak-based analysis
- Quality recommendations
- Still available for compatibility

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 👥 Contributors

- **Development:** YCP Lab
- **System Design:** Based on FTIR spectroscopy best practices
- **AI Integration:** Claude Sonnet 4.5 consultation

---

## 🔗 Related Resources

- FTIR Spectroscopy Principles
- Grease Analysis Standards
- PyQt6 Documentation

---

**Last Updated:** November 9, 2025  
**Version:** 2.0 - Deviation Analysis System
