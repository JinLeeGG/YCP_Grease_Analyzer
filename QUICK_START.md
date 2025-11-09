# FTIR Deviation Analysis System - Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd D:\GitHub\YCP_Grease_Analyzer

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Run the Application

```powershell
# Start GUI
python app.py
```

### Step 3: Analyze Samples

1. **Load Baseline**

   - Click "Select Baseline CSV"
   - Choose your reference/fresh grease spectrum

2. **Load Samples**

   - Click "Select Sample CSV(s)"
   - Select one or more samples to analyze

3. **View Results**

   - Analysis runs automatically
   - See deviation metrics, multi-metric category
   - Review critical regions

4. **Export**
   - Save graphs and reports as needed

---

## 📊 Understanding Your First Results

### Example Output:

```
╔═══════════════════════════════════════════════════════════════════╗
║ MULTI-METRIC CATEGORIZATION (Primary Decision System)            ║
╚═══════════════════════════════════════════════════════════════════╝

**Final Category:** ⚠️ REQUIRES_ATTENTION
**Confidence:** 80%

**Decision Logic:**
  1. Spectral correlation: r=0.952
  2. Minor deviations detected (ΔY=0.042 A, ΔX=6.3 cm⁻¹)
  → Monitor trends - may indicate early changes

**Metrics Used:**
  • Spectral Correlation (r): 0.952
  • Max ΔY (vertical): 0.042 A
  • Max ΔX (horizontal): 6.3 cm⁻¹
  • ΔX:ΔY ratio: 150.0
  • Critical outliers: 0
```

---

## 🎯 Key Metrics Quick Reference

### ΔY (Vertical Deviation) - Intensity Difference

| Value       | Level       | Meaning                      |
| ----------- | ----------- | ---------------------------- |
| < 0.03 A    | ✅ OK       | Spectra superimpose well     |
| 0.03-0.05 A | ⚠️ Minor    | Small intensity change       |
| 0.05-0.10 A | ❌ Major    | Significant intensity change |
| > 0.10 A    | 🚨 Critical | Large intensity change       |

### ΔX (Horizontal Shift) - Peak Position Change

| Value     | Level    | Meaning           |
| --------- | -------- | ----------------- |
| < 5 cm⁻¹  | ✅ OK    | Peaks aligned     |
| 5-10 cm⁻¹ | ⚠️ Minor | Notable shift     |
| > 10 cm⁻¹ | ❌ Major | Significant shift |

---

## 🤖 Optional: Enable AI Translation

The system works perfectly without AI. But if you want user-friendly summaries:

### Install Ollama (Windows)

```powershell
# Install Ollama
winget install Ollama.Ollama

# Pull LLaVA model
ollama pull llava:7b-v1.6
```

---

**System Version:** 2.0 - Deviation Analysis  
**Last Updated:** November 9, 2025
