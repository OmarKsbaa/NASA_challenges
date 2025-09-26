# 📋 NASA Kepler Data Analysis - Quick Reference

## 🎯 Key Statistics
- **Total Samples**: 9,564 KOIs
- **Target Classes**: 2 (FALSE POSITIVE 50.7%, CANDIDATE 49.3%)
- **Original Features**: 50 columns
- **Processed Features**: 43 engineered features
- **Missing Data**: 8.4% overall, structured patterns

## 🎯 Target Variable Analysis

| **Class** | **Label** | **Count** | **Percentage** | **Description** |
|-----------|-----------|-----------|----------------|-----------------|
| 0 | **FALSE POSITIVE** | 4,847 | 50.7% | Transit-like signals that are **not planets** |
| 1 | **CANDIDATE** | 4,717 | 49.3% | Objects that **may be planets** (pending confirmation) |

### 🔍 Target Characteristics:
- **Perfect Balance**: No class imbalance issues - can use accuracy as primary metric
- **Binary Problem**: Simplified from expected 3-class to 2-class classification
- **Clear Distinction**: FALSE POSITIVE = definitely not a planet, CANDIDATE = potentially a planet
- **Business Impact**: High precision for FALSE POSITIVE saves telescope time, high recall for CANDIDATE finds planets

## 🔍 Critical Insights
1. **Binary Problem**: Only 2 classes, not 3 as expected
2. **Perfect Balance**: No class imbalance issues
3. **Quality Data**: Professional observatory measurements
4. **Structured Missing**: Missingness follows astronomical protocols
5. **Physics Matters**: Domain knowledge essential for preprocessing

## 📊 Missing Value Categories
| Type | Count | Strategy |
|------|--------|----------|
| Complete (100%) | 2 cols | Drop |
| High (71%) | 1 col | Convert to binary |
| Moderate (5-16%) | 2 cols | Iterative imputation |
| Low (<5%) | 31 cols | Median imputation |
| None (0%) | 14 cols | No action needed |

## 🎪 Feature Groups
- **Core Transit**: Always complete, fundamental measurements
- **Derived Planetary**: Some missing, calculated parameters  
- **Stellar Properties**: Moderate missing, host star characteristics
- **Quality Flags**: Complete, false positive indicators

## ⚙️ Processing Pipeline
1. Target encoding (2 classes)
2. Column selection (27 → 43 features after engineering)
3. Feature engineering (logs, ratios, flags)
4. **Astronomy-aware missing value handling**
5. Feature scaling (RobustScaler)
6. Train/test split (80/20, stratified)

## 🎯 Most Important Features & Target Relationship

| **Rank** | **Feature** | **Correlation** | **Type** | **Interpretation** |
|----------|-------------|-----------------|----------|-------------------|
| 1 | `koi_score` | 0.829 | Disposition Score | **Strong predictor** - higher score = more likely CANDIDATE |
| 2 | `koi_fpflag_ss` | -0.537 | Quality Flag | **Stellar contamination** - flag indicates FALSE POSITIVE |
| 3 | `koi_fpflag_co` | -0.493 | Quality Flag | **Centroid offset** - flag indicates FALSE POSITIVE |
| 4 | `log_koi_prad` | -0.400 | Engineered | **Planet radius** - larger planets more likely FALSE POSITIVE |
| 5 | `koi_fpflag_ec` | -0.364 | Quality Flag | **Ephemeris match** - flag indicates FALSE POSITIVE |
| 6 | `koi_teq` | -0.293 | Planetary | **Equilibrium temp** - hotter planets more likely FALSE POSITIVE |
| 7 | `log_koi_depth` | -0.287 | Engineered | **Transit depth** - deeper transits more likely FALSE POSITIVE |
| 8 | `koi_depth` | -0.266 | Core Transit | **Raw transit depth** - correlates with log version |
| 9 | `log_koi_insol` | -0.258 | Engineered | **Insolation flux** - higher flux more likely FALSE POSITIVE |
| 10 | `log_koi_teq` | -0.256 | Engineered | **Log temperature** - correlates with raw temperature |

### 🔍 Key Insights from Feature Analysis:
- **`koi_score`** is by far the strongest predictor (0.83 correlation)
- **Quality flags** (`koi_fpflag_*`) are highly discriminative - they directly indicate false positives
- **Engineered log features** often outperform raw measurements
- **Negative correlations** indicate larger/hotter planets tend to be false positives
- **Feature types represented**: Quality Flags (3), Engineered (4), Core Transit (2), Planetary (1)

## 🚀 Ready for Modeling
- ✅ Clean datasets: `data/processed_v2/`
- ✅ Balanced classes: 50.7% FALSE POSITIVE, 49.3% CANDIDATE
- ✅ No missing values in 43 features
- ✅ Domain-informed preprocessing with physics-aware imputation
- ✅ Strong predictive features identified (correlation up to 0.83)
- ✅ Comprehensive documentation

## 📝 Files Created
- `data_analysis.ipynb` - Complete EDA notebook
- `src/preprocess.py` - Production preprocessing pipeline
- `docs/data_insights.md` - Detailed analysis report
- `docs/missing_value_strategy.md` - Missing value approach
- `docs/data_insights_quick.md` - This quick reference