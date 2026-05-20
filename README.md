# 🏠 CSCI-415 Data Mining Projects — Ames Housing Dataset

> A collection of data mining projects analyzing the Ames Housing Dataset using statistical analysis, association rule mining (Apriori), and machine learning regression models. Built for CSCI-415 (Intro to Data Mining) at NYIT.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Weka](https://img.shields.io/badge/Weka-FF6B35?style=flat-square)

---

## 📂 Projects Overview

| Project | Topic | Tools |
|---------|-------|-------|
| [Project 1](#project-1--dataset-analysis--arff-conversion) | Dataset Analysis & ARFF Conversion | Python, Weka |
| [Project 3](#project-3--association-rule-mining-apriori) | Association Rule Mining (Apriori) | Weka |
| [Project 4](#project-4--house-price-prediction-regression) | House Price Prediction (Regression) | Python, Scikit-Learn |

---

## 📌 Dataset

**Ames Housing Dataset** — Kaggle  
🔗 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

- **1,459** residential property records from Ames, Iowa
- **80** attributes covering lot size, quality ratings, room counts, garage, basement, and sale conditions
- **Target variable:** `SalePrice` (mean: $180,921 · std: $79,442)

---

## Project 1 — Dataset Analysis & ARFF Conversion

**Goal:** Explore the Ames Housing Dataset, perform statistical analysis on key attributes, and convert the data from CSV to ARFF format for use with Weka.

### Selected Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| LotArea | Numeric | Lot size in square feet |
| GrLivArea | Numeric | Above-ground living area (sq ft) |
| OverallQual | Numeric | Material & finish quality (1–10) |
| SalePrice | Numeric | Final sale price (target) |
| MSZoning | Categorical | Zoning classification |

### Key Statistical Findings

| Attribute | Mean | Min | Max | Std Dev |
|-----------|------|-----|-----|---------|
| LotArea | 9,819 | 1,470 | 56,600 | 4,955 |
| SalePrice | $180,921 | $34,900 | $755,000 | $79,442 |
| OverallQual | 6.1 | 1 | 10 | 1.38 |

- **LotArea** shows high variability (std = 4,955) — extreme outliers present (large rural lots)
- **SalePrice** is right-skewed — luxury homes pull the mean above the median
- **OverallQual** is the strongest single predictor of SalePrice

### Preprocessing
- Separated numeric and categorical attributes for analysis
- Reviewed missing values — no major structural issues in selected columns
- Converted CSV → ARFF format for Weka compatibility

---

## Project 3 — Association Rule Mining (Apriori)

**Goal:** Apply the Apriori algorithm in Weka to discover association rules in supermarket transaction data, and experiment with different support/confidence thresholds.

### Parameter Experiments

| Support (s) | Confidence (c) | Rules Generated |
|-------------|----------------|-----------------|
| 0.3 | 0.5 | 145 |
| 0.3 | 0.8 | 72 |
| 0.3 | 0.9 | 34 |
| 0.4 | 0.5 | 96 |
| 0.4 | 0.8 | 41 |
| 0.4 | 0.9 | 18 |

### Sample Rules Discovered

```
bread and cake  →  milk-cream
frozen foods    →  soft drinks
fruit           →  vegetables
dairy foods     →  bread and cake
coffee          →  biscuits
```

### Key Observations
- Increasing **confidence** dramatically reduces the number of rules — only the strongest associations survive
- Increasing **support** also reduces rules but less aggressively than confidence
- The most stable rules (appear across all thresholds) represent the strongest market basket associations
- Grocery staples (bread, milk, dairy) consistently appear as frequent itemsets

---

## Project 4 — House Price Prediction (Regression)

**Goal:** Train and compare three regression models to predict house sale prices using selected features from the Ames Housing Dataset.

### Features Used

```python
features = [
    'Lot Area',       # Lot size in sq ft
    'Gr Liv Area',    # Above-ground living area
    'Overall Qual',   # Quality rating (1-10)
    'Year Built',     # Construction year
    'Garage Area',    # Garage size in sq ft
    'Total Bsmt SF'   # Total basement area
]
target = 'SalePrice'
```

### Models

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline — fits a linear relationship between features and target |
| **Decision Tree Regressor** | Non-linear — partitions data using recursive splits |
| **Random Forest Regressor** | Ensemble of 100 decision trees — reduces overfitting via averaging |

### Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | ~$22,000 | ~$35,000 | ~0.78 |
| Decision Tree | ~$25,000 | ~$38,000 | ~0.74 |
| Random Forest | ~$17,000 | ~$27,000 | **~0.88** |

### Key Conclusions
- **Random Forest** outperforms both models on all metrics — ensemble averaging reduces variance
- **Linear Regression** is competitive and more interpretable — good baseline
- **Decision Tree** alone overfits more than Random Forest due to high variance
- `Overall Qual` and `Gr Liv Area` are the most predictive features

---

## 🚀 Running the Code

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Download dataset from Kaggle and place in project directory
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# Run Project 4 regression models
python Final_Project4_Code.py
```

### Expected Output

```
==================================================
Model: Linear Regression
Mean Absolute Error (MAE): 22XXX.XX
Mean Squared Error (MSE): XXXXXXXX.XX
Root Mean Squared Error (RMSE): 35XXX.XX
R² Score: 0.78XX
==================================================
Model: Decision Tree Regressor
...
==================================================
Model: Random Forest Regressor
...
==================================================
Project completed successfully.
```

---

## 📁 File Structure

```
csci415-data-mining/
├── Final_Project4_Code.py        # Regression models (Project 4)
├── AmesHousing.csv               # Dataset (download from Kaggle)
├── README.md
└── reports/
    ├── Project1_Dataset_Analysis.docx
    ├── Project3_Apriori_Mining.docx
    └── Project4_House_Price_Prediction.docx
```

---

## 🏫 Course

CSCI-415 — Intro to Data Mining | NYIT | Spring 2026  
Instructor: Dr. Helen Gu
