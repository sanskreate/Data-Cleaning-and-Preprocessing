# 📊 Customer Personality Analysis - Data Cleaning and Preprocessing

This project involves data cleaning, preprocessing, feature engineering, and outlier detection on a **Customer Personality Analysis** dataset. The aim is to understand customer behavior and preferences to support business decisions like customer segmentation, marketing strategy, and campaign targeting.

---

## 📁 Dataset Overview

The dataset contains the following information:

### 👤 Demographic Attributes
- `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`
- `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, `Complain`

### 🛒 Product Spend Attributes
- `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`

### 📢 Campaign Response
- `AcceptedCmp1` to `AcceptedCmp5`, `Response`, `NumDealsPurchases`

### 🌐 Purchase Channels
- `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`

> **Source:** `marketing_campaign.csv` (Tab-separated values)

---

## 🧰 Libraries Used

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
```

---

## 🛠️ Steps & Code Summary

### 📦 1. **Load the Dataset**

```python
df = pd.read_csv("marketing_campaign.csv", sep="\t")
```

### 🧹 2. **Initial Inspection and Cleaning**

- Checked for missing values
- Removed rows with `NaN`
- Cleaned column names: replaced spaces with underscores and converted to lowercase

```python
df = df.dropna()
def clean_column_names(df):
    ...
df = clean_column_names(df)
```

---

### 📆 3. **Datetime Conversion**

Converted `dt_customer` column to datetime and extracted the first and last enrollment dates.

```python
df["dt_customer"] = pd.to_datetime(df["dt_customer"], dayfirst=True)
```

---

### 🏗️ 4. **Feature Engineering**

New features created:

- `total_spend`: Total money spent across all product categories
- `total_children`: Combined value of kids and teens at home
- `age`: Calculated from `year_birth`
- `customer_seniority`: Duration since joining (in years)
- `total_purchases`: Combined online, catalog, and in-store purchases
- `campaign_acceptance_rate`: Ratio of accepted campaigns to total campaigns
- `family_size`: Assumed to be `total_children + 2`
- `age_group`: Binned age into custom age groups (18-29, 30-44, etc.)

```python
df['total_spend'] = ...
df['age_group'] = pd.cut(...)
```

---

### 🔠 5. **Categorical Encoding**

Identified and label-encoded all categorical features.

```python
LE = LabelEncoder()
for col in categorical_cols:
    df[col] = LE.fit_transform(df[col])
```

---

### 🧪 6. **Outlier Detection & Removal**

Used IQR (Interquartile Range) method to remove outliers from:
- `income`
- `age`
- `total_spend`

```python
def remove_outliers_iqr(data, columns):
    ...
df_clean = remove_outliers_iqr(df, numerical_cols)
```

Visual comparisons before and after outlier removal were done using:
- Boxplots
- Histograms

```python
compare_distributions(df, df_clean, numerical_cols)
```

---

### 🔗 7. **Correlation Analysis**

Created a heatmap to visualize correlations between numerical features post-cleaning.

```python
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
```

---

### ✅ Final Dataset

The cleaned and engineered dataset is now ready for:
- Customer segmentation (clustering)
- Predictive modeling
- Campaign targeting strategies
- Lifetime value analysis

---

## 📌 Key Insights & Next Steps

- Strong correlation found between `total_spend` and `income`
- Recency and campaign response may indicate customer engagement
- Family size and age group can be helpful in defining marketing personas

---

## 📂 Folder Structure

```
├── marketing_campaign.csv
├── customer_analysis.ipynb
├── README.md
```

---

## 💬 Suggestions for Future Work

- Apply EDA for further analysis
- Use PCA for dimensionality reduction
- Train models to predict campaign response or customer churn

---
