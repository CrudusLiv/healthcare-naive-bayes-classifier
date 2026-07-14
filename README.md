# healthcare-naive-bayes-classifier

BCS102 (Fundamentals of Artificial Intelligence) assignment: predicting patient `Test Results` (Normal / Abnormal / Inconclusive) from the [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) using Naive Bayes.

## Contents

- `BCS102_Healthcare_NaiveBayes.ipynb` — main notebook: data loading, EDA, preprocessing, model training, and evaluation
- `healthcare_dataset.csv` — raw dataset (55,500 rows, 15 features)
- `05 Assignment-Question-BCS102-May26-Final.docx` — assignment brief

## What the notebook does

1. **Load & describe** the raw dataset (15 features, 55,500 rows).
2. **Clean & type-convert**: strip column names, cast dates to `datetime64`, cast nominal fields to `category`; remove 534 exact-duplicate rows (54,966 rows remain).
3. **EDA**: target/feature distributions, target-vs-feature breakdowns, correlation heatmap, IQR-based outlier check, and formal feature-target association tests (ANOVA for numeric features, Cramer's V for categorical features).
4. **Feature engineering & preprocessing**: derive `Length of Stay` from admission/discharge dates, drop identifier columns (`Name`, `Doctor`, `Hospital`, `Room Number`), label-encode categorical features, scale numeric features, and split 80/20 (stratified) into train/test sets.
5. **Model training**: `GaussianNB` and `CategoricalNB`, each tuned via 5-fold stratified cross-validation (`GridSearchCV`).
6. **Evaluation**: accuracy, precision, recall, F1, multi-class ROC-AUC, and confusion matrices, compared against a majority-class baseline.
7. **Discussion**: ethical, social, and future implications of the results.

## Key finding

Both tuned Naive Bayes models perform statistically indistinguishably from a majority-class baseline (~33% accuracy, ROC-AUC ≈ 0.5), consistent with the EDA showing negligible association between any available feature and `Test Results`. The notebook concludes this reflects limited signal in the administrative features available (not a modelling error), and recommends richer clinical features and human oversight before any real-world use.

## Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

## Usage

1. Ensure `healthcare_dataset.csv` is in the repository root.
2. Open and run `BCS102_Healthcare_NaiveBayes.ipynb` top to bottom (e.g. in Jupyter or VS Code).
