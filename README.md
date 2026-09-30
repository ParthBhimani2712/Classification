# Classification: Telco customer churn

This project compares six classic scikit-learn classifiers on predicting **customer churn** (`Churn` = Yes/No) from the Telco Customer Churn dataset. The notebook cleans the data, one-hot encodes the categorical features, standardises everything, trains KNN, Decision Tree, Random Forest, Naive Bayes, SVM and Logistic Regression models, and compares them by accuracy and per-class precision/recall/F1. It also plots the fitted decision tree.

## Dataset

- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`. This is the IBM Telco Customer Churn dataset (commonly distributed on Kaggle). It is **not included** in this repo, so download it and place it next to the notebook.
- **Size:** 7,043 rows × 21 columns.
- **Columns:** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn` (target).

## Approach

1. Convert `TotalCharges` to a numeric type (blank strings become NaN).
2. Drop `customerID` and `SeniorCitizen`.
3. One-hot encode every column except `tenure`, `MonthlyCharges`, `TotalCharges` and `Churn` with `pd.get_dummies(drop_first=True)`.
4. Drop rows with missing values, which leaves 7,032 rows.
5. Split the data 75/25 into train and test sets (1,758 test rows) with no fixed `random_state`.
6. Standardise the features with `StandardScaler`, fitted on the training set only.
7. Train and evaluate these models, all with default hyperparameters except the Random Forest:
   - K-Nearest Neighbours (`KNeighborsClassifier()`)
   - Decision Tree (`DecisionTreeClassifier()`), also visualised with `plot_tree`
   - Random Forest (`RandomForestClassifier(n_estimators=400)`)
   - Bernoulli Naive Bayes (`BernoulliNB()`)
   - Support Vector Machine (`SVC()`)
   - Logistic Regression (`LogisticRegression()`)

## Results

Results on the test split (1,758 customers: 1,289 "No", 469 "Yes"), taken from the saved notebook outputs:

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | Macro F1 |
|---|---|---|---|---|---|
| **SVM (SVC)** | **81.63%** | 0.72 | 0.51 | 0.60 | 0.74 |
| Logistic Regression | 81.23% | 0.69 | 0.54 | 0.61 | 0.74 |
| Random Forest (400 trees) | 79.29% | 0.65 | 0.48 | 0.55 | 0.71 |
| KNN | 77.30% | 0.58 | 0.54 | 0.56 | 0.70 |
| Decision Tree | 73.78% | 0.51 | 0.49 | 0.50 | 0.66 |
| Bernoulli Naive Bayes | 73.04% | 0.50 | **0.79** | 0.61 | 0.70 |

Takeaways:
- SVM and Logistic Regression are the most accurate, both at about 81%.
- Every model has much lower recall on the minority "Yes" (churn) class than on "No". Naive Bayes has the best churn recall (0.79) but the lowest accuracy.
- About 73% of the test set did not churn, so a model that always predicts "No" would already score about 73% accuracy. Read the accuracy figures against that baseline.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook Classificaiton.ipynb
```

Put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder first. The notebook metadata shows it was written in Google Colab, so you can also upload the notebook and the CSV to Colab.

## Repository structure

```
Classification/
├── Classificaiton.ipynb   # the whole analysis (the filename typo is in the original file name)
└── README.md
```

## Notes and limitations

- **Missing import:** the KNN cell uses `KNeighborsClassifier` but the notebook never imports it. Add `from sklearn.neighbors import KNeighborsClassifier` before running it top to bottom.
- `df.dropna()` in an early cell is not assigned back. The NaN rows are actually removed later by `df_dummies.dropna(inplace=True)`.
- There is no `random_state` on the train/test split or the models, so re-running will give slightly different numbers.
- Only a single hold-out split is used: no cross-validation, hyperparameter tuning or class-imbalance handling (for example class weights or resampling).
- `BernoulliNB` binarises features at 0, so applying it to standardised continuous features is a rough approximation.
- `SeniorCitizen` is dropped without any stated reason.
