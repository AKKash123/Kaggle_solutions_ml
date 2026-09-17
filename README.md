# 🏠 House Prices — Advanced Regression Techniques

A complete **Python Machine Learning solution** for the Kaggle **House Prices: Advanced Regression Techniques** competition.

The objective is to predict the final sale price of houses using features such as overall quality, living area, neighborhood, year built, garage information, basement information, and many other property characteristics.

---

## 📌 Competition

**Kaggle Competition:** House Prices: Advanced Regression Techniques

**Problem Type:** Supervised Machine Learning — Regression

**Target Variable:** `SalePrice`

**Evaluation Metric:** Root Mean Squared Error (RMSE) on the logarithm of predicted and actual house prices.

---

## 🎯 Objective

For every house in the test dataset, predict its `SalePrice`.

The final submission must contain:

```csv
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
...
```

---

## 📊 Evaluation Metric

The competition evaluates predictions using logarithmic RMSE:

```text
RMSE(log(predicted SalePrice), log(actual SalePrice))
```

Because of this, the model is trained using:

```python
y = np.log1p(train["SalePrice"])
```

After prediction, the logarithmic values are converted back to the original price scale:

```python
predictions = np.expm1(predictions)
```

This approach makes the model optimize the same type of error used by the Kaggle competition.

---

## 🧰 Technologies Used

* Python 3
* NumPy
* Pandas
* Scikit-learn
* Kaggle Notebook
* Gradient Boosting Regression
* One-Hot Encoding
* Median Imputation
* Log Transformation

---

## 📁 Dataset

The competition provides two main datasets:

```text
train.csv
test.csv
```

### Training Dataset

`train.csv` contains:

* `Id`
* House/property features
* `SalePrice`

The `SalePrice` column is the target variable.

### Test Dataset

`test.csv` contains:

* `Id`
* House/property features

The test dataset does not contain `SalePrice`.

---

## 🔎 Dataset Structure

The dataset contains both numerical and categorical features.

### Numerical Features

Examples:

```text
LotArea
YearBuilt
YearRemodAdd
OverallQual
OverallCond
GrLivArea
GarageCars
GarageArea
TotalBsmtSF
1stFlrSF
2ndFlrSF
```

### Categorical Features

Examples:

```text
Neighborhood
HouseStyle
Exterior1st
Exterior2nd
KitchenQual
GarageType
BsmtQual
HeatingQC
SaleType
```

---

# 🚀 Machine Learning Workflow

The solution follows these steps:

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Separate Target Variable
     ↓
Remove Id
     ↓
Apply Log Transformation
     ↓
Identify Numerical/Categorical Features
     ↓
Handle Missing Values
     ↓
One-Hot Encode Categorical Features
     ↓
Train Gradient Boosting Model
     ↓
Validate Model
     ↓
Train Final Model on Full Dataset
     ↓
Predict Test Data
     ↓
Reverse Log Transformation
     ↓
Create submission.csv
```

---

## 1. Import Libraries

```python
import os
import glob

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
```

---

## 2. Locate Kaggle Dataset

Instead of hard-coding a particular dataset folder, the notebook searches for the CSV files automatically:

```python
train_files = glob.glob(
    "/kaggle/input/**/train.csv",
    recursive=True
)

test_files = glob.glob(
    "/kaggle/input/**/test.csv",
    recursive=True
)
```

This makes the notebook more flexible if Kaggle changes the mounted directory name.

---

## 3. Load Data

```python
train = pd.read_csv(train_files[0])
test = pd.read_csv(test_files[0])
```

Check the dataset:

```python
print(train.shape)
print(test.shape)

display(train.head())
```

---

## 4. Separate Features and Target

The target variable is:

```python
SalePrice
```

The `Id` column is retained separately for the submission file.

```python
test_ids = test["Id"].copy()

X = train.drop(
    columns=["SalePrice", "Id"]
)

X_test = test.drop(
    columns=["Id"]
)
```

---

## 5. Log Transform SalePrice

Since the competition evaluates logarithmic RMSE:

```python
y = np.log1p(train["SalePrice"])
```

`log1p()` calculates:

```text
log(1 + SalePrice)
```

This is useful because house prices have a strongly right-skewed distribution.

---

## 6. Identify Feature Types

Numerical columns:

```python
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
```

Categorical columns:

```python
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()
```

---

## 7. Handle Missing Values

### Numerical Features

Missing numerical values are replaced using the median:

```python
SimpleImputer(
    strategy="median"
)
```

### Categorical Features

Missing categorical values are replaced using the most frequent value:

```python
SimpleImputer(
    strategy="most_frequent"
)
```

---

## 8. One-Hot Encoding

Categorical features are converted into numerical features using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

`handle_unknown="ignore"` prevents errors if a category appears in the test dataset but was not present in the training split.

---

# 🤖 Model

The solution uses:

```python
GradientBoostingRegressor
```

with the following configuration:

```python
model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    max_features="sqrt",
    min_samples_leaf=10,
    min_samples_split=10,
    loss="huber",
    random_state=42
)
```

### Important Parameters

| Parameter           |   Value | Purpose                                |
| ------------------- | ------: | -------------------------------------- |
| `n_estimators`      |    1000 | Number of boosting stages              |
| `learning_rate`     |    0.03 | Learning rate                          |
| `max_depth`         |       4 | Tree depth                             |
| `max_features`      |  `sqrt` | Features considered per split          |
| `min_samples_leaf`  |      10 | Minimum samples per leaf               |
| `min_samples_split` |      10 | Minimum samples required for splitting |
| `loss`              | `huber` | Robust regression loss                 |
| `random_state`      |      42 | Reproducibility                        |

---

# 🔗 Scikit-learn Pipeline

Preprocessing and model training are combined into one pipeline:

```python
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)
```

This ensures that preprocessing is consistently applied during:

* Training
* Validation
* Test prediction

---

# 🧪 Model Validation

The training dataset is divided into:

```text
80% → Training
20% → Validation
```

using:

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The validation metric is calculated as:

```python
valid_predictions = pipeline.predict(X_valid)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        valid_predictions
    )
)
```

The resulting value represents **log RMSE**, which corresponds to the competition's evaluation approach.

---

# 🏋️ Final Training

After validation, the model is trained using the complete training dataset:

```python
pipeline.fit(X, y)
```

This allows the final model to use all available training examples before generating test predictions.

---

# 🔮 Generate Predictions

The model generates predictions in logarithmic space:

```python
test_log_predictions = pipeline.predict(X_test)
```

The predictions are converted back to normal house prices:

```python
test_predictions = np.expm1(
    test_log_predictions
)
```

Negative values are prevented:

```python
test_predictions = np.maximum(
    test_predictions,
    0
)
```

---

# 📄 Create Submission File

The final submission DataFrame is:

```python
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})
```

The file is saved as:

```python
submission.to_csv(
    "/kaggle/working/submission.csv",
    index=False
)
```

---

# ✅ Submission Format

The final file must contain exactly two columns:

```text
Id
SalePrice
```

Example:

```csv
Id,SalePrice
1461,169000.123
1462,187724.456
1463,175221.789
1464,145321.321
1465,198765.432
```

---

# 📂 Output File

After running the notebook, the following file should be created:

```text
/kaggle/working/submission.csv
```

Before submitting, verify:

```python
import os

print(os.path.exists(
    "/kaggle/working/submission.csv"
))
```

Expected result:

```text
True
```

---

# ⚠️ Kaggle Notebook Submission

If Kaggle displays:

> "Submission file and the selected Notebook Version does not have any output files."

make sure you create the submission file inside:

```text
/kaggle/working/
```

For example:

```python
submission.to_csv(
    "/kaggle/working/submission.csv",
    index=False
)
```

Then:

1. Run the complete notebook.
2. Make sure the notebook finishes successfully.
3. Click **Save Version**.
4. Select **Save & Run All**.
5. Wait until execution completes.
6. Open the saved notebook version.
7. Check the **Output** section.
8. Confirm that `submission.csv` is present.
9. Submit `submission.csv` to the competition.

---

# 🧪 Final Verification

Use the following code before saving the notebook:

```python
submission_check = pd.read_csv(
    "/kaggle/working/submission.csv"
)

print(submission_check.head())
print(
    "Rows:",
    len(submission_check)
)

print(
    "Columns:",
    submission_check.columns.tolist()
)

print(
    "Missing values:",
    submission_check.isna().sum().sum()
)
```

Expected columns:

```text
['Id', 'SalePrice']
```

Expected missing values:

```text
0
```

---

# 📌 Project Structure

A simple Kaggle notebook structure:

```text
House-Prices-Kaggle/
│
├── house_prices_solution.ipynb
├── README.md
│
└── submission.csv
```

On Kaggle, the generated submission file will be located at:

```text
/kaggle/working/submission.csv
```

---

# 📚 Key Machine Learning Concepts

This project demonstrates:

* Regression
* Supervised learning
* Exploratory data handling
* Missing-value imputation
* Categorical encoding
* Feature preprocessing
* Log transformation
* Gradient boosting
* Train-validation split
* RMSE evaluation
* Scikit-learn pipelines
* Kaggle submission generation

---

# 🔧 Possible Improvements

The baseline can be improved by experimenting with:

* XGBoost
* LightGBM
* CatBoost
* Random Forest
* Random Forest + Gradient Boosting
* Lasso Regression
* Ridge Regression
* ElasticNet
* Feature engineering
* Hyperparameter tuning
* K-Fold cross-validation
* Model blending
* Stacking
* Outlier analysis
* Additional skewed-feature transformations

For this competition, model combinations and careful feature engineering can be explored after establishing the baseline.

---

# 📝 Conclusion

This project provides a complete end-to-end Python solution for the Kaggle House Prices competition.

The main steps are:

```text
Dataset
   ↓
Preprocessing
   ↓
Missing Value Handling
   ↓
One-Hot Encoding
   ↓
Log Transformation
   ↓
Gradient Boosting
   ↓
Validation
   ↓
Full Training
   ↓
Prediction
   ↓
Inverse Log Transformation
   ↓
submission.csv
```

The final submission file is:

```text
submission.csv
```

---

## 👨‍💻 Author

**Akash Dhar**

Python • Machine Learning • Web Development • Data Science

---

## ⭐ If this project helped you

Consider starring the repository and experimenting with different regression models and feature-engineering techniques to improve your understanding of machine learning competitions.
# 🚢 Titanic — Machine Learning from Disaster

A complete **Python Machine Learning solution** for the Kaggle **Titanic — Machine Learning from Disaster** competition.

The objective is to predict whether a passenger survived the Titanic disaster based on information such as passenger class, sex, age, fare, family relationships, and port of embarkation.

---

# 📌 Competition

**Kaggle Competition:** Titanic — Machine Learning from Disaster

**Problem Type:** Supervised Machine Learning — Classification

**Target Variable:** `Survived`

**Prediction:**

```text
0 → Did not survive
1 → Survived
```

**Evaluation Metric:** Accuracy

---

# 🎯 Objective

For every passenger in the test dataset, predict whether the passenger survived.

The final submission file must contain:

```csv
PassengerId,Survived
892,0
893,1
894,0
895,1
...
```

The `PassengerId` identifies the passenger, while `Survived` contains the model's prediction.

---

# 📊 Evaluation Metric

The Titanic competition uses **classification accuracy**.

Accuracy is calculated as:

```text
Accuracy =
Correct Predictions / Total Predictions
```

For example, if the model correctly predicts 780 passengers out of 891:

```text
Accuracy = 780 / 891
```

The final Kaggle score is calculated on the competition's hidden test labels.

---

# 🧰 Technologies Used

* Python 3
* NumPy
* Pandas
* Scikit-learn
* Kaggle Notebook
* Logistic Regression
* Data preprocessing
* Missing-value imputation
* One-Hot Encoding
* Train/Test Split

---

# 📁 Dataset

The Titanic dataset generally contains:

```text
train.csv
test.csv
```

---

## Training Dataset

`train.csv` contains the target variable:

```text
Survived
```

along with passenger information.

Important columns include:

| Column        | Description                       |
| ------------- | --------------------------------- |
| `PassengerId` | Unique passenger identifier       |
| `Survived`    | Target variable                   |
| `Pclass`      | Passenger class                   |
| `Name`        | Passenger name                    |
| `Sex`         | Passenger sex                     |
| `Age`         | Passenger age                     |
| `SibSp`       | Number of siblings/spouses aboard |
| `Parch`       | Number of parents/children aboard |
| `Ticket`      | Ticket number                     |
| `Fare`        | Passenger fare                    |
| `Cabin`       | Cabin number                      |
| `Embarked`    | Port of embarkation               |

---

## Test Dataset

`test.csv` contains passenger information but does not contain:

```text
Survived
```

The model predicts this value.

---

# 🔎 Feature Description

### Passenger Class

```text
Pclass
```

Represents the passenger's ticket class:

```text
1 → First Class
2 → Second Class
3 → Third Class
```

---

### Sex

```text
Sex
```

Passenger sex:

```text
male
female
```

---

### Age

```text
Age
```

Passenger age.

Some values may be missing, so missing values must be handled before training.

---

### Family Features

```text
SibSp
Parch
```

These describe family relationships aboard the Titanic.

---

### Fare

```text
Fare
```

The amount paid for the ticket.

---

### Embarked

```text
Embarked
```

Port where the passenger boarded.

Common values:

```text
C → Cherbourg
Q → Queenstown
S → Southampton
```

---

# 🚀 Machine Learning
