# Manual Implementation of Iterative Imputation Using Pandas and Regression

## 1. Introduction

Iterative Imputation is a missing-value imputation technique in which we use machine learning models to predict missing values in different columns.

Instead of replacing missing values with a fixed statistical value such as mean or median, we predict them using the relationships between other features.

### Libraries Required

```python
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
```

## 2. Create a Sample Dataset

Let's create a DataFrame containing missing values in multiple numerical columns.

```python
df = pd.DataFrame({
    'Age': [20, 25, np.nan, 30, 35, 40],
    'Salary': [20000, np.nan, 30000, 40000, 45000, 50000],
    'Experience': [1, 2, 3, np.nan, 5, 6]
})

print(df)
```

Output:

| Index | Age | Salary | Experience |
| ----- | --: | -----: | ---------: |
| 0     |  20 |  20000 |          1 |
| 1     |  25 |    NaN |          2 |
| 2     | NaN |  30000 |          3 |
| 3     |  30 |  40000 |        NaN |
| 4     |  35 |  45000 |          5 |
| 5     |  40 |  50000 |          6 |

Our objective is to predict the missing values using the other available columns.

---

## 3. Step 1: Identify Columns Containing Missing Values

First, we need to identify which columns contain missing values.

```python
missing_cols = df.columns[df.isnull().any()].tolist()

print(missing_cols)
```

### Understanding the Code

**A. `df.isnull()`**

Checks every element in the DataFrame and returns True wherever a missing value exists.

```python
df.isnull()
```

**B. `df.isnull().any()`**

The `any()` function checks each column.

It returns True if at least one missing value exists in that column.

```python
df.isnull().any()
```

**C. `df.columns[...]`**

Selects the column names corresponding to True values.

**D. `.tolist()`**

Converts the resulting column names into a Python list.

Output:

```python
['Age', 'Salary', 'Experience']
```

Now we know which columns require imputation.

---

## 4. Step 2: Initial Imputation Using Median

Before training regression models, we need an initial complete dataset.

Why?

Because when we use other columns as independent features, those columns might also contain missing values.

Therefore, we initially replace missing values with the median of their respective columns.

```python
df_initial = df.copy()

for col in missing_cols:

    median_value = df_initial[col].median()

    df_initial[col] = df_initial[col].fillna(median_value)
```

### Understanding the Code

**A. Create a copy**

```python
df_initial = df.copy()
```

Creates a separate copy of the original DataFrame.

We don't want to modify our original dataset.

**B. Iterate over missing columns**

```python
for col in missing_cols:
```

The loop processes each incomplete column.

**C. Calculate the median**

```python
median_value = df_initial[col].median()
```

Calculates the median of the current column.

**D. Replace missing values**

```python
df_initial[col] = df_initial[col].fillna(median_value)
```

Replaces missing values with the calculated median.

After this operation, our initial DataFrame contains no missing values.

**Important:** We will use this initial DataFrame for training features and predictions. However, we must preserve the original missing-value locations because we only want to replace values that were originally missing.

---

## 5. Step 3: Select One Column as the Target

Now we begin the actual iterative imputation process.

Let's assume we want to predict missing values in the Age column.

```python
target_col = 'Age'
```

Our target variable is Age.

The remaining columns will become independent features.

```python
feature_cols = df.columns.drop(target_col)
```

Output:

```python
Index(['Salary', 'Experience'])
```

Therefore:

* X = Salary and Experience.
* y = Age.

We will train a regression model to predict Age using Salary and Experience.

---

## 6. Step 4: Separate Training Rows and Prediction Rows

This is one of the most important parts of the implementation.

We need to divide our rows into two groups.

1. Rows where the target value is originally available.
2. Rows where the target value is originally missing.

### A. Create a Missing Value Mask

```python
missing_mask = df[target_col].isnull()
```

This creates a Boolean Series.

Example:

```python
False
False
True
False
False
False
```

Here:

* True means the original target value was missing.
* False means the original target value was available.

### B. Create Training Features

```python
X_train = df_initial.loc[~missing_mask, feature_cols]
```

Explanation:

* `df_initial` contains our initially completed dataset.
* `.loc[]` selects particular rows and columns.
* `~missing_mask` selects rows where the target was originally available.
* `feature_cols` selects all columns except the target column.

### C. Create Training Target

```python
y_train = df.loc[~missing_mask, target_col]
```

We use the original DataFrame here because we want the actual observed target values, not the median-imputed values.

### D. Create Prediction Features

```python
X_missing = df_initial.loc[missing_mask, feature_cols]
```

This selects rows where the target was originally missing.

We use the other columns as input features to predict the missing target values.

At this point, we have three important variables:

| Variable  | Purpose                                |
| --------- | -------------------------------------- |
| X_train   | Features for training                  |
| y_train   | Actual target values for training      |
| X_missing | Features for predicting missing values |

---

## 7. Step 5: Train the Linear Regression Model

Now we have prepared our training data.

Let's import and create the regression model.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

### Train the Model

```python
model.fit(X_train, y_train)
```

The `fit()` method trains the regression model.

It learns the relationship between:

* Salary and Age.
* Experience and Age.

The model learns regression coefficients using the rows where Age was originally available.

The learned relationship can then be used to predict Age for rows where it is missing.

---

## 8. Step 6: Predict the Missing Values

After training the model, we can predict the missing Age values.

```python
predictions = model.predict(X_missing)
```

The `predict()` method uses the trained regression model to calculate the missing values.

For example, suppose the model predicts:

```python
[27.5]
```

This means the estimated Age for the missing row is 27.5.

### Replace the Initial Median Value

Now we replace the initially median-imputed value with our regression prediction.

```python
df_initial.loc[missing_mask, target_col] = predictions
```

Understanding this statement:

* `df_initial` selects our working DataFrame.
* `missing_mask` selects originally missing rows.
* `target_col` selects the Age column.
* `predictions` contains the predicted values.

Now the Age column has been updated using regression predictions.

---

## 9. Step 7: Repeat the Process for Every Incomplete Column

We don't want to write the same code separately for Age, Salary and Experience.

Instead, we can use a loop.

```python
for target_col in missing_cols:

    # Identify originally missing rows
    missing_mask = df[target_col].isnull()

    # Select all other columns as features
    feature_cols = df.columns.drop(target_col)

    # Create training features
    X_train = df_initial.loc[~missing_mask, feature_cols]

    # Create actual training target
    y_train = df.loc[~missing_mask, target_col]

    # Create prediction features
    X_missing = df_initial.loc[missing_mask, feature_cols]

    # Create regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict missing values
    predictions = model.predict(X_missing)

    # Replace initial median values
    df_initial.loc[missing_mask, target_col] = predictions
```

### Understanding the Execution Order

The loop processes each incomplete column.

| Iteration within the cycle | Target Column | Independent Features |
| -------------------------- | ------------- | -------------------- |
| 1                          | Age           | Salary, Experience   |
| 2                          | Salary        | Age, Experience      |
| 3                          | Experience    | Age, Salary          |

During the process:

1. Age is predicted using Salary and Experience.
2. Salary is predicted using the updated Age and Experience.
3. Experience is predicted using the updated Age and Salary.

Once all incomplete columns have been processed, one complete imputation cycle is finished.

---

## 10. Step 8: Repeat the Complete Process

One cycle might not be sufficient.

The predictions generated during the first cycle can be improved using the updated values from other columns.

Therefore, we repeat the complete process.

We introduce an outer loop.

```python
for iteration in range(10):

    for target_col in missing_cols:

        missing_mask = df[target_col].isnull()

        feature_cols = df.columns.drop(target_col)

        X_train = df_initial.loc[~missing_mask, feature_cols]

        y_train = df.loc[~missing_mask, target_col]

        X_missing = df_initial.loc[missing_mask, feature_cols]

        model = LinearRegression()

        model.fit(X_train, y_train)

        predictions = model.predict(X_missing)

        df_initial.loc[missing_mask, target_col] = predictions
```

### Understanding the Two Loops

**Outer Loop**

```python
for iteration in range(10):
```

Controls the maximum number of complete imputation cycles.

**Inner Loop**

```python
for target_col in missing_cols:
```

Processes every column containing missing values.

Execution:

Iteration 1:

Age → Salary → Experience

Iteration 2:

Age → Salary → Experience

Iteration 3:

Age → Salary → Experience

This continues until the maximum number of iterations is reached.

---

## 11. Step 9: Understand Convergence

We don't necessarily want to execute the algorithm for a fixed number of iterations.

We want to check whether the imputed values have stopped changing significantly.

For this, we calculate the difference between the imputed values from consecutive iterations.

### A. Save Previous Imputed Values

Before starting each complete iteration:

```python
old_values = df_initial.where(
    original_missing_mask
).to_numpy()
```

This extracts the values at the positions that were originally missing.

### B. Get New Imputed Values

After completing the entire iteration:

```python
new_values = df_initial.where(
    original_missing_mask
).to_numpy()
```

This extracts the newly updated imputed values.

### C. Calculate the Difference

```python
difference = np.nanmax(
    np.abs(new_values - old_values)
)
```

Understanding each operation:

* `new_values - old_values`: Calculates the changes.
* `np.abs()`: Converts negative differences into positive differences.
* `np.nanmax()`: Finds the maximum change while ignoring NaN values.

### D. Check Convergence

```python
if difference < 0.001:
    break
```

If the maximum change becomes smaller than the specified tolerance, we stop the iterations.

Example:

| Iteration | Maximum Change |
| --------- | -------------: |
| 1         |           2.50 |
| 2         |           0.80 |
| 3         |           0.12 |
| 4         |          0.008 |
| 5         |         0.0005 |

The changes are becoming smaller.

This indicates that the imputed values are stabilizing.

**Important:** Convergence does not mean that the predictions are equal to the actual missing values. It only means that the predictions are no longer changing significantly between iterations.

---

# 12. Complete Manual Implementation

Now let's combine all the steps into one reusable function.

```python
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def iterative_imputer(df, max_iter=10, tol=0.001):

    # Step 1: Create a copy of original DataFrame
    df_initial = df.copy()

    # Step 2: Identify columns containing missing values
    missing_cols = df.columns[df.isnull().any()].tolist()

    # If no missing values exist, return original DataFrame
    if not missing_cols:
        return df_initial

    # Step 3: Store original missing-value locations
    original_missing_mask = df.isnull()

    # Step 4: Initial imputation using median
    for col in missing_cols:

        median_value = df_initial[col].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Column {col} contains no valid values."
            )

        df_initial[col] = df_initial[col].fillna(median_value)

    # Step 5: Start iterative process
    for iteration in range(max_iter):

        # Store previous imputed values
        old_values = df_initial.where(
            original_missing_mask
        ).to_numpy().copy()

        # Step 6: Process every incomplete column
        for target_col in missing_cols:

            # Identify originally missing rows
            missing_mask = original_missing_mask[target_col]

            # Select remaining columns as features
            feature_cols = df.columns.drop(target_col)

            # Training features
            X_train = df_initial.loc[
                ~missing_mask, feature_cols
            ]

            # Original observed target values
            y_train = df.loc[
                ~missing_mask, target_col
            ]

            # Features for predicting missing values
            X_missing = df_initial.loc[
                missing_mask, feature_cols
            ]

            # Create regression model
            model = LinearRegression()

            # Train regression model
            model.fit(X_train, y_train)

            # Predict missing values
            predictions = model.predict(X_missing)

            # Update the working DataFrame
            df_initial.loc[
                missing_mask, target_col
            ] = predictions

        # Step 7: Get updated imputed values
        new_values = df_initial.where(
            original_missing_mask
        ).to_numpy()

        # Step 8: Calculate maximum change
        difference = np.nanmax(
            np.abs(new_values - old_values)
        )

        # Step 9: Check convergence
        if difference < tol:

            print(
                f"Converged at iteration {iteration + 1}"
            )

            break

    return df_initial
```

---

## 13. Test Our Function

Let's create our dataset again.

```python
df = pd.DataFrame({
    'Age': [20, 25, np.nan, 30, 35, 40],
    'Salary': [20000, np.nan, 30000, 40000, 45000, 50000],
    'Experience': [1, 2, 3, np.nan, 5, 6]
})
```

Call our function:

```python
result = iterative_imputer(df)
```

Display the original DataFrame:

```python
print("Original DataFrame:")
print(df)
```

Display the imputed DataFrame:

```python
print("Imputed DataFrame:")
print(result)
```

Check whether any missing values remain:

```python
print(result.isnull().sum())
```

All originally missing numerical values should now have predictions.

---

# 14. Understanding the Complete Implementation

| Component               | Responsibility                             |
| ----------------------- | ------------------------------------------ |
| `missing_cols`          | Identifies incomplete columns              |
| `original_missing_mask` | Remembers original missing-value locations |
| `median()`              | Performs initial imputation                |
| `feature_cols`          | Selects independent features               |
| `X_train`               | Contains training features                 |
| `y_train`               | Contains actual observed target values     |
| `X_missing`             | Contains features for prediction           |
| `LinearRegression()`    | Creates the regression estimator           |
| `fit()`                 | Trains the regression model                |
| `predict()`             | Predicts missing values                    |
| `difference`            | Measures changes between iterations        |
| `break`                 | Stops when convergence is achieved         |

---

# 15. Important Limitations of Our Implementation

This implementation is designed to understand the internal working of Iterative Imputation.

It is not yet a production-ready replacement for sklearn's IterativeImputer.

### 1. Numerical Columns Only

Our implementation assumes that all features are numerical.

Categorical features require encoding before they can be used by Linear Regression.

### 2. Regression Estimator

We are using Linear Regression.

Other estimators can also be used, such as:

* Bayesian Ridge Regression
* Decision Tree Regressor
* Random Forest Regressor
* Extra Trees Regressor

### 3. Data Leakage

We currently calculate medians and train regression models using the entire DataFrame.

In an actual ML project, we must split the data into training and testing sets first.

We should learn the imputation parameters using training data and apply the learned transformations to test data.

### 4. Separate Fit and Transform Methods

Our current function performs fitting and transformation together.

A proper sklearn-compatible transformer should implement:

```python
class CustomIterativeImputer:

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass
```

This will allow us to integrate our custom imputer into:

* Pipeline
* ColumnTransformer
* GridSearchCV

---

# 16. Final Workflow

The complete algorithm can be summarized as follows:

```text
Original DataFrame
        |
        v
Identify Missing Columns
        |
        v
Store Original Missing Locations
        |
        v
Initial Median Imputation
        |
        v
Start Iterative Process
        |
        v
Select One Incomplete Column
        |
        v
Separate Features (X) and Target (y)
        |
        v
Train Regression Model
        |
        v
Predict Missing Values
        |
        v
Update DataFrame
        |
        v
Move to Next Incomplete Column
        |
        v
Complete One Iteration
        |
        v
Calculate Difference
        |
        v
Check Convergence
        |
        +------ Not Converged ------+
        |                           |
        |                           v
        |                     Repeat Process
        |
        +------ Converged
                    |
                    v
             Final DataFrame
```

**Key Takeaway:** Iterative Imputation repeatedly uses the relationships between features to estimate missing values. Each incomplete column becomes a target variable in turn, while the remaining columns act as independent features.




















# Implement Iterative Imputation from scratch using Pandas and Regression

We're going to build our own version of `IterativeImputer`, without using sklearn's `IterativeImputer`.

We'll use:

* **Pandas:** To manipulate our DataFrame.
* **NumPy:** To identify missing values and work with arrays.
* **Linear Regression:** To predict missing values.
* **Python functions:** To make our implementation reusable.

I'll teach this as a practical coding lesson. We'll understand every operation before moving to the next one.

## 1. Understand what we are building

Imagine we have a dataset containing three numerical features:

Output:

| Index | Age | Salary | Experience |
| ----- | --- | ------ | ---------- |
| 0     | 20  | 20000  | 1          |
| 1     | 25  | NaN    | 2          |
| 2     | NaN | 30000  | 3          |
| 3     | 30  | 40000  | NaN        |
| 4     | 35  | 45000  | 5          |
| 5     | 40  | 50000  | 6          |

We have three columns with missing values.

Our objective is to predict those missing values using the other columns.

## 2. Understand the complete algorithm

## Our algorithm

1. **Identify missing values**

   Find which columns contain NaN values.
2. **Initial imputation**

   Fill missing values using the median of each column.
3. **Select one target column**

   For example, select Age.
4. **Create X and y**

   Other columns become features. Age becomes the target.
5. **Train regression model**

   Train only on rows where the original Age was available.
6. **Predict missing Age values**

   Replace the initial median values with predictions.
7. **Repeat for other columns**

   Salary, then Experience.
8. **Repeat iterations**

   Continue until the imputed values stabilize or we reach the iteration limit.

Now let's code it.

---

## 3. Step one: Find columns containing missing values

We need to know which columns actually have missing values.

Let's understand each part.

**First:**

This produces a Boolean DataFrame.

| Index | Age   | Salary | Experience |
| ----- | ----- | ------ | ---------- |
| 0     | False | False  | False      |
| 1     | False | True   | False      |
| 2     | True  | False  | False      |
| 3     | False | False  | True       |
| 4     | False | False  | False      |
| 5     | False | False  | False      |

**Second:**

`any()` checks each column.

It returns True if at least one missing value exists in that column.

Output:

**Third:**

It extracts the names of columns where the result is True.

Output:

So now we know which columns need imputation.

## 4. Step two: Initial imputation using median

Before training regression models, we need an initial complete dataset.

Why?

Because when we select other columns as features, they might also contain missing values.

Let's calculate the median of each column.

Let's break this down.

**Line 1:**

We create a copy so that our original dataset remains unchanged.

**Line 2:**

We loop through each column:

* First Age
* Then Salary
* Then Experience

**Line 3:**

We calculate the median of the current column.

**Line 4:**

We replace missing values with the calculated median.

Now our DataFrame has no missing values.

This is our initial dataset.

(Important) We will use this initial DataFrame for predictions, but we must preserve the original missing-value locations so that we know which values need to be predicted.

## 5. Step three: Understand how we select X and y

This is the most important part of the implementation.

Suppose we want to predict missing values in Age.

Our target:

Now select every other column as features.

Output:

Now we need two groups of rows.

* Rows where Age was originally available: training data.
* Rows where Age was originally missing: prediction data.

Let's create a mask.

Output:

A mask is simply a Boolean series used to select rows.

Now create training features:

And training target:

Why `~missing_mask`?

Because `~` reverses Boolean values.

| Original mask | `~missing_mask` |
| ------------- | --------------- |
| False         | True            |
| False         | True            |
| True          | False           |
| False         | True            |
| False         | True            |
| False         | True            |

So we select rows where Age was originally available.

Now select the rows whose Age needs prediction:

We now have:

**Training data**

Rows where Age is known

| Salary | Experience | Age |
| ------ | ---------- | --- |
| 20000  | 1          | 20  |
| 20000? | 2          | 25  |
| 40000  | 4?         | 30  |
| 45000  | 5          | 35  |
| 50000  | 6          | 40  |

Illustrative layout only: actual feature values come from the initial median-filled DataFrame.

---

**Prediction data**

Rows where Age was originally missing

| Salary | Experience |
| ------ | ---------- |
| 30000  | 3          |

Notice: The target column Age is not included in `X_missing`. We want to predict it, not use it as an input.

## 6. Step four: Train the Linear Regression model

Now we have:

* `X_train`: Salary and Experience.
* `y_train`: Age.
* `X_missing`: Salary and Experience for rows where Age is missing.

Let's import Linear Regression.

Create our model:

Train it:

What happens here?

The model learns the relationship between:

* Salary and Age.
* Experience and Age.

It learns its regression coefficients from the available Age observations.

The model learns a relationship that can be used to estimate Age when Salary and Experience are known.

## 7. Step five: Predict the missing values

Now we have trained our model.

Let's predict Age:

Suppose the model predicts:

This means the missing Age at index 2 is estimated as 27.5.

But we need to put this prediction back into our DataFrame.

Understand this carefully:

* `df_initial` is the DataFrame we are updating.
* `missing_mask` selects the originally missing Age rows.
* `target_col` selects the Age column.
* `predictions` contains the predicted values.

We have now completed the imputation for Age.

## 8. Step six: Repeat for every missing column

Now comes the interesting part.

We don't want to write the same code separately for Age, Salary and Experience.

We use a loop.

Let's understand what happens during this loop.

| Loop | Target     | Features           |
| ---- | ---------- | ------------------ |
| 1    | Age        | Salary, Experience |
| 2    | Salary     | Age, Experience    |
| 3    | Experience | Age, Salary        |

After the first column is imputed, its predictions are available when we move to the next column.

For example:

When predicting Salary, the model can now use the newly predicted Age.

This completes one full iteration.

**One important detail:** We use the original DataFrame `df` to identify which target values were originally missing. We use `df_initial` for the current feature values and predictions.

This distinction prevents us from accidentally treating our own predictions as actual training targets.

---

## 9. Step seven: Repeat the process using a while/for loop

One iteration is not necessarily enough.

We want to repeat the complete cycle.

What does this do?

The outer loop controls the number of iterations.

The inner loop processes every incomplete column.

**Execution order**

**Iteration 1**

Age → Salary → Experience

**Iteration 2**

Age → Salary → Experience

**Iteration 3 ... 10**

Repeat the same process

But we have one missing piece: How do we know when to stop?

## 10. Step eight: Understand convergence

Remember what you said earlier about calculating the difference?

You're right that we need to observe how much the imputed values change.

We can save the previous missing values before an iteration:

After completing the entire iteration:

Now calculate the maximum change:

Let's understand this:

* `new_values - old_values`: Calculates changes.
* `np.abs()`: Removes negative signs.
* `np.max()`: Finds the largest change.

For example:

| Iteration | Maximum change |
| --------- | -------------- |
| 1         | 2.50           |
| 2         | 0.80           |
| 3         | 0.12           |
| 4         | 0.008          |
| 5         | 0.0005         |

We can stop when:

This is our convergence condition.

Remember: convergence means the predictions have stabilized. It does not prove that they equal the actual missing values.

---

# 11. Now combine everything into one reusable function

This is our final educational implementation.

Read the comments carefully. Each section corresponds to something we learned above.

```python
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

def iterative_imputer(df, max_iter=10, tol=0.001):

    # Step 1: Create a copy of original data
    df_initial = df.copy()

    # Step 2: Find columns containing missing values
    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        return df_initial

    # Step 3: Save the original missing-value locations
    original_missing_mask = df.isnull()

    # Step 4: Initial imputation using median
    for col in missing_cols:

        median_value = df_initial[col].median()

        if pd.isna(median_value):
            raise ValueError(f"Column {col} contains no valid values.")

        df_initial[col] = df_initial[col].fillna(median_value)

    # Step 5: Start iterative process
    for iteration in range(max_iter):

        # Save previous imputed values
        old_values = df_initial.where(
            original_missing_mask
        ).to_numpy()

        # Step 6: Iterate over every incomplete column
        for target_col in missing_cols:

            # Identify rows where target was originally missing
            missing_mask = original_missing_mask[target_col]

            # Select all other columns as features
            feature_cols = df.columns.drop(target_col)

            # Create training features
            X_train = df_initial.loc[~missing_mask, feature_cols]

            # Create actual training target
            y_train = df.loc[~missing_mask, target_col]

            # Create prediction features
            X_missing = df_initial.loc[missing_mask, feature_cols]

            # Train regression model
            model = LinearRegression()

            model.fit(X_train, y_train)

            # Predict missing values
            predictions = model.predict(X_missing)

            # Replace median values with predictions
            df_initial.loc[missing_mask, target_col] = predictions

        # Step 7: Calculate difference between iterations
        new_values = df_initial.where(
            original_missing_mask
        ).to_numpy()

        difference = np.nanmax(
            np.abs(new_values - old_values)
        )

        # Step 8: Check convergence
        if difference < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

    return df_initial
```

## 12. Test our function

Let's use our earlier dataset.

Now compare:

You should see that the missing values have been replaced with regression predictions, while the originally observed values remain unchanged.

# 13. Important: What have we actually built?

Our function follows this workflow:

| Component               | Responsibility                       |
| ----------------------- | ------------------------------------ |
| `missing_cols`          | Finds incomplete columns             |
| `original_missing_mask` | Remembers original missing locations |
| `median()`              | Performs initial imputation          |
| `feature_cols`          | Selects independent features         |
| `LinearRegression()`    | Learns relationships                 |
| `fit()`                 | Trains the regression model          |
| `predict()`             | Predicts missing values              |
| `difference`            | Measures changes between iterations  |
| `break`                 | Stops when convergence is achieved   |

## 14. Three things you should understand before moving to sklearn

**1. This is an educational implementation.**

It uses Linear Regression as the estimator. sklearn's `IterativeImputer` is more configurable and has additional features.

**2. This function is for learning, not yet production ML.**

We currently calculate medians and train models using the entire DataFrame. In a real ML project, this could cause data leakage.

We will eventually separate `fit()` and `transform()` so that:

* Training data learns the medians and regression models.
* Test data only receives the learned transformations.
* No information from test data leaks into training.

**3. This implementation assumes numerical columns.**

Categorical features require additional handling, such as encoding, and categorical target columns need a suitable estimator.

---

# 15. Check whether you understood the most important part

**Test your understanding** (/4)

[Check my understanding]

**Your next learning milestone:** Convert this function into a proper sklearn-compatible transformer with `fit()` and `transform()`, then use it inside `ColumnTransformer` and `Pipeline`. That is where you'll understand how to build your own preprocessing classes rather than just call existing sklearn functions.
