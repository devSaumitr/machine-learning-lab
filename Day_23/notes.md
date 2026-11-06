# Day 23 — What Is Feature Engineering?

> **Course:** CampusX 100 Days of Machine Learning, Day 23  
> **Official notebook:** none. CampusX has no `day23-*` folder; this is a concept day.  
> **Personal notebook:** `my_notebook.ipynb` (9 markdown cells, 0 code cells) — a checklist, not a runnable implementation.  
> **Dataset:** none for this day. Examples below use the Titanic table already studied in Days 19–22.  
> **Reference status:** see `REFERENCE_STATUS.md`.

This chapter has two parts. **Part A** is the main learning chapter: after reading it, you should be able to explain what feature engineering is, why it matters, how the main techniques work, and how to apply them without data leakage. **Part B** is advanced reference: formal notation, deeper mathematics, library internals, extra interview questions, and extended exam practice. Part B is optional for a first reading.

The day itself contains no code. That is not an accident and it is not missing work. The course uses Day 23 to give the vocabulary; Days 24–29 and 31–45 provide the implementations. This chapter keeps that distinction: explanations are teaching material, while any code shown here is an illustrative pattern from the later scikit-learn workflow, not an official Day-23 notebook output.

---

# Part A — Main Learning Chapter

## 1. The problem: a table is not a model-ready matrix

A machine-learning model does not see a spreadsheet the way a person does. It sees a mathematical function that maps an input vector to an output.

A single row of the Titanic data looks like this:

| PassengerId | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked | Survived |
|---:|---:|---|---|---:|---:|---:|---|---:|---|---|---:|
| 1 | 3 | Braund, Mr. Owen Harris | male | 22 | 1 | 0 | A/5 21171 | 7.25 | missing | S | 0 |

A model such as logistic regression computes something like

$$
\hat{y} = \sigma\left(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p\right),
$$

where $\hat{y}$ is the predicted probability, the $x_j$ are numeric features, and the $\beta_j$ are learned coefficients. The raw row contains strings (`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`), a missing value (`Cabin`, and sometimes `Age`), and columns with very different ranges (`Age` versus `Fare`). Most scikit-learn estimators cannot use that row directly.

We need a **feature map**:

$$
\mathbf{x}_i = \phi(\mathbf{r}_i),
$$

where $\mathbf{r}_i$ is the raw row and $\mathbf{x}_i$ is the numeric feature vector for that row. Stacking the vectors for $n$ rows gives the **design matrix**

$$
\mathbf{X} \in \mathbb{R}^{n \times p},
$$

with one row per observation and one column per feature.

### The vocabulary

- **Observation / sample / row:** one passenger, one customer, one transaction.
- **Feature / predictor / input:** a column used to predict.
- **Target / label / output:** the column being predicted, such as `Survived`.
- **Design matrix $\mathbf{X}$:** the numeric table given to the model.
- **Target vector $\mathbf{y}$:** the labels, one per row.
- **Feature engineering:** the process of turning raw data into a useful design matrix.

A model-ready matrix usually needs four properties:

1. **Numeric or safely encodable.** Strings must become numbers, or the estimator must explicitly support categorical columns.
2. **Finite and complete enough.** Most estimators cannot handle `NaN` or infinity.
3. **Consistent schema.** Training and test data must have the same columns, order, and meaning.
4. **Informative and leak-free.** The features must help predict the target without containing information that would not be available at prediction time.

Feature engineering is not cosmetic. It changes the hypothesis space available to the model. A linear model can only learn a linear function of the features you give it. If the useful pattern is nonlinear, you must either use a nonlinear model or construct a feature that makes the pattern linear.

### A first example

Suppose we want to predict survival from `Age` and `Fare`:

$$
\hat{y} = \sigma(\beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Fare}).
$$

This equation can use `Age` and `Fare` because they are numeric. It cannot directly use `Sex = "male"` or `Embarked = "S"`. It also cannot compute anything for a passenger whose `Age` is missing. Before fitting the model we must decide:

- How to fill or flag missing ages.
- How to represent `Sex` and `Embarked`.
- Whether `Fare` should be scaled or transformed.
- Whether to create new columns such as family size or fare per person.

Those decisions are feature engineering.

---

## 2. Why feature engineering matters

### 2.1 The same model can succeed or fail depending on representation

Consider a tiny classification problem with two binary inputs:

| $x_1$ | $x_2$ | $y$ |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This is the XOR pattern. No straight line in the $(x_1,x_2)$ plane separates the positive and negative examples. A linear classifier using only $x_1$ and $x_2$ cannot represent it exactly.

Now construct one new feature:

$$
x_3 = x_1 x_2.
$$

The function

$$
\hat{y} = x_1 + x_2 - 2x_3
$$

gives the correct answer:

- $(0,0)$: $0+0-0=0$
- $(0,1)$: $0+1-0=1$
- $(1,0)$: $1+0-0=1$
- $(1,1)$: $1+1-2=0$

The model did not become more powerful. The **representation** became more suitable. This is the central idea of feature construction.

### 2.2 Scale can dominate distance

Suppose two customers differ by 40 years in age and by 60,000 dollars in salary. A distance-based algorithm such as KNN computes something like

$$
d(\mathbf{a},\mathbf{b})
=
\sqrt{(a_{\text{age}}-b_{\text{age}})^2
+
(a_{\text{salary}}-b_{\text{salary}})^2}.
$$

The age difference contributes $40^2=1{,}600$. The salary difference contributes $60{,}000^2=3{,}600{,}000{,}000$. The salary term is more than two million times larger. The algorithm is effectively ignoring age unless salary is scaled.

This is why standardization, normalization, and robust scaling exist.

### 2.3 Models cannot learn from columns they cannot use

- Missing values break many numerical operations.
- Strings cannot be multiplied by coefficients unless encoded.
- A categorical column with 500 unique values can create 500 columns if one-hot encoded.
- A column with extreme outliers can dominate a mean, a variance, a distance, or a gradient.
- A noisy column can add variance without adding signal.

Feature engineering removes these practical blockers and shapes the signal the model can see.

### 2.4 Better features can beat a more complex model

A simple model with well-designed features often generalizes better than a complex model fed raw, noisy columns. This is not a law; it is a common engineering observation. A complex model can also overfit irrelevant detail. The right comparison is always empirical: use a proper validation split and compare against a baseline.

### 2.5 Feature engineering affects bias and variance

In learning theory, the expected error can be decomposed into bias, variance, and irreducible noise. Feature engineering influences the first two:

- A missing interaction term can create **approximation bias**: the model family cannot represent the true relationship.
- Adding many noisy or highly correlated features can increase **estimation variance**: the model fits accidental patterns in the training sample.
- Feature selection and extraction try to reduce variance by removing irrelevant or redundant dimensions.

There is no universally best feature set. The useful representation depends on the data, the model, and the prediction task.

### 2.6 The leakage warning

Feature engineering is also where data leakage most often enters a project. If a fill value, a scaler, an encoder, an outlier threshold, a target encoding, or a selected feature set is learned from the full dataset before the train/test split, information from the test set can influence training. The model then looks better in validation than it will in production.

The rule is simple:

> **Learn parameters on training data only; apply the learned transformation to validation and test data.**

A scikit-learn `Pipeline` is the standard way to enforce this rule.

---

## 3. The feature-engineering job list

The Day-23 notebook lists four jobs: **transformation**, **construction**, **selection**, and **extraction**. Around them sits the cleaning work: missing values, types, outliers, and scale.

| Job | Question it answers | Typical techniques | Where it appears later |
|---|---|---|---|
| Cleaning | Can the model read the value at all? | imputation, type conversion, invalid-value handling | Days 35–40 |
| Transformation | Can the value be represented better? | scaling, encoding, binning, log/power transforms | Days 24–27, 31–34 |
| Construction | Can we create a more useful signal? | ratios, differences, interactions, date parts, text features | Day 45, later feature engineering |
| Selection | Which existing features should remain? | variance threshold, chi-square, mutual information, RFE, Lasso | Day 29 and later |
| Extraction | Can we build a smaller set of new axes? | PCA, LDA, matrix factorization | Day 47 and later |

The boundaries are not rigid. One-hot encoding is a transformation and also a type conversion. Imputation is cleaning and transformation. Binning can be transformation and construction. The labels are useful for planning, not for building a mathematical taxonomy.

### 3.1 The notebook's four jobs

1. **Feature transformation** — change the representation of an existing column.
   - Scaling: `StandardScaler`, `MinMaxScaler`, `RobustScaler`.
   - Encoding: `OrdinalEncoder`, `OneHotEncoder`.
   - Binning: `KBinsDiscretizer`.
   - Distribution transforms: log, square root, Box–Cox, Yeo–Johnson.
   - Date/time extraction: year, month, weekday, cyclical encodings.

2. **Feature construction** — create a new column from existing columns.
   - `FamilySize = SibSp + Parch + 1`.
   - `FarePerPerson = Fare / FamilySize`.
   - `IsAlone = 1` if `FamilySize == 1`, else `0`.
   - `Title` extracted from `Name`.
   - `Deck` extracted from `Cabin`.
   - Interaction terms such as `Age × Pclass`.

3. **Feature selection** — keep a subset of the original columns.
   - Remove near-constant columns.
   - Remove highly correlated duplicates.
   - Rank by chi-square, ANOVA F, or mutual information.
   - Recursive feature elimination.
   - L1 regularization or tree importance.

4. **Feature extraction** — create a smaller set of new columns that are combinations of the old ones.
   - PCA projects onto directions of maximum variance.
   - LDA projects onto directions that separate classes.
   - Autoencoders and matrix factorization learn latent representations.

### 3.2 The workflow picture

```text
Raw data
   |
   v
Inspect / EDA  (Days 19-22)
   |
   v
Train/test split
   |
   v
Fit preprocessing on training data only
   |
   +--> missing-value imputation
   +--> type conversion / encoding
   +--> outlier treatment
   +--> scaling / distribution transform
   +--> feature construction
   +--> feature selection
   |
   v
Transform training and test data with the same learned objects
   |
   v
Model training and validation
   |
   v
Error analysis and iteration
```

The order is not sacred. Some operations must happen before others:

- You cannot scale a column that is still a string.
- You usually impute before scaling so the scaler sees finite numbers.
- You must split before learning any parameter that depends on the data.
- Feature selection should be inside cross-validation when it uses the target.

---

## 4. Missing values

### 4.1 Why values go missing

Missingness has many causes:

- A field was optional and the user skipped it.
- A sensor failed.
- A value was not recorded in the source system.
- A value was removed for privacy.
- A value is genuinely not applicable.
- A data-join failed and produced a blank.

The statistical name for the missingness mechanism matters because it determines whether a simple fix is defensible.

### 4.2 MCAR, MAR, and MNAR

Let $M$ be the missingness indicator: $M=1$ if the value is missing, $M=0$ if observed.

- **MCAR — Missing Completely At Random.** The probability of missingness is independent of both observed and unobserved values:
  $$
  P(M=1 \mid X_{\text{obs}}, X_{\text{miss}}) = P(M=1).
  $$
  Example: a random sample of survey forms was lost.

- **MAR — Missing At Random.** The probability of missingness depends on observed variables but not on the missing value itself:
  $$
  P(M=1 \mid X_{\text{obs}}, X_{\text{miss}}) = P(M=1 \mid X_{\text{obs}}).
  $$
  Example: older passengers are more likely to have a missing age, and age is missing after conditioning on observed class and fare.

- **MNAR — Missing Not At Random.** The probability of missingness depends on the unobserved value:
  $$
  P(M=1 \mid X_{\text{obs}}, X_{\text{miss}}) \neq P(M=1 \mid X_{\text{obs}}).
  $$
  Example: people with very high incomes are less likely to report income.

The names are slightly counterintuitive: “missing at random” does **not** mean the value is random. It means that, after conditioning on the observed data, the missingness no longer depends on the missing value.

In practice we cannot prove the mechanism from the data alone. We use domain knowledge and sensitivity analysis.

### 4.3 Univariate versus multivariate imputation

**Univariate imputation** uses only the column being filled:

- mean for a symmetric numeric column;
- median for a skewed numeric column;
- mode for a categorical column;
- a constant such as `"Unknown"` or `-1`.

**Multivariate imputation** uses other columns in the same row:

- KNN imputation: average the values of similar rows.
- Iterative imputation (MICE): model each incomplete column as a function of the others and iterate.
- Regression imputation: predict the missing value from the other features.

Multivariate methods can preserve relationships between columns, but they are more complex and can also introduce bias if the imputation model is wrong.

### 4.4 Mean imputation shrinks variance

Suppose a column has $n$ rows, $n_o$ observed values, and $n_m = n-n_o$ missing values. We fill every missing value with the observed mean $\bar{x}_o$. The new column has the same mean, because the imputed values equal the mean. But each imputed value has zero deviation from the mean. The completed sample variance is therefore

$$
s_{\text{imputed}}^2
=
\frac{(n_o-1)s_o^2}{n-1},
$$

where $s_o^2$ is the sample variance of the observed values. The completed variance is smaller than the observed variance. The standard deviation shrinks by roughly the factor

$$
\sqrt{\frac{n_o-1}{n-1}}.
$$

For the Titanic `Age` column, $n_o=714$, $n=891$, and the observed standard deviation is about $14.5265$:

$$
s_{\text{imputed}}
=
14.5265\sqrt{\frac{713}{890}}
\approx 13.00.
$$

So mean imputation makes the column look more concentrated than it really is. This can distort distances, correlations, and model coefficients.

### 4.5 A tiny worked example

Take observed ages

$$
[20,\ 25,\ 30,\ 40].
$$

The mean is

$$
\bar{x} = \frac{20+25+30+40}{4}=28.75.
$$

The median is

$$
\frac{25+30}{2}=27.5.
$$

If the fifth row is missing and we fill it with the mean, the completed column is

$$
[20,\ 25,\ 30,\ 40,\ 28.75].
$$

The mean remains $28.75$, but the new value adds no information about the true missing age. The median fill would produce $[20,25,30,40,27.5]$, which is less affected by the two larger values.

### 4.6 Missing indicators

Sometimes the fact that a value is missing is itself informative. For example, if `Cabin` is missing for 687 of 891 Titanic passengers, “no cabin recorded” may be associated with passenger class and survival. A **missing indicator** adds a binary column:

$$
I_{\text{missing}} =
\begin{cases}
1, & \text{if the original value is missing},\\
0, & \text{otherwise}.
\end{cases}
$$

The imputed value and the indicator can both be given to the model. This is especially useful when missingness is MAR or MNAR and the pattern may carry signal.

### 4.7 The scikit-learn pattern

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_train_age = imputer.fit_transform(train[["Age"]])
X_test_age = imputer.transform(test[["Age"]])
```

What happens:

- `SimpleImputer(strategy="median")` creates an unfitted transformer.
- `fit` computes the median of the training `Age` column and stores it in `imputer.statistics_`.
- `transform` replaces missing values using that stored median.
- `fit_transform` is a convenience that calls `fit` and then `transform` on the same data.

The test set must use `transform`, not `fit_transform`. If we call `fit_transform` on the test set, the imputer learns a test-set median, which is leakage.

For categorical columns:

```python
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_embarked = cat_imputer.fit_transform(train[["Embarked"]])
X_test_embarked = cat_imputer.transform(test[["Embarked"]])
```

### 4.8 Common missing-value mistakes

1. **Imputing before splitting.** This lets test-set information enter the fill value.
2. **Using the target to impute.** A feature that uses `Survived` to fill `Age` is leakage unless it is part of a carefully cross-fitted model.
3. **Filling with zero when zero is meaningful.** A missing salary is not a salary of zero.
4. **Dropping every row with a missing value.** On Titanic, dropping rows with missing `Age` or `Cabin` can remove a large, non-random part of the data.
5. **Ignoring the missingness pattern.** The missingness itself may be predictive.
6. **Assuming mean imputation preserves variance.** It does not.

---

## 5. Categorical variables and encoding

### 5.1 Types of categorical variables

- **Nominal:** categories with no natural order. Examples: `Sex`, `Embarked`, `color`.
- **Ordinal:** categories with a meaningful order. Examples: `small < medium < large`, `low < medium < high`.
- **Binary:** two categories. A special case of nominal or ordinal.
- **High-cardinality:** many distinct categories. Examples: `Ticket`, `Name`, `zip_code`.
- **Target:** the label being predicted, such as `Survived`.

The type determines the encoding. There is no single correct encoder for every categorical column.

### 5.2 Why arbitrary integer codes are dangerous

Suppose `color` has categories `red`, `green`, `blue`. If we map them to $0,1,2$, a linear model sees

$$
\hat{y} = \beta_0 + \beta_1 \cdot \text{color}.
$$

It treats green as halfway between red and blue, and blue as twice red. That is not meaningful for a nominal variable. A distance-based model also treats the numerical gap as real distance.

For an ordinal variable, integer codes can be appropriate because the order is real. But the spacing between codes is still an assumption. `small=1, medium=2, large=3` assumes equal spacing; sometimes that is acceptable, sometimes it is not.

### 5.3 One-hot encoding

One-hot encoding creates one binary column per category. For `color`:

| Row | red | green | blue |
|---|---:|---:|---:|
| red | 1 | 0 | 0 |
| green | 0 | 1 | 0 |
| blue | 0 | 0 | 1 |

For $k$ categories, one-hot encoding uses $k$ columns. If the model has an intercept, the columns sum to 1 for every row, which creates perfect collinearity with the intercept. This is the **dummy variable trap**. A common fix is to drop one category, leaving $k-1$ columns. The dropped category becomes the reference level.

For linear models with an intercept, dropping one dummy avoids a rank-deficient design matrix. For tree-based models, dropping a dummy is usually unnecessary; the tree can split on any encoding.

### 5.4 Ordinal encoding

Ordinal encoding maps ordered categories to integers:

$$
\text{small}\mapsto 0,\quad
\text{medium}\mapsto 1,\quad
\text{large}\mapsto 2.
$$

Use it when the order is meaningful. Do not use it for nominal categories unless the model is tree-based and you have a specific reason, because the numeric order can create false relationships.

### 5.5 Label encoding

Label encoding usually refers to encoding the **target** labels, not the input features:

$$
\text{yes}\mapsto 1,\quad
\text{no}\mapsto 0.
$$

Some libraries also call integer encoding of features “label encoding.” In this course, be precise: for input features, say **ordinal encoding** or **one-hot encoding**; for the target, say **label encoding**.

### 5.6 High-cardinality categories

One-hot encoding a column with 10,000 categories creates 10,000 columns. That is expensive and sparse, and many categories appear only a few times. Alternatives include:

- **Frequency encoding:** replace each category with its count or frequency.
- **Target encoding:** replace each category with a smoothed average of the target, computed with cross-fitting to reduce leakage.
- **Hashing:** map categories into a fixed number of buckets.
- **Embeddings:** learn a low-dimensional vector for each category in a neural network.
- **Grouping rare categories:** combine categories with very few observations into `"Other"`.

Target encoding is powerful but dangerous. If you compute the target mean for each category on the full training data and then evaluate on the same rows, the encoding contains the answer. Use out-of-fold encoding or smoothing, and never use the test target.

### 5.7 scikit-learn pattern

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

X_train_cat = encoder.fit_transform(train[["Sex", "Embarked"]])
X_test_cat = encoder.transform(test[["Sex", "Embarked"]])
```

Key details:

- `handle_unknown="ignore"` prevents an error when the test set contains a category not seen in training. The unseen category becomes an all-zero row.
- `sparse_output=False` returns a dense array. In older scikit-learn versions the argument is `sparse=False`.
- `encoder.categories_` stores the categories learned from training.
- `encoder.get_feature_names_out()` gives the output column names.

If you use `OneHotEncoder` inside a `ColumnTransformer`, you usually do not need to convert to a DataFrame manually; the transformer handles the array.

### 5.8 Titanic example

- `Sex` has two categories: `male`, `female`. One-hot encoding creates two columns, or one if you drop a reference.
- `Embarked` has three categories: `S`, `C`, `Q`, plus two missing values. Impute the missing values first, then encode.
- `Name` and `Ticket` have many unique values. One-hot encoding them directly is usually a bad idea. Extract a useful part (`Title` from `Name`, maybe ticket prefix from `Ticket`) or use a targeted encoding.
- `Cabin` is missing for 687 of 891 passengers. A missing indicator plus a deck letter extracted from the first character may be more useful than one-hot encoding the full cabin string.

### 5.9 Common encoding mistakes

1. One-hot encoding high-cardinality identifiers.
2. Ordinal encoding nominal categories and interpreting the codes as distances.
3. Fitting the encoder on training and test data together.
4. Forgetting `handle_unknown` and crashing on unseen categories.
5. Encoding the target as an input feature.
6. Dropping a dummy without understanding the model's intercept.
7. Using target encoding without cross-fitting.

---

## 6. Scaling

### 6.1 Why scale?

Features can have wildly different units:

- `Age`: roughly 0–80 years.
- `Fare`: roughly 0–512, with a long tail.
- `Salary`: tens of thousands.
- `Number of children`: 0–10.

Algorithms that use distances, dot products, gradients, or regularization can be dominated by the largest-scale feature. Scaling puts features on comparable footing.

### 6.2 Standardization

Standardization converts a column to zero mean and unit variance:

$$
z = \frac{x-\mu}{\sigma},
$$

where $\mu$ is the training mean and $\sigma$ is the training standard deviation. The transformed column has mean 0 and standard deviation 1 (up to sample/population conventions). It is not bounded; values can be negative and can be large if the original distribution has a long tail.

### 6.3 Min–max normalization

Min–max scaling maps the training range to $[0,1]$:

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}.
$$

It preserves the shape of the distribution but is sensitive to outliers: one extreme value can compress all other values into a tiny interval.

### 6.4 Robust scaling

Robust scaling uses the median and interquartile range:

$$
x' = \frac{x-\operatorname{median}(x)}{\operatorname{IQR}(x)},
$$

where

$$
\operatorname{IQR} = Q_3 - Q_1.
$$

It is less sensitive to outliers than standardization or min–max scaling, but it does not make the data normal.

### 6.5 A distance example

Suppose two rows have:

| Feature | Row A | Row B | Difference |
|---|---:|---:|---:|
| Age | 20 | 60 | 40 |
| Salary | 30,000 | 90,000 | 60,000 |

Euclidean distance:

$$
d = \sqrt{40^2 + 60{,}000^2} \approx 60{,}000.013.
$$

After standardizing both columns, suppose the age difference becomes 1.2 standard deviations and the salary difference becomes 0.8 standard deviations. The distance is then

$$
d = \sqrt{1.2^2 + 0.8^2} \approx 1.44.
$$

Now both features contribute meaningfully.

### 6.6 Which models need scaling?

| Model family | Usually needs scaling? | Why |
|---|---|---|
| KNN | Yes | Distances are the prediction mechanism. |
| k-means | Yes | Clusters are defined by distances. |
| SVM with RBF/polynomial kernel | Yes | Kernel distances and margins. |
| Linear/logistic regression with regularization | Yes | Penalty is applied to coefficients; scale changes their size. |
| PCA | Yes | PCA maximizes variance; unscaled features dominate. |
| Neural networks | Yes | Gradient descent and activation functions are sensitive to scale. |
| Decision trees | Usually no | Splits depend on order, not magnitude. |
| Random forests / gradient boosting | Usually no | Same reason; scaling is mostly harmless. |
| Naive Bayes | Usually no | Likelihoods are estimated per feature. |

“Usually no” does not mean “forbidden.” Scaling a tree model is harmless in most cases, but it adds unnecessary computation and can make interpretation slightly less direct.

### 6.7 scikit-learn pattern

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train[["Age", "Fare"]])
X_test_scaled = scaler.transform(test[["Age", "Fare"]])
```

The fitted scaler stores:

- `mean_` and `scale_` for `StandardScaler`;
- `data_min_` and `data_max_` for `MinMaxScaler`;
- `center_` and `scale_` for `RobustScaler`.

`inverse_transform` maps scaled values back to the original units.

### 6.8 Scaling does not fix skew

Scaling changes location and spread, not shape. A highly skewed `Fare` column remains skewed after standardization. If the goal is to reduce skew, use a distribution transform such as `log1p`, Box–Cox, or Yeo–Johnson (Day 31), not a scaler.

### 6.9 Common scaling mistakes

1. Fitting the scaler on the full dataset before splitting.
2. Scaling the test set with its own mean and standard deviation.
3. Using min–max scaling on a column with extreme outliers.
4. Assuming scaling makes a distribution normal.
5. Forgetting to scale inside cross-validation; a pipeline solves this.
6. Scaling the target when the model expects the original units.

---

## 7. Outliers

### 7.1 What is an outlier?

An outlier is an observation that is unusually far from the bulk of the data. It may be:

- a data-entry error (`Age = 999`);
- a measurement failure;
- a genuine rare event (a very expensive house);
- a member of a different population.

“Outlier” is not automatically “bad value.” The right treatment depends on why the value exists and what the model needs.

### 7.2 Why outliers matter

- The mean and standard deviation are sensitive to them.
- Distance-based models can be pulled toward them.
- Linear regression can give them high leverage.
- k-means can devote a cluster to a single extreme point.
- PCA can align a component with an outlier.
- Tree-based models are more robust because splits use order, but extreme values can still affect binning and regularization.

### 7.3 Detection methods

**Z-score.** For an approximately normal column,

$$
z = \frac{x-\mu}{\sigma}.
$$

A common rule flags $|z|>3$. This assumes normality and uses the mean and standard deviation, which are themselves affected by outliers.

**IQR rule.** Compute

$$
Q_1,\quad Q_3,\quad \operatorname{IQR}=Q_3-Q_1.
$$

Flag values below

$$
Q_1 - 1.5\,\operatorname{IQR}
$$

or above

$$
Q_3 + 1.5\,\operatorname{IQR}.
$$

The multiplier 1.5 is a convention, not a law. A multiplier of 3 gives a more conservative rule.

**Percentiles.** Flag values below the 1st percentile or above the 99th percentile. This always removes a fixed fraction, whether or not the values are genuinely unusual.

**Modified z-score.** Use the median and median absolute deviation:

$$
M_i = \frac{0.6745(x_i-\tilde{x})}{\operatorname{MAD}},
$$

where $\tilde{x}$ is the median and $\operatorname{MAD}=\operatorname{median}(|x_i-\tilde{x}|)$. This is more robust than the ordinary z-score.

**Multivariate methods.** Mahalanobis distance, isolation forests, local outlier factor, and one-class SVM can detect outliers in more than one dimension.

### 7.4 Treatment options

1. **Remove** the row if it is an error or irrelevant to the task.
2. **Cap/winsorize** values at a chosen percentile or threshold.
3. **Transform** the column (log, square root, Box–Cox) to reduce the influence of the tail.
4. **Use a robust scaler** so the model is less sensitive.
5. **Keep the outlier** if it is a genuine and important case.
6. **Model it separately** if it belongs to a different population.

Removing outliers changes the population you are modeling. If the business cares about extreme cases, removing them can be a serious mistake.

### 7.5 Leakage warning

The mean, standard deviation, quartiles, and thresholds used to detect outliers must be learned from the training data only. If you compute them on the full dataset, the test set influences which training rows are removed or capped.

### 7.6 Common outlier mistakes

1. Removing outliers without checking whether they are errors.
2. Using test-set statistics to set thresholds.
3. Applying a z-score rule to a heavily skewed column.
4. Treating a high-leverage point as automatically invalid.
5. Forgetting that outlier removal changes the target distribution.
6. Capping so aggressively that real signal is lost.

---

## 8. Binning and discretization

### 8.1 What is binning?

Binning converts a numeric column into a small number of ordered intervals. For example, age might become:

| Age range | Bin |
|---|---|
| 0–12 | child |
| 13–19 | teenager |
| 20–39 | young adult |
| 40–59 | middle-aged |
| 60+ | senior |

The new column is categorical, often ordinal. Binning is also called **discretization**.

### 8.2 Why bin?

- To capture nonlinear relationships with a linear model.
- To make a feature more interpretable.
- To reduce the influence of small measurement noise.
- To match domain-specific cutoffs (tax brackets, age groups).
- To handle a skewed distribution with a few broad bands.

Binning trades resolution for robustness. Once a value is placed in a bin, the model no longer sees its exact value.

### 8.3 Methods

**Equal-width binning.** Divide the range into $k$ intervals of equal width:

$$
w = \frac{x_{\max}-x_{\min}}{k}.
$$

The edges are

$$
x_{\min}, x_{\min}+w, x_{\min}+2w, \ldots, x_{\max}.
$$

Equal-width bins can be empty or badly unbalanced when the data are skewed.

**Equal-frequency binning.** Choose edges so that each bin contains roughly the same number of observations. This uses quantiles. It handles skew better but can put very different values in the same bin.

**Custom binning.** Use domain knowledge to choose edges. This is often the best choice when meaningful thresholds exist.

### 8.4 A worked example

Take ages

$$
[18,\ 22,\ 25,\ 30,\ 35,\ 40,\ 50,\ 60,\ 70].
$$

Equal-width binning into three bins:

$$
x_{\min}=18,\quad x_{\max}=70,\quad w=\frac{70-18}{3}=17.33.
$$

The edges are approximately $18, 35.33, 52.67, 70$. The bins are:

- $[18,35.33)$: 18, 22, 25, 30, 35
- $[35.33,52.67)$: 40, 50
- $[52.67,70]$: 60, 70

Equal-frequency binning into three bins would put roughly three values in each bin.

### 8.5 scikit-learn pattern

```python
from sklearn.preprocessing import KBinsDiscretizer

binner = KBinsDiscretizer(
    n_bins=5,
    encode="ordinal",
    strategy="quantile",
)

X_train_binned = binner.fit_transform(train[["Age"]])
X_test_binned = binner.transform(test[["Age"]])
```

- `strategy="uniform"` uses equal-width bins.
- `strategy="quantile"` uses equal-frequency bins.
- `strategy="kmeans"` uses one-dimensional k-means to choose edges.
- `encode="ordinal"` returns integer bin labels.
- `encode="onehot"` returns one-hot columns.
- `bin_edges_` stores the learned edges.

### 8.6 Common binning mistakes

1. Using too many bins, which recreates the original noise.
2. Using equal-width bins on a skewed column.
3. Choosing edges on the full dataset instead of the training set.
4. Forgetting that binning loses information.
5. Treating ordinal bins as nominal, or nominal bins as ordinal.
6. Using target-based edges without cross-fitting.

---

## 9. Feature construction

### 9.1 What is feature construction?

Feature construction creates a new column from one or more existing columns. It is where domain knowledge enters the model.

Examples:

- `FamilySize = SibSp + Parch + 1`.
- `IsAlone = 1` if `FamilySize == 1`, else `0`.
- `FarePerPerson = Fare / FamilySize`.
- `Title` extracted from `Name`.
- `Deck` extracted from `Cabin`.
- `HasCabin = 0` if `Cabin` is missing, else `1`.
- `AgeGroup` from `Age`.
- `IsWeekend` from a date.
- `PricePerSquareFoot = Price / Area`.
- `Interaction = FeatureA × FeatureB`.

### 9.2 Why construction works

A model can only use the columns it receives. If the useful signal is a ratio, a difference, a count, or an interaction, construction makes that signal explicit.

The XOR example from Section 2 is the cleanest mathematical demonstration: a linear model cannot represent XOR in the original two features, but it can represent it after adding the product feature.

### 9.3 Interaction terms

An interaction means that the effect of one feature depends on another. For two features $x_1$ and $x_2$, a model with an interaction includes

$$
\beta_1 x_1 + \beta_2 x_2 + \beta_{12}x_1x_2.
$$

The coefficient $\beta_{12}$ captures the dependence. Without the product term, the model assumes the effect of $x_1$ is the same at every value of $x_2$.

Polynomial features automate this. For $p$ input features and degree $d$, the number of monomials of degree at most $d$ is

$$
\binom{p+d}{d}.
$$

If we exclude the bias term, the count is

$$
\binom{p+d}{d}-1.
$$

For $p=3$ and $d=2$:

$$
\binom{5}{2}-1 = 10-1 = 9.
$$

The nine features are $x_1,x_2,x_3,x_1^2,x_2^2,x_3^2,x_1x_2,x_1x_3,x_2x_3$.

### 9.4 Date and time features

A timestamp is not directly useful to most models. Extract:

- year, month, day, hour, minute;
- day of week, is_weekend;
- day of year, week of year;
- time since a reference event;
- cyclical encodings for periodic features.

For a periodic feature such as hour of day, a raw integer treats 23 and 0 as far apart even though they are adjacent on a clock. Cyclical encoding uses

$$
x_{\sin} = \sin\left(\frac{2\pi t}{T}\right),\qquad
x_{\cos} = \cos\left(\frac{2\pi t}{T}\right),
$$

where $T$ is the period (24 for hours, 12 for months, 7 for weekdays).

### 9.5 Text features

A text column can produce:

- length in characters or words;
- counts of digits, punctuation, or uppercase letters;
- presence of specific keywords;
- TF–IDF or embedding vectors (later study).

Text feature engineering is a large topic of its own; the Day-23 lesson is simply that raw text is not a numeric feature.

### 9.6 Aggregation features

For grouped data, you can compute:

- count per group;
- mean, median, min, max per group;
- difference between a row's value and its group mean;
- time since the previous event.

Aggregation features can be powerful, but they are a major leakage risk. If the group statistic includes the current row's target or future rows, the feature contains information that would not be available at prediction time.

### 9.7 scikit-learn pattern

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(train[["Age", "Fare"]])
X_test_poly = poly.transform(test[["Age", "Fare"]])
```

For two input columns, `PolynomialFeatures(degree=2, include_bias=False)` produces

$$
[\text{Age},\ \text{Fare},\ \text{Age}^2,\ \text{Age}\cdot\text{Fare},\ \text{Fare}^2].
$$

For custom construction, a `FunctionTransformer` can apply a plain function:

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_transform = FunctionTransformer(np.log1p)
X_train_log = log_transform.fit_transform(train[["Fare"]])
X_test_log = log_transform.transform(test[["Fare"]])
```

`FunctionTransformer` receives the function object, not the result of calling it. The function is applied later, when `transform` runs.

### 9.8 Common construction mistakes

1. Using the target to build a feature.
2. Computing group aggregates on the full dataset.
3. Creating too many interactions and overfitting.
4. Dividing by zero or by a missing value.
5. Creating features that are unavailable at prediction time.
6. Forgetting to apply the same construction to test data.

---

## 10. Feature selection

### 10.1 Why select?

More features are not always better. Irrelevant features can:

- add noise and variance;
- increase training and inference cost;
- make the model harder to interpret;
- create collinearity;
- worsen the curse of dimensionality.

Feature selection keeps a subset of the original features. It does not create new columns.

### 10.2 Filter methods

Filter methods score each feature independently of the model.

**Variance threshold.** Remove features whose variance is below a threshold. A constant feature has zero variance and carries no information.

**Correlation.** Remove one of two highly correlated features. Correlation measures linear association:

$$
r_{xy} =
\frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}.
$$

Correlation near $+1$ or $-1$ indicates a strong linear relationship. It does not capture nonlinear dependence.

**Chi-square.** For non-negative features and a categorical target, the chi-square statistic measures dependence between a feature and the target. It is often used with one-hot encoded counts.

**ANOVA F-test.** Compares the mean of a numeric feature across target classes. It assumes approximately normal distributions and equal variances.

**Mutual information.** Measures how much knowing a feature reduces uncertainty about the target. It can capture nonlinear relationships, but it is harder to interpret and can be sensitive to estimation choices.

### 10.3 Wrapper methods

Wrapper methods train a model repeatedly and evaluate subsets.

- **Forward selection:** start with no features, add the best one at each step.
- **Backward elimination:** start with all features, remove the least useful one at each step.
- **Recursive Feature Elimination (RFE):** fit a model, rank features by importance, remove the weakest, repeat.

Wrapper methods can find interactions and model-specific usefulness, but they are computationally expensive and prone to overfitting if the selection is not cross-validated.

### 10.4 Embedded methods

Embedded methods perform selection during model training.

- **Lasso (L1 regularization):** adds a penalty proportional to the absolute value of coefficients:
  $$
  \text{Loss} + \lambda\sum_j |\beta_j|.
  $$
  The L1 penalty can push some coefficients exactly to zero, producing a sparse model.

- **Tree importance:** decision trees and forests rank features by how much they reduce impurity or error.

- **ElasticNet:** combines L1 and L2 penalties.

### 10.5 Selection versus extraction

- **Selection:** keep a subset of the original columns. The selected columns remain interpretable.
- **Extraction:** create new columns that are combinations of the originals. The new columns are usually less interpretable.

### 10.6 scikit-learn pattern

```python
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_train_filtered = selector.fit_transform(X_train)
X_test_filtered = selector.transform(X_test)
```

```python
selector = SelectKBest(score_func=chi2, k=10)
X_train_selected = selector.fit_transform(X_train_nonnegative, y_train)
X_test_selected = selector.transform(X_test_nonnegative)
```

`SelectKBest` stores `scores_` and `pvalues_`. The test set must be transformed with the selector fitted on training data.

### 10.7 Leakage warning

If feature selection uses the target, it must be inside the cross-validation loop. Selecting the top features on the full training set and then cross-validating can still overfit because the selection saw all training labels. In scikit-learn, put the selector inside a `Pipeline`.

### 10.8 Common selection mistakes

1. Selecting features on the test set.
2. Selecting features on the full dataset before splitting.
3. Using correlation and concluding there is no relationship; correlation only measures linear association.
4. Dropping a feature because it is correlated with another without checking whether both carry different information.
5. Using target-based selection outside cross-validation.
6. Assuming feature importance is causal.

---

## 11. Feature extraction

### 11.1 What is extraction?

Feature extraction creates a new set of features by combining or projecting the original columns. The new features are not a subset of the old ones.

The most common example is **Principal Component Analysis (PCA)**. PCA finds directions in which the data vary the most and projects the data onto those directions.

### 11.2 PCA intuition

Imagine a two-dimensional cloud of points stretched along a diagonal. Most of the variation is along the diagonal; the perpendicular direction contains mostly noise. PCA finds the diagonal as the first principal component, then the perpendicular direction as the second. If the second component is mostly noise, we can keep only the first and reduce the data from two dimensions to one.

PCA is unsupervised: it does not use the target. It preserves variance, not class separation.

### 11.3 PCA mathematics

Assume the data matrix $\mathbf{X}$ has been centered (each column has mean zero). The sample covariance matrix is

$$
\mathbf{C} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}.
$$

PCA solves for unit vectors $\mathbf{v}_1,\mathbf{v}_2,\ldots$ that maximize the projected variance:

$$
\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1}\mathbf{v}^\top\mathbf{C}\mathbf{v}.
$$

The solution is the eigenvector of $\mathbf{C}$ with the largest eigenvalue. The next component is the eigenvector with the next largest eigenvalue, subject to being orthogonal to the previous components.

The fraction of variance explained by component $j$ is

$$
\text{EV}_j = \frac{\lambda_j}{\sum_k \lambda_k}.
$$

If the first two components explain 95% of the variance, projecting onto them retains most of the variability while reducing the dimension.

### 11.4 scikit-learn pattern

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pca_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
])

X_train_pca = pca_pipeline.fit_transform(X_train)
X_test_pca = pca_pipeline.transform(X_test)
```

`PCA(n_components=0.95)` keeps enough components to explain 95% of the variance. `pca.explained_variance_ratio_` shows the contribution of each component. `pca.components_` shows the loadings.

### 11.5 When to use extraction

- High-dimensional data with correlated features.
- Visualization in two or three dimensions.
- Denoising by discarding low-variance components.
- Speeding up a downstream model.
- Reducing multicollinearity.

### 11.6 When not to use extraction

- When interpretability of the original features is essential.
- When the components are hard to explain to stakeholders.
- When the data are not standardized and the scale would dominate PCA.
- When the signal is in low-variance directions; PCA could discard it.
- When a nonlinear manifold is involved; kernel PCA or other methods may be better.

### 11.7 Common extraction mistakes

1. Fitting PCA on the full dataset before splitting.
2. Running PCA on unscaled features.
3. Choosing the number of components using the test set.
4. Assuming PCA components are interpretable.
5. Using PCA when the target depends on a low-variance direction.
6. Forgetting that PCA is unsupervised and may not separate classes.

---

## 12. Putting it together: a leakage-safe scikit-learn workflow

The following example is an illustrative pattern for the Titanic data. It is not an official Day-23 notebook result. It shows how the ideas combine.

### 12.1 Split first

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

`stratify=y` keeps the survival proportion similar in both splits.

### 12.2 Define the preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Sex", "Embarked"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```

What each object does:

- `SimpleImputer(strategy="median")` learns a median for each numeric column.
- `StandardScaler()` learns a mean and standard deviation for each numeric column.
- `SimpleImputer(strategy="most_frequent")` learns the most frequent category.
- `OneHotEncoder(handle_unknown="ignore")` learns the training categories and safely handles unseen test categories.
- `ColumnTransformer` sends the numeric columns to one pipeline and the categorical columns to another, then concatenates the results.
- `Pipeline` chains steps so that `fit` and `transform` are applied in the correct order.

### 12.3 Add a model

```python
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The pipeline's `fit` does the following:

1. Fits the numeric imputer on the training numeric columns.
2. Fits the scaler on the imputed training numeric columns.
3. Fits the categorical imputer on the training categorical columns.
4. Fits the one-hot encoder on the imputed training categories.
5. Transforms the training data.
6. Fits the logistic regression on the transformed training data.

The pipeline's `predict` applies the already-fitted transformations to the test data and then predicts.

### 12.4 Cross-validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

Cross-validation refits the entire pipeline on each training fold. This is important: the imputer, scaler, encoder, and model are all learned inside each fold, so validation data does not leak into preprocessing.

### 12.5 Data flow

```text
Raw training data
   |
   +--> numeric columns --> imputer --> scaler --+
   |                                             |
   +--> categorical columns --> imputer --> one-hot --+
                                                     |
                                                     v
                                             ColumnTransformer
                                                     |
                                                     v
                                             LogisticRegression
                                                     |
                                                     v
                                               Predictions
```

For test data, the same fitted imputers, scaler, and encoder are applied. No test statistic is learned.

### 12.6 `fit`, `transform`, and `fit_transform`

- `fit(X, y)` learns parameters from data and returns the fitted object.
- `transform(X)` applies the learned parameters and returns transformed data.
- `fit_transform(X, y)` calls `fit` and then `transform` on the same data.

Use `fit_transform` on the training set and `transform` on validation/test data. In a pipeline, you usually call `fit` or `fit_transform` on the whole pipeline, not on individual steps.

### 12.7 Where construction and selection fit

Construction can be a custom transformer or a `FunctionTransformer` before scaling. Selection can be a step after preprocessing:

```python
from sklearn.feature_selection import SelectKBest, f_classif

model = Pipeline([
    ("preprocess", preprocess),
    ("select", SelectKBest(score_func=f_classif, k=10)),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

`f_classif` works with the standardized numeric output above. `chi2` is another common choice, but it requires non-negative features. Because `StandardScaler` produces negative values, using `chi2` after that scaler would be a modelling error; you would need a non-negative representation or a different score function. This is a practical example of why the order and the mathematical assumptions matter.

---

## 13. Evaluating feature engineering

Feature engineering is an empirical activity. You need a baseline and a fair comparison.

### 13.1 Baseline first

Before adding complexity, measure a simple baseline:

- majority-class prediction for classification;
- mean prediction for regression;
- a model with minimal preprocessing.

If a new feature does not beat the baseline under the same validation protocol, it is not obviously useful.

### 13.2 Compare with the same folds

Use the same cross-validation splits for every experiment. If you change the split, you cannot tell whether the difference comes from the feature or from the split.

### 13.3 Ablation

Remove one feature group at a time and measure the change. This tells you whether the group contributes. Ablation is often more trustworthy than a single feature-importance ranking, especially when features are correlated.

### 13.4 Feature importance

- **Tree impurity importance** measures how much a feature reduces impurity across splits. It can favor high-cardinality features.
- **Permutation importance** shuffles a feature and measures the drop in performance. It is model-agnostic but can be misleading when features are correlated.
- **Coefficient size** in a linear model depends on feature scale; standardize before comparing.
- **SHAP** values provide local explanations but are not a causal proof.

Feature importance answers “what the model uses,” not “what causes the outcome.”

### 13.5 A comparison table

Use a table like this for your own experiments:

| Experiment | Features | CV mean | CV std | Test score | Notes |
|---|---|---:|---:|---:|---|
| Baseline | raw numeric only | fill from your run | fill | fill | same folds |
| + imputation | add median imputation | fill | fill | fill | no leakage |
| + encoding | add one-hot | fill | fill | fill | handle unknown |
| + scaling | add standardization | fill | fill | fill | pipeline |
| + construction | add FamilySize | fill | fill | fill | domain feature |
| + selection | add SelectKBest | fill | fill | fill | inside CV |

Do not invent numbers. Fill the table from an actual run and record the random seed, data version, and preprocessing.

### 13.6 What not to conclude

- A higher test score does not prove the feature is generally useful; the test set is finite.
- A lower training score after regularization is not necessarily bad.
- A feature with high importance is not necessarily causal.
- A feature that helps one model may hurt another.
- A feature that helps on one dataset may fail under distribution shift.

---

## 14. Common mistakes and debugging

| Mistake | Symptom | Fix |
|---|---|---|
| Fit preprocessing on full data | Validation score looks too good; production performance drops | Split first; use `Pipeline` |
| `fit_transform` on test data | Test distribution changes; leakage | Use `transform` on test |
| One-hot encoding high-cardinality IDs | Thousands of columns; overfitting | Group rare categories, frequency/target encoding, or drop IDs |
| Ordinal encoding nominal data | Model learns false order | Use one-hot or a model-appropriate encoding |
| Target encoding without cross-fitting | Extremely high training score; leakage | Out-of-fold encoding with smoothing |
| Scaling before imputation | Errors or distorted values | Impute first, then scale |
| Removing outliers using full-data thresholds | Leakage; unstable evaluation | Learn thresholds on train |
| Dropping all rows with missing values | Small, biased dataset | Analyze missingness; impute or use indicators |
| Selecting features on test data | Optimistic test score | Select inside CV/pipeline |
| PCA before splitting | Leakage; optimistic score | Fit PCA on train only |
| Forgetting `handle_unknown` | Crash on unseen categories | `handle_unknown="ignore"` |
| Mismatched train/test columns | Shape errors at prediction | Use `ColumnTransformer`/`Pipeline` |
| NaN after transformation | Model error | Inspect `isna().sum()` after each step |
| Sparse/dense mismatch | TypeError or memory issues | Check encoder output and model support |
| Assuming correlation means causation | Wrong feature decisions | Use experiments and domain knowledge |

### 14.1 A debugging checklist

1. Print `X_train.shape` and `X_test.shape`.
2. Print `X_train.dtypes`.
3. Print `X_train.isna().sum()` before and after preprocessing.
4. Inspect `preprocess.transformers_` and `get_feature_names_out()`.
5. Verify that the test set has the same columns as the training set.
6. Check for unseen categories in test data.
7. Compare training and validation scores; a large gap suggests overfitting.
8. Run a sanity-check model with a tiny subset.
9. Verify that no target-derived column is in `X`.
10. Confirm that the random seed is fixed for reproducibility.

---

## 15. A practical feature-engineering checklist

Use this checklist for a new tabular dataset.

### 15.1 Understand the problem

- What is the target?
- Is it classification or regression?
- What is the prediction time?
- Which columns would actually be available at prediction time?
- What is the cost of a false positive versus a false negative?

### 15.2 Inspect the data

- Shape, dtypes, missingness.
- Unique values per categorical column.
- Distributions and outliers.
- Relationships with the target.
- Potential leakage columns (IDs, timestamps, post-outcome fields).

### 15.3 Split

- Hold out a test set before learning any preprocessing parameters.
- Use stratification for classification when appropriate.
- Keep a validation strategy that matches the deployment setting (random, time-based, group-based).

### 15.4 Handle missing values

- Measure the missing rate per column.
- Ask why the value is missing.
- Choose mean/median/mode, indicator, KNN, or iterative imputation.
- Fit on training data only.

### 15.5 Handle types

- Numeric columns: keep numeric.
- Ordinal columns: ordinal encoding.
- Nominal columns: one-hot or another appropriate encoding.
- Dates: extract useful parts and cyclical features.
- Text: length/counts/TF-IDF/embeddings.
- IDs: usually drop or handle separately; do not one-hot high-cardinality IDs.

### 15.6 Handle outliers

- Detect with z-score, IQR, percentiles, or multivariate methods.
- Decide whether each outlier is an error, a rare event, or a different population.
- Remove, cap, transform, or keep.
- Learn thresholds on training data.

### 15.7 Scale and transform

- Scale distance/gradient/regularized models.
- Use log/power transforms for skew when appropriate.
- Do not scale tree models out of habit.
- Fit scalers on training data only.

### 15.8 Construct features

- Use domain knowledge.
- Add ratios, differences, counts, interactions, date parts, and aggregates.
- Check that each feature is available at prediction time.
- Avoid target leakage.

### 15.9 Select or extract

- Remove constant and duplicate columns.
- Use filter, wrapper, or embedded selection.
- Use PCA/LDA when a lower-dimensional representation is useful.
- Perform target-based selection inside cross-validation.

### 15.10 Validate

- Compare against a baseline.
- Use the same folds.
- Check train/validation/test performance.
- Inspect errors and feature importance.
- Test on a genuinely held-out set.
- Document the feature definitions and the preprocessing pipeline.

---

## 16. Interview essentials

### Level 1 — Definitions

**Q1. What is feature engineering?**

Feature engineering is the process of transforming raw data into features that a model can use effectively. It includes handling missing values, encoding categorical variables, scaling, outlier treatment, binning, constructing new features, selecting useful features, and extracting lower-dimensional representations. The goal is not to add complexity; it is to make the relevant signal easier for the model to learn.

**Q2. Feature selection versus feature extraction?**

Feature selection keeps a subset of the original columns. The selected columns remain interpretable. Feature extraction creates new columns that combine the originals, such as principal components. Extraction can reduce dimension more aggressively, but the new features are usually harder to interpret.

**Q3. Why do we need to encode categorical variables?**

Most numerical models perform arithmetic on features. A string cannot be multiplied by a coefficient or used in a distance. Encoding converts categories into numbers. One-hot encoding avoids implying an order for nominal categories; ordinal encoding preserves a real order.

### Level 2 — Mechanics

**Q4. Explain `fit`, `transform`, and `fit_transform`.**

`fit` learns parameters from data. For an imputer, it learns the fill value; for a scaler, the mean and standard deviation; for an encoder, the categories. `transform` applies those learned parameters. `fit_transform` does both on the same data. Use `fit_transform` on training data and `transform` on validation/test data to prevent leakage.

**Q5. Why is one-hot encoding not always the best choice?**

It creates one column per category. For high-cardinality columns, this produces a very large, sparse matrix, increases overfitting risk, and makes some models slow. Alternatives include grouping rare categories, frequency encoding, target encoding with cross-fitting, hashing, and embeddings.

**Q6. Why does scaling matter for KNN but not for a decision tree?**

KNN uses distances, so a feature with a large numeric range can dominate the distance. A decision tree splits on thresholds and only uses the order of values, so monotonic scaling does not change the splits. Standardization is therefore important for KNN, SVM, k-means, PCA, and regularized linear models, but usually unnecessary for trees.

### Level 3 — Mathematics

**Q7. What happens to variance under mean imputation?**

If $n_o$ values are observed with variance $s_o^2$ and $n-n_o$ values are replaced by the observed mean, the completed sample variance is

$$
s_{\text{imputed}}^2 = \frac{(n_o-1)s_o^2}{n-1}.
$$

The variance shrinks because every imputed value has zero deviation from the mean. This can distort distances and model coefficients.

**Q8. Write the standardization and min–max formulas.**

Standardization:

$$
z = \frac{x-\mu}{\sigma}.
$$

Min–max:

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}.
$$

Both must use statistics learned from the training set.

**Q9. How does PCA choose components?**

PCA centers the data, computes the covariance matrix, and finds its eigenvectors. The eigenvector with the largest eigenvalue is the first principal component, the direction of maximum projected variance. The next component is orthogonal and captures the next largest variance. The explained variance ratio is the eigenvalue divided by the sum of eigenvalues.

### Level 4 — Diagnosis

**Q10. Your model has 99% training accuracy and 60% validation accuracy. What could be wrong?**

This is a large generalization gap. Possible causes: overfitting, too many features, high-cardinality one-hot columns, leakage in the training pipeline that is not present in validation, or a validation split that is not representative. Check the learning curve, simplify the model, inspect feature importance, and verify that preprocessing is fitted inside cross-validation.

**Q11. Your test set contains a category not seen in training and the encoder crashes. How do you fix it?**

Use `OneHotEncoder(handle_unknown="ignore")`, which maps unseen categories to an all-zero encoding. Also inspect whether the unseen category is a data-quality issue or a sign that the training sample is not representative. For high-cardinality features, consider grouping rare categories or using a more robust encoding.

**Q12. Your imputation improved validation but the model fails in production. What should you check?**

Check for leakage: was the imputer fitted on the full dataset? Was the fill value based on the target? Was the production missingness mechanism different from training? Also check schema changes, unseen categories, and distribution shift. Compare the production missingness pattern with the training pattern.

### Level 5 — Design

**Q13. How would you build a preprocessing pipeline for Titanic?**

Split the data first. For numeric columns such as `Age`, `Fare`, `SibSp`, and `Parch`, impute the median and standardize. For categorical columns such as `Sex` and `Embarked`, impute the most frequent category and one-hot encode with `handle_unknown="ignore"`. Use a `ColumnTransformer` to route columns, then a `Pipeline` with the classifier. Fit the whole pipeline on the training set and evaluate with cross-validation.

**Q14. How would you handle a categorical column with 5,000 unique values?**

Start by asking whether the column is an identifier. If it is, drop it or use it only for grouping. If it carries signal, try grouping rare categories, frequency encoding, target encoding with out-of-fold fitting and smoothing, hashing, or embeddings. Evaluate each option with the same validation protocol.

**Q15. When would you not remove outliers?**

When the outliers are genuine rare events that matter to the business, when the model is tree-based and robust to them, when removing them would change the target population, or when the sample is already small. In those cases, use robust scaling, transformation, or a model that handles extreme values.

### Level 6 — Advanced

**Q16. Why is target encoding dangerous, and how do you make it safer?**

Target encoding replaces a category with a statistic of the target. If computed on the same rows used for training, the feature contains the answer and the model overfits. Use out-of-fold encoding: compute the encoding for each fold from the other folds only. Add smoothing toward the global mean, especially for rare categories. Never use the test target.

**Q17. Why can feature selection leak information even if it does not use the test set?**

If selection uses the training labels and is performed before cross-validation, the selection has seen all training labels. The cross-validation folds are no longer independent of the selection step, so the validation score can be optimistic. Put the selector inside the pipeline so it is refitted on each fold.

**Q18. What are the assumptions and limitations of PCA?**

PCA assumes linear structure, is sensitive to feature scale, and preserves variance rather than class separation. It can discard low-variance directions that are still predictive. The components are linear combinations and may be hard to interpret. It is unsupervised, so it does not use the target. Standardize first and fit only on training data.

### Level 7 — Synthesis

**Q19. A model performs well in cross-validation but poorly on a new month of data. How do you debug the feature pipeline?**

Check for temporal leakage: features that use future information, group aggregates that include future rows, or target encoding computed across time. Check for distribution shift in feature distributions and missingness. Compare the schema and category sets. Evaluate with a time-based split that mirrors deployment. Revisit whether the features are available at prediction time.

**Q20. How would you explain feature engineering to a non-technical stakeholder?**

Feature engineering is the step where we turn raw records into the specific pieces of information a model can use. For example, a date becomes “weekday” and “month,” a family column becomes “family size,” and a missing cabin becomes “cabin recorded or not.” Better information helps the model make better predictions, but we must be careful not to use information that would not be available when the prediction is made.

---

## 17. Exam practice

### Q1. Define feature transformation, construction, selection, and extraction with one example each.

- **Transformation:** change the representation of an existing feature. Example: standardize `Age`.
- **Construction:** create a new feature from existing ones. Example: `FamilySize = SibSp + Parch + 1`.
- **Selection:** keep a subset of existing features. Example: keep the top 10 features by mutual information.
- **Extraction:** create new features as combinations of old ones. Example: PCA components.

### Q2. A column has 10% missing values. Compare mean and median imputation.

Mean imputation preserves the observed mean but shrinks variance and is sensitive to skew and outliers. Median imputation is more robust to skew and outliers but does not preserve the mean. The choice depends on the distribution and the model. In both cases, fit the fill value on training data only, and consider a missing indicator if missingness may be informative.

### Q3. Why does one-hot encoding create collinearity, and how do you fix it?

For $k$ categories, the $k$ dummy columns sum to 1 for every row. If the model also has an intercept, the design matrix is rank-deficient: one column is a linear combination of the others. Drop one dummy category to leave $k-1$ columns, or use a model that handles the collinearity (for example, regularization).

### Q4. A feature has values 1, 2, 3, 4, 1000. Which scaler would you choose and why?

Standardization and min–max scaling are both affected by the extreme value. Min–max compresses the first four values into a tiny interval. Standardization still uses the mean and standard deviation, which the outlier influences. A robust scaler based on the median and IQR is usually a better first choice. Alternatively, investigate whether 1000 is an error and consider a log transform if the value is genuine and the distribution is skewed.

### Q5. How many features does `PolynomialFeatures(degree=2, include_bias=False)` create for 4 input features?

The number of monomials of degree at most 2 in 4 variables is

$$
\binom{4+2}{2}=15.
$$

Excluding the bias term gives $15-1=14$ features: 4 original, 4 squares, and 6 pairwise interactions.

### Q6. Explain why feature selection must be inside cross-validation when it uses the target.

If you select features using all training labels and then cross-validate, each fold's validation score is computed on features that were chosen partly because they fit the validation labels. The selection step has seen information from the validation fold. Refitting the selector inside each fold keeps the validation fold unseen during selection.

### Q7. What is the difference between a filter, wrapper, and embedded feature-selection method?

- **Filter:** score features independently of the model, using statistics such as variance, correlation, chi-square, or mutual information.
- **Wrapper:** repeatedly train a model on subsets and choose the best subset, such as RFE.
- **Embedded:** perform selection during training, such as Lasso or tree importance.

### Q8. A model uses `Age`, `Fare`, and `Pclass`. You add `FamilySize`, and validation accuracy improves. Does that prove `FamilySize` is causally important?

No. The improvement could be due to chance, a particular split, or correlation with other features. It could also be specific to this dataset. To make a stronger claim, repeat with multiple folds and seeds, compare against a baseline, and use ablation or permutation importance. Even then, predictive importance is not causal evidence.

---

## 18. One-page revision

### Definitions

- **Feature engineering:** turning raw data into a useful design matrix.
- **Transformation:** change an existing feature's representation.
- **Construction:** create a new feature from existing ones.
- **Selection:** keep a subset of original features.
- **Extraction:** create new features as combinations of old ones.

### Core formulas

$$
\text{standardization: } z=\frac{x-\mu}{\sigma}
$$

$$
\text{min--max: } x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

$$
\text{IQR: } Q_3-Q_1
$$

$$
\text{z-score: } z=\frac{x-\mu}{\sigma}
$$

$$
\text{mean-imputation variance: }
s_{\text{imputed}}^2=\frac{(n_o-1)s_o^2}{n-1}
$$

$$
\text{PCA explained variance: }
\text{EV}_j=\frac{\lambda_j}{\sum_k\lambda_k}
$$

### Workflow

```text
EDA -> split -> fit preprocessing on train -> transform train/test
    -> construct -> select/extract -> model -> cross-validate
    -> final test -> error analysis -> iterate
```

### Model sensitivity

| Model | Scale needed? | Encoding needed? |
|---|---|---|
| KNN, k-means, SVM | Yes | Yes |
| Regularized linear models | Yes | Yes |
| PCA | Yes | Yes |
| Neural networks | Yes | Yes |
| Decision trees / forests | Usually no | Yes |
| Naive Bayes | Usually no | Yes |

### Leakage rules

1. Split before learning any parameter.
2. Fit imputers, scalers, encoders, selectors, and PCA on training data only.
3. Use `transform` on validation/test data.
4. Put target-based steps inside cross-validation.
5. Never use future information or post-outcome columns.

### Final checklist

- [ ] Missing values measured and handled.
- [ ] Categorical types identified and encoded appropriately.
- [ ] Outliers investigated, not blindly removed.
- [ ] Scaling applied only where it matters.
- [ ] Constructed features are available at prediction time.
- [ ] Selection/extraction is inside the validation protocol.
- [ ] Pipeline prevents leakage.
- [ ] Baseline and final model compared on the same folds.
- [ ] Feature definitions documented.
- [ ] Test set used once, at the end.

---

# Part B — Advanced Reading and Reference

## B1. Formal setup and notation

Let the raw data be a set of $n$ records

$$
\mathcal{D} = \{(\mathbf{r}_i, y_i)\}_{i=1}^{n},
$$

where $\mathbf{r}_i$ is the raw record and $y_i$ is the target. A feature map

$$
\phi:\mathcal{R}\to\mathbb{R}^{p}
$$

produces the design matrix

$$
\mathbf{X} =
\begin{bmatrix}
\phi(\mathbf{r}_1)^\top\\
\vdots\\
\phi(\mathbf{r}_n)^\top
\end{bmatrix}
\in\mathbb{R}^{n\times p}.
$$

A supervised model learns a function $f_\theta:\mathbb{R}^p\to\mathcal{Y}$ by minimizing an empirical loss

$$
\hat{\theta} =
\arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\ell\left(f_\theta(\phi(\mathbf{r}_i)), y_i\right)
+
\lambda \Omega(\theta).
$$

The feature map is part of the model design. Changing $\phi$ changes the hypothesis space, the optimization problem, and the generalization behavior.

## B2. Missingness theory and imputation bias

### B2.1 Missingness mechanisms

Using the indicator $M_j$ for feature $j$:

- MCAR: $P(M_j=1\mid X)=P(M_j=1)$.
- MAR: $P(M_j=1\mid X)=P(M_j=1\mid X_{\text{obs}})$.
- MNAR: the probability depends on $X_{\text{miss},j}$ itself.

A complete-case analysis is unbiased for the regression coefficients under MCAR, but it can be biased under MAR or MNAR. Mean imputation can attenuate correlations and distort variances even under MCAR. Multiple imputation is often preferable when the missingness model is plausible.

### B2.2 Variance under mean imputation

Let $n_o$ observed values have sample variance $s_o^2$. Replace $n_m=n-n_o$ missing values by $\bar{x}_o$. The completed sum of squared deviations is the same as the observed sum, because each imputed value equals the mean. The completed sample variance uses denominator $n-1$:

$$
s_{\text{imputed}}^2
=
\frac{(n_o-1)s_o^2}{n-1}.
$$

The ratio is

$$
\frac{s_{\text{imputed}}^2}{s_o^2}
=
\frac{n_o-1}{n-1}.
$$

As $n_m$ grows, the variance shrinks. This is why mean imputation can make a feature look more informative than it is.

### B2.3 Indicator method

Adding a missing indicator and the imputed value is equivalent to allowing the imputed group to have a different intercept. If the missingness is informative, the indicator can recover some of the signal. If the missingness is MCAR, the indicator may add noise; validate it like any other feature.

### B2.4 Multiple imputation

Multiple imputation creates $m$ completed datasets, fits the model on each, and pools the estimates. The pooling accounts for both within-imputation and between-imputation variance. Rubin's rules combine the estimates:

$$
\bar{Q} = \frac{1}{m}\sum_{k=1}^{m}\hat{Q}_k,
$$

$$
T = \bar{U} + \left(1+\frac{1}{m}\right)B,
$$

where $\bar{U}$ is the average within-imputation variance and $B$ is the between-imputation variance. This is more statistically principled than a single fill, but it is also more work.

## B3. Encoding deep dive

### B3.1 Dummy variable trap

For $k$ categories, the one-hot matrix $\mathbf{D}\in\{0,1\}^{n\times k}$ satisfies

$$
\mathbf{D}\mathbf{1}_k = \mathbf{1}_n.
$$

If the design matrix includes an intercept column $\mathbf{1}_n$, then the columns of $\mathbf{D}$ are linearly dependent with the intercept. The matrix $\mathbf{X}^\top\mathbf{X}$ is singular. Dropping one dummy column gives a full-rank design for a linear model with an intercept.

Regularized models can tolerate the collinearity, but the coefficients become non-unique and harder to interpret.

### B3.2 Target encoding and smoothing

A simple target encoding for category $c$ is

$$
\text{TE}(c) = \frac{1}{n_c}\sum_{i:\,c_i=c} y_i.
$$

For rare categories, this estimate is noisy. A smoothed version shrinks toward the global mean:

$$
\text{TE}_{\text{smooth}}(c)
=
\frac{n_c\bar{y}_c + m\bar{y}}{n_c+m},
$$

where $\bar{y}_c$ is the category mean, $\bar{y}$ is the global mean, and $m$ is a smoothing strength. Larger $m$ gives more shrinkage.

Even with smoothing, computing the encoding on the same rows used for training leaks target information. Use out-of-fold encoding.

### B3.3 High-cardinality strategies

| Strategy | Advantage | Risk |
|---|---|---|
| One-hot | Simple, interpretable | Huge sparse matrix; overfitting |
| Frequency | Cheap, no target | Ignores category identity |
| Target encoding | Compact, can be strong | Leakage; unstable for rare categories |
| Hashing | Fixed dimension | Collisions; less interpretable |
| Embeddings | Learns similarity | Needs more data and a neural model |
| Rare grouping | Reduces cardinality | Loses fine-grained information |

## B4. Scaling, gradients, and conditioning

### B4.1 Gradient descent

For a linear model with mean squared error, the gradient with respect to a coefficient is proportional to the feature values. If one feature has a much larger scale, its gradient is much larger, and gradient descent can zigzag or converge slowly. Standardization makes the curvature more uniform and often speeds up convergence.

### B4.2 Regularization

For ridge regression,

$$
\hat{\beta}
=
\arg\min_\beta
\|\mathbf{y}-\mathbf{X}\beta\|_2^2
+
\lambda\|\beta\|_2^2.
$$

The penalty treats all coefficients on the same scale. If features have different units, the penalty is not comparable across them. Standardization makes the regularization fair.

### B4.3 Condition number

The condition number of $\mathbf{X}^\top\mathbf{X}$ measures how sensitive the solution is to small changes in the data. Highly correlated or badly scaled features increase the condition number, which can make coefficient estimates unstable. Scaling and removing redundant features improve conditioning.

## B5. Outlier mathematics

### B5.1 Ordinary z-score

$$
z_i = \frac{x_i-\bar{x}}{s}.
$$

The rule $|z_i|>3$ is based on an approximate normal distribution. For a normal distribution, about 99.73% of values lie within three standard deviations. The rule is not appropriate for heavily skewed data.

### B5.2 Modified z-score

$$
M_i = \frac{0.6745(x_i-\tilde{x})}{\operatorname{MAD}},
$$

where $\operatorname{MAD}=\operatorname{median}(|x_i-\tilde{x}|)$. The constant 0.6745 makes the MAD comparable to a standard deviation under normality. A common threshold is $|M_i|>3.5$.

### B5.3 IQR fences

$$
\text{lower} = Q_1 - k\cdot\operatorname{IQR},\qquad
\text{upper} = Q_3 + k\cdot\operatorname{IQR}.
$$

The usual $k=1.5$ flags “possible outliers”; $k=3$ flags “extreme outliers.” The choice is a convention, not a universal rule.

### B5.4 Mahalanobis distance

For a multivariate observation $\mathbf{x}$ with mean $\boldsymbol{\mu}$ and covariance $\mathbf{\Sigma}$,

$$
D_M(\mathbf{x})
=
\sqrt{(\mathbf{x}-\boldsymbol{\mu})^\top
\mathbf{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})}.
$$

It accounts for correlations and scale. It is sensitive to the estimated covariance and can be unreliable when the number of dimensions is large relative to the sample size.

## B6. Binning mathematics

### B6.1 Equal-width bins

For $k$ bins over $[x_{\min},x_{\max}]$:

$$
w = \frac{x_{\max}-x_{\min}}{k}.
$$

Bin $j$ is

$$
[x_{\min}+(j-1)w,\ x_{\min}+jw).
$$

The last bin includes the maximum. Equal-width bins can be empty when the data are concentrated in a small region.

### B6.2 Equal-frequency bins

Choose edges at quantiles $1/k,2/k,\ldots,(k-1)/k$. Each bin contains approximately $n/k$ observations. This is useful for skewed data, but it can group very different values together.

### B6.3 Information loss

Binning is a many-to-one map. All values in a bin become indistinguishable. If the target depends on fine-grained differences within a bin, binning destroys that signal. The trade-off is robustness and interpretability versus resolution.

### B6.4 Entropy and bin purity

For a categorical target, the entropy of a bin is

$$
H(B) = -\sum_{c} p_c\log_2 p_c,
$$

where $p_c$ is the fraction of class $c$ in the bin. A pure bin has entropy 0. Binning can be evaluated by how much it reduces entropy relative to the parent distribution, but this is a supervised criterion and must be cross-validated.

## B7. Construction and interaction mathematics

### B7.1 Polynomial feature count

The number of monomials of total degree at most $d$ in $p$ variables is

$$
\binom{p+d}{d}.
$$

With bias included, that is the total number of columns. With `include_bias=False`, subtract one.

For $p=10$ and $d=2$:

$$
\binom{12}{2}=66,
$$

so 65 non-bias features. For $p=100$ and $d=2$:

$$
\binom{102}{2}=5151,
$$

which is a large increase. High-degree polynomial expansion is expensive and prone to overfitting.

### B7.2 Interaction example

Suppose the true model is

$$
y = 2x_1 + 3x_2 + 5x_1x_2.
$$

A model with only $x_1$ and $x_2$ cannot represent the $5x_1x_2$ term. Adding the product feature lets a linear model represent the interaction exactly. This is why construction can change the achievable bias.

### B7.3 Cyclical encoding

For a periodic variable $t$ with period $T$:

$$
x_{\sin}=\sin\left(\frac{2\pi t}{T}\right),\qquad
x_{\cos}=\cos\left(\frac{2\pi t}{T}\right).
$$

The pair lies on the unit circle. The Euclidean distance between adjacent times (for example, 23:00 and 00:00) is small, unlike a raw integer encoding.

## B8. Selection theory

### B8.1 Variance threshold

For a feature with values $x_i$,

$$
\operatorname{Var}(x) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2.
$$

A threshold of 0 removes constant features. A small positive threshold removes near-constant features that are unlikely to be useful.

### B8.2 Chi-square statistic

For a contingency table with observed counts $O_{ij}$ and expected counts $E_{ij}$ under independence,

$$
\chi^2 = \sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}.
$$

A large statistic indicates dependence between the feature and the target. The test assumes independent observations and adequate expected counts.

### B8.3 Lasso

The Lasso objective is

$$
\hat{\beta}
=
\arg\min_\beta
\frac{1}{2n}\|\mathbf{y}-\mathbf{X}\beta\|_2^2
+
\lambda\|\beta\|_1.
$$

The L1 penalty has a corner at zero, so some coefficients become exactly zero. This performs selection and regularization simultaneously. Features must be standardized for the penalty to be meaningful.

### B8.4 Selection stability

A selected feature set can change substantially across folds or random seeds. Stability selection repeats selection on many subsamples and keeps features chosen frequently. This is more reliable than a single run, especially with correlated features.

## B9. PCA derivation and extensions

### B9.1 Variance maximization

For centered data $\mathbf{X}$, the projected variance onto a unit vector $\mathbf{v}$ is

$$
\operatorname{Var}(\mathbf{X}\mathbf{v})
=
\frac{1}{n-1}\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v}
=
\mathbf{v}^\top\mathbf{C}\mathbf{v}.
$$

Maximizing this subject to $\|\mathbf{v}\|=1$ gives the eigenvector equation

$$
\mathbf{C}\mathbf{v} = \lambda\mathbf{v}.
$$

The largest eigenvalue corresponds to the first principal component.

### B9.2 Explained variance

The total variance is the trace of $\mathbf{C}$:

$$
\operatorname{tr}(\mathbf{C}) = \sum_j \lambda_j.
$$

The fraction explained by component $j$ is

$$
\frac{\lambda_j}{\sum_k\lambda_k}.
$$

### B9.3 Whitening

PCA whitening scales each component to unit variance:

$$
\mathbf{z} = \mathbf{\Lambda}^{-1/2}\mathbf{V}^\top\mathbf{x},
$$

where $\mathbf{V}$ contains the eigenvectors and $\mathbf{\Lambda}$ contains the eigenvalues. Whitening can help some models but removes the variance ordering.

### B9.4 Kernel PCA

Kernel PCA applies PCA in a feature space defined by a kernel. It can capture nonlinear structure, but it is more expensive and less interpretable than linear PCA.

### B9.5 LDA

Linear Discriminant Analysis is supervised: it finds directions that maximize between-class separation relative to within-class scatter. Unlike PCA, it uses the target. It assumes approximately Gaussian class distributions with shared covariance for the standard version.

## B10. Leakage and cross-validation

### B10.1 The leakage taxonomy

- **Train/test contamination:** fitting preprocessing on the full dataset.
- **Target leakage:** a feature contains information about the target that would not be available at prediction time.
- **Temporal leakage:** using future information to predict the past.
- **Group leakage:** the same entity appears in training and validation.
- **Selection leakage:** selecting features using the validation labels.
- **Duplicate leakage:** near-duplicate rows across splits.

### B10.2 Nested preprocessing

The correct pattern is:

```python
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

Each fold fits its own imputer, scaler, encoder, selector, and model. The validation fold is transformed with objects fitted only on the training fold.

### B10.3 Time-series splits

For time-dependent data, use `TimeSeriesSplit` or a custom forward-chaining split. Random cross-validation can leak future information into the past. Feature engineering must respect the same boundary: a rolling mean must use only past values.

### B10.4 Group splits

If multiple rows belong to the same patient, customer, or device, use `GroupKFold` so the same group does not appear in both training and validation. Otherwise the model can memorize group-specific patterns.

## B11. scikit-learn API details

### B11.1 Transformers

A transformer implements:

- `fit(X, y=None)` — learn parameters;
- `transform(X)` — apply parameters;
- `fit_transform(X, y=None)` — fit and transform.

After fitting, learned attributes end with an underscore, such as `statistics_`, `mean_`, `scale_`, `categories_`, `bin_edges_`, `components_`, and `explained_variance_ratio_`.

### B11.2 `ColumnTransformer`

```python
ColumnTransformer(
    transformers=[
        ("name", transformer, columns),
        ...
    ],
    remainder="drop",  # or "passthrough"
)
```

Each tuple is `(name, transformer, columns)`. The name is a string used in `named_transformers_` and in pipeline parameter grids. The columns can be names, indices, slices, or boolean masks.

### B11.3 `Pipeline`

```python
Pipeline([
    ("step_name", transformer_or_estimator),
    ...
])
```

All steps except the last must implement `transform`. The last step can be an estimator with `fit` and `predict`. Pipeline parameters are accessed with double underscores: if the step is named `classifier`, its regularisation strength is `classifier__C`. Nested pipelines add another prefix, such as `preprocess__num__imputer__strategy`.

### B11.4 `FunctionTransformer`

`FunctionTransformer(func)` stores the function and applies it in `transform`. Pass the function object, not the result:

```python
FunctionTransformer(np.log1p)   # correct
FunctionTransformer(np.log1p(X))  # wrong: calls the function immediately
```

### B11.5 Sparse versus dense

One-hot encoding often produces a sparse matrix. Some models accept sparse input; others require dense. `OneHotEncoder(sparse_output=False)` returns dense output. Check the model documentation and memory cost.

### B11.6 Feature names

`get_feature_names_out()` is available on many transformers and on `ColumnTransformer`. It helps verify the output schema and debug column mismatches.

## B12. Extended interview bank

### Level 1 — Definitions

1. What is a feature?
2. What is the difference between a feature and a target?
3. What is a design matrix?
4. What is imputation?
5. What is one-hot encoding?
6. What is standardization?
7. What is an outlier?
8. What is feature selection?
9. What is feature extraction?
10. What is data leakage?

### Level 2 — Mechanics

11. Explain `fit`, `transform`, and `fit_transform`.
12. Why do we split before fitting a scaler?
13. How does `SimpleImputer` store its learned values?
14. What does `handle_unknown="ignore"` do?
15. Why does `ColumnTransformer` use tuples?
16. What is the difference between `remainder="drop"` and `remainder="passthrough"`?
17. How does a `Pipeline` prevent leakage?
18. What is the dummy variable trap?
19. Why does `FunctionTransformer` receive a function object?
20. What is the difference between sparse and dense output?

### Level 3 — Mathematics

21. Derive the variance shrinkage under mean imputation.
22. Write the standardization formula and explain each term.
23. Write the min–max formula and explain its sensitivity to outliers.
24. Write the IQR outlier rule.
25. How many polynomial features does degree 2 create for 5 inputs?
26. What does a PCA eigenvalue represent?
27. What is the explained variance ratio?
28. Write the Lasso objective.
29. What is the chi-square statistic measuring?
30. What is the condition number, and why does scaling help?

### Level 4 — Diagnosis

31. Training accuracy is high but validation accuracy is low. What do you check?
32. The encoder crashes on unseen categories. What do you do?
33. The model performs well in validation but poorly in production. What could be wrong?
34. A feature has 80% missing values. How do you decide whether to keep it?
35. A column has 5,000 categories. What are your options?
36. A scaler was fitted on the full dataset. How do you detect and fix the leakage?
37. A feature importance ranking changes every run. What does that suggest?
38. A model is unstable across folds. What could cause it?
39. A test set has a different schema. How do you make the pipeline robust?
40. A transformed column contains NaN after preprocessing. How do you debug it?

### Level 5 — Design

41. Design a preprocessing pipeline for a tabular classification problem.
42. How would you handle missing numeric and categorical values?
43. How would you encode an ordinal column?
44. How would you handle a high-cardinality categorical column?
45. When would you use PCA?
46. When would you not use PCA?
47. How would you select features for a linear model?
48. How would you evaluate whether a new feature helps?
49. How would you handle time-dependent features without leakage?
50. How would you document a feature pipeline for deployment?

### Level 6 — Advanced

51. Explain the bias–variance trade-off in feature construction.
52. Why can target encoding leak even with smoothing?
53. How does cross-fitting reduce target-encoding leakage?
54. Why can feature selection be unstable with correlated features?
55. What are the assumptions of PCA?
56. What are the assumptions of LDA?
57. How does regularization interact with feature scaling?
58. What is the curse of dimensionality?
59. How does binning affect information content?
60. How would you detect multivariate outliers?

### Level 7 — Synthesis

61. A model works in cross-validation but fails on next month's data. Diagnose.
62. A stakeholder wants an interpretable model but the best model uses PCA. What do you do?
63. You have a small dataset and 10,000 features. Design a safe workflow.
64. You suspect leakage but cannot find it. How do you audit the pipeline?
65. How would you explain feature engineering trade-offs to a non-technical audience?

## B13. Extended examination questions

1. **Numerical:** A column has 800 observed values with sample variance 25 and 200 missing values filled with the mean. Compute the completed sample variance.
2. **Numerical:** A feature ranges from 10 to 90. Apply min–max scaling to 30 and 70.
3. **Numerical:** For a column with $Q_1=20$, $Q_3=50$, compute the 1.5-IQR fences.
4. **Numerical:** How many non-bias polynomial features of degree 3 exist for 4 inputs?
5. **Derivation:** Show that mean imputation preserves the sample mean.
6. **Derivation:** Show that mean imputation shrinks the sample variance.
7. **Design:** Write a leakage-safe pipeline for Titanic with numeric and categorical columns.
8. **Diagnosis:** A one-hot encoder produces 12,000 columns. Explain the risks and propose two alternatives.
9. **Comparison:** Compare filter, wrapper, and embedded selection.
10. **Synthesis:** Explain how feature engineering, model choice, and validation interact.

### Worked answers to selected questions

**1.** The completed variance is

$$
\frac{(800-1)\cdot 25}{999}
=
\frac{19975}{999}
\approx 19.99.
$$

**2.** Min–max:

$$
\frac{30-10}{90-10}=0.25,\qquad
\frac{70-10}{90-10}=0.75.
$$

**3.** IQR $=50-20=30$. Lower fence $=20-1.5(30)=-25$. Upper fence $=50+1.5(30)=95$.

**4.** $\binom{4+3}{3}-1=\binom{7}{3}-1=35-1=34$.

**5.** Let the observed values be $x_1,\ldots,x_{n_o}$ with mean $\bar{x}_o$. Replacing each missing value by $\bar{x}_o$ gives a completed mean

$$
\frac{\sum_{i=1}^{n_o}x_i+(n-n_o)\bar{x}_o}{n}
=
\frac{n_o\bar{x}_o+(n-n_o)\bar{x}_o}{n}
=
\bar{x}_o.
$$

**6.** The completed sum of squared deviations is the observed sum, because each imputed value equals the mean. Dividing by $n-1$ instead of $n_o-1$ gives the shrinkage formula.

## B14. Coverage and source fidelity

- **Official Day-23 notebook:** none. The CampusX repository has no `day23-*` folder.
- **Personal notebook:** `Day_23/my_notebook.ipynb`, 9 markdown cells, 0 code cells. It is a concept checklist.
- **Dataset:** none.
- **Implemented later:** imputation in Days 35–40, scaling in Days 24–25, encoding in Days 26–27, `ColumnTransformer` in Day 28, pipelines and selection in Day 29, power transforms in Day 31, binning in Day 32, mixed variables in Day 33, dates in Day 34, feature construction/splitting in Day 45, PCA in Day 47.
- **This chapter:** teaching expansion based on the Day-23 outline, the later course notebooks, and standard scikit-learn behavior. Code snippets are illustrative patterns, not saved Day-23 outputs.
- **No invented experimental results:** no accuracy numbers, dataset sizes, or report values are claimed for Day 23.

## B15. References

- scikit-learn documentation: `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `KBinsDiscretizer`, `PolynomialFeatures`, `FunctionTransformer`, `ColumnTransformer`, `Pipeline`, `SelectKBest`, `RFE`, `PCA`.
- CampusX 100 Days of Machine Learning, Day 23 outline and later implementation days.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning* — feature selection, shrinkage, PCA, bias–variance.
- James, Witten, Hastie, and Tibshirani, *An Introduction to Statistical Learning* — preprocessing and model assessment.
- Rubin, *Multiple Imputation for Nonresponse in Surveys* — missing-data theory.
- Little and Rubin, *Statistical Analysis with Missing Data* — MCAR, MAR, MNAR, and imputation.
- Jolliffe, *Principal Component Analysis* — PCA theory and practice.
