# Day 41 - What Are Outliers?

> **Course:** CampusX 100 Days of Machine Learning, Day 41  
> **Format:** theory class; there is no official notebook and no dataset for this day.  
> **Video:** [What are Outliers | Outliers in Machine Learning](https://youtu.be/Lln1PKgGr_M)  
> **Video length:** 17:07 (about 1,027 seconds)  
> **Transcript used:** auto-generated English transcript, 382 snippets. The transcript is noisy in places, so the examples below are reconstructed from it rather than quoted verbatim.  
> **Scope:** this chapter explains the ideas introduced in the video and adds clearly marked supplementary mathematics and practice. It does not claim that the video covered every advanced extension.

This chapter has two parts. **Part A** is the main learning chapter: it explains what outliers are, when they are dangerous, how they affect machine-learning models, how they can be treated, and how they can be detected. **Part B** is an advanced reference: it contains formal definitions, robust statistics, influence diagnostics, multivariate methods, code patterns, leakage discipline, and an extended interview and examination bank. Part A is sufficient for understanding the video. Part B is optional.

The video is an introduction, not a complete treatment. Its own roadmap says that four follow-up techniques will be studied separately:

1. Z-score treatment,
2. IQR-based filtering,
3. percentile-based detection,
4. Winsorization.

Those four techniques are previewed here and developed in Part B.

---

# Part A - Main Learning Chapter

## 1. The problem: one observation can change the conclusion

An outlier is not merely a large number. It is a value that behaves very differently from the rest of the observations in a way that matters for the question being asked. The danger is easiest to see with an average.

### 1.1 The salary example

Imagine a small class with four salaries:

| Person | Salary |
|---|---:|
| A | 20,000 |
| B | 30,000 |
| C | 40,000 |
| D | 50,000 |

The mean salary is

$$
\bar{x}=\frac{20{,}000+30{,}000+40{,}000+50{,}000}{4}=35{,}000.
$$

The median is also 35,000 because the middle two values are 30,000 and 40,000.

Now suppose a person with a salary of 1,000,000 joins the class. The new mean is

$$
\bar{x}=\frac{1{,}140{,}000}{5}=228{,}000.
$$

The median becomes 40,000. The mean has moved from 35,000 to 228,000, while four of the five people still earn between 20,000 and 50,000. The mean is no longer a useful summary of a typical salary. This is the instructor's point: one extremely different observation can pull an average away from the majority.

The video uses Bill Gates or Elon Musk as the extreme salary. The exact name does not matter; the statistical lesson does. A single observation can dominate a statistic that is not robust.

### 1.2 The study-hours example

The second example is a regression problem. Suppose students have a study time $x$ and a mark $y$. The original data follow a clear upward pattern:

| Hours studied | Marks |
|---:|---:|
| 1 | 50 |
| 2 | 55 |
| 3 | 60 |
| 4 | 65 |
| 5 | 70 |

A least-squares line through these five points is

$$
\hat{y}=45+5x.
$$

The slope is $+5$: one more hour is associated with five more marks. At 6 hours, the prediction is 75.

Now add two students who studied very little but scored very high:

| Hours studied | Marks |
|---:|---:|
| 1 | 95 |
| 2 | 90 |

With all seven points, the least-squares line becomes approximately

$$
\hat{y}=73.44-1.61x.
$$

The slope has changed from $+5$ to $-1.61$. The correlation changes from $1.00$ to about $-0.142$. The line has been pulled toward the two unusual students. At 6 hours, the new prediction is about 63.75 instead of 75. The model now gives a worse description of the ordinary students, and it will probably perform poorly on new ordinary test data.

This is the central warning of the video: outliers are dangerous not because they are large, but because many models try to accommodate them. The model spends capacity explaining a few unusual points and loses the pattern that describes the majority.

### 1.3 The hidden danger

The video stresses that outliers are often hidden. A few bad values can sit inside thousands of rows and never appear in a quick `head()`. A model can train without an error message and still be distorted. The damage appears later as poor test performance, unstable coefficients, or a decision boundary that does not make sense.

The practical lesson is not "always delete unusual values." It is:

> Find unusual observations deliberately, understand why they exist, and decide whether they should be kept, corrected, treated, or removed.

---

## 2. What an outlier is

### 2.1 Working definition

An **outlier** is an observation that is unusually far from the majority of the observations in a dataset, or that does not follow the pattern followed by the majority.

The definition has two parts:

1. **Unusualness:** the value is far away in the feature space, or in a combination of features.
2. **Context:** the value is unusual relative to the population and the problem, not relative to an arbitrary universal rule.

There is no single number that makes a value an outlier in every dataset. A salary of 1,000,000 may be an outlier in a class of students and completely normal in a list of company executives. A temperature of 40°C may be normal in one city and impossible in another. Context is part of the definition.

### 2.2 The instructor's analogy

The video uses the phrase "Sharma ji ka beta" for the student who cries after getting 99 marks while the rest of the class is struggling to pass. The student is unusual relative to the class. The analogy is memorable, but it also contains an important warning: being unusual is not the same as being wrong. The 99-mark student may be a genuine high performer, not a data-entry error.

The same is true in data. A high value can be:

- a legitimate rare event,
- a measurement error,
- a different population mixed into the data,
- a data-entry mistake,
- a sign that an important variable is missing,
- or exactly the event we are trying to detect.

The number alone does not tell us which.

### 2.3 Outlier, anomaly, and noise

These words are related but not identical.

| Term | Meaning | Typical response |
|---|---|---|
| **Outlier** | A statistical observation far from the majority | Investigate; keep, correct, cap, or remove |
| **Anomaly** | An observation that is abnormal for a process or domain | Often the target of detection, not an error |
| **Noise** | Random variation or measurement error without useful signal | Reduce or model it; do not over-interpret it |

A credit-card fraud transaction is an anomaly and an outlier. A typo of 300 for age is an outlier and an error. A genuinely exceptional athlete's performance may be an outlier but not an error. A noisy sensor reading may be neither interesting nor a meaningful outlier.

The video's credit-card example makes the distinction concrete. In fraud detection, the unusual transaction is the signal. Removing all unusual transactions would destroy the task. In a model that predicts ordinary customer spending, the same transaction might be treated as an extreme value that needs special handling.

### 2.4 Outliers can be univariate or multivariate

A **univariate outlier** is unusual in one column. For example, an age of 300 is unusual in the `Age` column.

A **multivariate outlier** is unusual in the combination of several columns even if each individual value is ordinary. For example, a person may have a normal age, a normal income, and a normal purchase amount, but the combination may be extremely unlikely. A two-dimensional scatterplot can reveal such a point even when neither column's histogram flags it.

The video focuses on one-dimensional examples, but the idea extends to many dimensions. Part B covers multivariate detection.

---

## 3. When should an outlier be removed, and when should it be kept?

The video says that deciding what to do with an outlier is often harder than detecting it. This is correct. Detection is a measurement problem; treatment is a judgment problem that requires domain knowledge.

### 3.1 Cases where removal is usually justified

An outlier is a good candidate for removal or correction when it is clearly invalid.

**1. Impossible values**

The video's example is an age of 300. A human age of 300 is impossible. The value is not a rare observation; it is a data error. We cannot infer whether the true age was 30, 3, or something else. If the correct value cannot be recovered, removing the row or marking the value as missing is reasonable.

Other examples:

- a negative height,
- a negative price for a new product,
- a human age above 120,
- a date in the future for a historical event,
- a percentage above 100 when the definition forbids it,
- a count that is negative.

**2. Measurement or recording errors**

A sensor may malfunction, a unit may be entered incorrectly, or a decimal point may be misplaced. If the error is identifiable, correct it. If it cannot be corrected, remove or treat it as missing.

**3. Wrong-population contamination**

If the dataset is supposed to contain adult customers but includes a few records from a children's program, those records may be outliers. They are valid records, but they belong to a different population. Whether to remove them depends on the target population.

**4. Data corruption**

A value such as `999999` used as a missing-value code is not a real measurement. It should be converted to missing, not treated as a genuine large value.

**5. Target leakage or impossible combinations**

An observation that could not exist at prediction time, or that contains information unavailable when the model will be used, is a data-quality problem. This is related to leakage, not just extremeness.

### 3.2 Cases where removal can be harmful

An outlier should not be removed automatically when it is a genuine part of the population.

**1. Rare but real events**

A very large purchase, a very long hospital stay, or a very high salary may be rare but real. Removing it changes the population the model is supposed to serve.

**2. Anomaly detection**

In fraud detection, intrusion detection, equipment-failure detection, and disease outbreak detection, the unusual observation is the event of interest. Removing it destroys the task.

**3. Heavy-tailed distributions**

Income, insurance claims, city populations, and sales often have heavy tails. A value can be extreme but still consistent with the distribution. A normal-distribution rule may label too many legitimate values as outliers.

**4. Small samples**

In a small dataset, one unusual value may be a large fraction of the data. Removing it can change the conclusion dramatically. The video's phrase is that the data can become "thin": too few observations remain for reliable modeling.

**5. When an omitted variable explains the value**

The video gives an important example: the two students who study little but score high may not be errors. If we add an `IQ` column, their high marks may become explainable. The apparent outlier may be a sign that the model is missing an important feature. In that case, the right response is feature construction, not row deletion.

This is a deep lesson: **an outlier can be a symptom of a missing variable or a wrong model, not a bad row.** Removing the row may hide the real problem.

**6. When the outlier is the target**

If the target itself is extreme, removing the row changes the target distribution and may bias the model. For example, in insurance claim prediction, the largest claims are often the most important. Robust loss functions or specialized models may be better than deleting the largest targets.

### 3.3 Questions to ask before treatment

For every suspected outlier, ask:

1. Is the value physically or logically possible?
2. Is it a measurement, entry, or unit error?
3. Does it belong to the population the model will serve?
4. Is it a genuine rare event that matters for the decision?
5. Is it explained by another variable that is missing from the model?
6. Is the task anomaly detection, where the outlier is the signal?
7. How much data will remain if I remove it?
8. Would removing it change the conclusion on a validation set?
9. Can I correct it instead of deleting it?
10. Can I keep it but use a method that is robust to it?

If these questions cannot be answered, the safest first step is usually to **flag** the observation, not silently delete it. Create an `is_outlier` indicator and compare models with and without the treatment.

---

## 4. Why outliers affect machine-learning models

The video divides algorithms into two broad groups:

- algorithms that learn weights and are strongly affected by outliers,
- tree-based algorithms that split the data space and are less affected.

This is a useful first approximation, not an absolute law.

### 4.1 Weight-based algorithms

The video names four important examples:

1. linear regression,
2. logistic regression,
3. AdaBoost,
4. deep learning.

The common pattern is that these algorithms adjust parameters to reduce a loss that depends on the errors of individual observations. A very large error can produce a very large gradient or a very large weight update. The model may distort its parameters to reduce that one error.

#### Linear regression

Ordinary least squares (OLS) minimizes the sum of squared errors:

$$
L(\beta)=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
=\sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}\beta_j x_{ij}\right)^2.
$$

The square makes large errors disproportionately important. A point with a large residual contributes much more to the loss than a point with a small residual. The fitted line is therefore pulled toward extreme observations.

The video's study-hours example is exactly this phenomenon. Two high-mark, low-study students can change the slope from positive to negative.

#### Logistic regression

Logistic regression minimizes log loss:

$$
L(\beta)=-\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right],
$$

where $p_i=\sigma(\beta_0+\beta^T x_i)$ and $\sigma(z)=1/(1+e^{-z})$.

An observation with an extreme feature value can have a very small or very large predicted probability, producing a large gradient. The decision boundary may move to accommodate it. Logistic regression is therefore sensitive to extreme feature values and to mislabeled points near the boundary.

#### AdaBoost

AdaBoost combines weak learners and gives more weight to observations that are repeatedly misclassified. Its exponential loss is

$$
L=\sum_{i=1}^{n}\exp\left(-y_i f(x_i)\right),
$$

where $y_i\in\{-1,+1\}$ and $f(x_i)$ is the combined score. A misclassified outlier can receive an enormous weight, so later weak learners may focus almost entirely on that one point. This is why the video says AdaBoost is significantly affected by outliers.

#### Deep learning

Neural networks are also weight-based. Large inputs or large target errors can produce large gradients, unstable updates, and overfitting to a few examples. Batch normalization, gradient clipping, robust losses, and data cleaning can reduce the problem, but they do not make a network immune.

### 4.2 Tree-based algorithms

The video names decision trees, random forests, gradient boosting, and XGBoost as less affected by outliers. The reason is that trees split the feature space using thresholds:

$$
x_j \le t.
$$

A split cares about the order of values and the purity of the resulting groups, not about the squared distance from a mean. Moving an extreme value from 100 to 1,000 may not change which side of a split it falls on. This makes trees more robust to monotone transformations and to extreme values in individual features.

But "less affected" does not mean "immune."

- An extreme value can still change the best split threshold.
- A small group of outliers can create a split that isolates them, using up model capacity.
- Tree ensembles can overfit unusual points if the trees are deep.
- Probability estimates and leaf means can be distorted by extreme targets.
- Multivariate outliers can still be isolated by trees.

The practical conclusion is: trees are often a safer default when outliers cannot be cleaned, but they are not a substitute for understanding the data.

### 4.3 Other algorithm families

The video does not discuss every family, but the same reasoning extends:

| Family | Sensitivity | Reason |
|---|---|---|
| Linear regression, logistic regression | High | Squared/log loss; weights react to large errors |
| AdaBoost | High | Exponential loss; misclassified points get huge weight |
| Neural networks | High to moderate | Large gradients; sensitive to input scale and label noise |
| Decision trees, random forests, gradient boosting, XGBoost | Lower | Axis-aligned splits; rank-based thresholds |
| KNN, K-means | High | Distance-based; extreme values dominate distances |
| SVM with RBF kernel | Moderate to high | Kernel similarity and margin can be affected |
| Naive Bayes | Model-dependent | Gaussian assumptions can be distorted by extreme values |

This table is a guide, not a guarantee. The actual effect depends on the data, the loss function, the regularization, and the hyperparameters.

### 4.4 Outliers can also distort evaluation

An outlier can change the mean and standard deviation used in metrics. For example, mean squared error is dominated by large errors. A single extreme test point can make a good model look bad, or a model that ignores the tail can look better than it is.

For this reason, evaluation should include:

- median absolute error,
- mean absolute error,
- robust metrics,
- per-group analysis,
- inspection of the largest residuals,
- comparison with and without the flagged observations.

The video's focus is on training, but evaluation is part of the same problem.

---

## 5. How to treat outliers

The video describes four treatments and says that the course will focus mainly on the first two.

### 5.1 Trimming

**Trimming** means removing the outlier rows from the dataset.

The video's example: if two students are outliers, remove both students completely.

Advantages:

- simple and fast,
- removes the extreme values entirely,
- easy to explain.

Disadvantages:

- the dataset becomes smaller ("thin"),
- genuine rare events are lost,
- removal can bias the sample if the outliers are not missing at random,
- the model no longer represents the full population,
- removing rows can change the target distribution.

Trimming is most defensible when the values are clearly invalid and cannot be corrected.

### 5.2 Capping

**Capping** means replacing values beyond a boundary with the boundary value.

The video explains that an outlier is usually on the low side or the high side, not in the middle. For salary, a person earning too much or too little may be an outlier; an average earner is not. Capping puts a limit on each side:

$$
x_i^{\text{capped}}=
\begin{cases}
L, & x_i<L,\\
x_i, & L\le x_i\le U,\\
U, & x_i>U.
\end{cases}
$$

Here $L$ is the lower cap and $U$ is the upper cap.

Example: if the lower limit is 1 and the upper limit is 90, every value below 1 becomes 1, and every value above 90 becomes 90.

Advantages:

- keeps the row,
- preserves the sample size,
- reduces the influence of extreme values,
- simple to implement.

Disadvantages:

- creates a pile-up of identical values at the boundary,
- changes the shape of the distribution,
- the choice of boundary is subjective,
- a capped value may be interpreted by a model as a real measurement,
- it can hide genuine extremes.

Capping is closely related to **Winsorization**. In winsorization, extreme values are replaced by a chosen percentile (for example, the 1st and 99th percentiles). Capping can also use a domain limit rather than a percentile. The video uses "capping" broadly; later videos study Winsorization specifically.

### 5.3 Treating outliers as missing values

Another option is to replace the outlier with `NaN` and then apply a missing-value method.

This is useful when the outlier is believed to be invalid but the rest of the row is still useful. For example, an age of 300 can be set to missing and then imputed.

Advantages:

- keeps the row,
- uses the missing-data machinery already learned,
- can preserve other columns.

Disadvantages:

- imputation introduces assumptions,
- imputed values may be wrong,
- the missingness itself may be informative,
- a missing indicator may be needed,
- imputation can shrink variance or distort relationships.

This approach should be used carefully. Replacing a real extreme value with a mean or median can erase a meaningful signal.

### 5.4 Discretization

**Discretization** converts a numerical column into bins. For example, age might be grouped as 0-10, 10-20, 20-30, and so on.

The video explains that if 100 is an outlier and the top bin is 90-100, then 100 is merged into the same bin as 90. The value is no longer extreme relative to the bin.

Advantages:

- can reduce the influence of extreme values,
- can capture non-linear relationships,
- easy to explain.

Disadvantages:

- loses information,
- bin boundaries are subjective,
- can hide real extremes,
- can create abrupt jumps at boundaries,
- may reduce predictive power.

Discretization is a feature-engineering choice, not a universal outlier cure.

### 5.5 Transformations and robust methods

The video focuses on trimming and capping, but other approaches are common:

- **Log transformation:** compresses large values; useful for right-skewed positive data.
- **Square-root transformation:** milder compression.
- **Box-Cox / Yeo-Johnson:** power transformations that reduce skew.
- **Robust scaling:** center by median and scale by IQR instead of mean and standard deviation.
- **Robust loss:** Huber loss, quantile loss, or absolute loss instead of squared error.
- **Model choice:** tree-based models or anomaly-detection methods.
- **Separate model:** model the tail separately if extreme values are important.

These are supplementary extensions. They are not substitutes for understanding why the outlier exists.

### 5.6 Comparison of treatments

| Treatment | Keeps row? | Keeps extreme value? | Main risk | Good use |
|---|---|---|---|---|
| Trim | No | No | Loses data and can bias | Invalid values that cannot be corrected |
| Cap / Winsorize | Yes | No, replaced by boundary | Distorts distribution; arbitrary boundary | Reduce influence while keeping rows |
| Treat as missing | Yes | No, becomes `NaN` | Imputation assumptions | Invalid value but useful row |
| Discretize | Yes | Merged into bin | Loses granularity | Non-linear binning is useful anyway |
| Transform | Yes | Compressed | Changes scale and interpretation | Skewed positive data |
| Robust model | Yes | Down-weighted | Model may still be affected | Tail values are genuine |
| Do nothing | Yes | Yes | Model distortion | Outlier is genuine and important |

There is no universally best row. The right choice depends on the problem, the data-generating process, and the cost of different errors.

---

## 6. How to detect outliers

The video presents three main detection methods and says they are the most important ones for the course.

### 6.1 Visual inspection first

Before applying a formula, look at the data:

- histogram,
- boxplot,
- scatterplot,
- time-series plot,
- group-wise summaries,
- sorted values,
- domain rules.

A plot often reveals whether the distribution is normal, skewed, multimodal, or contaminated. It also shows whether a value is isolated or part of a cluster. A formula cannot replace this judgment.

### 6.2 Normal distribution and the three-sigma rule

If a column is approximately normally distributed, the empirical rule says:

- about 68% of observations lie within $\mu\pm\sigma$,
- about 95% within $\mu\pm2\sigma$,
- about 99.7% within $\mu\pm3\sigma$.

The exact values are 68.27%, 95.45%, and 99.73%.

A common rule is to flag an observation as an outlier when

$$
x<\mu-3\sigma
\quad\text{or}\quad
x>\mu+3\sigma.
$$

Equivalently, the **Z-score**

$$
z_i=\frac{x_i-\mu}{\sigma}
$$

is flagged when $|z_i|>3$.

The video says this is strictly applicable when the data is normal or close to normal. That caveat is important. If the distribution is skewed, the mean and standard deviation are themselves affected by the tail, and the rule may fail.

### 6.3 IQR and the boxplot rule

If the data is skewed or not normal, the video recommends the **interquartile range (IQR)** method.

Let

- $Q_1$ be the 25th percentile,
- $Q_3$ be the 75th percentile,
- $\operatorname{IQR}=Q_3-Q_1$.

The lower and upper fences are

$$
L=Q_1-1.5\,\operatorname{IQR},
\qquad
U=Q_3+1.5\,\operatorname{IQR}.
$$

Values below $L$ or above $U$ are flagged.

The boxplot draws the box from $Q_1$ to $Q_3$, the median inside the box, and whiskers to the fences. Points beyond the whiskers are plotted individually.

The multiplier 1.5 is a convention, not a law. A multiplier of 3.0 defines "extreme" outliers in Tukey's terminology. The choice depends on how aggressive the detection should be.

### 6.4 Percentile method

The percentile method chooses lower and upper percentiles directly. For example:

- lower 1st percentile and upper 99th percentile,
- lower 2.5th and upper 97.5th,
- lower 5th and upper 95th.

Values outside the chosen range are treated as outliers.

The video mentions 97th/99th on the upper side and 2.5/5 on the lower side, and says the choice depends on the business problem. This is correct: there is no universal percentile. A stricter threshold flags fewer points; a looser threshold flags more.

### 6.5 Comparing the methods

| Method | Center/scale | Best when | Main weakness |
|---|---|---|---|
| 3-sigma / Z-score | Mean, standard deviation | Approximately normal data | Sensitive to outliers; masking |
| IQR / boxplot | Median, quartiles | Skewed data | Arbitrary 1.5 multiplier; many flags in heavy tails |
| Percentile | Chosen quantiles | Business-defined tail | Arbitrary cut-off; loses information |
| Visual/domain | Human judgment | Always | Not scalable alone |

The methods can disagree. A value may be flagged by IQR but not by a 3-sigma rule because the outlier inflates the standard deviation. This is called **masking**. A worked example appears in Section 7.4.

### 6.6 A note on "outlier" versus "rare"

A fixed percentile always flags a fixed fraction of the data, even if the data is clean. If you choose the top 1%, you will flag 1% of observations whether or not any of them are errors. This is useful for tail analysis but dangerous as an automatic cleaning rule.

The IQR rule can also flag many points in a heavily skewed distribution. A log transformation may make the distribution more symmetric and the rule more meaningful.

---

## 7. Worked numerical examples

These examples are constructed to show the arithmetic. They are not results from an official notebook.

### 7.1 Salary mean and median

Base data: 20,000, 30,000, 40,000, 50,000.

$$
\bar{x}=35{,}000,
\qquad
\operatorname{median}=35{,}000.
$$

Add 1,000,000:

$$
\bar{x}=228{,}000,
\qquad
\operatorname{median}=40{,}000.
$$

The mean moves by 193,000; the median moves by 5,000. The median is more robust to the single extreme value.

### 7.2 Regression line with and without outliers

Base points: $(1,50),(2,55),(3,60),(4,65),(5,70)$.

Without outliers:

$$
\hat{y}=45+5x,
\qquad r=1.00.
$$

With $(1,95)$ and $(2,90)$ added:

$$
\hat{y}=73.44-1.61x,
\qquad r\approx-0.142.
$$

The two points change the sign of the slope. This is a dramatic illustration of influence: the outliers are few, but they control the fitted line.

### 7.3 Three-sigma example

Suppose a column has $\mu=50$ and $\sigma=10$. The three-sigma bounds are

$$
50-3(10)=20,
\qquad
50+3(10)=80.
$$

A value of 85 is flagged; a value of 75 is not. The intervals are:

| Interval | Approximate share of a normal distribution |
|---|---:|
| $40$ to $60$ | 68.27% |
| $30$ to $70$ | 95.45% |
| $20$ to $80$ | 99.73% |

This rule assumes the normal shape. It is not a universal cleaning rule.

### 7.4 IQR example and the masking problem

Consider the values

$$
10,\ 12,\ 14,\ 15,\ 16,\ 18,\ 20,\ 22,\ 100.
$$

Using the linear percentile convention:

$$
Q_1=14,
\qquad
Q_3=20,
\qquad
\operatorname{IQR}=6.
$$

The fences are

$$
L=14-1.5(6)=5,
\qquad
U=20+1.5(6)=29.
$$

The value 100 is above 29, so IQR flags it.

Now compute the mean and standard deviation of the same nine values:

$$
\bar{x}=25.22,
\qquad
s=28.29.
$$

The Z-score of 100 is

$$
z=\frac{100-25.22}{28.29}\approx2.64.
$$

It is **not** flagged by the $|z|>3$ rule. The extreme value has inflated the mean and standard deviation enough to hide itself. This is masking. It is one reason the IQR method is preferred for skewed or contaminated data.

A modified Z-score based on the median and median absolute deviation (MAD) is more robust:

$$
M_i=\frac{0.6745(x_i-\tilde{x})}{\operatorname{MAD}},
\qquad
\operatorname{MAD}=\operatorname{median}(|x_i-\tilde{x}|).
$$

For this data, $\tilde{x}=16$ and $\operatorname{MAD}=4$. The modified Z-score of 100 is about 14.16, so it is clearly flagged. The constant 0.6745 makes the MAD comparable to the standard deviation under normality.

### 7.5 Percentile example

For the same nine values, some percentile cut-offs are:

| Rule | Lower cut-off | Upper cut-off | Is 100 flagged? |
|---|---:|---:|---|
| 1st / 99th | 10.16 | 93.76 | Yes |
| 2.5th / 97.5th | 10.40 | 84.40 | Yes |
| 5th / 95th | 10.80 | 68.80 | Yes |

The exact cut-offs depend on the percentile interpolation method. The important point is that the percentile rule always chooses a fixed tail fraction; it does not ask whether the tail is an error.

### 7.6 Trim, cap, and winsorize

For the nine values:

**Trim** the value 100. The remaining eight values have

$$
\bar{x}=15.875,
\qquad
\operatorname{median}=15.5.
$$

**Cap** at the IQR upper fence 29. The value 100 becomes 29. The capped data have

$$
\bar{x}=17.33,
\qquad
\operatorname{median}=16.
$$

**Winsorize** at the 5th and 95th percentiles (10.8 and 68.8). The value 100 becomes 68.8. The winsorized data have

$$
\bar{x}=21.84,
\qquad
\operatorname{median}=16.
$$

The three treatments produce different means and different distribution shapes. Trimming removes the row; capping and winsorizing keep it but change the value. None of them is automatically correct.

---

## 8. A practical decision workflow

The video is an introduction, so it does not give a full production workflow. The following workflow is a safe extension.

### Step 1: Understand the column

Ask what the column means, its units, its valid range, and how it was collected. A value that is impossible in one domain may be normal in another.

### Step 2: Visualize

Plot the raw distribution before transforming or cleaning. Look for skew, multiple modes, clusters, and isolated points.

### Step 3: Detect with more than one method

Use a robust method (IQR or MAD) and a non-robust method (mean/standard deviation) and compare. If they disagree, investigate why.

### Step 4: Classify the cause

Is the value:

- an error,
- a different population,
- a genuine rare event,
- a missing-value code,
- an omitted-variable effect,
- or the target of an anomaly-detection task?

### Step 5: Choose a treatment

Consider trimming, capping, missing-value treatment, discretization, transformation, or a robust model. Prefer correction over deletion when the true value can be recovered.

### Step 6: Fit only on training data

If a threshold, cap, mean, standard deviation, or percentile is learned from data, learn it from the training set only. Applying it to the test set with test-derived statistics is data leakage. Put the rule inside a preprocessing pipeline.

### Step 7: Evaluate with and without treatment

Compare validation performance, coefficient stability, residual behavior, and subgroup performance. A treatment that improves one aggregate metric but damages an important subgroup may not be acceptable.

### Step 8: Document

Record:

- the detection rule,
- the threshold,
- how many rows were affected,
- what was changed,
- the reason,
- the version of the data,
- the effect on the model.

Reproducibility matters. "I removed the outliers" is not a sufficient record.

---

## 9. Common mistakes and misconceptions

1. **"An outlier is always an error."**  
   False. It may be a genuine rare event, an anomaly, or a different population.

2. **"Always remove values beyond 3 standard deviations."**  
   False. The rule assumes approximate normality and is sensitive to the outliers themselves.

3. **"The IQR rule is objective."**  
   The formula is objective, but the 1.5 multiplier and the decision to treat are choices.

4. **"A fixed percentile cleans the data."**  
   A fixed percentile always removes a fixed fraction, even when the data is clean. It is a tail-selection rule, not an error detector.

5. **"Tree models do not care about outliers."**  
   They are less sensitive to monotone transformations and extreme feature values, but they can still overfit unusual points and distort leaf estimates.

6. **"Capping and trimming are the same."**  
   Trimming removes rows; capping keeps rows but changes values. The sample size and distribution differ.

7. **"Outliers should be removed before train-test split."**  
   Detection can be explored on training data, but treatment thresholds must be learned from training data only. Otherwise test information leaks into preprocessing.

8. **"Outliers in the target can be deleted safely."**  
   Removing extreme targets changes the prediction problem and can bias the model. Investigate whether the tail is the point of the task.

9. **"A value is either an outlier or not."**  
   Outlierness is a matter of degree and context. Different methods produce different flags.

10. **"Removing outliers always improves the model."**  
    It can improve a metric on ordinary data while destroying performance on the tail that matters.

11. **"Univariate checks are enough."**  
    A point can be normal in every column but unusual in combination. Multivariate methods are needed for that case.

12. **"Outlier detection is only a preprocessing step."**  
    In fraud, security, health monitoring, and quality control, outlier detection is the modeling task itself.

---

## 10. Interview essentials

### Level 1 - definitions

**Q1. What is an outlier?**  
An observation that is unusually far from the majority of the data or does not follow the dominant pattern. The definition depends on context; there is no universal threshold.

**Q2. Why are outliers a problem in machine learning?**  
They can distort parameter estimates, inflate loss, shift decision boundaries, dominate distance calculations, and make evaluation misleading. Weight-based models are especially sensitive.

**Q3. Are outliers always errors?**  
No. They can be genuine rare events, anomalies of interest, a different population, or signs of a missing variable.

### Level 2 - mechanics

**Q4. How does the Z-score method detect outliers?**  
It computes $z=(x-\mu)/\sigma$ and flags values with $|z|$ above a threshold, commonly 3. It assumes an approximately normal distribution and uses the mean and standard deviation, which are not robust.

**Q5. How does the IQR method work?**  
It computes $Q_1$, $Q_3$, and $\operatorname{IQR}=Q_3-Q_1$. Values below $Q_1-1.5\operatorname{IQR}$ or above $Q_3+1.5\operatorname{IQR}$ are flagged. It is more robust because quartiles are less affected by extremes.

**Q6. What is the difference between trimming and capping?**  
Trimming removes the outlier rows. Capping replaces values beyond a boundary with the boundary value, keeping the rows.

**Q7. What is Winsorization?**  
A capping method that replaces extreme values with chosen percentile values, for example the 1st and 99th percentiles.

### Level 3 - mathematics

**Q8. Why can an outlier hide from the Z-score rule?**  
Because the outlier inflates the mean and standard deviation. The Z-score is computed using contaminated estimates, so the extreme value may appear less extreme. This is masking.

**Q9. What is a robust alternative to the Z-score?**  
The modified Z-score using the median and MAD:
$$
M_i=\frac{0.6745(x_i-\tilde{x})}{\operatorname{MAD}}.
$$
A common threshold is $|M_i|>3.5$.

**Q10. Why are tree-based models less sensitive to outliers?**  
They split on thresholds and use the ordering of values rather than squared distances from a mean. A monotone transformation of a feature does not change the possible splits. They are still not immune.

### Level 4 - diagnosis and scenarios

**Q11. A dataset contains an age of 300. What do you do?**  
Treat it as invalid. If the true age cannot be recovered, set it to missing or remove the row, and document the decision. Do not cap it at 100 and pretend it is a real age without justification.

**Q12. A fraud-detection model has extreme transactions. Should you remove them?**  
Usually not. The extreme transactions may be the fraud signal. Use anomaly-detection methods, robust models, or cost-sensitive evaluation instead of deleting the events of interest.

**Q13. Two students study little but score high. Does that justify deleting them?**  
Not automatically. They may be genuine high performers or the dataset may be missing an explanatory variable such as IQ or prior knowledge. Investigate before deleting.

**Q14. A tree model and a linear regression give very different results on the same data. Why?**  
The linear model may be distorted by outliers, while the tree model is more robust to monotone extremes. The difference is diagnostic: inspect the residuals and the extreme points.

### Level 5 - design

**Q15. How would you build an outlier-treatment pipeline without leakage?**  
Split the data first. Learn detection thresholds, caps, medians, and other statistics from the training set only. Apply the same fitted rule to validation and test data. Put the rule inside a scikit-learn transformer or pipeline.

**Q16. How would you decide between trimming and capping?**  
Ask whether the outlier is invalid or genuine, how much data would be lost, whether the tail matters, and whether the row contains useful information in other columns. Compare validation performance and subgroup effects.

**Q17. When would you not treat outliers at all?**  
When they are genuine and important, when the task is anomaly detection, when the model is robust enough, or when treatment would remove the very signal the business needs.

---

## 11. Exam practice

### Short answers

1. Define an outlier in one sentence.
2. State the empirical rule for 1, 2, and 3 standard deviations.
3. Write the IQR fence formulas.
4. What is the difference between an outlier and an anomaly?
5. Why is the mean sensitive to outliers?
6. Why is the median more robust?
7. What does Winsorization do?
8. Name two weight-based algorithms that are sensitive to outliers.
9. Name two tree-based algorithms that are less sensitive.
10. What is data leakage in the context of outlier treatment?

### Worked questions

**Q1.** A column has mean 100 and standard deviation 15. Using the three-sigma rule, what are the lower and upper bounds? Is 150 an outlier?

**Answer.** Bounds are $100\pm45$, so 55 and 145. The value 150 is above 145, so it is flagged.

**Q2.** A column has $Q_1=20$, $Q_3=40$. What is the IQR? What are the 1.5-IQR fences? Is 75 an outlier?

**Answer.** $\operatorname{IQR}=20$. Lower fence $20-30=-10$; upper fence $40+30=70$. The value 75 is above 70, so it is flagged.

**Q3.** Why can the Z-score and IQR methods disagree?

**Answer.** They use different center and scale estimates. The Z-score uses the mean and standard deviation, which are affected by extreme values; the IQR uses quartiles, which are more robust. A value can be flagged by one method and not the other.

**Q4.** A dataset has 1,000 rows. You remove the top 1% as outliers. What is the main risk?

**Answer.** You remove 10 rows even if the data is clean, and you may remove genuine rare events. A fixed percentile is a tail-selection rule, not an error detector.

**Q5.** Why should outlier thresholds be learned on the training set only?

**Answer.** If the test set influences the threshold, information from the test set leaks into preprocessing. The reported test performance is then optimistic and the pipeline is not reproducible on truly unseen data.

---

## 12. One-page revision

- An outlier is an observation that is unusually different from the majority in context.
- Outliers can be errors, genuine rare events, anomalies, or symptoms of a missing variable.
- Do not remove automatically. Investigate the cause first.
- Weight-based models (linear regression, logistic regression, AdaBoost, deep learning) are strongly affected.
- Tree-based models are less affected but not immune.
- Treatments: trim, cap/Winsorize, treat as missing, discretize, transform, use a robust model.
- Detection: visual inspection, 3-sigma/Z-score, IQR fences, percentiles.
- Z-score assumes approximate normality and is sensitive to masking.
- IQR uses quartiles and is more robust for skewed data.
- Percentiles choose a fixed tail fraction; the cut-off is a business decision.
- Learn thresholds on training data only; use a pipeline.
- Compare models with and without treatment; document every decision.
- In anomaly detection, the outlier is often the signal, not the error.

---

# Part B - Advanced Reading and Reference

> **Note:** Part B extends beyond the video. It is included for interview preparation, practical implementation, and deeper understanding. The video itself is an introductory overview.

## B1. Formal taxonomy

### B1.1 Point, contextual, and collective outliers

- **Point outlier:** a single observation is extreme relative to the rest of the data. Example: age 300.
- **Contextual outlier:** an observation is extreme only in a given context. Example: a temperature of 35°C is normal in summer but unusual in winter.
- **Collective outlier:** a group of observations is unusual together even if each individual value is not extreme. Example: a sudden sequence of small transactions that together indicate fraud.

### B1.2 Outlier, anomaly, novelty

- **Outlier detection** is often unsupervised: find unusual points in a dataset.
- **Anomaly detection** usually has a domain meaning: find points that represent abnormal behavior.
- **Novelty detection** is semi-supervised: learn the normal region from clean training data, then flag new points outside it.

The statistical method may be the same, but the evaluation and cost of errors differ.

### B1.3 Outlierness is not a binary property

Most methods produce a score. A threshold converts the score into a flag. The threshold is a decision, not a discovery. Two analysts can reasonably choose different thresholds for the same data.

---

## B2. Robust statistics and influence

### B2.1 Breakdown point

The **breakdown point** of an estimator is the fraction of observations that can be made arbitrarily large before the estimate becomes arbitrarily large.

- The mean has breakdown point 0: one extreme value can move it without bound.
- The median has breakdown point about 0.5: up to half the data can be contaminated before the median is unreliable.
- The trimmed mean has a breakdown point equal to the trimming fraction.
- The IQR is robust but not infinitely so.

This explains why the median and IQR are preferred for contaminated or skewed data.

### B2.2 Influence in linear regression

For OLS,

$$
\hat{\beta}=(X^T X)^{-1}X^T y.
$$

The fitted values are

$$
\hat{y}=Hy,
\qquad
H=X(X^T X)^{-1}X^T,
$$

where $H$ is the hat matrix. The diagonal element $h_{ii}$ is the **leverage** of observation $i$. High leverage means the observation has an unusual combination of predictor values. Leverage alone is not enough: an observation can have high leverage but still follow the trend.

The **residual** is

$$
e_i=y_i-\hat{y}_i.
$$

A point with high leverage and a large residual is especially influential. **Cook's distance** combines leverage and residual size:

$$
D_i=\frac{e_i^2}{p\,s^2}\cdot\frac{h_{ii}}{(1-h_{ii})^2},
$$

where $p$ is the number of parameters and $s^2$ is the estimated error variance. Large $D_i$ means removing observation $i$ would change the fitted coefficients substantially.

This is the formal version of the video's regression example: a few points can control the line.

### B2.3 Robust regression

Alternatives to OLS include:

- **Huber regression:** squared loss for small residuals, absolute loss for large residuals.
- **Quantile regression:** models a conditional quantile, often the median.
- **RANSAC:** repeatedly fits on random subsets and chooses a consensus set.
- **Theil-Sen:** median of pairwise slopes; robust but slower.

These methods down-weight or ignore extreme residuals instead of deleting rows.

---

## B3. Z-score, modified Z-score, and their limits

### B3.1 Standard Z-score

$$
z_i=\frac{x_i-\mu}{\sigma}.
$$

Under normality, about 99.73% of values lie within $\pm3\sigma$. A threshold of 3 is common, but it is not universal.

Limitations:

- $\mu$ and $\sigma$ are not robust.
- The rule assumes approximate normality.
- In small samples, the estimate of $\sigma$ is unstable.
- In heavy-tailed data, many genuine values exceed 3 standard deviations.
- Masking can hide extreme values.
- Swamping can flag ordinary values because the outlier distorts the scale.

### B3.2 Modified Z-score

$$
M_i=\frac{0.6745(x_i-\tilde{x})}{\operatorname{MAD}},
\qquad
\operatorname{MAD}=\operatorname{median}(|x_i-\tilde{x}|).
$$

A common threshold is $|M_i|>3.5$. The constant 0.6745 makes the scale comparable to the standard deviation for normal data. The modified Z-score is more robust but still univariate and still assumes a roughly symmetric shape.

### B3.3 When to use which

| Situation | Preferred starting point |
|---|---|
| Approximately normal, large sample | Z-score or IQR |
| Skewed data | IQR, percentile, or transform first |
| Contaminated data | Modified Z-score, IQR, or robust model |
| Small sample | Visual inspection; avoid aggressive thresholds |
| Heavy-tailed business data | Percentile or domain-specific tail analysis |

---

## B4. IQR details

### B4.1 Quartile definitions

Different software uses slightly different quartile conventions. For $n$ sorted values, the median splits the data; the lower and upper halves are then split again. The exact interpolation matters for small samples and for reproducibility.

In pandas, `Series.quantile(0.25)` uses linear interpolation by default. In NumPy, `np.percentile` also uses linear interpolation by default. Other conventions exist, so report the method when exact reproducibility matters.

### B4.2 Fence multipliers

- $1.5$ is the standard boxplot rule for "outliers."
- $3.0$ is often used for "extreme" outliers.
- The multiplier controls the trade-off between false positives and false negatives.

A larger multiplier flags fewer points. A smaller multiplier flags more. There is no universal correct value.

### B4.3 IQR on skewed data

The IQR is more robust than the standard deviation, but it is not a complete solution for skew. In a strongly right-skewed distribution, the upper fence may still flag many legitimate values. A log transform can make the distribution more symmetric, after which the IQR rule may be more meaningful.

The IQR rule also assumes that the bulk of the data is reasonably continuous. In a discrete or mixture distribution, the fences may behave oddly.

---

## B5. Percentiles, trimming, capping, and Winsorization

### B5.1 Quantiles

The $p$-th quantile $q_p$ is a value such that approximately a fraction $p$ of the data is at or below it. Common quantiles:

- $q_{0.01}, q_{0.99}$: 1st and 99th percentiles,
- $q_{0.025}, q_{0.975}$: 2.5th and 97.5th,
- $q_{0.05}, q_{0.95}$: 5th and 95th.

The choice is a business decision. A stricter tail (1%/99%) removes fewer points; a looser tail (5%/95%) removes more.

### B5.2 Trimming

**Truncation** or **trimming** removes observations outside a quantile range:

$$
\{x_i: q_{\alpha}\le x_i\le q_{1-\alpha}\}.
$$

The trimmed mean is the mean of the remaining values. Trimming reduces the sample size and can bias the estimate if the tails are informative.

### B5.3 Capping and Winsorization

Capping replaces values outside a range with the boundary:

$$
x_i^{\text{cap}}=\min(\max(x_i,L),U).
$$

Winsorization replaces values outside a quantile range with the corresponding quantile:

$$
x_i^{\text{win}}=
\begin{cases}
q_{\alpha}, & x_i<q_{\alpha},\\
x_i, & q_{\alpha}\le x_i\le q_{1-\alpha},\\
q_{1-\alpha}, & x_i>q_{1-\alpha}.
\end{cases}
$$

The two are similar. "Capping" often refers to domain limits; "Winsorization" usually refers to percentile limits.

### B5.4 Effects on distribution

Capping and Winsorization create a point mass at the boundary. The mean moves toward the center, the standard deviation shrinks, and the tail becomes truncated. A model may interpret the boundary value as a real measurement unless you also add an indicator for "was capped."

### B5.5 Trimming versus Winsorization

| Property | Trimming | Winsorization |
|---|---|---|
| Row kept? | No | Yes |
| Sample size | Decreases | Unchanged |
| Extreme value | Removed | Replaced |
| Distribution | Truncated | Point mass at boundary |
| Risk | Loses data | Distorts tail |
| Common use | Invalid values | Reduce influence while keeping rows |

---

## B6. Multivariate outlier detection

### B6.1 Why univariate checks are insufficient

A point can be normal in every column but unusual in combination. For example:

- age = 30 (normal),
- income = 50,000 (normal),
- spending = 2,000 (normal),

but a 30-year-old with that income and spending pattern may be extremely rare. A univariate histogram will not flag the row.

### B6.2 Scatterplots and pairwise views

For two or three features, plot the data. A point far from the main cloud is a multivariate outlier. For many features, use a projection method such as PCA or a pairwise scatterplot matrix.

### B6.3 Mahalanobis distance

For a vector $x$, mean $\mu$, and covariance matrix $\Sigma$, the squared Mahalanobis distance is

$$
D^2=(x-\mu)^T\Sigma^{-1}(x-\mu).
$$

Under a multivariate normal distribution, $D^2$ follows a chi-square distribution with $p$ degrees of freedom. A threshold can be chosen from the chi-square quantile, for example the 99.9th percentile.

Limitations:

- $\mu$ and $\Sigma$ are not robust,
- the covariance matrix can be singular or unstable when features are correlated or $p$ is large,
- the method assumes an elliptical distribution,
- it can be affected by the outliers it is trying to detect.

Robust covariance estimators (for example, minimum covariance determinant) can help.

### B6.4 Isolation Forest

An isolation forest builds random trees that split the data. Outliers are isolated in fewer splits because they are few and different. The anomaly score is based on the average path length. It is unsupervised, scales well, and does not assume a distribution, but it has hyperparameters and can struggle with local density variation.

### B6.5 Local Outlier Factor

LOF compares the local density of a point with the local densities of its neighbors. A point in a sparse region relative to its neighbors gets a high score. It is useful for local outliers but is sensitive to the choice of $k$ and to scaling.

### B6.6 DBSCAN

DBSCAN labels points in low-density regions as noise. It is a clustering method, not a dedicated outlier detector, but it can be used for anomaly detection when the normal data forms dense clusters. It is sensitive to `eps` and `min_samples`.

### B6.7 One-Class SVM

One-Class SVM learns a boundary around normal data and flags points outside it. It is useful for novelty detection when clean normal training data is available. It is sensitive to kernel and hyperparameters.

---

## B7. Model-specific effects in more detail

### B7.1 Linear regression

OLS is sensitive to both leverage and residual size. A high-leverage point that lies off the trend can change the slope dramatically. A high-leverage point that lies on the trend may be harmless.

Diagnostics:

- residual vs fitted plot,
- leverage vs residual plot,
- Cook's distance,
- DFBETAS,
- added-variable plots.

### B7.2 Logistic regression

Logistic regression can be affected by:

- extreme feature values,
- complete separation,
- mislabeled observations,
- rare-event imbalance.

Regularization (L2) can stabilize coefficients, but it does not solve label noise. Robust logistic regression and data cleaning are alternatives.

### B7.3 AdaBoost

AdaBoost's exponential loss gives large weight to misclassified points. A noisy outlier can dominate the ensemble. Methods to mitigate:

- clean labels,
- cap extreme features,
- use a robust boosting variant,
- limit the number of estimators,
- use learning-rate shrinkage.

### B7.4 Deep learning

Neural networks can be affected by:

- large input magnitudes,
- label noise,
- extreme target values,
- batch statistics,
- gradient explosions.

Mitigations:

- normalization,
- gradient clipping,
- robust losses (Huber, log-cosh),
- data cleaning,
- outlier-aware training,
- ensemble methods.

### B7.5 Tree-based models

Trees are invariant to monotone transformations of individual features, so a log transform does not change the possible splits. They are less sensitive to extreme feature values, but they can still:

- isolate outliers in small leaves,
- overfit noisy points,
- produce unstable splits when many values are tied,
- be affected by extreme targets in regression trees.

Random forests average many trees and are usually more stable than a single tree. Gradient boosting can still chase outliers if the loss is sensitive.

### B7.6 Distance-based models

KNN, K-means, and RBF SVMs use distances or similarities. A single extreme value can dominate the distance calculation and change neighbors, centroids, or the kernel. Scaling and robust distances matter.

---

## B8. Code patterns

These snippets are supplementary examples. They are not from an official Day 41 notebook.

### B8.1 Z-score

```python
import numpy as np

x = np.array([10, 12, 14, 15, 16, 18, 20, 22, 100], dtype=float)
z = (x - x.mean()) / x.std(ddof=1)
flag = np.abs(z) > 3
print(z)
print(flag)
```

### B8.2 Modified Z-score

```python
import numpy as np

x = np.array([10, 12, 14, 15, 16, 18, 20, 22, 100], dtype=float)
median = np.median(x)
mad = np.median(np.abs(x - median))
modified_z = 0.6745 * (x - median) / mad
flag = np.abs(modified_z) > 3.5
print(modified_z)
print(flag)
```

### B8.3 IQR fences

```python
import numpy as np

x = np.array([10, 12, 14, 15, 16, 18, 20, 22, 100], dtype=float)
q1, q3 = np.percentile(x, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
flag = (x < lower) | (x > upper)
print(q1, q3, iqr, lower, upper)
print(flag)
```

### B8.4 Capping and Winsorization

```python
import numpy as np

x = np.array([10, 12, 14, 15, 16, 18, 20, 22, 100], dtype=float)

capped = np.clip(x, 5, 29)

lower = np.percentile(x, 5)
upper = np.percentile(x, 95)
winsorized = np.clip(x, lower, upper)

print(capped)
print(winsorized)
```

### B8.5 Leakage-safe transformer sketch

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class IQRClip(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * self.iqr_
        self.upper_ = self.q3_ + self.factor * self.iqr_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)
```

This sketch is deliberately small. In a real pipeline, you would validate shapes, handle missing values, and test the transformer. The important point is that `fit` learns the fences from training data, and `transform` applies them to any later data.

### B8.6 Multivariate sketch

```python
import numpy as np
from scipy.stats import chi2

X = np.array([
    [30, 50000, 2000],
    [31, 52000, 2100],
    [29, 48000, 1900],
    [30, 51000, 2050],
    [30, 50000, 90000],
], dtype=float)

mu = X.mean(axis=0)
cov = np.cov(X, rowvar=False)
inv_cov = np.linalg.inv(cov)

diff = X - mu
d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
threshold = chi2.ppf(0.999, df=X.shape[1])
print(d2)
print(d2 > threshold)
```

This example is illustrative. A real analysis should use a robust covariance estimator and check the distributional assumptions.

---

## B9. Leakage and workflow discipline

Outlier treatment is a preprocessing step, so the usual leakage rules apply.

### B9.1 Split before fitting

If you compute a cap, a mean, a standard deviation, a percentile, or an IQR fence from the full dataset, the test set influences preprocessing. The correct order is:

1. split into training and test,
2. fit the detector or cap on training,
3. apply the fitted rule to test.

### B9.2 Cross-validation

Inside cross-validation, the threshold must be learned inside each training fold. If you compute the threshold once on the full training set and then cross-validate, the validation folds have influenced the threshold indirectly.

Use a pipeline so that the transformer is refitted in each fold.

### B9.3 Target leakage

An outlier may be caused by information that will not be available at prediction time. For example, a feature may contain the target or a post-outcome measurement. Treating it as a normal extreme value misses the deeper leakage problem.

### B9.4 Distribution shift

A value that was an outlier in training may be normal in production, or vice versa. Monitor the distribution and re-fit preprocessing when the population changes. A fixed cap learned years ago may be inappropriate today.

### B9.5 Documentation

Record the detection rule, the fitted thresholds, the number of affected rows, and the treatment. A model that cannot reproduce its preprocessing cannot be audited.

---

## B10. Contextual and time-series outliers

### B10.1 Contextual outliers

A value can be normal globally but abnormal in context. For example:

- a temperature of 35°C in summer versus winter,
- a sales spike on a holiday,
- a high heart rate during exercise.

Detection should condition on the context. This may mean group-wise thresholds, regression residuals, or models with context features.

### B10.2 Time-series outliers

Time-series data adds temporal dependence. A point may be extreme relative to a rolling mean, a seasonal pattern, or a forecast residual. Simple global Z-scores can be misleading because the mean and variance change over time.

Methods include:

- rolling mean and standard deviation,
- seasonal decomposition,
- forecast residuals,
- STL decomposition,
- change-point detection,
- state-space models.

### B10.3 Anomaly detection versus cleaning

In time-series monitoring, the unusual point is often the event of interest. The goal is not to clean it away but to detect it quickly and with few false alarms. Evaluation must consider detection delay, precision, recall, and the cost of false alarms.

---

## B11. Extended interview bank

### Level 1 - definitions

1. What is an outlier?
2. What is the difference between an outlier and an anomaly?
3. Why can an outlier be useful?
4. What is a univariate outlier?
5. What is a multivariate outlier?

### Level 2 - mechanics

6. How does the IQR rule work?
7. How does the Z-score rule work?
8. What is the empirical rule?
9. What is Winsorization?
10. What is the difference between trimming and capping?
11. Why is the median more robust than the mean?
12. What is the MAD?

### Level 3 - mathematics

13. Derive the Z-score.
14. Why can the Z-score fail to detect an outlier?
15. What is the breakdown point of the median?
16. What is leverage in linear regression?
17. What is Cook's distance?
18. Write the Mahalanobis distance formula.
19. Why does the chi-square distribution appear in Mahalanobis detection?
20. What is the difference between a point mass from capping and a truncated distribution from trimming?

### Level 4 - diagnosis

21. A value is flagged by IQR but not by Z-score. What could explain the disagreement?
22. A tree model performs well while a linear model performs poorly. What might be happening?
23. A fraud model flags too many normal transactions after capping. Why?
24. A feature has a heavy tail and the IQR rule flags 8% of rows. What would you check?
25. A model's test error is dominated by a few large residuals. What does that suggest?

### Level 5 - design

26. How would you build a leakage-safe outlier pipeline?
27. How would you decide between trimming and capping?
28. When would you use a robust model instead of cleaning?
29. How would you evaluate the effect of outlier treatment?
30. How would you handle outliers in a time series?
31. How would you handle multivariate outliers in a high-dimensional dataset?
32. How would you explain an outlier treatment to a stakeholder?

### Level 6 - advanced

33. Compare isolation forest, LOF, and One-Class SVM.
34. What are the assumptions of Mahalanobis distance?
35. How does AdaBoost's loss make it sensitive to outliers?
36. Why are trees less sensitive to monotone transformations?
37. What is the difference between masking and swamping?
38. How can robust covariance estimation improve multivariate detection?
39. What are the risks of removing outliers from the target?
40. How would you detect anomalies when labels are unavailable?

### Level 7 - synthesis

41. A business wants to remove all extreme customers to improve a churn model. How would you challenge the request?
42. You have a fraud dataset with 0.1% positives and extreme transaction amounts. Design an evaluation plan.
43. A model is stable on random splits but fails in production after a marketing campaign. How could outliers and distribution shift explain this?
44. You must choose between a simple IQR rule and a learned anomaly detector. What factors decide?
45. How would you make outlier treatment reproducible and auditable across a team?

---

## B12. Extended examination questions

### Question 1

A dataset contains the values

$$
2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 100.
$$

Compute $Q_1$, $Q_3$, the IQR, and the fences. Is 100 an outlier by the 1.5-IQR rule? What is the Z-score of 100? Why do the methods disagree?

**Answer.** Using linear percentiles, $Q_1=4$, $Q_3=8$, $\operatorname{IQR}=4$. Fences are $4-6=-2$ and $8+6=14$. The value 100 is flagged. The mean is 16, the standard deviation is about 31.7, and the Z-score is about 2.65, so the 3-sigma rule does not flag it. The outlier inflates the mean and standard deviation, which is masking.

### Question 2

Explain why capping at the 99th percentile can make a model worse even though it reduces the influence of extremes.

**Answer.** Capping creates a point mass at the boundary and changes the relationship between the feature and the target. If the extreme values carry genuine signal, capping removes that signal. The model may also treat the cap as a real value rather than a censored observation. The effect must be evaluated on validation data and on the subgroup that contains the tail.

### Question 3

A hospital wants to remove the longest hospital stays because they are outliers. What is the danger?

**Answer.** The longest stays may be the most important cases for resource planning, cost prediction, or clinical risk. Removing them changes the target population and can make the model useless for the patients who matter most. A better approach may be a robust model, a separate model for the tail, or a cost-sensitive evaluation.

### Question 4

Why is it incorrect to compute a cap on the full dataset before splitting?

**Answer.** The cap depends on the test set's distribution. The test information leaks into preprocessing, so the test set is no longer a fair estimate of unseen performance. The cap must be learned from training data and applied unchanged to test data.

### Question 5

A point is not an outlier in any single feature but is an outlier in combination. How can you detect it?

**Answer.** Use a multivariate method: a scatterplot matrix, Mahalanobis distance, isolation forest, LOF, or a density-based method. Univariate rules cannot see the joint pattern.

### Question 6

Why can a fixed percentile rule be dangerous on a clean dataset?

**Answer.** It always flags a fixed fraction of the data. If the data is clean, it still removes or changes the top and bottom 1% or 5%, which may be genuine values. A percentile is a tail-selection rule, not evidence of an error.

### Question 7

Compare trimming and Winsorization for a regression model.

**Answer.** Trimming removes rows and changes the sample and target distribution; it can reduce influence but loses information. Winsorization keeps rows and replaces extreme values with quantiles; it reduces influence but creates a point mass and can distort the tail. The choice depends on whether the extreme values are errors or genuine.

### Question 8

What does it mean that trees are less sensitive to outliers?

**Answer.** Trees split on thresholds and use the ordering of values. A monotone transformation or a change in an extreme value's magnitude may not change the split. However, outliers can still create small leaves, affect split selection, and distort regression leaf means.

---

## B13. Coverage and source fidelity

This chapter is based on the auto-generated English transcript of the video. The following table maps the main video sections to the notes.

| Video time | Topic | Notes section |
|---|---|---|
| 00:00 | Introduction and roadmap | Header, Part A introduction |
| 00:51 | What are outliers? | Section 2 |
| 04:40 | When to remove or keep outliers | Section 3 |
| 07:14 | Effects on ML algorithms | Section 4 |
| 09:14 | How to treat outliers | Section 5 |
| 12:01 | How to detect outliers | Section 6 |
| 15:58 | Techniques for the next videos | Section 6, Part B |

The video is a theory class. It has no official notebook, no dataset, and no saved outputs. The numerical examples in Section 7 are constructed for teaching, not reproduced from a notebook. Part B is supplementary and is clearly separated from the transcript-based material.

The transcript is auto-generated and contains recognition errors and colloquial phrasing. Where the transcript is ambiguous, the notes use the mathematically standard interpretation and say so. No experimental result is invented.

---

## B14. References

- CampusX, [What are Outliers | Outliers in Machine Learning](https://youtu.be/Lln1PKgGr_M), 2021.
- scikit-learn, [Outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html).
- NIST/SEMATECH, [e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/).
- pandas, [quantile documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html).
- NumPy, [percentile documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html).
- Tukey, J. W., *Exploratory Data Analysis*, Addison-Wesley, 1977.
- Iglewicz, B., and Hoaglin, D. C., *How to Detect and Handle Outliers*, ASQC Quality Press, 1993.
