#import "lib.typ": *

#let body = [
#chapter-divider(36, "Numerical Imputation with SimpleImputer", "Mean, median, arbitrary constants and the end of the distribution", "Chapter 35 deleted rows. This chapter fills them. The lecture groups every numerical method into univariate and multivariate families, then works through the univariate four: mean/median, arbitrary constant, end-of-distribution, and (deferred to Chapter 38) random sample. The centrepiece is a full audit of what mean imputation does to variance, distribution, covariance and boxplots.")

#heading(level: 2)[36.1 Two families: univariate and multivariate]

The lecture opens with the distinction that organises the whole imputation block of the course. A row has a missing entry in column 2 of 1, 2, 3, 4.

#definition([Univariate imputation])[
Fill the gap using only values from *the same column*: its mean, its median, or a randomly drawn value. Any statistical technique applied to a single column is univariate.
]

#definition([Multivariate imputation])[
Fill the gap using the *other columns* of the row. KNN imputation and iterative imputation (MICE) are the two multivariate methods, covered in Chapters 39 and 40.
]

Univariate methods are cheap and easy to reuse in production: you store one number per column. Multivariate methods are more accurate because they respect relationships between columns, at the cost of speed and memory. The lecture's agenda for this chapter is the first three univariate techniques plus the sklearn machinery for choosing among them; random sample imputation and automatic model selection were moved to a later video and are covered in Chapter 38.

#heading(level: 2)[36.2 Mean and median imputation]

Replace every missing entry in a column with a single location statistic of that column:

$ x_i <- 1/n sum_(k in "observed") x_k = xbar " (mean) ", #h(1em) "or" #h(1em) x_i <- "median"(x) $

The choice between them follows the shape of the distribution:

- If the distribution is roughly normal, mean, median and mode nearly coincide, so the mean is fine.
- If the distribution is even slightly skewed, the mean drifts toward the tail and the median is the better candidate.

*Advantages.* The lecture gives two, and both are about simplicity: the technique is trivial to compute, and the deployed model needs only one stored statistic per column to reproduce it on a server. There is no special predictive benefit beyond that; it works well when missingness is rare.

*Disadvantages.* Three things change, and all three are bad for machine learning:

1. *The shape of the distribution changes.* Squeeze a spike of mass into the centre and the density curve fattens at the mean and thins in the shoulders.
2. *Spurious outliers appear.* Compressed quartiles shrink the interquartile range, and boxplot fences (`Q1 – 1.5·IQR`, `Q3 + 1.5·IQR`) then flag ordinary observations as extreme.
3. *Covariance and correlation with other columns shift.* The imputed variable's relationship with its neighbours is recomputed on fake data.

*When to use it.* Two conditions, stated by the lecture as rules of thumb: the data is missing completely at random, and the missingness is under 5% for that column. Above that, reliability falls off and the three red flags above start to bite. The lecture's blunt caveat: many people on the internet use mean imputation anyway because it is simple; it is rarely appropriate in production.

#keybox(title: [What "less than 5%" buys you])[
With 5% of the values replaced by one constant, roughly 19 out of every 20 points still carry real information, so the damage to moments and shapes stays small. At 20% missing, as in Age below, the same operation rewrites the histogram.
]

#heading(level: 2)[36.3 The four diagnostics]

The lecture does not ask you to trust mean imputation. It asks you to measure it. Four checks, in order:

*1. Variance.* Variance is the squared distance from the mean; pushing extreme values inward can only shrink it. Some shrinkage is unavoidable; a drastic collapse is a red flag.

*2. Distribution.* Overlay the density of the original column on the densities after mean and after median fill. Overlapping curves are a green flag; a new central peak is a red flag.

*3. Covariance and correlation.* Compare `df.cov()` and `df.corr()` before and after. Correlation is bounded in `[-1, 1]`, so its movements look small; covariance has no such bound and shows the shift clearly.

*4. Boxplot.* Compare the five-number summary and the outlier count. If filling shrinks the IQR and manufactures whisker outliers on both sides, the imputation invented structure that was never in the data.

#figure(
  image("figures/ch36_mean_vs_median.png", width: 100%),
  caption: [Mean versus median fill. The median ignores the tail; the mean is dragged with it. Both concentrate mass where the constant was inserted.],
)

#heading(level: 2)[36.4 Worked audit: a four-column Titanic]

The lecture works on a reduced Titanic variant with four numerical columns: `Age`, `Fare`, `Family` (the sum of parents/children and siblings/spouses) and the target `Survived`. `Age` arrives with about 19% of its values missing; about 5% of `Fare` values were deleted on purpose; `Family` and `Survived` are complete.

The first action is always the train/test split: 712 rows in `X_train`, 179 in `X_test`, with the missingness spread evenly across both (roughly 20% of Age and 5% of Fare inside `X_train`).

Four statistics are computed *from the training data only* (`age_mean`, `age_median`, `fare_mean`, `fare_median`), four new columns are created with `fillna`, and the diagnostics run:

#table(
  columns: (1.2fr, 1fr, 1fr, 1.4fr),
  inset: (x: 8pt, y: 5pt),
  align: (left + horizon, right + horizon, right + horizon, left + horizon),
  stroke: (x, y) => (bottom: 0.4pt + hairline),
  fill: (x, y) => if y == 0 { primary } else if calc.odd(y) { panel } else { white },
  table.header([Quantity], [Original], [After fill], [Verdict]),
  [Variance of Age], [204], [161.98 median / 161.82 mean], [Shrinks a lot: red flag],
  [Variance of Fare], [2448], [almost unchanged], [Green flag: only 5% missing],
  [Covariance Age–Fare], [70], [55 after mean fill], [Relationship weakened: red flag],
  [Covariance Age–Family], [-6.49], [changed], [Red flag],
  [Age density], [smooth], [new central peak], [Red flag],
  [Fare density], [smooth], [all three curves overlap], [Green flag],
  [Age boxplot], [few outliers], [IQR shrinks, outliers both sides], [Red flag],
  [Fare boxplot], [-], [no change], [Green flag],
)

The lecture's conclusion is a per-column verdict, not a global one: mean/median imputation is acceptable for `Fare`, where the red flags never appear, and unacceptable for `Age`, where every check fires. The honest response to `Age` is a better technique: an indicator, a multivariate method, or a model that can carry the gap.

#heading(level: 2)[36.5 Implementing it: fillna versus SimpleImputer]

Two routes. Pandas `fillna` is direct but lives outside the sklearn ecosystem:

```python
x_train["age_median"] = x_train["Age"].fillna(age_median)
x_train["fare_mean"]  = x_train["Fare"].fillna(fare_mean)
```

sklearn's `SimpleImputer` is the production answer. Its parameters are exactly the four the lecture lists: `missing_values` (default `np.nan`; you may register `0` or any sentinel), `strategy` (`"mean"`, `"median"`, `"most_frequent"` for categorical columns, `"constant"`), `fill_value` (used by `strategy="constant"`), and `add_indicator` (Chapter 38).

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_median = SimpleImputer(strategy="median")     # Age
num_mean   = SimpleImputer(strategy="mean")       # Fare
trf = ColumnTransformer([("age", num_median, ["Age"]),
                         ("fare", num_mean,  ["Fare"])], remainder="passthrough")

trf.fit(X_train)
trf.named_transformers_["age"].statistics_        # 28.75  -> Age median
trf.named_transformers_["fare"].statistics_       # 32.61  -> Fare mean
X_train_t = trf.transform(X_train)
X_test_t  = trf.transform(X_test)
```

The two `statistics_` values, 28.75 and 32.61, are the lecture's printed results: a median age of 28.75 years and a mean fare of 32.61. Using *different* strategies on *different* columns in one fit is precisely what `ColumnTransformer` exists for.

#warnbox(title: [The rule people break])[
`fit` on `X_train` only, then `transform` on both `X_train` and `X_test`, and the test set must be filled with the *training* statistics. Computing the mean of the test set and filling with it leaks test information into training and makes the two sets inconsistent. The lecture flags this as the classic mistake.
]

The trade-off the lecture notes for sklearn: `transform` returns a numeric array rather than a labelled DataFrame, which is harder to eyeball, but only the estimator form plugs into `Pipeline` and `GridSearchCV`.

#heading(level: 2)[36.6 Arbitrary-value imputation]

Replace every gap with one number that is deliberately outside the data: `0`, `-1`, `99`, `999`. For age, the lecture uses `99` or `-1`; for categorical columns the equivalent is the literal string `"Missing"`.

The mechanism's purpose is not to estimate anything. It is to *surface the missingness*:

#notebox(title: [Why a constant?])[
The whole idea is to create a distinction between observations that have data and observations that don't, so the model can tell them apart. The value itself carries no information; the fact that the gap is marked does.
]

That is why the technique is preferred when the data is *not* missing completely at random: if blanks cluster in chunks, a constant keeps the gap visible and lets tree splits use it as a signal.

The costs are the same three as before, only larger. The variance of Age explodes from 204 into the thousands when `99` is inserted, and `Fare` jumps from 2448 to roughly 47,000 with a far-out constant. The density curve develops a second peak at the sentinel. Covariance and correlation move. Easy to apply, distorted to interpret.

In sklearn:

```python
imputer = SimpleImputer(strategy="constant", fill_value=-1)
X_train_t = imputer.fit_transform(X_train)
```

#heading(level: 2)[36.7 End-of-distribution imputation]

The lecture presents this as the same technique with a principled constant: instead of guessing `99`, *compute* a value that sits at the end of the column's own distribution.

#definition([Normal-looking column])[
Fill with `$mu + 3 sigma$` or `$mu - 3 sigma$`, where `$sigma$` is the standard deviation. Under the 68–95–99.7 rule, about 99.7% of a normal distribution lies within three standard deviations of the mean, so the fill value behaves like an extreme observation.
]

#definition([Skewed column])[
Apply the IQR proximity rule. With `IQR = Q3 - Q1`, fill with `$Q_1 - 1.5 "IQR"$` or `$Q_3 + 1.5 "IQR"$` — the exact fences a boxplot uses to define outliers.
]

Both variants achieve the arbitrary-value goal: the imputed point sits where a real extreme value would sit, unmistakably separated from the bulk of the data. The advantages and disadvantages are unchanged (easy to compute; distribution, variance and covariance all move), and the same condition applies: use it when the missingness is not at random and you want the model to see the gap.

#table(
  columns: (1.1fr, 1.3fr, 1.2fr, 1.1fr),
  inset: (x: 8pt, y: 5pt),
  align: (left + horizon, left + horizon, left + horizon, left + horizon),
  stroke: (x, y) => (bottom: 0.4pt + hairline),
  fill: (x, y) => if y == 0 { primary } else if calc.odd(y) { panel } else { white },
  table.header([Technique], [Fills with], [Assumption], [Signature]),
  [Mean / median], [One location statistic], [MCAR, under 5%], [strategy="mean" / "median"],
  [Arbitrary constant], [A value absent from the data], [Not MCAR: mark the gap], [strategy="constant"],
  [End of distribution], [μ ± 3σ, or Q1 – 1.5·IQR / Q3 + 1.5·IQR], [Not MCAR: mark the gap], [strategy="constant" with a computed fill_value],
  [Random sample], [A drawn observed value], [MCAR, distribution preserved], [Chapter 38],
)

#figure(
  image("figures/ch36_simpleimputer_flow.png", width: 90%),
  caption: [One `SimpleImputer`, two steps. `fit` stores a single statistic per column from the training data; `transform` writes it into every gap of whatever frame is passed, including the test set.],
)

#heading(level: 2)[36.8 Recap, practice and sources]

Univariate methods ignore the row and read the column. Mean and median are for MCAR gaps under 5%, and they must be audited four ways: variance, density overlay, covariance, boxplot. Arbitrary and end-of-distribution fills deliberately live outside the data so the model can see the gap; they distort moments far more. Whatever the strategy, `fit` on training data, `transform` everywhere else, and store the training statistic for the server.

#heading(level: 3)[Practice]
1. Reproduce the lecture's audit: split your data, fill Age and Fare with mean and median, and compare `var()`, `cov()`, `corr()` and overlaid densities.
2. Prove to yourself that filling with a constant cannot increase variance: write the three-line argument using $sum (x_i - xbar)^2$.
3. Compute `mean + 3*std` and `Q1 - 1.5*IQR` for a skewed column. Which is more extreme? Why?
4. Build a `ColumnTransformer` that applies `median` to one column, `mean` to another and `constant` to a third, and inspect `.statistics_`.
5. Fill the test set with its own mean instead of the training mean. Explain the leakage in two sentences.
6. Run the boxplot check before and after mean filling. If the outlier count doubles, what has the imputation created?

#heading(level: 3)[Sources and scope]
Video 36, "Numerical Imputation / SimpleImputer", CampusX 100 Days of Machine Learning. The lecture covers the univariate/multivariate split, mean versus median choice, the three disadvantages, the two use conditions (MCAR and under 5%), the four-column Titanic audit with all printed numbers, `fillna` versus `SimpleImputer`, `ColumnTransformer` with per-column strategies and the fit-on-train rule, arbitrary constants and end-of-distribution imputation. Random sample imputation and automatic strategy selection were explicitly deferred to a later video and appear in Chapter 38. The 68–95–99.7 framing of the three-sigma rule is a supplementary correction of the lecture's informal phrasing.
]
