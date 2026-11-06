#import "lib.typ": *

#let body = [
#chapter-divider(31, "Power Transformer", "Skewness, Box-Cox and Yeo--Johnson", "This chapter starts from the failure mode: a long tail. It then derives the transform, shows what lambda does, and ends with a traced Scikit-learn pipeline that must be fitted on training data only.")

#heading(level: 2)[31.1 The problem with a long tail]
A feature with a long right tail is not merely ugly. It changes which observations dominate a squared-error loss, it stretches distances, and it makes a mean a poor summary of the typical case. Consider ten incomes:

#table(
  columns: (1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  stroke: (x, y) => (bottom: 0.4pt + hairline),
  fill: (x, y) => if y == 0 { primary } else if calc.odd(y) { panel } else { white },
  table.header([Value], [Deviation from mean]),
  [10], [$-25.4$],
  [11], [$-24.4$],
  [12], [$-23.4$],
  [90], [$+54.6$],
)

#keybox(title: [The mean is not robust])[One large value moves the mean, and squared error moves it even more. The median stays near the bulk. A transform does not make the outlier disappear; it changes how much leverage the outlier has.]

#figure(
  image("../figures/ch31_power_transform_effect.png", width: 100%),
  caption: [Power transforms change the shape of the distribution while preserving order. The same data is shown before and after several values of lambda.],
)

#heading(level: 2)[31.2 Box--Cox]
#definition[Box--Cox transform][For positive $x$,
$ y^((lambda)) = cases((x^lambda - 1)/lambda & "if " lambda != 0, ln x & "if " lambda = 0) $
The case at zero is the continuous limit, not an arbitrary exception.]

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="box-cox")
X_train_t = pt.fit_transform(X_train[["income"]])
X_test_t = pt.transform(X_test[["income"]])
```
]
