#set page(paper: "a4", margin: 2cm, header: context [Header])
#set text(font: ("Cambria", "Segoe UI"), size: 10.5pt)
#set par(justify: true)
#show raw.where(block: true): it => block(fill: luma(245), inset: 10pt, stroke: 0.5pt + luma(200), radius: 3pt, it)
#set heading(numbering: none)

= Test

$ "Skewness"(X) = frac(E[(X - mu)^3], sigma^3) $

$ y^((lambda)) = cases((x^lambda - 1)/lambda & "if " lambda != 0, ln x & "if " lambda = 0) $

```python
import numpy as np
print("hi")
```

#table(columns: (1fr, 1fr), table.header([*A*], [*B*]), [1], [2])
