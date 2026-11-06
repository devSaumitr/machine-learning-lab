# Day 33 — Handling Mixed Variables

> **Course:** CampusX — 100 Days of Machine Learning, Day 33  
> **Official notebook:** `vendor/100-days-of-machine-learning/day33-handling-mixed-variables/Untitled.ipynb` (14 cells)  
> **Personal notebook:** `mixedata.ipynb` (11 cells)  
> **Dataset:** `DAtasets/titanic.csv` — 891 rows and 4 columns  
> **Transcript:** `ML_Notes_Videos_31_40/transcripts/33_transcript.txt` exists but is a noisy Hindi auto-transcript; it is used for topic orientation, not for exact values  
> **Central question:** A single column can contain a number in one row and a category in another. How do we recognise that mixed meaning, split it without losing rows, and decide which extracted parts are useful features rather than identifier noise?

The dataset in this lesson is deliberately small: `Cabin`, `Ticket`, `number`, and `Survived`. Its columns look like ordinary strings, but each of the first three contains more than one kind of information. The goal is not to memorise three extraction tricks. The goal is to learn how to read a column's meaning before choosing a dtype or a model representation.

**Source labels used in this chapter**

- **Official** — the saved output in the CampusX notebook. It may have been produced with an older pandas version.
- **Rerun** — a result reproduced in this environment with Python 3.14.2, pandas 3.0.5, NumPy 2.5.3, and scikit-learn 1.9.1.
- **Supplementary** — an explanation or experiment added for learning; it is not presented as original course output.

---

# 1. Learning objectives

After studying this chapter I should be able to:

1. **Recognise a mixed variable** whose rows encode more than one type of information.
2. **Explain why the pandas dtype alone does not reveal semantic type**, especially when a column contains both numbers and letters.
3. **Split a mixed column into numeric and categorical components** without dropping rows that belong to either component.
4. **Use `pd.to_numeric(..., errors='coerce')`** and its missing-value mask correctly.
5. **Extract patterns with `str.extract`**, understand capture groups, and use raw strings for regular expressions.
6. **Extract a category from the first character** of an identifier such as a cabin number.
7. **Separate the numeric and categorical parts of a ticket identifier** and explain why the numeric part is not automatically a meaningful measurement.
8. **Distinguish a feature from an identifier**, and explain why high-cardinality IDs can cause overfitting and leakage.
9. **Analyse missingness** before and after extraction, including structural missingness created by parsing.
10. **Connect extraction to the train/test workflow**: deterministic parsing, category vocabulary, train-only encoding, group leakage, and production consistency.
11. **Interpret the official and personal notebooks accurately**, including their missing normalisation and warning issues.
12. **Answer interview and examination questions** with reasons, not only code fragments.

---

# 2. The problem: one column, several meanings

## 2.1 The raw table

The dataset begins as:

| Cabin | Ticket | number | Survived |
|---|---|---|---|
| NaN | A/5 21171 | 5 | 0 |
| C85 | PC 17599 | 3 | 1 |
| NaN | STON/O2. 3101282 | 6 | 1 |
| C123 | 113803 | 3 | 1 |
| NaN | 373450 | A | 0 |

Three columns look like strings, but they contain different structures:

- **`Cabin`:** sometimes missing; sometimes a deck letter plus digits (`C85`); sometimes several cabins (`C23 C25 C27`); sometimes a single letter (`D`).
- **`Ticket`:** sometimes a prefix plus a number (`PC 17599`); sometimes a number alone (`113803`); sometimes a special word (`LINE`).
- **`number`:** sometimes a digit character (`5`, `3`, `6`); sometimes the letter `A`.

A pandas dtype is a storage description, not a statement that every row has the same semantic type. In this environment, pandas 3.0.5 infers the string columns as `str` dtype because the columns contain text. The old official output shows an object-dtype array such as `array(['5', '3', ..., 'A'], dtype=object)`. Both descriptions mean the same practical thing: the column is a container of strings, and some of those strings happen to look like numbers.

## 2.2 What “mixed” means

A **mixed variable** is a column whose values encode different kinds of information in different rows. Common examples are:

- `"5"` versus `"A"` in a family-size or companion field;
- `"C85"` versus `"C23 C25 C27"` in a cabin field;
- `"PC 17599"` versus `"113803"` versus `"LINE"` in a ticket field;
- `"10+ years"` versus `"5 years"` in an experience field;
- `"$50,000"` versus `"not disclosed"` in an income field.

The mixed nature is not necessarily a data-entry error. It may be the real encoding scheme. The correct response is to identify the components, decide whether each component is meaningful, and keep provenance so that the transformation can be audited.

## 2.3 Why not simply convert the whole column?

If we call `pd.to_numeric` on the entire column without a missing policy:

```python
pd.to_numeric(df['number'])
```

pandas raises an error when it reaches `'A'`, because the column is not uniformly numeric. Two common alternatives are:

```python
# Coerce non-numeric values to NaN
numeric = pd.to_numeric(df['number'], errors='coerce')

# Ignore the error and leave the original value
mixed = pd.to_numeric(df['number'], errors='ignore')
```

`errors='coerce'` is usually the right choice for extraction because it converts every parseable value and marks every non-parseable value as missing. Crucially, the missing mask is not a failure; it is the signal that identifies the categorical component:

```python
is_categorical = numeric.isna()
categorical = df.loc[is_categorical, 'number']
```

`errors='ignore'` is usually less useful in modern pandas because it returns the original mixed column, so the caller does not get a clean numeric result. It also hides the parsing failure instead of making it explicit.

## 2.4 The two-component model

For a mixed column $x_1,\ldots,x_N$, define a numeric mask

$$
m_i =
\begin{cases}
1, & \text{if } x_i \text{ parses as a number},\\
0, & \text{otherwise}.
\end{cases}
$$

Then the numeric component is

$$
x_i^{\text{num}} =
\begin{cases}
\operatorname{parse}(x_i), & m_i=1,\\
\text{missing}, & m_i=0,
\end{cases}
$$

and the categorical component is

$$
x_i^{\text{cat}} =
\begin{cases}
x_i, & m_i=0,\\
\text{missing}, & m_i=1.
\end{cases}
$$

The two components partition the rows:

$$
N = N_{\text{num}} + N_{\text{cat}}.
$$

This is the key invariant. Extraction must not silently duplicate or lose rows. In the `number` column:

$$
891 = 752 + 139.
$$

There are 752 numeric rows and 139 rows whose value is `A`, with no row in both groups and no row in neither.

---

# 3. Foundations: dtype, semantics, identifiers, and missingness

## 3.1 Storage dtype versus semantic type

A **dtype** answers “how are the values stored and what operations does pandas allow?” A **semantic type** answers “what does the value mean?”

`Ticket` stored as `str` can contain:

- a booking prefix (`PC`),
- a numeric identifier (`17599`),
- a special code (`LINE`).

The dtype is one: string. The semantic types are multiple: category, identifier number, and special status. A model cannot interpret the semantic distinction unless we extract it.

A useful rule is:

> Inspect the values, not only `df.dtypes`.

`df.dtypes` would show `Cabin`, `Ticket`, and `number` as string-like in this environment. It would not tell us that `number` contains a genuine category `A`, that `Cabin` has multiple cabins in one cell, or that `Ticket` has inconsistent prefixes.

## 3.2 Nominal, ordinal, and numeric information

After extraction, ask what kind of information each component carries.

- **Numeric measurement:** a quantity where arithmetic has meaning, such as fare or age. Cabin number 85 is not a measurement in this sense; it is an identifier.
- **Ordinal:** categories with a meaningful order, such as deck A–G if the deck order is meaningful. The order must come from domain knowledge, not from the letters alone.
- **Nominal:** categories without an inherent order, such as a ticket prefix or a booking office.
- **Identifier:** a label used to look up or group records, such as a ticket number or passenger ID. Its numeric magnitude usually has no predictive meaning, and its cardinality can be high.

The same extracted string can be treated in more than one way. A cabin number can be an identifier, a frequency count, or an ordinal position on a deck. The choice must be justified by the problem, not by the fact that it contains digits.

## 3.3 Missing values have several meanings

A missing value can mean:

1. **Not recorded:** the information existed but was not captured.
2. **Not applicable:** the question does not apply to this record.
3. **Unknown:** the true value is unknown.
4. **Parse failure:** the information was recorded in a form the parser did not recognise.
5. **Structural missingness:** a value is absent by design, such as the numeric component of an alphabetic code.

These meanings need different handling. For example:

- `Cabin` is missing for 687 passengers. That may mean the cabin was not recorded, not that those passengers had no cabin. Dropping all 687 rows would discard 77% of the data and probably change the population.
- `number_numerical` is missing for the 139 rows whose `number` is `A`. That is structural: the numeric parse does not apply.
- `cabin_num` is missing for the 687 missing cabins plus four cabins with no digits (`D`, `D`, `D`, `T`). The four extra missing values are parse failures, not the same kind of missingness as an absent cabin.

Before filling or dropping, count and explain the missingness.

## 3.4 Cardinality

**Cardinality** is the number of distinct values in a column. High cardinality is a modelling concern because:

- one-hot encoding creates many columns;
- rare categories may have too few examples to estimate an effect;
- target encoding can overfit;
- identifiers can let a model memorise individual records;
- train/test splits may contain categories unseen during training.

In this dataset:

| Column | Non-missing rows | Distinct values | Distinct / non-missing |
|---|---:|---:|---:|
| `Cabin` | 204 | 147 | 72.1% |
| `Ticket` | 891 | 681 | 76.4% |
| `number` | 891 | 7 | 0.8% |

`Cabin` and `Ticket` are high-cardinality identifiers. `number` is low-cardinality and looks more like a categorical code or a small count. The raw uniqueness is not the only thing to check: after extracting a prefix, `ticket_cat` has 43 observed categories, still high but more manageable than 681 ticket values.

## 3.5 Identifiers are not automatically features

A feature should vary in a way that is plausibly related to the target and should generalise to new records.

A ticket number such as `21171` is an identifier. The fact that it is larger than `17599` does not mean the passenger is “more” of anything. Treating it as a continuous feature asks the model to find a linear or threshold relationship in an arbitrary numbering scheme. That is usually meaningless. A better representation is often:

- drop the identifier;
- use its frequency or group structure;
- extract a meaningful prefix;
- treat it as a categorical value with regularisation;
- use it only for grouping in validation, not as a predictor.

The same caution applies to cabin numbers. `C85` and `C123` share a deck letter; the numeric portion may encode a location, but the order of cabin numbers is not guaranteed to be a meaningful numeric scale.

---

# 4. The dataset: counts and missingness

## 4.1 Shape and columns

A **Rerun** gives:

```text
(891, 4)
Cabin         str
Ticket        str
number        str
Survived    int64
```

The source notebook is a reduced Titanic table, not the full 12-column Titanic dataset used on Day 32. It contains only:

- `Cabin`,
- `Ticket`,
- `number`,
- `Survived`.

There is no `Age`, `Fare`, `Sex`, or `Pclass` in this file. Do not import conclusions from the full Titanic dataset without checking that the columns exist.

## 4.2 Missingness before extraction

| Column | Missing | Non-missing | Missing fraction |
|---|---:|---:|---:|
| `Cabin` | 687 | 204 | 77.10% |
| `Ticket` | 0 | 891 | 0% |
| `number` | 0 | 891 | 0% |
| `Survived` | 0 | 891 | 0% |

The `Cabin` missing fraction is

$$
\frac{687}{891}=0.77104377\ldots \approx 77.10\%.
$$

That is a large fraction. Any feature derived from `Cabin` inherits this missingness unless it is paired with a missing indicator or an explicit “unknown cabin” category. A model that drops every row with a missing extracted cabin feature would be trained on only about 23% of the data.

## 4.3 Target balance

The target counts are:

| Survived | Count | Share |
|---:|---:|---:|
| 0 | 549 | 61.62% |
| 1 | 342 | 38.38% |

A majority-class classifier would achieve about 61.6% accuracy. Accuracy alone would therefore be misleading if these extracted features were used in a survival model. Later evaluation should include recall, precision, F1, and a confusion matrix.

## 4.4 `number` counts

The seven values and their counts are:

| `number` | Count | Numeric? |
|---|---:|---|
| A | 139 | No |
| 1 | 129 | Yes |
| 2 | 126 | Yes |
| 3 | 117 | Yes |
| 4 | 126 | Yes |
| 5 | 123 | Yes |
| 6 | 131 | Yes |

The numeric rows total

$$
129+126+117+126+123+131=752,
$$

and the categorical rows total 139, so

$$
752+139=891.
$$

The course plot is titled “Passengers travelling with”. The notebook does not establish exactly what `A` means or whether the numbers are counts, group codes, or something else. Treat the field as a low-cardinality mixed code until the data dictionary confirms its semantics. Do not invent a causal interpretation such as “A means travelling alone”.

---

# 5. Splitting the `number` column

## 5.1 The extraction code

The official and personal notebooks use the same idea:

```python
df['number_numerical'] = pd.to_numeric(
    df['number'],
    errors='coerce',
    downcast='integer'
)

df['number_categorical'] = np.where(
    df['number_numerical'].isnull(),
    df['number'],
    np.nan
)
```

Step by step:

1. `pd.to_numeric` attempts to parse each string.
2. `errors='coerce'` returns `NaN` for values that cannot be parsed.
3. `downcast='integer'` asks pandas to use a smaller integer dtype when possible.
4. `isnull()` creates a Boolean mask of failed parses.
5. `np.where(mask, original, NaN)` copies the original string for failed parses and places `NaN` in the numeric rows.
6. The two new columns are stored in the DataFrame.

The official head is:

| number | number_numerical | number_categorical |
|---|---:|---|
| 5 | 5.0 | NaN |
| 3 | 3.0 | NaN |
| 6 | 6.0 | NaN |
| 3 | 3.0 | NaN |
| A | NaN | A |

## 5.2 Why the numeric result is float

A **Rerun** gives `number_numerical` dtype `float64`, despite `downcast='integer'`. The reason is missingness: the column contains `NaN` for `A`, and ordinary NumPy integer arrays cannot represent `NaN`. The downcast hint cannot produce a plain integer extension dtype when missing values are present, so the result is float.

This is a common source of confusion:

- the values look like integers (`5.0`, `3.0`);
- the dtype is `float64`;
- the trailing `.0` is not a change in meaning.

If integer semantics are required, use a nullable integer dtype after deciding how to handle missing values, or keep a separate mask. Do not silently replace `NaN` with 0, because 0 may be a meaningful code.

## 5.3 The partition invariant

The masks are complementary:

$$
\text{numeric rows} = \{i: \text{number\_numerical}_i \text{ is not missing}\},
$$

$$
\text{categorical rows} = \{i: \text{number\_numerical}_i \text{ is missing}\}.
$$

Every row belongs to exactly one group. A **Rerun** confirms:

- numeric: 752 rows;
- categorical: 139 rows;
- total: 891 rows.

This is the check to run after any extraction. If the two counts do not sum to the original number of rows, the extraction has either duplicated or lost records.

## 5.4 What the categorical component contains

In this dataset, `number_categorical` is `A` for 139 rows and `NaN` for the other 752. It is effectively a binary flag for the presence of the code `A`. That does not mean every mixed column will behave this way. In another dataset, the categorical component could contain `A`, `B`, `Unknown`, and several text codes. The extraction code is general; the resulting cardinality is data-specific.

A **Rerun** gives:

| `number_categorical` | Count |
|---|---:|
| A | 139 |
| NaN | 752 |

If `A` is a genuine category, a missing indicator is not needed for the categorical part; the `NaN` values mean “not applicable because the row was numeric”. If `A` is actually a missing-value code, then the interpretation changes. That is a data-dictionary question, not a pandas question.

## 5.5 Should `number_numerical` be used as a number?

Not automatically. The values 1–6 might be counts of companions, group codes, or another ordinal scheme. A model may treat them as numeric if the data dictionary says they are counts. If they are arbitrary codes, treat them as categorical.

A useful diagnostic is to compare target rates by value:

```python
df.groupby('number', observed=True)['Survived'].agg(['count', 'mean'])
```

This is an association check, not proof of a useful feature. The result can guide representation choices, but it must not be computed using the test target during model selection.

## 5.6 Why `np.where` is used

`np.where(condition, a, b)` returns an array. It does not align by pandas index; it combines the inputs positionally. Here all inputs have the same length and order, so the result is correct. If the Series were reordered independently, positional combination could misalign values.

An index-aligned alternative is:

```python
df['number_categorical'] = df['number'].where(
    df['number_numerical'].isnull()
)
```

`Series.where(cond)` keeps values where the condition is true and replaces the rest with `NaN` by default. This operates through pandas alignment rules. Either approach works for this simple ordered DataFrame; the important point is to know whether the operation is positional or index-aligned.

---

# 6. Extracting the cabin information

## 6.1 The raw cabin forms

The official notebook shows values such as:

```text
NaN
C85
C123
E46
G6
C103
D56
A6
C23 C25 C27
B78
D33
F G73
D
T
```

There are four structural patterns:

1. **Missing** — `NaN`.
2. **Deck letter + digits** — `C85`, `E46`, `B78`.
3. **Several cabins** — `C23 C25 C27`, `F G73`, `B57 B59 B63 B66`.
4. **A deck letter without digits** — `D`, `T`.

A full treatment might extract the deck, the first cabin number, all cabin numbers, and the number of cabins. The course extracts two components: the first numeric run and the first character.

## 6.2 The extraction code

```python
df['cabin_num'] = df['Cabin'].str.extract(r'(\d+)')
df['cabin_cat'] = df['Cabin'].str[0]
```

`str.extract(r'(\d+)')` uses a regular expression:

- `\d` means a digit;
- `+` means one or more digits;
- parentheses create a capture group;
- the first match in each string is returned.

`str[0]` takes the first character. For `C85`, that is `C`. For `C23 C25 C27`, that is also `C`. For a missing cabin, both operations return missing.

The raw-string prefix `r` is important. Without it, Python treats `\d` as an escape sequence and may emit a `SyntaxWarning` in modern Python. The local notebook's non-raw `'(\d+)'` produced exactly that warning in this environment. The pattern happened to work because Python preserved the backslash for compatibility, but the warning is a signal to write `r'(\d+)'` and avoid future ambiguity.

## 6.3 A worked mapping

| Cabin | `cabin_num` | `cabin_cat` | Observation |
|---|---|---|---|
| `NaN` | NaN | NaN | no cabin recorded |
| `C85` | `'85'` | `C` | one cabin |
| `C123` | `'123'` | `C` | one cabin |
| `C23 C25 C27` | `'23'` | `C` | only the first number is kept |
| `F G73` | `'73'` | `F` | spaces before the number |
| `D` | NaN | `D` | letter-only cabin |
| `T` | NaN | `T` | rare letter code |

`cabin_num` is extracted as text, not converted to a numeric dtype in the course. In a modelling pipeline, decide whether to convert it. Conversion is appropriate only if the numeric cabin position has a defensible meaning; otherwise treat it as an identifier or categorical value.

## 6.4 Counts after extraction

A **Rerun** gives:

| Quantity | Count |
|---|---:|
| Rows | 891 |
| `Cabin` non-missing | 204 |
| `cabin_cat` non-missing | 204 |
| `cabin_num` non-missing | 200 |
| `cabin_num` missing because cabin missing | 687 |
| `cabin_num` missing despite cabin present | 4 |
| `cabin_num` distinct values (non-missing) | 92 |

The four rows with a cabin but no digits are the letter-only cabins `D`, `D`, `D`, and `T`. Their numeric component is structurally missing. The categorical component is present.

The cabin category counts are:

| `cabin_cat` | Count |
|---|---:|
| NaN | 687 |
| C | 59 |
| B | 47 |
| D | 33 |
| E | 32 |
| A | 15 |
| F | 13 |
| G | 4 |
| T | 1 |

The counts sum to 891. The distribution is highly unbalanced, and `T` is a rare category. A one-hot encoder with `handle_unknown='ignore'` will produce a column for `T` in training, but a new dataset may contain other rare letters or a missing value.

## 6.5 What the extraction loses

`C23 C25 C27` becomes cabin number `23` and category `C`. The other two cabin numbers, 25 and 27, disappear. This is a lossy transformation. If the number of cabins or the set of locations matters, extract all numbers:

```python
cabin_numbers = df['Cabin'].str.findall(r'\d+')
cabin_count = cabin_numbers.str.len()
```

This is a **Supplementary** extension, not part of the course notebook. It illustrates the important point: the first match is not always the only useful information. The course's extraction is a reasonable first pass for a deck-level feature, but it is not a complete representation of multi-cabin records.

## 6.6 Is `cabin_num` an ordinal number?

A cabin number such as 85 is written with digits, but that does not make it a measurement. The number may encode a location, a sequential assignment, or an internal identifier. The order of cabin numbers is not guaranteed to correspond to distance, quality, or survival risk.

Before treating `cabin_num` as numeric, ask:

- Does a one-unit increase have a consistent meaning?
- Are 85 and 86 more similar than 85 and 185?
- Is the number assigned by a physical layout or by an arbitrary booking sequence?

If the answer is not clear, treat the cabin number as categorical, or use the deck letter plus a missing indicator. A decision tree can split a numeric code, but it can also overfit arbitrary numeric order.

## 6.7 Missingness and the missing indicator

Because `Cabin` is missing for 77.1% of rows, “cabin not recorded” may itself be informative. A model may learn that missing cabin is associated with a different survival rate, but that association can be confounded by passenger class, ticket type, or other unobserved variables.

A safe representation is:

- `cabin_cat` with an explicit `Unknown` category;
- `cabin_num` as a categorical or numeric value plus a missing indicator;
- a separate `cabin_missing` flag;
- no mean imputation of a cabin identifier.

Never replace a missing cabin number with the mean. The mean of an identifier is not a meaningful cabin.

---

# 7. Extracting the ticket information

## 7.1 The raw ticket forms

The official values include:

```text
A/5 21171
PC 17599
STON/O2. 3101282
113803
373450
PP 9549
C.A. 24579
LINE
```

The common structure is:

- an optional prefix made of letters and punctuation,
- a space,
- a final token that is usually numeric.

A numeric-only ticket has no prefix. `LINE` is a special word with no numeric token.

## 7.2 The extraction code

```python
# last token as a number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(
    df['ticket_num'],
    errors='coerce',
    downcast='integer'
)

# first token as a category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(
    df['ticket_cat'].str.isdigit(),
    np.nan,
    df['ticket_cat']
)
```

`str.split()` without an argument splits on whitespace. For `A/5 21171`:

- first token: `A/5`;
- last token: `21171`.

For `113803`:

- first and last token: `113803`;
- the last token parses as a number;
- the first token is digits, so `str.isdigit()` is true and the category is set to `NaN`.

For `LINE`:

- first and last token: `LINE`;
- numeric parsing returns `NaN`;
- the category remains `LINE`.

The `apply(lambda s: ...)` form calls the function once per row. It is readable for a small dataset but slower than vectorised string methods. A vectorised alternative is:

```python
tokens = df['Ticket'].str.split()
df['ticket_num'] = pd.to_numeric(tokens.str[-1], errors='coerce')
df['ticket_cat'] = tokens.str[0].where(~tokens.str[0].str.isdigit())
```

Both versions operate on strings; the vectorised version avoids a Python-level loop.

## 7.3 A worked mapping

| Ticket | `ticket_num` | `ticket_cat` | Observation |
|---|---|---:|---|
| `A/5 21171` | 21171.0 | `A/5` | prefix and number |
| `PC 17599` | 17599.0 | `PC` | prefix and number |
| `STON/O2. 3101282` | 3101282.0 | `STON/O2.` | longer prefix |
| `113803` | 113803.0 | NaN | numeric-only ticket |
| `373450` | 373450.0 | NaN | numeric-only ticket |
| `PP 9549` | 9549.0 | `PP` | prefix and number |
| `LINE` | NaN | `LINE` | special word |

## 7.4 Counts after extraction

A **Rerun** gives:

| Quantity | Count |
|---|---:|
| Rows | 891 |
| `ticket_num` non-missing | 887 |
| `ticket_num` missing | 4 |
| `ticket_num` distinct values | 678 |
| `ticket_cat` non-missing | 230 |
| `ticket_cat` missing | 661 |
| `ticket_cat` distinct values | 43 |
| Duplicate ticket strings | 210 rows |
| Distinct ticket strings | 681 |

All four missing numeric values come from the special ticket `LINE`. The first token for 661 rows is numeric, so those rows have no categorical prefix. The 230 prefixed rows have 43 raw prefixes. The numeric ticket component has 678 distinct values among 887 non-missing rows, which is very high cardinality.

## 7.5 Why the numeric ticket value is an identifier

A ticket number such as 21171 is not a measurement. The difference between ticket 21171 and 21172 is not a meaningful one-unit change in a quantity. The number identifies a booking or a ticket block. Treating it as continuous asks the model to find an ordering in an arbitrary numbering scheme.

Useful alternatives include:

- drop the numeric ID;
- use frequency encoding of the ticket string;
- extract a meaningful prefix;
- group rare tickets;
- use the ticket as a grouping key for validation, not as a predictor.

The prefix may be more meaningful than the number because it can represent a booking agent, a port, or a ticket series. However, the raw prefixes are inconsistent: `PC`, `C.A.`, `CA`, `CA.`, `A/5`, `A/5.`, `A./5.`, and `A.5.` are all present. Without normalisation, the model sees many near-duplicate categories with few observations each.

## 7.6 Duplicate tickets and group leakage

`Ticket` is not unique. A **Rerun** finds 210 duplicate rows and 681 distinct tickets among 891 rows. Passengers sharing a ticket are likely travelling together, so they may share survival outcomes through common circumstances.

If the train/test split is random at the row level, the same ticket can appear in both training and test. The model can then learn group-specific patterns that would not generalise to a new group. This is **group leakage**. For a group-aware evaluation, split by ticket or by cabin group:

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['Ticket']))
```

This is a **Supplementary** modelling consideration; the course notebook does not train a model. It belongs in the notes because the extraction creates a natural group key and because row-level random splitting is not always appropriate.

## 7.7 Prefix normalisation

The official `ticket_cat` unique output contains variants such as:

```text
A/5, A/5., A./5., A.5.
C.A., CA, CA.
SC/Paris, SC/PARIS
SOTON/OQ, SOTON/O.Q.
```

These are probably the same or related booking prefixes written in different punctuation styles. A model that treats each variant as a separate category will spread the evidence across many rare levels. A **Supplementary** normalisation step could:

1. convert to upper case;
2. remove punctuation or standardise separators;
3. map known variants to a canonical prefix;
4. group rare prefixes into `Other`;
5. fit the mapping on training data only if it is data-dependent.

The course does not specify a canonical mapping. Do not invent one without the data dictionary. The safe conclusion is that the raw prefix is noisy and needs inspection before use.

---

# 8. String and regular-expression mechanics

## 8.1 The `.str` accessor

Pandas Series with string dtype provide a `.str` accessor:

```python
df['Cabin'].str[0]
df['Cabin'].str.extract(r'(\d+)')
df['Ticket'].str.split()
```

The accessor applies a string method element by element. It is different from calling Python string methods directly on the Series:

```python
df['Cabin'][0][0]      # first character of the first value only
df['Cabin'].str[0]     # first character of every value
```

The `.str` accessor propagates missing values. If `Cabin` is `NaN`, `str[0]` returns `NaN`; it does not raise an exception. This is one reason vectorised string methods are convenient for extraction.

The accessor is not a general replacement for every Python operation. Some methods return a Series; some return a DataFrame; some have `expand` arguments. Read the return type before assigning it to a column.

## 8.2 What a regular expression is doing

A regular expression is a pattern language for matching text. In:

```python
r'(\d+)'
```

the pieces are:

- `\d` — one digit character;
- `+` — one or more repetitions;
- `(...)` — a **capture group**, which tells `str.extract` what part to return.

For `C23 C25 C27`, the regex engine scans from the left and finds the first run of digits, `23`. For `F G73`, it finds `73`. For `D`, it finds nothing and returns `NaN`.

The parentheses are not cosmetic. `str.extract` without a capture group raises an error because it does not know which part of the match to return. With multiple groups, it returns one column per group.

```python
# One capture group -> one column
df['Cabin'].str.extract(r'(\d+)')

# Two groups -> two columns named 0 and 1
df['Cabin'].str.extract(r'([A-Z])(\d+)')

# All matches, not only the first
df['Cabin'].str.findall(r'\d+')
```

`str.extract` returns only the **first** match. This is why `C23 C25 C27` becomes `23` and loses 25 and 27. If all matches matter, use `str.findall` and aggregate the resulting lists.

## 8.3 Raw strings

Python strings use backslashes for escape sequences. In a normal string, `\d` is not a recognised Python escape, so modern Python may emit a `SyntaxWarning`:

```text
SyntaxWarning: "\d" is an invalid escape sequence.
```

The local notebook produced exactly this warning in cell 6. The official notebook used the same non-raw pattern in an older environment where the warning was not shown. The semantically safe form is:

```python
r'(\d+)'
```

The `r` prefix makes the backslash literal for the regex engine. Use raw strings for every regular expression unless there is a specific reason not to. This is not only a style issue: future Python versions may change how invalid escapes are handled.

## 8.4 `str[0]`, `str.split`, and `str[-1]`

These operations have different jobs:

| Operation | Meaning | Example on `A/5 21171` | Result |
|---|---|---|---|
| `str[0]` | first character | first character | `A` |
| `str.split()` | split on whitespace | `['A/5', '21171']` | list-like |
| `.str[-1]` | last character of each string | last character | `1` |
| `str.split().str[-1]` | last token | last token | `21171` |
| `str.split().str[0]` | first token | first token | `A/5` |

A common confusion is between `str[0]` (first character) and `str.split().str[0]` (first token). For a cabin such as `C85`, `str[0]` gives the deck `C`. For a ticket such as `PC 17599`, the first character is `P`, but the first token is `PC`. The course uses the token for the ticket category and the character for the cabin category because those choices match the observed structure.

## 8.5 `str.isdigit` is not `pd.to_numeric`

`str.isdigit()` checks whether every character is a digit. It is appropriate for deciding whether a ticket's first token is a numeric-only token:

```python
tokens = df['Ticket'].str.split()
first = tokens.str[0]
is_numeric_token = first.str.isdigit()
```

But `str.isdigit()` is not a general numeric parser. It rejects:

- negative numbers (`-5`),
- decimals (`5.0`),
- numbers with spaces or thousands separators,
- scientific notation.

For general numeric parsing, use `pd.to_numeric` and decide an error policy. `isdigit()` is a string classification test, not a conversion.

## 8.6 `np.where` and index alignment

The ticket category code is:

```python
df['ticket_cat'] = np.where(
    df['ticket_cat'].str.isdigit(),
    np.nan,
    df['ticket_cat']
)
```

`np.where(condition, x, y)` returns an array, choosing `x` where the condition is true and `y` elsewhere. It combines values **positionally**, not by index label. If the inputs are aligned Series with the same order, this is fine. If one Series was reindexed, the values could align incorrectly.

The pandas alternative:

```python
df['ticket_cat'] = df['ticket_cat'].where(
    ~df['ticket_cat'].str.isdigit()
)
```

`Series.where(condition)` keeps values where the condition is true and replaces the rest with `NaN`, using pandas alignment. Both are correct here; know which semantics the code uses.

## 8.7 Vectorised methods versus `apply`

The course uses:

```python
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
```

This calls a Python function once per row. It is readable and works for a small table. A vectorised version is:

```python
tokens = df['Ticket'].str.split()
df['ticket_num'] = pd.to_numeric(tokens.str[-1], errors='coerce')
```

The vectorised version avoids a Python loop and handles missing values through the string accessor. For 891 rows the difference is negligible; for millions of rows it matters. The result must still be validated: vectorisation changes syntax, not semantics.

## 8.8 Execution flow

The extraction pipeline for one column is:

```text
Raw string values
      |
      v
Parse / match / split
      |
      +-----------------------+
      |                       |
      v                       v
Numeric component       Categorical component
      |                       |
      v                       v
Missing where parse     Missing where numeric parse
failed                  succeeded
      |                       |
      +-----------+-----------+
                  |
                  v
        Row-count validation
                  |
                  v
        Decide feature representation
```

For `number`:

```text
"5"  -> parse 5.0  -> numeric 5.0, categorical NaN
"A"  -> parse NaN  -> numeric NaN, categorical "A"
```

For `Cabin`:

```text
"C85"          -> numeric "85", categorical "C"
"C23 C25 C27"  -> numeric "23", categorical "C"   (other numbers lost)
"F G73"        -> numeric "73", categorical "F"
"D"            -> numeric NaN, categorical "D"
NaN            -> numeric NaN, categorical NaN
```

For `Ticket`:

```text
"A/5 21171"   -> ticket_num 21171.0, ticket_cat "A/5"
"PC 17599"    -> ticket_num 17599.0, ticket_cat "PC"
"113803"      -> ticket_num 113803.0, ticket_cat NaN
"LINE"        -> ticket_num NaN, ticket_cat "LINE"
```

The diagrams show the important principle: **the two outputs are not independent columns to be filled with arbitrary defaults.** Their missing patterns encode which branch each row took.

## 8.9 Edge cases to check

Before trusting an extraction, inspect:

- values with no match (`D`, `LINE`);
- values with multiple matches (`C23 C25 C27`);
- values with unusual separators;
- values that are numeric but should not be treated as magnitudes;
- values that are missing before extraction;
- values that parse successfully but have leading zeros or very large integers;
- categories that differ only by case or punctuation.

A `head()` shows the common cases. `value_counts()`, `unique()`, and explicit masks show the edge cases.

---

# 9. Validation: did extraction preserve the data?

## 9.1 The post-extraction table

A **Rerun** of the course code gives these non-missing counts:

| Column | Non-missing | Missing |
|---|---:|---:|
| `Cabin` | 204 | 687 |
| `Ticket` | 891 | 0 |
| `number` | 891 | 0 |
| `number_numerical` | 752 | 139 |
| `number_categorical` | 139 | 752 |
| `cabin_num` | 200 | 691 |
| `cabin_cat` | 204 | 687 |
| `ticket_num` | 887 | 4 |
| `ticket_cat` | 230 | 661 |
| `Survived` | 891 | 0 |

The final DataFrame has 891 rows and 10 columns: the four original columns plus six extracted columns.

## 9.2 The arithmetic checks

The following identities should hold:

$$
752+139=891,
$$

$$
200+4=204,
$$

$$
204+687=891,
$$

$$
230+661=891,
$$

$$
887+4=891.
$$

Interpretation:

- `number`: 752 numeric rows plus 139 categorical rows partition all rows.
- `cabin_cat`: 204 observed cabins plus 687 missing cabins partition all rows.
- `cabin_num`: 200 cabins with a numeric part plus 4 cabins without digits equals the 204 observed cabins.
- `ticket_cat`: 230 prefixed tickets plus 661 numeric-only tickets partition all rows.
- `ticket_num`: 887 parseable final tokens plus 4 `LINE` rows partition all rows.

These are stronger checks than “the notebook ran without an error”. A silent extraction bug often appears as a wrong count, not an exception.

## 9.3 Original missingness versus created missingness

It is essential to separate:

1. **Pre-existing missingness:** `Cabin` is missing in 687 rows.
2. **Parse-created missingness:** `cabin_num` gains four additional missing values because `D` and `T` contain no digits.
3. **Structural missingness:** `number_numerical` is missing for `A`; `ticket_cat` is missing for numeric-only tickets.

These are not interchangeable. If `cabin_num` is missing, it could mean:

- the cabin was not recorded (687 cases),
- the cabin was recorded but had no digits (4 cases).

If `Cabin` is present but `cabin_num` is missing, the correct action is not “impute the mean”; it is “recognise a different cabin form”. A missing indicator plus the cabin category is more faithful than a fabricated number.

## 9.4 Example assertions

```python
n = len(df)
assert n == 891

assert (
    df['number_numerical'].notna().sum()
    + df['number_categorical'].notna().sum()
) == n

assert (
    df['cabin_num'].notna().sum()
    + (df['Cabin'].notna() & df['cabin_num'].isna()).sum()
) == df['Cabin'].notna().sum()

assert (
    df['ticket_num'].notna().sum()
    + df['ticket_num'].isna().sum()
) == n
```

The cabin assertion says: numeric cabin parts plus cabins that were present but had no numeric part equal all observed cabins. It deliberately excludes the 687 originally missing cabins.

## 9.5 Dtypes after extraction

A **Rerun** gives:

```text
number_numerical      float64
number_categorical        str
cabin_num                 str
cabin_cat                 str
ticket_num            float64
ticket_cat                str
```

Two consequences:

- `cabin_num` is still text. The output `85` is a string, not an integer. This is easy to miss because it looks numeric.
- `ticket_num` is float because it contains `NaN` for `LINE`, despite `downcast='integer'`.
- The categorical columns use pandas 3.0 string dtype in this environment; the saved official output may show object dtype from an older pandas. The semantic category values are the same.

If a model requires numeric `cabin_num`, convert explicitly and decide how to represent missing values. If a model requires categorical strings, keep them as strings and use an encoder.

## 9.6 Do not validate only with `head()`

`df.head()` confirms that the first few rows look plausible. It does not confirm:

- that all rows were covered;
- that rare values were parsed;
- that missingness was preserved;
- that no row was duplicated;
- that the category vocabulary is consistent.

Validation needs counts, masks, and invariants. The course notebook stops after display; the notes add the checks because a production workflow cannot rely on visual inspection of five rows.

---

# 10. Official notebook walkthrough

## 10.1 Cell map

The official `Untitled.ipynb` has 14 cells:

| Cell | Type | Main code | Purpose / output |
|---:|---|---|---|
| 0 | code | `import numpy as np`, `import pandas as pd` | imports |
| 1 | code | `df = pd.read_csv('titanic.csv')` | loads 891×4 data |
| 2 | code | `df.head()` | first five rows |
| 3 | code | `df['number'].unique()` | `array(['5', '3', '6', 'A', '2', '1', '4'], dtype=object)` |
| 4 | code | `df['number'].value_counts().plot.bar()` | bar chart of `number` counts |
| 5 | code | `pd.to_numeric(..., errors='coerce', downcast='integer')` | creates `number_numerical` |
| 6 | code | `np.where(...)`, `df.head()` | creates `number_categorical`, shows first rows |
| 7 | code | `df['Cabin'].unique()` | 147 non-missing values plus `NaN` |
| 8 | code | `df['Ticket'].unique()` | 681 ticket strings; very long output |
| 9 | code | `str.extract('(\d+)')`, `str[0]`, `head()` | creates `cabin_num`, `cabin_cat` |
| 10 | code | `df['cabin_cat'].value_counts().plot(kind='bar')` | bar chart of deck counts |
| 11 | code | ticket split, `to_numeric`, `np.where`, `head(20)` | creates `ticket_num`, `ticket_cat` |
| 12 | code | `df['ticket_cat'].unique()` | 43 non-missing prefix values plus `NaN` |
| 13 | code | empty | no operation |

## 10.2 What the official notebook does well

- It starts with values, not with an assumption about dtype.
- It shows a quick visual count for `number` and `cabin_cat`.
- It uses `errors='coerce'` so parsing failures become visible missing values.
- It separates numeric and categorical components rather than forcing one representation.
- It preserves the original columns, so the extraction can be audited.
- It demonstrates the same pattern on three different structures:
  - a simple mixed code (`number`),
  - a structured identifier (`Cabin`),
  - an optional-prefix identifier (`Ticket`).

## 10.3 Source-fidelity limitations

The official notebook is a demonstration, not a complete preprocessing pipeline. Its limitations include:

1. **No train/test split** and no model. There is no evaluation of whether the extracted features help.
2. **No normalisation of ticket prefixes.** `C.A.`, `CA`, `CA.`, `A/5`, `A/5.`, and `A./5.` remain separate categories.
3. **No handling of multi-cabin strings.** Only the first number is extracted from `C23 C25 C27`.
4. **No missing indicator** for `Cabin`; the 77.1% missingness is left for the modeller to notice.
5. **No conversion of `cabin_num` to numeric**. It is still text after `str.extract`.
6. **No discussion of identifier semantics.** `ticket_num` is displayed as a number, but its numeric magnitude is not meaningful.
7. **No category grouping.** Rare prefixes and rare deck letters remain rare.
8. **No output validation.** The notebook does not check the row-count invariants.
9. **Non-raw regex** in cell 9. It worked in the saved environment but produces a `SyntaxWarning` in the current environment.
10. **Version-dependent display.** The saved `number.unique()` is an object array; pandas 3.0.5 returns a `StringArray`. The values are the same; the repr is not.

These are not reasons to dismiss the notebook. They are the boundary between a teaching demonstration and a production feature-engineering process.

## 10.4 Why `ticket_cat.unique()` is useful

The official cell 12 output reveals the raw category vocabulary:

```text
A/5, PC, STON/O2., PP, A/5., C.A., A./5., SC/Paris,
S.C./A.4., A/4., CA, S.P., S.O.C., SO/C, W./C.,
SOTON/OQ, W.E.P., STON/O, A4., C, SOTON/O.Q.,
SC/PARIS, S.O.P., A.5., Fa, CA., LINE, F.C.C., W/C,
SW/PP, SCO/W, P/PP, SC, SC/AH, A/S, A/4, WE/P,
S.W./PP, S.O./P.P., F.C., SOTON/O2, S.C./PARIS,
C.A./SOTON
```

This output is more informative than a `head()` because it shows the near-duplicates and rare forms. It also shows why a simple one-hot encoder is risky: many categories occur only a few times.

## 10.5 Why the official `Cabin.unique()` output is long

`Cabin` has 147 non-missing distinct strings, so the unique array has 148 entries including `NaN`. The notebook prints all of them. This is a legitimate exploratory command for a small dataset, but a production report would use a frequency table, not a raw array of 148 values.

The array includes:

- ordinary deck-plus-number values;
- multi-cabin strings;
- letter-only cabins `D` and `T`;
- `NaN`.

The extracted `cabin_cat` collapses these into eight non-missing deck categories plus missing. That is a substantial dimensionality reduction, but it is also a loss of information.

---

# 11. Personal notebook walkthrough

## 11.1 Cell map

The personal `mixedata.ipynb` has 11 cells:

| Cell | Type | Main code | Purpose / note |
|---:|---|---|---|
| 0 | Markdown | `# handle Mixed_variables` | notebook title |
| 1 | code | imports and `pd.read_csv(r'E:\MACHINE_LEARNING\DAY_33\DAtasets\titanic.csv')` | loads the local file |
| 2 | code | `df.head(10)` | shows more rows than the official notebook |
| 3 | code | `df['number'].unique()` | checks mixed values |
| 4 | code | `pd.to_numeric(..., errors='coerce', downcast='integer')` | creates `number_numerical`; no display output |
| 5 | code | `np.where(...)`, `df.head()` | creates `number_categorical` |
| 6 | code | `df['Cabin'].unique()`, then cabin extraction | combines inspection and extraction |
| 7 | code | `df.head()` | displays extracted cabin columns |
| 8 | code | `df['cabin_cat'].value_counts().plot(kind='bar')` | deck counts |
| 9 | code | ticket split, `to_numeric`, `np.where` | creates `ticket_num`, `ticket_cat` |
| 10 | code | `df['ticket_num'].value_counts().plot(kind='bar')` | plots all ticket-number counts |

## 11.2 Differences from the official notebook

- The personal notebook uses an **absolute Windows path** to the dataset. It runs on this machine but is not portable. A relative path such as `DAtasets/titanic.csv` would be better if the notebook is always launched from `DAY_33`.
- The personal notebook has an extra `df.head(10)` early on, which is useful for seeing more raw rows.
- It does not plot the `number` counts. The official notebook does.
- Cell 6 combines `df['Cabin'].unique()` with the two extraction assignments. Because the last statement is an assignment, the unique output is not displayed; the cell's saved output is the regex `SyntaxWarning`, not the cabin inventory. Splitting exploration and transformation into separate cells would make the output visible.
- It does not display `ticket_cat.unique()` or a ticket-category frequency table. The official notebook does.
- It plots `ticket_num.value_counts()`, which has hundreds of bars because ticket numbers are high-cardinality identifiers. The plot is not informative. A top-10 or top-20 frequency plot would be more useful.
- It produces a `SyntaxWarning` for the non-raw regex. Use `r'(\d+)'` in future edits.
- It uses the same extraction logic and preserves the original columns.

## 11.3 The dataset path and README correction

The notebook reads:

```python
pd.read_csv(r'E:\MACHINE_LEARNING\DAY_33\DAtasets\titanic.csv')
```

The actual directory is `DAtasets` with a capital `D` and lowercase `a` in `Atasets`. The earlier README wording described the folder as `Datasets`; the documentation pass has corrected it to `DAtasets`. The dataset folder should not be renamed casually because the notebook's absolute path depends on it.

A better long-term layout is:

```text
DAY_33/
├── mixedata.ipynb
├── notes.md
├── README.md
└── DAtasets/
    └── titanic.csv
```

or, after a deliberate migration:

```text
DAY_33/
├── mixedata.ipynb
├── notes.md
├── README.md
└── datasets/
    └── titanic.csv
```

The second layout is cleaner, but it requires updating the notebook path and README together. This documentation pass corrected the README without moving the dataset or rewriting the notebook.

## 11.4 What the personal notebook demonstrates

The personal notebook follows the same conceptual sequence:

1. inspect the mixed column;
2. extract numeric values with a coercive parser;
3. extract the categorical complement;
4. extract cabin number and deck;
5. extract ticket number and prefix;
6. visualise the resulting deck and ticket-number distributions.

It is a practical implementation, not a complete modelling notebook. Its strongest parts are the direct use of the raw data and the visible comparisons. Its biggest gaps are the missing ticket-category inspection, the unreadable ticket-number plot, the non-raw regex, and the untested path portability.

---

# 12. Interpreting the extracted features

## 12.1 What the counts establish

The counts establish coverage and structure:

- `number` is split without loss: 752 numeric + 139 `A`.
- `Cabin` is observed for 204 rows; 687 rows have no recorded cabin.
- Four observed cabins have no digits and therefore no numeric component.
- `Ticket` has 887 parseable numeric endings and four `LINE` values.
- Only 230 tickets have a non-numeric prefix; 661 are numeric-only.
- `Cabin` and `Ticket` are high-cardinality identifiers.

These are data facts. They are not evidence that the features predict survival.

## 12.2 What the counts do not establish

The notebook does not show:

- whether `number`, `cabin_cat`, `cabin_num`, `ticket_cat`, or `ticket_num` improves a model;
- whether the relationship is causal;
- whether missing cabin is informative after controlling for class or fare;
- whether ticket prefixes are stable across time or ports;
- whether the categories will appear in new data;
- whether the extraction generalises beyond this 891-row file.

A feature can be interesting and still be useless or harmful in a model. Feature extraction creates candidates; validation decides whether they are useful.

## 12.3 The class balance context

The target has 342 positive and 549 negative cases. A model that always predicts “not survived” gets 61.6% accuracy. Any later evaluation should compare against that baseline and should not rely on accuracy alone.

For a small dataset, a difference of a few percentage points can be one or two rows. The Day 32 lesson showed that a 0.006993 accuracy difference was exactly one test prediction. Day 33 has no model, but the same caution applies when someone later experiments with these features: report counts and uncertainty, not only a single score.

## 12.4 Identifier semantics

`ticket_num` and `cabin_num` look numeric because they contain digits. Their values are labels, not measurements. A model that uses them as continuous variables can:

- fit arbitrary thresholds;
- memorise particular ticket ranges;
- learn patterns that do not generalise;
- give false importance to an identifier.

Before using such a column as numeric, test whether its magnitude has a defensible meaning. If not, use a categorical, frequency, or group representation, or drop it.

---

# 13. Choosing a feature representation

## 13.1 Feature-by-feature decisions

| Extracted feature | Likely semantic type | Reasonable representation | Main risk |
|---|---|---|---|
| `number_numerical` | count or code | numeric if counts; categorical if arbitrary codes | assuming a false numeric scale |
| `number_categorical` | nominal code | one-hot / binary indicator | unseen codes; `A` may be a missing code |
| `cabin_num` | identifier / location code | categorical, frequency, or binned; rarely continuous | arbitrary numeric order |
| `cabin_cat` | nominal or ordinal deck | one-hot; ordinal only with domain order | rare categories and missingness |
| `ticket_num` | identifier | frequency, group key, or drop | high cardinality and overfitting |
| `ticket_cat` | nominal booking prefix | normalise, group rare, then one-hot/target encode | near-duplicate categories |
| `Cabin` missing flag | missingness indicator | binary | confounding by class/other variables |
| `Ticket` group size | group structure | numeric count or frequency | leakage if computed with target |

## 13.2 Encoding choices

**One-hot encoding** creates one binary column per category. It is transparent and works well for low-cardinality nominal variables. It can create many columns for `ticket_cat` and `cabin_num`.

**Ordinal encoding** assigns an integer to each category. It is appropriate only when the order is real. Deck letters might be ordered by deck, but that must come from domain knowledge; ticket prefixes have no obvious order.

**Frequency encoding** replaces a category with its count or share in the training data. It handles high cardinality but can leak if computed on the full dataset. It also collapses different categories with the same frequency.

**Target encoding** replaces a category with an estimate of the target mean for that category. It can be powerful for high-cardinality features but is prone to leakage and overfitting. Use out-of-fold estimates and smoothing on training data only.

A smoothed target encoding can be written as:

$$
\hat{p}_c =
\frac{\sum_{i:c_i=c} y_i + m\bar{y}}{n_c + m},
$$

where:

- $c$ is a category;
- $n_c$ is the number of training rows in category $c$;
- $\sum y_i$ is the number of positive outcomes in that category;
- $\bar{y}$ is the global training positive rate;
- $m$ is a smoothing strength.

The formula shrinks rare categories toward the global mean. The smoothing parameter must be chosen inside cross-validation. The encoding must be fitted on training folds only; computing it on the full training set before cross-validation leaks target information into the validation folds.

For this dataset, target encoding is a **Supplementary** idea, not something the official notebook does. It is included because high-cardinality `ticket_cat` naturally raises the question.

## 13.3 Missing indicators

A missing indicator can be added for:

- `cabin_cat` / `Cabin` missingness;
- `cabin_num` parse failure;
- `ticket_num` failure (`LINE`);
- `ticket_cat` absence (numeric-only ticket).

Do not add indicators mechanically. Ask whether the missingness mechanism is plausibly informative and whether it is stable in new data. A missing indicator that is constant in production is useless; a missing indicator that encodes a data-entry change can be misleading.

## 13.4 Model-family considerations

- **Linear models** need explicit numeric encodings, scaling for continuous features, and careful handling of high-cardinality one-hot columns. Regularisation is important.
- **Tree-based models** can split categorical codes and may handle non-linearities, but they can also overfit identifiers and rare categories. High-cardinality integer codes are especially risky because trees can isolate individual values.
- **Regularised linear models** with one-hot encoding can work for moderate cardinality; rare categories should be grouped.
- **Gradient boosting** can handle mixed feature types but still benefits from normalised categories and group-aware validation.
- **Neural networks** may use embeddings for high-cardinality categories, but the dataset is far too small to justify that complexity here.

## 13.5 What to drop

Candidates for dropping or not using directly:

- `ticket_num` as a continuous feature;
- `cabin_num` as a continuous feature without a domain justification;
- raw `Ticket` and raw `Cabin` after extraction if they are not needed for grouping;
- rare ticket prefixes with only one or two observations;
- any feature derived from the target after the split.

The original columns should generally be kept in an audit table even if they are not passed to the model. They let you trace an extracted value back to its source.

---

# 14. Connection to the machine-learning workflow

## 14.1 Where extraction belongs

A practical sequence is:

```text
Raw data
   |
   v
Data understanding: inspect values, missingness, uniqueness
   |
   v
Deterministic extraction: parse strings into components
   |
   v
Train/test split (or cross-validation folds)
   |
   v
Fit encoders/scalers on training data only
   |
   v
Train model
   |
   v
Evaluate on held-out data
   |
   v
Error analysis and feature review
```

The extraction itself is usually deterministic and can be done before the split. Encoding vocabularies, target encodings, frequency counts, and imputation statistics are **learned** and must be fitted on training data only.

## 14.2 Train/test consistency

The same parsing function must be applied to training and test data. If the function changes between training and serving, the model receives a different feature definition. Keep the extraction logic in one reusable function or transformer rather than copying ad hoc cells.

A minimal function might return a DataFrame with the same columns:

```python
def extract_titanic_features(frame):
    out = frame.copy()

    out['number_numerical'] = pd.to_numeric(
        out['number'], errors='coerce'
    )
    out['number_categorical'] = out['number'].where(
        out['number_numerical'].isna()
    )

    out['cabin_num'] = out['Cabin'].str.extract(r'(\d+)')
    out['cabin_cat'] = out['Cabin'].str[0]

    tokens = out['Ticket'].str.split()
    out['ticket_num'] = pd.to_numeric(tokens.str[-1], errors='coerce')
    out['ticket_cat'] = tokens.str[0].where(
        ~tokens.str[0].str.isdigit()
    )

    return out
```

This function is deterministic: the same input row produces the same output row. It can be tested with unit tests and reused in a pipeline.

## 14.3 Leakage checklist

Ask these questions before fitting a model:

1. Was any statistic (mean, frequency, target rate) computed using the test set?
2. Were one-hot categories learned on the full dataset?
3. Is the same ticket or cabin group present in both training and test?
4. Was a feature derived from `Survived`?
5. Was feature selection performed on the test set?
6. Does the extraction use future information that would not be available at prediction time?

For Day 33, the regex extraction itself does not use the target, so it is not target leakage. But ticket/cabin group structure can cause **group leakage** under a random row split, and any later target encoding must be fold-aware.

## 14.4 Production parsing

In production, the parser must handle:

- unseen ticket prefixes;
- new cabin letter codes;
- lowercase or punctuation variants;
- missing strings;
- non-string values (for example, a numeric ID read as an integer);
- empty strings and whitespace;
- schema changes in the source system.

A robust parser should:

- normalise types before regex operations;
- use raw regex strings;
- return explicit missing values rather than raising;
- log parse failure rates;
- validate row counts and category vocabularies;
- version the extraction rules;
- keep a fallback for unknown categories.

A model is only as stable as its feature contract. If the parser silently changes, the model's inputs change even when the model file is unchanged.

---

# 15. Common mistakes and debugging

## 15.1 Conversion mistakes

**Mistake:** `pd.to_numeric(df['number'])` without an error policy.  
**Result:** an exception at `'A'`.  
**Fix:** use `errors='coerce'` and preserve the failed mask.

**Mistake:** `astype(int)` on a column with `NaN`.  
**Result:** `ValueError` or an unexpected cast.  
**Fix:** handle missing values first or use a nullable integer dtype.

**Mistake:** assuming `downcast='integer'` guarantees an integer dtype.  
**Result:** float output because `NaN` forces float.  
**Fix:** inspect `dtype`; choose a nullable integer or keep float.

## 15.2 Missing-value mistakes

**Mistake:** replacing all `NaN` with 0.  
**Result:** 0 becomes a fake numeric code and missingness is lost.  
**Fix:** keep a missing indicator or an explicit `Unknown` category.

**Mistake:** dropping every row with a missing `Cabin`.  
**Result:** losing 687 of 891 rows.  
**Fix:** model missingness explicitly; use the deck category and a flag.

**Mistake:** treating parse-created missingness as original missingness.  
**Result:** four letter-only cabins are confused with 687 unrecorded cabins.  
**Fix:** compare `Cabin` presence with `cabin_num` presence.

**Mistake:** imputing a cabin identifier with its mean.  
**Result:** a fabricated cabin number with no physical meaning.  
**Fix:** use categorical representation or a missing category.

## 15.3 Regex and string mistakes

**Mistake:** writing `'(\d+)'` without `r`.  
**Result:** `SyntaxWarning` in modern Python and future fragility.  
**Fix:** `r'(\d+)'`.

**Mistake:** using `str.extract` without a capture group.  
**Result:** an error because there is no group to return.  
**Fix:** add parentheses or use `str.findall`.

**Mistake:** assuming `str.extract` returns all matches.  
**Result:** only the first cabin number is kept.  
**Fix:** use `str.findall` and aggregate if all matches matter.

**Mistake:** using `str.isdigit()` as a general numeric validator.  
**Result:** decimals, negatives, and formatted numbers are misclassified.  
**Fix:** use `pd.to_numeric` with an explicit policy.

**Mistake:** mixing `str[0]` with `str.split().str[0]`.  
**Result:** the first character is used when the first token was intended.  
**Fix:** choose based on the observed structure and name the feature clearly.

## 15.4 Identifier mistakes

**Mistake:** treating `ticket_num` as a continuous feature.  
**Result:** arbitrary numeric ordering and overfitting.  
**Fix:** use frequency, grouping, or drop it.

**Mistake:** one-hot encoding every raw ticket prefix.  
**Result:** many rare, near-duplicate columns.  
**Fix:** normalise variants, group rare categories, and validate.

**Mistake:** random row split when passengers share tickets.  
**Result:** group leakage and optimistic evaluation.  
**Fix:** use `GroupShuffleSplit` with `Ticket` or a suitable group key.

## 15.5 Workflow mistakes

**Mistake:** fitting an encoder on the full dataset before splitting.  
**Result:** test information leaks into training.  
**Fix:** fit on training data; transform test data with the same vocabulary.

**Mistake:** inspecting the test set repeatedly to choose features.  
**Result:** test-set overfitting.  
**Fix:** use training/CV for decisions; touch test once at the end.

**Mistake:** reporting only accuracy on an imbalanced target.  
**Result:** the majority-class baseline looks strong.  
**Fix:** report precision, recall, F1, confusion matrix, and a baseline.

**Mistake:** trusting a `head()` display as validation.  
**Result:** rare parse failures and missingness go unnoticed.  
**Fix:** check counts, masks, uniqueness, and row invariants.

## 15.6 Environment mistakes

**Mistake:** comparing saved official output byte-for-byte with a modern rerun.  
**Result:** false alarms about object versus string dtypes or display formatting.  
**Fix:** compare values, counts, and semantics; label version differences.

**Mistake:** using an absolute path in a shared notebook.  
**Result:** the notebook fails on another machine.  
**Fix:** use a relative path and document the working directory.

**Mistake:** plotting 678 ticket values as a bar chart.  
**Result:** an unreadable figure.  
**Fix:** plot the top-k categories or aggregate identifiers.

---

# 16. Interview preparation

The questions below are grouped by difficulty. The answers are deliberately explanatory: an interviewer is usually testing whether the reasoning survives a follow-up question, not whether a particular pandas method name was memorised.

## Level 1 — definitions and direct observations

### 1. What is a mixed variable?

A mixed variable is a column whose values encode different kinds of information in different rows. In this dataset, `number` contains digit strings such as `"5"` and a letter code `"A"`; `Cabin` contains missing values, deck-plus-number strings, multi-cabin strings, and letter-only strings; `Ticket` contains prefixed identifiers, numeric-only identifiers, and the special value `LINE`.

The important point is that the column has one storage dtype but more than one semantic type. The extraction separates the components according to meaning, not merely according to whether a value “looks unusual”.

### 2. Why is the pandas dtype not enough?

The dtype describes storage and available operations. It does not describe what the values mean. `Ticket` can be stored as text while containing a booking prefix, a numeric identifier, and a special status code. `number` can be stored as text while containing a count-like digit and a categorical letter. Two columns with the same dtype can require completely different preprocessing.

A useful answer is: “I inspect values, uniqueness, and patterns before choosing a representation. `df.dtypes` is a starting point, not a data dictionary.”

### 3. What does `errors='coerce'` do?

`pd.to_numeric(values, errors='coerce')` tries to parse each value as a number. Values that cannot be parsed become `NaN`; no exception is raised. The resulting missing mask is useful because it identifies the non-numeric branch:

```python
numeric = pd.to_numeric(df['number'], errors='coerce')
categorical_mask = numeric.isna()
```

The important consequence is that a parse failure is made explicit. It is not silently ignored, and it is not silently converted to zero.

### 4. What is a capture group in `str.extract`?

A capture group is the parenthesised part of a regular expression whose matched text is returned. In `r'(\d+)'`, the group captures one or more digits. `str.extract` returns the first captured match for each string.

For `C85`, it returns `85`. For `C23 C25 C27`, it returns `23`, because `str.extract` takes the first match, not all matches. For `D`, it returns `NaN`.

### 5. Why should regular expressions use raw strings?

A raw string such as `r'(\d+)'` prevents Python from interpreting `\d` as a Python escape sequence. Without the `r`, modern Python may warn that `\d` is invalid, and future Python versions may change the behavior. The regex engine needs the backslash, so the raw string expresses the intent unambiguously.

### 6. What is the difference between original and created missingness?

Original missingness exists before parsing: `Cabin` is missing for 687 passengers. Created missingness appears because a parser cannot extract a component: `cabin_num` is missing for those 687 rows plus four observed cabins that contain no digits (`D` and `T`). Structural missingness appears because the component does not apply: `number_numerical` is missing for the categorical code `A`.

These distinctions matter because they imply different fixes. The 687 missing cabins should not be mean-imputed; the four letter-only cabins need a category representation; the `A` rows need a categorical branch.

## Level 2 — mechanics and row accounting

### 7. Why do 752 numeric rows plus 139 categorical rows equal 891?

Because the two masks are complements. `number_numerical` is non-missing exactly where parsing succeeded; `number_categorical` is non-missing exactly where parsing failed. A row cannot be in both branches, and every row is in one branch. The arithmetic is a validation of the partition, not a coincidence.

### 8. Why is `number_numerical` a float even though the values are integers?

The column contains `NaN` for the 139 `A` rows. A standard NumPy integer array cannot represent `NaN`, so pandas uses float64. `downcast='integer'` is only a hint and cannot override the need to represent missing values. A nullable integer extension dtype is an alternative if integer semantics are required.

The practical lesson: inspect `dtype` after conversion; do not infer the dtype from the printed values.

### 9. Why is `cabin_num` missing for `D` and `T`?

`str.extract(r'(\d+)')` looks for at least one digit. `D` and `T` contain no digits, so there is no match and the result is `NaN`. This is a parse-created missing value, not evidence that the cabin was absent. The categorical extraction `str[0]` still returns `D` or `T`.

### 10. Why is `ticket_num` missing for `LINE`?

The last whitespace-separated token of `LINE` is `LINE`. `pd.to_numeric(..., errors='coerce')` cannot parse it, so the numeric result is `NaN`. The ticket category remains `LINE`. This is a meaningful special code, not an ordinary missing ticket.

### 11. Why is `ticket_cat` missing for a numeric-only ticket such as `113803`?

The first token is `113803`, and `str.isdigit()` returns true. The code uses `np.where(isdigit, np.nan, first_token)`, so numeric-only tickets are assigned `NaN` in the category column. The assumption is that a numeric-only ticket has no prefix. That is a data-specific rule; it should be documented and tested.

### 12. What is the difference between `str[0]` and `str.split().str[0]`?

`str[0]` returns the first character of the whole string. `str.split().str[0]` splits on whitespace and returns the first token. For `C85`, the first character is `C`; for `PC 17599`, the first character is `P` while the first token is `PC`. The course uses the character for the cabin deck and the token for the ticket prefix because those structures differ.

## Level 3 — feature engineering and missingness

### 13. `Cabin` is missing for 77% of rows. What would you do?

First, preserve the missingness instead of dropping it. A missing indicator and an explicit `Unknown` cabin category allow the model to learn whether “not recorded” is associated with the outcome. For the numeric cabin component, do not mean-impute an identifier. Use a categorical or frequency representation, or leave it missing with an indicator. Then test whether the feature improves validation performance; the 77% missingness may make a cabin-number feature weak or unstable.

Dropping all rows with missing `Cabin` would leave 204 rows, which is a different and much smaller population. That decision would need a strong justification.

### 14. Is `cabin_num` a continuous feature?

Not by default. The digits identify a cabin location or assignment; a one-unit increase does not necessarily mean “one unit more” of a meaningful quantity. The value is more like a label. Convert it from text explicitly, then consider categorical, frequency, or binned representations. Use a numeric representation only if the data dictionary establishes an ordered meaning and the model's assumptions are appropriate.

### 15. Is `ticket_num` useful as a numeric feature?

Usually not as a continuous magnitude. It is an identifier with 678 distinct values among 887 non-missing rows. A model may memorise ticket ranges or split on arbitrary numeric thresholds. Prefer the prefix, frequency/group features, or dropping the identifier. If the ticket number encodes a meaningful sequence, that must be established from domain knowledge, not assumed from the digits.

### 16. How would you normalise ticket prefixes?

Start by inspecting the unique values. Convert to a consistent case, strip surrounding whitespace, and standardise punctuation. Map obvious variants such as `C.A.`, `CA`, and `CA.` to a canonical form if the data dictionary supports it. Group very rare prefixes into `Other`, and fit any data-dependent mapping on training data only. Record the mapping so that serving uses the same rules.

Do not invent a mapping solely from visual similarity without checking what the prefixes mean; two similar strings could represent different booking channels.

### 17. How would you encode `cabin_cat` and `ticket_cat`?

`cabin_cat` has eight non-missing deck values plus missingness. One-hot encoding with an explicit missing category is a reasonable default. If deck order is meaningful, an ordinal encoding may be justified, but the order must be documented.

`ticket_cat` has 43 raw prefixes and many rare variants. Normalise first, group rare levels, then consider one-hot for the common levels or target encoding with out-of-fold estimation. Use `handle_unknown='ignore'` or an explicit unknown category so that unseen prefixes do not break inference.

### 18. How do you validate that extraction preserved the data?

Check row counts and complementary masks. For `number`, verify that numeric plus categorical counts equal 891. For `Cabin`, verify that numeric components plus letter-only components equal the 204 observed cabins, and that the 687 missing cabins remain missing. For `Ticket`, verify that numeric plus `LINE` counts equal 891, and that prefix-present plus numeric-only counts equal 891. Also inspect dtypes, unique categories, and a few edge-case rows.

The visual `head()` is not sufficient validation.

## Level 4 — leakage and evaluation

### 19. Does the regex extraction itself cause data leakage?

No, not if it is a deterministic per-row transformation that does not use the target or test-set statistics. Applying `str.extract`, `str.split`, or `pd.to_numeric` to each row is safe in principle.

Leakage enters later when a transformation learns from data: one-hot vocabularies, frequency counts, target encodings, imputation statistics, or feature selection. Those must be fitted on training data only. Group structure can also leak if the same ticket appears in training and test.

### 20. Why can duplicate tickets cause group leakage?

Passengers sharing a ticket may be travelling together and may share survival outcomes through common circumstances. If the split is random by row, some members of a ticket group can land in training and others in test. The model can then use group-specific patterns that would not be available for a genuinely new group. The test score becomes optimistic.

Use a group-aware split, such as `GroupShuffleSplit` with `Ticket` as the group, when the group is part of the prediction unit. Report both random and group-aware results if both are relevant.

### 21. How can target encoding leak?

If the target mean for a category is computed using the same rows that are used to validate the model, the validation target has influenced the feature. This can produce unrealistically strong validation performance. Use out-of-fold encoding: compute the encoding for each training fold from the other folds, and fit the final encoding on the full training set only after model selection. Smoothing rare categories toward the global mean reduces variance but does not remove the need for fold-aware estimation.

### 22. Is frequency encoding leakage?

Frequency encoding uses the distribution of the feature, not the target. It is not target leakage, but computing the frequencies on the full dataset still lets test-set distribution information enter training. In a strict workflow, learn the frequency map on training data and map unseen categories to a fallback. The effect is usually smaller than target leakage, but the discipline is the same.

### 23. Why is accuracy misleading for this target?

The target has 342 positive and 549 negative cases. A majority-class predictor achieves about 61.6% accuracy without learning anything. Accuracy alone can therefore make an uninformative model look decent. Report precision, recall, F1, the confusion matrix, and the majority baseline. The choice of metric should reflect the cost of false positives and false negatives.

### 24. How would you test whether an extracted feature is useful?

Start with a baseline model using only the original or established features. Add the extracted feature inside cross-validation and compare a pre-specified metric. Use repeated folds or a held-out set, and report variability. Inspect whether the gain is stable across folds and whether it survives a group-aware split. For high-cardinality features, check whether the gain comes from memorising identifiers. Do not choose the feature by repeatedly scoring the final test set.

## Level 5 — design and reasoning

### 25. Design a complete pipeline for this dataset.

A reasonable design is:

1. **Inspect** the raw columns, missingness, uniqueness, and edge cases.
2. **Extract deterministically** into a reusable function: numeric/categorical `number`, cabin number/deck, ticket number/prefix.
3. **Split** into training and test, or create cross-validation folds. Use groups if ticket/cabin sharing matters.
4. **Fit preprocessing** inside the training folds: missing indicators, category grouping, one-hot or target encoding, scaling if needed.
5. **Train** a baseline and then a model with extracted features.
6. **Evaluate** with appropriate metrics and error analysis.
7. **Persist** the extraction and encoding logic together so that serving matches training.

The extraction function should be covered by unit tests for missing values, no-match strings, multi-cabin strings, `LINE`, numeric-only tickets, and unseen categories.

### 26. Why not one-hot encode the raw `ticket_num`?

There are 678 non-missing distinct ticket numbers among 887 values. One-hot encoding would create hundreds of sparse columns, many with only one observation. A model could memorise individual tickets and overfit. Regularisation may reduce the damage but does not make an arbitrary identifier meaningful.

Better options are to drop it, use frequency/group features, extract the prefix, or treat it as a categorical feature only with strong regularisation and group-aware validation.

### 27. Compare one-hot, ordinal, frequency, and target encoding.

| Encoding | What it represents | Strength | Weakness |
|---|---|---|---|
| One-hot | one binary column per category | transparent; no false order | many columns; rare levels |
| Ordinal | integer code per category | compact | implies order that may not exist |
| Frequency | count/share of category | handles high cardinality; simple | ignores target; collisions |
| Target | estimated target mean per category | can capture signal efficiently | leakage and overfitting risk |

The choice depends on cardinality, whether order is real, the model, and the validation design. For `cabin_cat`, one-hot is natural. For `ticket_cat`, normalisation plus rare grouping is needed before any encoding. For `ticket_num`, dropping or frequency/group representation is often safer than target encoding.

### 28. What can go wrong if a tree uses `cabin_num` as a continuous feature?

A tree can split on arbitrary numeric thresholds. If cabin numbers are identifiers, those splits may isolate small groups and memorise noise. The tree may appear to find structure because it can always split a high-cardinality numeric column. Cross-validation may look good by chance on a small dataset, especially without group awareness. The feature's importance may reflect identifier memorisation rather than a generalisable relationship.

### 29. How would you handle unseen patterns in production?

Use schema and format validation, explicit unknown categories, and fallbacks. Log parse-failure rates by pattern. Preserve the extraction version so that a sudden change in failures can be traced to a code or data change. Retrain or recalibrate when the category distribution shifts. Unit-test known patterns and a set of malformed inputs.

An unseen prefix should not crash inference; it should map to `Unknown` or a defined fallback, and the event should be monitored.

### 30. Missing cabin is associated with survival. Is that a causal effect?

Not necessarily. Missing cabin may be associated with passenger class, fare, booking channel, or other unmeasured variables. A predictive model can use the association, but a causal claim requires a causal design, not a correlation in an observational dataset. The correct answer is to state the association, adjust for available confounders, and avoid claiming that “recording a cabin” would change survival.

## Level 6 — deeper statistical and modelling judgement

### 31. When is target encoding appropriate?

Target encoding is most useful for high-cardinality categorical features with enough observations per category and a validation design that prevents leakage. It compresses many categories into one numeric feature and can capture target-related information efficiently. It is risky when categories are rare, when the target is imbalanced, or when the encoding is computed outside cross-validation. Smoothing and out-of-fold estimation are essential.

For this dataset, `ticket_cat` is a candidate, but the category counts are small and the raw prefixes are inconsistent. Normalisation and group-aware validation should come first.

### 32. How would you test whether a ticket prefix has signal?

Build a clean prefix mapping, then compare a baseline model with and without the prefix feature under the same cross-validation folds. Inspect target rates by prefix with counts, but do not trust rates from tiny categories. A chi-square or mutual-information statistic can screen for association, but it must be interpreted with small-count caution and multiple comparisons. The final decision should be based on out-of-sample performance and stability, not a single univariate p-value.

### 33. What is the difference between predictive association and causal effect?

A predictive association means the feature helps estimate the target in the observed distribution. A causal effect means intervening on the feature would change the target. Missing cabin can predict survival without causing survival. Predictive models can exploit confounded associations; causal claims require identification assumptions, study design, and adjustment. Interview answers should keep these claims separate.

### 34. How would you handle multi-cabin strings?

A first-pass feature can extract the first cabin number and the deck. A richer representation can extract all numbers with `str.findall(r'\d+')`, create a cabin count, and represent the set of cabins. The choice depends on the question. If the physical cabin location matters, the first cabin is a lossy summary. If the number of cabins reflects group size, the count may be more useful than any single cabin number.

Any richer extraction needs validation and a clear rule for duplicates or malformed strings.

### 35. How do you protect against parser drift?

Treat the parser as part of the model contract. Version it, test it on known examples, monitor parse-failure rates, and keep a schema of expected types. When a new format appears, log it instead of silently coercing it. Re-run feature-distribution checks after deployment. If the parser changes, retrain or re-validate the model rather than assuming the old coefficients remain meaningful.

## Level 7 — production and system design

### 36. How would you deploy this feature extraction in a production service?

Put the extraction in a deterministic function or transformer with tests. Validate the input schema, normalise types, and return explicit missing values for parse failures. Use the same code in training and serving, or serialize a pipeline that includes the transformation. Monitor missingness, category frequencies, and parse-failure rates. Keep an audit trail from raw values to extracted features. Handle unknown prefixes and new cabin codes without crashing.

### 37. What is train-serving skew, and how does this notebook risk it?

Train-serving skew is a difference between the feature computations used during training and those used at prediction time. If the notebook's ad hoc extraction is copied into a service and then edited independently, the definitions can diverge. An absolute path, a non-raw regex, or a different missing-value policy can also cause skew. The fix is a single shared transformation, schema tests, and monitoring of feature distributions in production.

### 38. Are ticket and cabin identifiers sensitive data?

They can be. Ticket numbers and cabin assignments may be personally identifying or linkable to passenger records. Before storing or sharing extracted features, consider hashing or pseudonymisation, access controls, retention limits, and the applicable privacy requirements. A model does not need the raw identifier if a frequency or group feature is sufficient.

### 39. What changes at 1000× the data size?

The extraction logic remains conceptually the same, but implementation changes: vectorised string operations instead of row-wise `apply`, chunked or distributed processing, efficient categorical dtypes, and careful memory use. High-cardinality categories become more stable but also more expensive to encode. Group-aware validation and leakage controls matter more, not less. Monitoring and schema validation become operational requirements rather than notebook niceties.

---

# 17. Exam-style questions with worked answers

## Question 1 — row accounting

The dataset has 891 rows. `number_numerical` is non-missing for 752 rows and `number_categorical` is non-missing for 139 rows.

1. How many rows are in neither column?
2. What does the answer imply about the extraction?
3. What would it mean if the two counts summed to 900?

**Answer**

1. Zero rows are in neither column, because the masks are complementary: 752 + 139 = 891.
2. The extraction partitions the rows without loss. Every row is either parseable as a number or assigned to the categorical branch.
3. A sum of 900 would mean the extraction had duplicated or misaligned rows, or that the original row count was not 891. It would require investigation before any modelling.

## Question 2 — predict the outputs

For each input, state the `number_numerical` and `number_categorical` values.

| `number` | `number_numerical` | `number_categorical` |
|---|---|---|
| `'5'` | 5.0 | NaN |
| `'A'` | NaN | `'A'` |
| `'12'` | 12.0 | NaN |
| `'B'` | NaN | `'B'` |

Now do the same for `Cabin` using `str.extract(r'(\d+)')` and `str[0]`.

| `Cabin` | `cabin_num` | `cabin_cat` |
|---|---|---|
| `'C85'` | `'85'` | `'C'` |
| `'C23 C25 C27'` | `'23'` | `'C'` |
| `'F G73'` | `'73'` | `'F'` |
| `'D'` | NaN | `'D'` |
| `NaN` | NaN | NaN |

The key observations:

- `cabin_num` is text, not an integer;
- only the first number is extracted;
- a letter-only cabin has a category but no numeric component;
- missing input remains missing in both outputs.

## Question 3 — find the data-loss bug

Consider:

```python
df['number_numerical'] = pd.to_numeric(
    df['number'], errors='coerce', downcast='integer'
)
df = df.dropna(subset=['number_numerical'])
```

What is wrong with this preprocessing, and how would you fix it?

**Answer**

The code drops the 139 rows whose `number` is `A`, because their numeric parse is missing. That removes an entire semantic branch and changes the population. The categorical information is discarded even though it was available in the original column.

A better approach is to keep all rows, create the categorical column for failed parses, and use a missing indicator if needed:

```python
df['number_numerical'] = pd.to_numeric(
    df['number'], errors='coerce', downcast='integer'
)
df['number_categorical'] = df['number'].where(
    df['number_numerical'].isna()
)
```

Then decide separately how to represent the two branches in the model.

## Question 4 — design the feature set

You are asked to build a model from this four-column dataset. Propose a feature set and explain the representation of each feature.

**Answer**

A defensible starting point is:

- `number_numerical`: numeric if the values 1–6 are documented as counts; otherwise categorical.
- `number_categorical`: binary indicator for `A`, or a one-hot category if other codes appear.
- `cabin_cat`: one-hot deck category with an explicit `Unknown` level; optionally ordinal if deck order is documented.
- `cabin_num`: categorical or frequency representation, because it is an identifier; include a parse-failure/missing indicator. Do not use it as continuous without justification.
- `ticket_cat`: normalise variants, group rare prefixes, then one-hot common prefixes or use out-of-fold target encoding.
- `ticket_num`: drop as a continuous feature; optionally use frequency/group features.
- `Cabin`/`Ticket` raw: keep for audit and group-aware splitting, not necessarily as model features.

The target `Survived` is never a feature. All learned encodings are fitted inside the training folds. Evaluation uses a held-out set or cross-validation, with a majority baseline and metrics beyond accuracy.

## Question 5 — debug the regex

The local notebook contains:

```python
df['cabin_num'] = df['Cabin'].str.extract('(\d+)')
```

It emits:

```text
SyntaxWarning: "\d" is an invalid escape sequence.
```

1. What is the cause?
2. Does the extraction still work in this environment?
3. What is the correct fix?
4. What should be checked after the fix?

**Answer**

1. `\d` is being interpreted by Python as an invalid string escape because the string is not raw.
2. In this environment, the pattern still worked, but the warning indicates future fragility.
3. Use `r'(\d+)'`.
4. Rerun the extraction and verify the same counts: 200 non-missing `cabin_num` values, 691 missing, 92 distinct non-missing values, and the four letter-only cabins still missing their numeric part.

## Question 6 — group leakage

A random train/test split gives a strong score for a model that uses `Ticket`. The same tickets appear in both sets. Explain the problem and propose an evaluation design.

**Answer**

Passengers sharing a ticket may share outcomes and feature patterns. If the same ticket appears in training and test, the model can memorise group-specific information, so the test score overstates performance on unseen groups. Use `GroupShuffleSplit` with ticket (or a suitable group key) so that all rows from a ticket stay in one split. Report the group-aware result and compare it with the random split; do not select the model on the optimistic random score.

## Question 7 — missingness mechanism

`Cabin` is missing for 687 rows. A colleague proposes filling every missing `cabin_num` with the mean of the observed values.

Explain two reasons this is inappropriate and give a better alternative.

**Answer**

First, `cabin_num` is an identifier or location code, not a measurement, so its mean has no meaningful interpretation. Second, filling with a single value erases the distinction between “cabin not recorded” and “cabin recorded but has no digits”, and it can create artificial clusters. A better alternative is to keep a missing indicator and an explicit `Unknown` cabin category, and to represent the numeric part as categorical or frequency-based if it is used at all.

## Question 8 — high cardinality

`ticket_num` has 678 distinct non-missing values among 887 rows. A colleague one-hot encodes it for a logistic regression.

1. What problem will this create?
2. Why might cross-validation look acceptable anyway?
3. Propose a safer representation.

**Answer**

1. It creates hundreds of sparse columns, many with one observation. The model can overfit rare categories and the coefficient estimates become unstable.
2. On a small dataset, cross-validation can be optimistic, especially if tickets share groups and the split is random. Rare categories may happen to align with the target in the validation folds.
3. Drop the identifier, use frequency encoding fitted on training data, use the prefix, or use group-aware validation with a strongly regularised categorical representation.

---

# 18. Chapter summary

## 18.1 The main ideas

1. **A dtype is not a semantic type.** Text columns can contain several kinds of information.
2. **Mixed variables need a partition, not a forced cast.** Numeric and categorical branches should cover every row exactly once.
3. **`errors='coerce'` makes parse failures visible.** The resulting mask is the categorical branch for a mixed code.
4. **Missingness has multiple origins.** Original missingness, parse-created missingness, and structural missingness need different treatment.
5. **Regex extraction is lossy unless designed otherwise.** `str.extract` returns the first match; multi-cabin strings lose information.
6. **Identifiers are not measurements.** Ticket numbers and cabin numbers can be high-cardinality labels.
7. **High cardinality changes the modelling problem.** One-hot encoding, target encoding, and validation must account for rare categories and leakage.
8. **Feature extraction creates candidates, not conclusions.** The course notebook does not train or evaluate a model.

## 18.2 The data facts to remember

- 891 rows, 4 original columns.
- `Cabin` missing: 687 (77.1%); observed: 204.
- `Cabin` categories: C 59, B 47, D 33, E 32, A 15, F 13, G 4, T 1, plus 687 missing.
- `cabin_num`: 200 non-missing, 691 missing, 92 distinct non-missing; four observed cabins have no digits.
- `number`: 752 numeric rows and 139 `A` rows; numeric counts are 129, 126, 117, 126, 123, 131 for values 1–6.
- `Ticket`: 887 numeric endings, 4 `LINE` values, 681 distinct tickets, 210 duplicate rows.
- `ticket_cat`: 230 non-missing, 661 missing, 43 distinct non-missing prefixes plus `NaN`.
- Target: 549 negative, 342 positive (38.4% positive).
- Official notebook: 14 cells, extraction only, no train/test or model.
- Personal notebook: 11 cells, absolute `DAtasets` path, non-raw regex warning, no ticket-category inspection.

## 18.3 A practical checklist

Before modelling extracted mixed features:

- [ ] Inspect raw values and uniqueness.
- [ ] Identify missingness before extraction.
- [ ] Use raw regex strings.
- [ ] Decide what each component means: measurement, ordinal, nominal, or identifier.
- [ ] Verify row-count invariants.
- [ ] Separate original, parse-created, and structural missingness.
- [ ] Check edge cases: no match, multiple matches, `LINE`, letter-only cabins, multi-cabin strings.
- [ ] Decide whether the identifier should be dropped, grouped, frequency-encoded, or target-encoded.
- [ ] Normalise near-duplicate categories.
- [ ] Group rare categories.
- [ ] Fit encoders on training data only.
- [ ] Use group-aware evaluation when tickets or cabins create groups.
- [ ] Report a baseline and metrics beyond accuracy.
- [ ] Keep the extraction function deterministic and reusable in serving.
- [ ] Document the dataset path and the exact extraction rules.

## 18.4 What not to conclude

Do not conclude from this notebook that:

- `cabin_num` is a continuous predictor;
- `ticket_num` is meaningful because it is numeric;
- missing cabin causes survival differences;
- extracted features improve a model;
- the official category list is stable across datasets or time;
- a random row split is appropriate for passengers sharing tickets.

Those are questions for validation, not assumptions to carry into a model.

---

# 19. Coverage matrix

| Topic | Official notebook | Personal notebook | These notes |
|---|---|---|---|
| Load and inspect data | Cells 1–2 | Cells 1–2 | Sections 4, 10, 11 |
| Mixed `number` values | Cells 3–4 | Cell 3 | Sections 5, 9 |
| Numeric extraction | Cell 5 | Cell 4 | Sections 5, 8 |
| Categorical complement | Cell 6 | Cell 5 | Sections 5, 8 |
| Cabin inspection | Cell 7 | Cell 6 | Sections 6, 9 |
| Cabin numeric/deck extraction | Cell 9 | Cell 6 | Sections 6, 8, 9 |
| Cabin distribution | Cell 10 | Cell 8 | Sections 6, 12 |
| Ticket numeric extraction | Cell 11 | Cell 9 | Sections 7, 8, 9 |
| Ticket category extraction | Cell 11 | Cell 9 | Sections 7, 8, 9 |
| Ticket category inventory | Cell 12 | Not inspected | Sections 7, 10, 12 |
| Ticket-number distribution | Not plotted | Cell 10 | Sections 7, 11 |
| Missingness analysis | Not explicit | Not explicit | Sections 3, 6, 7, 9 |
| Identifier/high-cardinality caution | Not explicit | Not explicit | Sections 3, 7, 12, 13 |
| Leakage and validation | Not covered | Not covered | Sections 14, 15, 16 |
| Feature encoding choices | Not covered | Not covered | Sections 13, 16 |
| Production concerns | Not covered | Not covered | Sections 14, 16 |

The matrix makes the relationship clear: the notebooks demonstrate the mechanics; the notes add the reasoning, validation, encoding, and deployment context.

---

# 20. References and glossary

## 20.1 Primary sources

- CampusX, *100 Days of Machine Learning*, Day 33: “Handling Mixed Variables”.
- Official notebook: `vendor/100-days-of-machine-learning/day33-handling-mixed-variables/Untitled.ipynb`.
- Official video link from the vendor readme: `https://youtu.be/9xiX-I5_LQY`.
- Local dataset: `DAY_33/DAtasets/titanic.csv`.
- Local notebook: `DAY_33/mixedata.ipynb`.
- Transcript: `ML_Notes_Videos_31_40/transcripts/33_transcript.txt` (noisy Hindi ASR; topic orientation only).

The official saved outputs were produced in an older environment. The counts and values were reproduced on the local dataset with pandas 3.0.5; dtype reprs and warnings may differ. All such differences are marked **Rerun** or **Supplementary** in this chapter.

## 20.2 Glossary

**Mixed variable:** a column whose values encode more than one semantic kind of information.

**Coercion:** converting an unparseable value to `NaN` rather than raising an error, as in `errors='coerce'`.

**Capture group:** the parenthesised portion of a regex whose matched text is returned by `str.extract`.

**Raw string:** a Python string prefixed with `r` in which backslashes are treated literally for the regex engine.

**Structural missingness:** a missing component that exists because the value does not have that component, such as the numeric part of the letter-only cabin `D`.

**Parse-created missingness:** a missing value produced because a parser could not extract a component, such as `cabin_num` for `D`.

**High cardinality:** a large number of distinct categories relative to the number of rows.

**Identifier:** a label used to identify or group records; its numeric magnitude usually has no measurement meaning.

**Target encoding:** replacing a category with an estimate of the target mean for that category, ideally with out-of-fold estimation and smoothing.

**Group leakage:** leakage caused by related records appearing in both training and test, for example passengers sharing a ticket.

**Train-serving skew:** a mismatch between the feature computations used during model training and those used at prediction time.

**Deterministic transformation:** a transformation whose output depends only on the current row's values, not on learned statistics or the target.

---

# 21. Final connection

Day 33 is a feature-engineering lesson disguised as three string operations. The durable skill is not the exact call to `str.extract`; it is the habit of asking:

1. What information does this value actually contain?
2. Which part is a measurement, a category, a missing code, or an identifier?
3. Does the extraction preserve every row and every meaningful distinction?
4. Will the representation generalise, or will it memorise identifiers?
5. Does the validation design prevent target and group leakage?

Those questions matter far beyond this Titanic file. The same reasoning applies to amounts with currency symbols, dates stored as text, addresses, product codes, free-text fields, and any other semi-structured column encountered in a real ML workflow.

The official notebook shows a working first pass. The personal notebook shows the same logic in practice. This chapter supplies the conceptual and engineering layer that turns the first pass into a defensible feature pipeline.


