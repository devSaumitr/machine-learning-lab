"""Build Chapters 36-40 PDFs — Missing Data Imputation Methods."""
import sys, os
sys.path.insert(0, r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40')
from pdf_framework import *

FIGDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40\figures'
OUTDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40'

def build_chapter(ch_num, ch_title, subtitle, sections_fn, output_name=None):
    if output_name is None:
        safe = ch_title.replace(' ', '_').replace(':', '').replace('/', '-')
        output_name = f'Chapter_{ch_num}_{safe}.pdf'
    output_path = os.path.join(OUTDIR, output_name)
    doc = MLDocTemplate(
        output_path, pagesize=A4,
        leftMargin=55, rightMargin=55, topMargin=55, bottomMargin=55,
        chapter_title=f'Chapter {ch_num} — {ch_title}',
        title=f'Chapter {ch_num}: {ch_title}'
    )
    story = []
    story.append(spacer(30))
    story.append(Paragraph(f'CHAPTER {ch_num}', STYLES['chapter_num']))
    story.append(Paragraph(ch_title, STYLES['title']))
    story.append(Paragraph(subtitle, STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=10, spaceAfter=10))
    story.append(Paragraph('CampusX — 100 Days of Machine Learning<br/>'
                          'Feature Engineering Series, Videos 31–40',
                          STYLES['body_center']))
    story.append(spacer(20))
    sections_fn(story, ch_num)
    doc.multiBuild(story)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'  Chapter {ch_num}: {output_name} ({size_kb:.0f} KB)')
    return output_path

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 36: NUMERICAL MISSING DATA — SIMPLEIMPUTER
# ═══════════════════════════════════════════════════════════════════════════════
def ch36_sections(story, num):
    story.extend(H1('36.1 &nbsp; Introduction to Imputation'))
    story.append(P(
        'When Complete Case Analysis is not appropriate — because too much data would '
        'be lost, or because the missingness mechanism is MAR — we turn to '
        '<b>imputation</b>: replacing missing values with estimated ones. The simplest '
        'and most widely used imputation method is <b>mean imputation</b> (replace NaN '
        'with the column mean) or <b>median imputation</b> (replace NaN with the column '
        'median).'
    ))
    
    story.append(P(
        'Scikit-learn provides the <font face="Courier" size="10">SimpleImputer</font> '
        'class for this purpose. Despite its name, SimpleImputer is a foundational tool '
        'that every ML practitioner must understand deeply.'
    ))
    
    story.extend(H1('36.2 &nbsp; Mean vs Median Imputation'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch36_mean_vs_median.png'),
        width=440,
        caption='Figure 36.1: Left: for symmetric distributions, mean and median are '
                'nearly identical — either works. Right: for skewed distributions, the '
                'mean is pulled toward the long tail while the median remains robust. '
                'Median imputation is preferred for skewed data.'
    ))
    
    story.append(H2('36.2.1 &nbsp; Mathematical Basis'))
    story.append(P(
        'Mean imputation replaces each missing value xⱼ with the sample mean of observed '
        'values in that feature:'
    ))
    story.append(math_block(
        r'\hat{x}_j = \bar{x}_{\text{obs}} = \frac{1}{n_{\text{obs}}} \sum_{i \in \text{obs}} x_{ij}', '36.1'))
    
    story.append(P(
        'Median imputation replaces each missing value with the median of observed values:'
    ))
    story.append(math_block(
        r'\hat{x}_j = \text{median}\left(\{x_{ij} : i \in \text{obs}\}\right)', '36.2'))
    
    story.append(P(
        'The mean is sensitive to outliers because it uses all values equally in its '
        'computation. The median — the middle value when data is sorted — is robust to '
        'outliers because it depends only on the rank order, not the magnitudes.'
    ))
    
    story.append(page_break())
    story.extend(H1('36.3 &nbsp; Scikit-learn SimpleImputer'))
    
    story.extend(make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['missing_values', 'np.nan', 'The placeholder for missing values'],
            ['strategy', '"mean"', 'Imputation strategy: "mean", "median", "most_frequent", "constant"'],
            ['fill_value', 'None', 'Value to use when strategy="constant"'],
            ['copy', 'True', 'If True, creates a copy; if False, imputes in place'],
            ['add_indicator', 'False', 'If True, adds binary missingness indicator columns'],
        ],
        col_widths=[100, 70, 310],
        caption='Table 36.1: SimpleImputer parameters.'
    ))
    
    story.extend(make_table(
        ['Attribute', 'Description'],
        [
            ['statistics_', 'Imputation value for each feature (array of shape [n_features])'],
            ['n_features_in_', 'Number of features seen during fit'],
            ['indicator_', 'MissingIndicator instance (if add_indicator=True)'],
        ],
        col_widths=[130, 350],
        caption='Table 36.2: SimpleImputer attributes after fitting.'
    ))
    
    story.append(H2('36.3.1 &nbsp; Full Implementation'))
    
    code = """import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Create dataset with missing numerical values
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'age': np.random.normal(35, 10, n),
    'income': np.random.lognormal(10.5, 0.8, n),  # Right-skewed
    'score': np.random.normal(70, 15, n)
})

# Introduce MCAR missing values (15%)
for col in df.columns:
    mask = np.random.random(n) < 0.15
    df.loc[mask, col] = np.nan

print("=== MISSING VALUES BEFORE ===")
print(df.isnull().sum())
print()

# ── Mean Imputation ──
imputer_mean = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(
    imputer_mean.fit_transform(df),
    columns=df.columns
)
print("Mean imputation statistics:", imputer_mean.statistics_)

# ── Median Imputation ──
imputer_median = SimpleImputer(strategy='median')
df_median = pd.DataFrame(
    imputer_median.fit_transform(df),
    columns=df.columns
)
print("Median imputation statistics:", imputer_median.statistics_)

print("\\n=== MISSING VALUES AFTER ===")
print(f"Mean imputer:   {df_mean.isnull().sum().sum()} remaining")
print(f"Median imputer: {df_median.isnull().sum().sum()} remaining")

# Compare distributions
print("\\n=== DISTRIBUTION COMPARISON ===")
for col in df.columns:
    print(f"\\n{col}:")
    print(f"  Original mean:   {df[col].mean():.2f}")
    print(f"  Mean-imputed:    {df_mean[col].mean():.2f}")
    print(f"  Median-imputed:  {df_median[col].mean():.2f}")
    print(f"  Original median: {df[col].median():.2f}")
    print(f"  Original skew:   {df[col].skew():.2f}")"""
    story.extend(code_block(code, 'Listing 36.1: Mean vs Median imputation with SimpleImputer.'))
    
    story.append(page_break())
    story.extend(H1('36.4 &nbsp; Column-Wise Imputation'))
    
    story.append(P(
        'SimpleImputer computes a <b>separate</b> statistic for each feature (column). '
        'This is the correct behaviour: each feature has its own distribution, so using '
        'a single global mean for all features would be wrong. The '
        '<font face="Courier" size="9">statistics_</font> attribute reveals the per-feature '
        'imputation value after fitting.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch36_simpleimputer_flow.png'),
        width=440,
        caption='Figure 36.2: SimpleImputer with strategy="mean" replaces each NaN with '
                'the mean of its own column. Column 1 mean = 3.0, column 2 mean = 4.0, '
                'column 3 mean = 3.5.'
    ))
    
    story.extend(H1('36.5 &nbsp; Statistical Implications'))
    
    story.append(warning_box(
        'Mean Imputation Distorts Variance',
        'Replacing missing values with the column mean <b>reduces the variance</b> of the '
        'feature. All imputed values are identical (the mean), so they add no spread. '
        'This artificial narrowing of the distribution can make relationships appear '
        'stronger than they really are and can underestimate standard errors in '
        'statistical models.'
    ))
    
    story.append(P(
        'Median imputation has a similar problem: it reduces variance and can create '
        'artificial spikes at the median value. Both methods treat imputed values as if '
        'they were observed with certainty, ignoring the uncertainty of the estimate.'
    ))
    
    story.append(note_box(
        'Better Alternatives',
        'Methods that preserve the distribution better include: random sample imputation '
        '(Chapter 38), KNN imputation (Chapter 39), and iterative/MICE imputation '
        '(Chapter 40). These methods produce more realistic imputed values by considering '
        'the relationships between features.'
    ))
    
    story.extend(H1('36.6 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'SimpleImputer replaces missing values with a per-column statistic (mean, median, '
        'most_frequent, or constant). Mean works for symmetric data; median is robust to '
        'skewness and outliers. Both reduce variance — a limitation to remember when '
        'interpreting model results.'
    ))
    
    story.extend(H1('36.7 &nbsp; Interview Questions'))
    story.extend(numbered([
        'Why is median preferred over mean for skewed distributions?',
        'What does SimpleImputer.statistics_ contain after fitting?',
        'How does mean imputation affect the variance of a feature?',
        'Can SimpleImputer handle categorical data? Which strategy would you use?',
        'What is the difference between fit() and transform() in SimpleImputer?',
    ]))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 37: MISSING CATEGORICAL DATA
# ═══════════════════════════════════════════════════════════════════════════════
def ch37_sections(story, num):
    story.extend(H1('37.1 &nbsp; Missing Categorical Values'))
    story.append(P(
        'Categorical features can also contain missing values. Examples include a '
        '"city" column where some rows are blank, a "gender" column with unrecorded '
        'responses, or a "product_category" column where the category was not assigned. '
        'The strategies for handling missing categorical data differ from numerical '
        'imputation because we cannot compute a "mean" of text labels.'
    ))
    
    story.extend(H1('37.2 &nbsp; Strategies for Missing Categorical Data'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch37_categorical_strategies.png'),
        width=440,
        caption='Figure 37.1: Four strategies for handling missing categorical data. '
                'Each produces a different distribution of categories after imputation.'
    ))
    
    story.extend(make_table(
        ['Strategy', 'What it does', 'When to use', 'Risk'],
        [
            ['Most Frequent', 'Replace NaN with the modal category',
             'Missingness is low and one category dominates',
             'Inflates the dominant category'],
            ['Constant', 'Replace NaN with a fixed string (e.g., "Unknown")',
             'Missingness might be informative',
             'Adds an artificial category'],
            ['Missing Category', 'Create a new "(Missing)" category',
             'The fact that a value is missing may predict the target',
             'High cardinality if applied to many features'],
            ['Drop', 'Remove rows with missing categories',
             'Very low missingness and MCAR',
             'Data loss (same as CCA)'],
        ],
        col_widths=[90, 140, 130, 120],
        caption='Table 37.1: Strategies for missing categorical data.'
    ))
    
    story.append(page_break())
    story.extend(H1('37.3 &nbsp; Implementation'))
    
    story.append(H2('37.3.1 &nbsp; Most Frequent Imputation'))
    
    code = """import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Categorical data with missing values
df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', np.nan, 'Chicago', np.nan,
             'NYC', 'Houston', np.nan, 'LA', 'NYC', 'Chicago'],
    'gender': ['M', 'F', np.nan, 'F', 'M', 'M',
               np.nan, 'F', 'M', np.nan, 'F', 'M'],
    'subscription': ['Basic', 'Premium', 'Basic', np.nan, 'Premium',
                     'Basic', 'Premium', np.nan, 'Basic', 'Premium', 'Basic', 'Premium']
})

print("=== MISSING VALUES ===")
print(df.isnull().sum())
print()

# Most frequent imputation
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print("Most frequent values (statistics_):", imputer.statistics_)
print("\\n=== AFTER MOST FREQUENT IMPUTATION ===")
print(df_imputed)
print("\\nCategory counts after imputation:")
for col in df.columns:
    print(f"\\n{col}:")
    print(df_imputed[col].value_counts())"""
    story.extend(code_block(code, 'Listing 37.1: Most frequent imputation for categorical data.'))
    
    story.append(H2('37.3.2 &nbsp; Constant / Missing Category Imputation'))
    
    code = """# Constant imputation — replace with a designated string
imputer_const = SimpleImputer(strategy='constant', fill_value='Unknown')
df_const = pd.DataFrame(
    imputer_const.fit_transform(df),
    columns=df.columns
)
print("=== CONSTANT IMPUTATION (fill_value='Unknown') ===")
print(df_const)
print()

# Manual approach: create a missing category
df_missing_cat = df.copy()
for col in df.columns:
    df_missing_cat[col] = df_missing_cat[col].fillna('(Missing)')

print("=== MISSING CATEGORY APPROACH ===")
print(df_missing_cat)
print("\\nCategory counts with missing category:")
for col in df.columns:
    print(f"\\n{col}:")
    print(df_missing_cat[col].value_counts())"""
    story.extend(code_block(code, 'Listing 37.2: Constant and missing-category imputation.'))
    
    story.append(page_break())
    story.extend(H1('37.4 &nbsp; Consequences of Different Approaches'))
    
    story.append(H2('37.4.1 &nbsp; Most Frequent Imputation'))
    story.extend(bullets([
        '<b>Pro:</b> Simple, preserves the existing category set.',
        '<b>Con:</b> Inflates the frequency of the most common category. If "NYC" was '
        'already 40% of the data and you impute 20 more NaNs as "NYC", it might become '
        '55% — distorting the true distribution.',
        '<b>Con:</b> Breaks the relationship between the feature and other features. '
        'If missingness depends on another feature (MAR), ignoring that dependence '
        'introduces bias.',
    ]))
    
    story.append(H2('37.4.2 &nbsp; Constant / Missing Category'))
    story.extend(bullets([
        '<b>Pro:</b> Preserves information that a value was missing (the missingness '
        'itself might be predictive).',
        '<b>Pro:</b> Does not inflate any existing category.',
        '<b>Con:</b> Adds a new category, increasing cardinality. For high-cardinality '
        'features, this can be problematic for one-hot encoding.',
        '<b>Con:</b> Tree-based models can handle the new category natively, but linear '
        'models require encoding — and the encoding of "(Missing)" carries no inherent '
        'ordinal meaning.',
    ]))
    
    story.append(warning_box(
        'Common Mistake',
        'Do not use <font face="Courier" size="9">strategy="mean"</font> or '
        '<font face="Courier" size="9">"median"</font> on categorical (string) columns. '
        'These strategies are designed for numerical data. Scikit-learn will either raise '
        'an error or produce meaningless results. Always use '
        '<font face="Courier" size="9">"most_frequent"</font> or '
        '<font face="Courier" size="9">"constant"</font> for categorical features.'
    ))
    
    story.extend(H1('37.5 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'For missing categorical data, use most_frequent imputation (SimpleImputer '
        'strategy="most_frequent") when the dominant category is appropriate, or create '
        'a dedicated "(Missing)" category when the missingness itself may be informative. '
        'Never apply mean/median strategies to categorical columns.'
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 38: MISSING INDICATOR AND RANDOM SAMPLE IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def ch38_sections(story, num):
    story.extend(H1('38.1 &nbsp; Missing Indicators'))
    
    story.append(P(
        'Sometimes the <b>fact that a value is missing</b> is itself predictive. For '
        'example, in a medical dataset, patients who did not report their income might '
        'systematically differ from those who did. A <b>missing indicator</b> is a binary '
        'feature (0 or 1) that records whether the original value was missing. This '
        'preserves the missingness information that would otherwise be destroyed by '
        'imputation.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch38_missing_indicator.png'),
        width=440,
        caption='Figure 38.1: Left: original data with missing values (red dots). '
                'Right: the corresponding missing indicator feature — 1 where the '
                'original was missing, 0 where it was observed.'
    ))
    
    story.extend(H1('38.2 &nbsp; MissingIndicator in Scikit-learn'))
    
    story.extend(make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['features', '"missing-only"', 'Which columns to generate indicators for: '
             '"missing-only" (only columns with NaN) or "all"'],
            ['error_on_new', 'True', 'If True, raises error on unseen features at transform time'],
        ],
        col_widths=[100, 90, 290],
        caption='Table 38.1: MissingIndicator parameters.'
    ))
    
    code = """import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer

df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, np.nan, 38, 52, np.nan],
    'income': [50000, np.nan, 60000, np.nan, 75000, 80000, np.nan, 90000],
    'score': [85, 92, 78, np.nan, 88, 95, 72, np.nan]
})

# Standalone MissingIndicator
indicator = MissingIndicator(features='missing-only')
indicator.fit(df)
missing_features = indicator.transform(df)

print("Missing indicator columns shape:", missing_features.shape)
print("Indicator for first 3 features:")
print(missing_features)

# ── add_indicator=True in SimpleImputer ──
imputer = SimpleImputer(strategy='mean', add_indicator=True)
df_imputed = imputer.fit_transform(df)

print(f"\\nOriginal shape:  {df.shape}")
print(f"Imputed shape:   {df_imputed.shape}")
print(f"(Added {df_imputed.shape[1] - df.shape[1]} indicator columns)")
print(f"\\nImputed data:")
print(pd.DataFrame(df_imputed,
                   columns=list(df.columns) + ['age_indicator', 'income_indicator', 'score_indicator']))"""
    story.extend(code_block(code, 'Listing 38.1: MissingIndicator and add_indicator in SimpleImputer.'))
    
    story.append(page_break())
    story.extend(H1('38.3 &nbsp; Random Sample Imputation'))
    
    story.append(P(
        '<b>Random sample imputation</b> fills each missing value by randomly selecting '
        'an observed value from the same feature. Unlike mean/median imputation, this '
        'preserves the original distribution of the feature because it draws from the '
        'actual observed values.'
    ))
    
    story.append(P(
        'The algorithm is straightforward:'
    ))
    story.extend(numbered([
        'For each feature, collect all non-missing values.',
        'For each missing value, randomly sample one value from the observed set '
        '(with replacement).',
        'Replace the missing value with the sampled value.',
    ]))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch38_random_sample.png'),
        width=440,
        caption='Figure 38.2: Random sample imputation preserves the distribution shape. '
                'Left: observed data before imputation. Centre: after random sample '
                'imputation. Right: the two distributions overlap closely.'
    ))
    
    story.append(H2('38.3.1 &nbsp; Implementation'))
    
    code = """import numpy as np
import pandas as pd

def random_sample_imputation(df, random_state=42):
    \"\"\"Fill missing values by sampling from observed values of the same column.\"\"\"
    np.random.seed(random_state)
    df_imputed = df.copy()
    for col in df.columns:
        # Get observed (non-missing) values
        observed = df[col].dropna().values
        if len(observed) == 0:
            continue  # Skip all-missing columns
        # Find missing positions
        missing_mask = df[col].isna()
        n_missing = missing_mask.sum()
        # Sample with replacement from observed values
        sampled = np.random.choice(observed, size=n_missing, replace=True)
        # Fill
        df_imputed.loc[missing_mask, col] = sampled
    return df_imputed

# Example
np.random.seed(42)
df = pd.DataFrame({
    'income': np.random.lognormal(10.5, 0.8, 200),
    'age': np.random.normal(40, 12, 200),
    'score': np.random.normal(70, 15, 200)
})

# Introduce missing values
for col in df.columns:
    mask = np.random.random(200) < 0.2
    df.loc[mask, col] = np.nan

print(f"Missing before: {df.isnull().sum().sum()}")

# Impute
df_filled = random_sample_imputation(df)
print(f"Missing after:  {df_filled.isnull().sum().sum()}")

# Compare distributions
print("\\n=== DISTRIBUTION COMPARISON ===")
print(f"{'Col':<10} {'Obs Mean':>10} {'Filled Mean':>12} {'Obs Std':>10} {'Filled Std':>11}")
for col in df.columns:
    obs = df[col].dropna()
    print(f"{col:<10} {obs.mean():>10.2f} {df_filled[col].mean():>12.2f} "
          f"{obs.std():>10.2f} {df_filled[col].std():>11.2f}")"""
    story.extend(code_block(code, 'Listing 38.2: Random sample imputation implementation.'))
    
    story.append(page_break())
    story.extend(H1('38.4 &nbsp; Advantages and Limitations'))
    
    story.append(H2('38.4.1 &nbsp; Missing Indicator'))
    story.extend(bullets([
        '<b>Advantage:</b> Preserves the information that a value was missing, which '
        'can be predictive in MAR/MNAR scenarios.',
        '<b>Advantage:</b> Easy to combine with any imputation method via '
        'add_indicator=True.',
        '<b>Limitation:</b> Adds extra features, increasing dimensionality. For datasets '
        'with many features and low missingness, this can add noise.',
        '<b>Limitation:</b> The indicator is only useful if the model can exploit it. '
        'Simple linear models may not benefit as much as tree-based models.',
    ]))
    
    story.append(H2('38.4.2 &nbsp; Random Sample Imputation'))
    story.extend(bullets([
        '<b>Advantage:</b> Preserves the original distribution (mean, variance, shape) '
        'better than mean/median imputation.',
        '<b>Advantage:</b> Simple to implement and understand.',
        '<b>Limitation:</b> Does not consider relationships between features. Each '
        'feature is imputed independently.',
        '<b>Limitation:</b> Introduces randomness — results vary across runs unless '
        'you fix the random seed.',
        '<b>Limitation:</b> Can create implausible combinations (e.g., imputing an '
        'income that does not match the education level of that observation).',
    ]))
    
    story.append(warning_box(
        'Key Limitation of Random Sample Imputation',
        'Because each feature is imputed independently, random sample imputation can '
        'create <b>internally inconsistent</b> records. An imputed age of 25 might be '
        'paired with an imputed retirement status of "retired" — a combination that '
        'would never occur in the real data. Multivariate methods like KNN and MICE '
        '(Chapters 39–40) address this by considering feature relationships.'
    ))
    
    story.extend(H1('38.5 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'Missing indicators preserve the information that a value was missing. Random '
        'sample imputation fills NaNs with randomly chosen observed values, preserving '
        'the marginal distribution. Both are useful additions to the imputation toolkit '
        'but have important limitations regarding feature relationships.'
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 39: KNN IMPUTER
# ═══════════════════════════════════════════════════════════════════════════════
def ch39_sections(story, num):
    story.extend(H1('39.1 &nbsp; Understanding K-Nearest Neighbors'))
    
    story.append(P(
        'The <b>K-Nearest Neighbors (KNN)</b> algorithm is one of the simplest '
        'machine-learning algorithms. To predict a value for a new observation, KNN '
        'finds the k most similar existing observations (nearest neighbours) and '
        'combines their values (e.g., by averaging for regression, or voting for '
        'classification).'
    ))
    
    story.append(P(
        '<b>KNN imputation</b> applies this idea to missing data: instead of using '
        'the global mean or median, it uses the values of the k most similar complete '
        'observations to estimate the missing value. This means the imputed value '
        'respects the relationships between features.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch39_knn_concept.png'),
        width=380,
        caption='Figure 39.1: KNN imputation concept. The red X marks an observation '
                'with a missing value. The 5 nearest neighbours (green circles) are '
                'identified, and the missing value is estimated from their values '
                '(orange star).'
    ))
    
    story.append(page_break())
    story.extend(H1('39.2 &nbsp; Mathematical Foundation: Distance'))
    
    story.append(P(
        'KNN relies on a <b>distance metric</b> to determine which observations are '
        '"near" each other. The most common metric is <b>Euclidean distance</b>:'
    ))
    story.append(math_block(
        r'd(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{j=1}^{p}(x_j - y_j)^2}', '39.1'))
    
    story.append(P(
        'where x and y are two observations with p features each. For two-dimensional '
        'points (x₁, y₁) and (x₂, y₂), this simplifies to:'
    ))
    story.append(math_block(
        r'd = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}', '39.2'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch39_euclidean_distance.png'),
        width=320,
        caption='Figure 39.2: Euclidean distance between P1=(2,3) and P2=(7,8). '
                'The horizontal and vertical differences form the legs of a right '
                'triangle; the distance is the hypotenuse.'
    ))
    
    story.append(H2('39.2.1 &nbsp; Distance with Missing Values'))
    
    story.append(P(
        'A critical subtlety in KNN imputation: when computing distances between '
        'observations, some feature values may be missing. Scikit-learn\'s KNNImputer '
        'handles this by computing the distance using <b>only the features that are '
        'observed in both observations</b>. Formally, for observations x and y, the '
        'distance is computed over the subset of features S where both xⱼ and yⱼ are '
        'observed:'
    ))
    story.append(math_block(
        r'd(\mathbf{x}, \mathbf{y}) = \sqrt{\frac{p}{|S|}} \cdot '
        r'\sqrt{\sum_{j \in S}(x_j - y_j)^2}', '39.3'))
    
    story.append(P(
        'The factor p/|S| rescales the distance to account for the missing features, '
        'so that distances computed from fewer features are not artificially small.'
    ))
    
    story.append(page_break())
    story.extend(H1('39.3 &nbsp; KNNImputer in Scikit-learn'))
    
    story.extend(make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['n_neighbors', '5', 'Number of neighbors to use for imputation'],
            ['weights', '"uniform"', 'Weight function: "uniform" (equal) or "distance" (inverse distance)'],
            ['metric', '"nan_euclidean"', 'Distance metric designed for data with NaN values'],
            ['copy', 'True', 'If True, creates a copy; if False, imputes in place'],
        ],
        col_widths=[90, 80, 310],
        caption='Table 39.1: KNNImputer parameters.'
    ))
    
    code = """import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Create a small example to trace the algorithm manually
data = np.array([
    [1.0, 2.0, np.nan],
    [2.0, 3.0, 6.0],
    [3.0, np.nan, 8.0],
    [4.0, 5.0, 10.0],
    [5.0, 6.0, np.nan],
    [np.nan, 7.0, 12.0]
])

print("Original data:")
print(pd.DataFrame(data, columns=['A', 'B', 'C']))
print()

# Apply KNN imputation with k=2
imputer = KNNImputer(n_neighbors=2, weights='distance')
imputed = imputer.fit_transform(data)

print("After KNN imputation (k=2, weights='distance'):")
print(pd.DataFrame(imputed, columns=['A', 'B', 'C']))
print()

# Trace one imputation manually
# For the missing C in row 0:
# Distance from row 0 to each complete row (using available features)
print("=== MANUAL TRACE: Imputing C[0] (missing) ===")
row0 = data[0]  # [1.0, 2.0, NaN]
print(f"Row 0 (with missing): {row0}")

# Compute nan_euclidean distance to each other row
for i in range(1, len(data)):
    other = data[i]
    # Features observed in both: A and B (C is missing in row 0)
    mask = ~np.isnan(row0) & ~np.isnan(other)
    n_features_total = data.shape[1]
    n_observed = mask.sum()
    if n_observed == 0:
        print(f"  Row {i}: no common features")
        continue
    sq_diff = np.sum((row0[mask] - other[mask]) ** 2)
    distance = np.sqrt(n_features_total / n_observed) * np.sqrt(sq_diff)
    print(f"  Row {i} {other}: distance = {distance:.4f} (using {n_observed} features)")"""
    story.extend(code_block(code, 'Listing 39.1: KNNImputer with manual distance trace.'))
    
    story.append(page_break())
    story.extend(H1('39.4 &nbsp; Weights Parameter'))
    
    story.append(P(
        'The <font face="Courier" size="9">weights</font> parameter controls how '
        'neighbouring observations contribute to the imputed value:'
    ))
    
    story.extend(make_table(
        ['Weights', 'Formula', 'Behaviour'],
        [
            ['"uniform"', 'Each neighbour contributes equally',
             'Imputed value = simple mean of k nearest neighbours'],
            ['"distance"', 'Closer neighbours contribute more (inverse distance)',
             'Imputed value = weighted mean, weighted by 1/d'],
        ],
        col_widths=[80, 180, 220],
        caption='Table 39.2: KNNImputer weights options.'
    ))
    
    story.append(P(
        'With <font face="Courier" size="9">weights="distance"</font>, the imputed value '
        'is computed as:'
    ))
    story.append(math_block(
        r'\hat{x} = \frac{\sum_{i \in N_k} \frac{x_i}{d_i}}{\sum_{i \in N_k} \frac{1}{d_i}}', '39.4'))
    
    story.append(P(
        'where Nₖ is the set of k nearest neighbours and dᵢ is the distance to '
        'neighbour i. Closer neighbours (smaller dᵢ) receive higher weight.'
    ))
    
    story.append(H2('39.4.1 &nbsp; Full Pipeline Example'))
    
    code = """import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Realistic dataset
np.random.seed(42)
n = 500
X = pd.DataFrame({
    'age': np.random.normal(40, 12, n),
    'income': np.random.lognormal(10.5, 0.7, n),
    'credit_score': np.random.normal(700, 50, n),
    'debt_ratio': np.random.beta(2, 5, n),
    'years_employed': np.random.exponential(8, n)
})

# Introduce MAR missingness (age missing more for low income)
for i in range(n):
    if X.loc[i, 'income'] < X['income'].quantile(0.3):
        if np.random.random() < 0.3:
            X.loc[i, 'age'] = np.nan
    if np.random.random() < 0.1:
        X.loc[i, 'credit_score'] = np.nan
    if np.random.random() < 0.15:
        X.loc[i, 'debt_ratio'] = np.nan

print("Missing per column:")
print(X.isnull().sum())
print(f"\\nTotal missing: {X.isnull().sum().sum()}")

# KNN imputation with different k values
for k in [3, 5, 10]:
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print(f"\\nk={k}: imputed shape={X_imputed.shape}, "
          f"age mean={X_imputed['age'].mean():.2f}, "
          f"credit mean={X_imputed['credit_score'].mean():.1f}")"""
    story.extend(code_block(code, 'Listing 39.2: KNN imputation pipeline with different k values.'))
    
    story.append(page_break())
    story.extend(H1('39.5 &nbsp; Advantages, Disadvantages and Computational Cost'))
    
    story.append(H2('39.5.1 &nbsp; Advantages'))
    story.extend(bullets([
        '<b>Respects feature relationships:</b> Unlike mean/median imputation, KNN '
        'considers all features when estimating a missing value.',
        '<b>No distributional assumption:</b> Does not assume the data follows any '
        'specific distribution.',
        '<b>Non-parametric:</b> Makes no assumptions about the functional form of the '
        'data-generating process.',
        '<b>Intuitive:</b> The imputed value is based on actual similar observations.',
    ]))
    
    story.append(H2('39.5.2 &nbsp; Disadvantages'))
    story.extend(bullets([
        '<b>Computationally expensive:</b> Requires computing distances between all '
        'pairs of observations: O(n² × p) time. For large datasets, this is prohibitive.',
        '<b>Memory intensive:</b> Must store the entire dataset for distance computation.',
        '<b>Sensitive to feature scaling:</b> Features with larger ranges dominate the '
        'distance calculation. <b>Always scale features before KNN imputation.</b>',
        '<b>Sensitive to k:</b> Too small → noisy estimates; too large → oversmoothing. '
        'The optimal k depends on the dataset.',
        '<b>Curse of dimensionality:</b> In high-dimensional spaces, all points become '
        'roughly equidistant, reducing the discriminative power of neighbours.',
    ]))
    
    story.append(warning_box(
        'Always Scale Before KNN',
        'KNN uses distance metrics, so features with larger scales dominate. If one '
        'feature ranges from 0–1,000,000 (income) and another from 0–1 (debt ratio), '
        'the income feature will completely determine the nearest neighbours. '
        'Apply StandardScaler or MinMaxScaler <b>before</b> KNN imputation.'
    ))
    
    story.extend(H1('39.6 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'KNNImputer estimates missing values from the k most similar observations using '
        'nan_euclidean distance. It respects feature relationships but is computationally '
        'expensive and sensitive to feature scaling. Always scale features before applying '
        'KNN imputation, and choose k through cross-validation.'
    ))
    
    story.extend(H1('39.7 &nbsp; Interview Questions'))
    story.extend(numbered([
        'How does KNNImputer handle missing values when computing distances between observations?',
        'Why must you scale features before KNN imputation?',
        'What is the computational complexity of KNN imputation?',
        'Compare uniform vs distance weights in KNNImputer.',
        'When would you choose KNN imputation over mean imputation?',
    ]))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 40: MICE AND ITERATIVE IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def ch40_sections(story, num):
    story.extend(H1('40.1 &nbsp; Why Univariate Imputation May Be Insufficient'))
    
    story.append(P(
        'Mean, median, and most-frequent imputation are <b>univariate</b> methods: they '
        'use only the information from a single feature to estimate missing values. '
        'Random sample imputation is also univariate. KNN imputation begins to use '
        'multivariate information but is limited to the k nearest neighbours.'
    ))
    
    story.append(P(
        'In many real-world datasets, features are <b>correlated</b>. Income is correlated '
        'with education. Age is correlated with years of experience. Blood pressure is '
        'correlated with weight and age. When features are correlated, knowing the values '
        'of other features gives you information about the missing value. '
        '<b>Multivariate imputation</b> methods exploit these relationships.'
    ))
    
    story.extend(H1('40.2 &nbsp; MICE: Multivariate Imputation by Chained Equations'))
    
    story.append(P(
        '<b>MICE</b> (also called the chained equations approach or iterative imputation) '
        'was proposed by van Buuren, Boshuizen, and Knook (1999). The core idea is elegant: '
        'each feature with missing values is treated as the <b>target</b> of a regression '
        'model, using all other features as <b>predictors</b>.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch40_mice_process.png'),
        width=440,
        caption='Figure 40.1: The MICE iterative process. Each feature with missing values '
                'is predicted using the other features. The process cycles through all '
                'features repeatedly until convergence.'
    ))
    
    story.append(page_break())
    story.extend(H1('40.3 &nbsp; The MICE Algorithm'))
    
    story.append(P('The algorithm proceeds as follows:'))
    
    story.extend(numbered([
        '<b>Initial imputation:</b> Fill all missing values with a simple method '
        '(e.g., mean or median) to create a complete initial dataset.',
        '<b>For each feature j with missing values:</b>',
    ]))
    story.extend(bullets([
        'Temporarily remove the observed values of feature j (treat them as missing).',
        'Build a regression model: feature j is the target; all other features are predictors.',
        'Predict the missing values of feature j using the model.',
        'Replace the imputed values of feature j with these new predictions.',
    ], style=STYLES['bullet2']))
    story.append(P(
        '<b>3. Cycle through all features with missing values.</b> Repeat step 2 for '
        'each such feature. One full pass through all features is called one '
        '<b>iteration</b>.'
    ))
    story.append(P(
        '<b>4. Repeat iterations</b> until the imputed values stabilise (converge). '
        'Convergence is typically assessed by monitoring the change in imputed values '
        'between iterations.'
    ))
    
    story.append(H2('40.3.1 &nbsp; Mathematical Intuition'))
    
    story.append(P(
        'Suppose we have three features A, B, C, and feature A has missing values. '
        'In the first iteration, we fit a regression model:'
    ))
    story.append(math_block(r'A_{\text{pred}} = f(B, C)', '40.1'))
    story.append(P(
        'and use it to predict the missing values of A. In the next step, if B also has '
        'missing values, we fit:'
    ))
    story.append(math_block(r'B_{\text{pred}} = g(A_{\text{new}}, C)', '40.2'))
    story.append(P(
        'using the <b>updated</b> values of A (including the newly imputed ones). This '
        'sequential updating — where each feature\'s imputation uses the most recent '
        'estimates of other features — is what makes MICE "chained equations".'
    ))
    
    story.append(page_break())
    story.extend(H1('40.4 &nbsp; IterativeImputer in Scikit-learn'))
    
    story.append(P(
        'Scikit-learn implements MICE as '
        '<font face="Courier" size="10">IterativeImputer</font> in '
        '<font face="Courier" size="10">sklearn.impute</font>. As of Scikit-learn 1.x, '
        'it is considered experimental and requires importing from '
        '<font face="Courier" size="9">sklearn.experimental</font>.'
    ))
    
    story.extend(make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['estimator', 'BayesianRidge()', 'The regression model used to predict each feature'],
            ['max_iter', '10', 'Maximum number of imputation rounds'],
            ['random_state', 'None', 'Controls randomness for reproducibility'],
            ['initial_strategy', '"mean"', 'Initial imputation: "mean", "median", "most_frequent"'],
            ['imputation_order', '"ascending"', 'Order in which features are imputed: '
             '"ascending", "descending", "arabic", "random"'],
            ['sample_posterior', 'False', 'If True, uses sampling from posterior; '
             'if False, uses point estimates'],
            ['tol', '1e-3', 'Convergence tolerance: stop when change < tol'],
            ['n_nearest_features', 'None', 'Number of features to use per estimator (for speed)'],
        ],
        col_widths=[110, 90, 280],
        caption='Table 40.1: IterativeImputer parameters.'
    ))
    
    code = """import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Dataset with correlated features and missing values
np.random.seed(42)
n = 300

# Create correlated features
education = np.random.normal(14, 3, n)           # Years of education
income = 30000 + 5000 * education + np.random.normal(0, 15000, n)
experience = np.random.normal(10, 5, n)
age = 22 + experience + np.random.normal(0, 3, n)
credit_score = 600 + 10 * education + 2 * income/1000 + np.random.normal(0, 40, n)

df = pd.DataFrame({
    'education': education,
    'income': income,
    'experience': experience,
    'age': age,
    'credit_score': credit_score
})

# Introduce missing values (MAR: less educated more likely to skip income)
for i in range(n):
    if education[i] < 12 and np.random.random() < 0.4:
        df.loc[i, 'income'] = np.nan
    if np.random.random() < 0.1:
        df.loc[i, 'credit_score'] = np.nan
    if np.random.random() < 0.08:
        df.loc[i, 'experience'] = np.nan

print("Missing per column:")
print(df.isnull().sum())

# Apply MICE imputation
imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42,
    sample_posterior=True
)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print(f"\\nShape before: {df.shape}, after: {df_imputed.shape}")
print(f"Missing after: {df_imputed.isnull().sum().sum()}")

# Compare original and imputed correlations
print("\\n=== CORRELATION COMPARISON ===")
print("Original (observed pairs):")
print(df.corr().round(3))
print("\\nAfter MICE imputation:")
print(df_imputed.corr().round(3))"""
    story.extend(code_block(code, 'Listing 40.1: MICE imputation with IterativeImputer.'))
    
    story.append(page_break())
    story.extend(H1('40.5 &nbsp; Comparison: KNN vs MICE'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch40_mice_vs_knn.png'),
        width=440,
        caption='Figure 40.2: KNN vs MICE imputation on correlated data. Both methods '
                'recover the relationship between true and imputed values, but MICE '
                'tends to produce slightly better-calibrated estimates when features '
                'are strongly correlated.'
    ))
    
    story.extend(make_table(
        ['Aspect', 'KNNImputer', 'IterativeImputer (MICE)'],
        [
            ['Approach', 'Find k similar observations, average their values',
             'Fit regression model per feature, predict missing values'],
            ['Feature relationships', 'Implicitly via distance metric',
             'Explicitly via regression coefficients'],
            ['Computation', 'O(n² × p) — expensive for large n',
             'O(iterations × n × p²) — depends on max_iter'],
            ['Scalability', 'Poor for large datasets',
             'Better for large datasets'],
            ['Hyperparameters', 'n_neighbors, weights',
             'estimator, max_iter, imputation_order'],
            ['Distribution preservation', 'Good (samples from observed values)',
             'Good (regression predictions vary)'],
            ['Best for', 'Small-to-medium datasets with moderate correlation',
             'Datasets with strong feature correlations'],
        ],
        col_widths=[100, 190, 190],
        caption='Table 40.2: KNNImputer vs IterativeImputer comparison.'
    ))
    
    story.append(page_break())
    story.extend(H1('40.6 &nbsp; Convergence and Stopping Criteria'))
    
    story.append(P(
        'IterativeImputer stops when one of two conditions is met:'
    ))
    story.extend(numbered([
        'The maximum number of iterations (<font face="Courier" size="9">max_iter</font>) '
        'is reached.',
        'The change in imputed values between consecutive iterations falls below '
        '<font face="Courier" size="9">tol</font> (default 1e-3).',
    ]))
    
    story.append(P(
        'In practice, 5–10 iterations are usually sufficient for convergence. If the '
        'imputed values are still changing significantly at max_iter, increase the '
        'iteration count or check whether the regression model is appropriate for the '
        'feature being imputed.'
    ))
    
    story.extend(H1('40.7 &nbsp; Advantages and Assumptions'))
    
    story.append(H2('40.7.1 &nbsp; Advantages'))
    story.extend(bullets([
        '<b>Uses all features:</b> Each feature\'s imputation leverages information from '
        'all other features.',
        '<b>Preserves correlations:</b> The iterative process maintains the multivariate '
        'relationships in the data.',
        '<b>Flexible estimator:</b> Any regression model can be used (BayesianRidge, '
        'DecisionTree, KNeighbors, etc.).',
        '<b>Handles different types:</b> Different estimators can be used for different '
        'feature types.',
    ]))
    
    story.append(H2('40.7.2 &nbsp; Assumptions'))
    story.extend(bullets([
        '<b>MAR assumption:</b> MICE assumes data is Missing at Random (missingness '
        'depends on observed data, not the missing values themselves).',
        '<b>Model specification:</b> The quality of imputation depends on whether the '
        'chosen regression model correctly captures the relationship between features.',
        '<b>Convergence:</b> The algorithm assumes the iterative process converges. '
        'For highly complex relationships, convergence may be slow or unstable.',
    ]))
    
    story.append(warning_box(
        'When NOT to Use MICE',
        'Avoid MICE when: (1) the dataset is very large and computation time is critical; '
        '(2) features are essentially univariate (no correlations to exploit); '
        '(3) the missingness is MNAR (MICE assumes MAR); (4) the relationship between '
        'features is highly non-linear and the default linear estimator cannot capture it. '
        'In such cases, use a tree-based estimator within IterativeImputer or fall back '
        'to simpler methods.'
    ))
    
    story.extend(H1('40.8 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'MICE treats each feature with missing values as a regression target, using all '
        'other features as predictors. The process iterates until convergence, preserving '
        'multivariate relationships. Scikit-learn implements this as IterativeImputer. '
        'It is more powerful than KNN for correlated data but assumes MAR and requires '
        'a suitable regression model.'
    ))
    
    story.extend(H1('40.9 &nbsp; Interview Questions'))
    story.extend(numbered([
        'Explain the MICE algorithm step by step.',
        'How does IterativeImputer differ from KNNImputer?',
        'What is the role of the estimator parameter in IterativeImputer?',
        'What does the max_iter parameter control, and what is a typical value?',
        'What assumption does MICE make about the missingness mechanism?',
        'How would you handle a mix of numerical and categorical features with MICE?',
    ]))

# Build chapters 36-40
print('Building Chapters 36-40...')
build_chapter(36, 'Numerical Missing Data - SimpleImputer',
              'Mean Imputation, Median Imputation, Strategy, Statistics',
              ch36_sections)
build_chapter(37, 'Missing Categorical Data',
              'Most Frequent, Constant, Missing Category Strategies',
              ch37_sections)
build_chapter(38, 'Missing Indicator and Random Sample Imputation',
              'Missingness Information, Distribution Preservation, Implementation',
              ch38_sections)
build_chapter(39, 'KNN Imputer',
              'K-Nearest Neighbors, Euclidean Distance, nan_euclidean, Weights',
              ch39_sections)
build_chapter(40, 'MICE and Iterative Imputation',
              'Chained Equations, Multivariate Imputation, Convergence, Comparison',
              ch40_sections)
print('Chapters 36-40 done.')
