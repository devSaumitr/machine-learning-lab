"""Build Chapters 32-40 PDFs."""
import sys, os
sys.path.insert(0, r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40')
from pdf_framework import *

FIGDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40\figures'
OUTDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40'

def build_chapter(ch_num, ch_title, subtitle, sections_fn, output_name=None):
    """Generic chapter builder."""
    if output_name is None:
        output_name = f'Chapter_{ch_num}_{ch_title.replace(" ", "_").replace("&", "and")}.pdf'
    output_path = os.path.join(OUTDIR, output_name)
    
    doc = MLDocTemplate(
        output_path, pagesize=A4,
        leftMargin=55, rightMargin=55, topMargin=55, bottomMargin=55,
        chapter_title=f'Chapter {ch_num} — {ch_title}',
        title=f'Chapter {ch_num}: {ch_title}'
    )
    
    story = []
    # Cover
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
    
    # Add sections
    sections_fn(story, ch_num)
    
    doc.multiBuild(story)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'  Chapter {ch_num}: {output_name} ({size_kb:.0f} KB)')
    return output_path

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 32: BINNING AND BINARIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def ch32_sections(story, num):
    story.extend(H1('32.1 &nbsp; Introduction: Continuous vs Discrete Variables'))
    story.append(P(
        'Machine-learning algorithms consume numerical features. However, not every '
        'continuous feature benefits from being continuous. Sometimes we deliberately '
        'convert a continuous variable into a discrete one — a process called '
        '<b>discretization</b> — to reduce noise, handle non-linear relationships, '
        'or make the feature more interpretable.'
    ))
    story.append(P(
        'A <b>continuous variable</b> can take any value within a range (e.g., age = 25.7, '
        'temperature = 36.6). A <b>discrete variable</b> takes only a finite set of values '
        '(e.g., age group = "young", "middle", "senior"). <b>Binning</b> is the process of '
        'converting the former into the latter by partitioning the value range into intervals.'
    ))
    story.append(P(
        '<b>Binarization</b> is the extreme case of binning where the variable is reduced '
        'to exactly two categories: 0 or 1, based on a threshold.'
    ))
    
    story.extend(H1('32.2 &nbsp; Why Bin Continuous Features?'))
    story.extend(bullets([
        '<b>Reduce noise:</b> Small fluctuations within a bin are absorbed, making the '
        'feature more robust to measurement error.',
        '<b>Handle non-linearity:</b> If the relationship between feature and target is '
        'stepwise rather than linear, binning captures the steps directly.',
        '<b>Improve interpretability:</b> "Age group: senior" is easier to communicate '
        'than "Age: 67.3".',
        '<b>Manage outliers:</b> Extreme values are compressed into the edge bins, '
        'reducing their influence.',
        '<b>Satisfy algorithm assumptions:</b> Some algorithms (e.g., naive Bayes with '
        'Gaussian assumption) work better with binned features.',
    ]))
    
    story.append(warning_box(
        'Caution: Information Loss',
        'Binning is a form of information loss. The continuous value within each bin '
        'is discarded. If the continuous feature is highly informative, binning can '
        'reduce model performance. Always validate with cross-validation whether '
        'binning improves your specific model.'
    ))
    
    story.append(page_break())
    story.extend(H1('32.3 &nbsp; Equal-Width Binning'))
    
    story.append(P(
        '<b>Equal-width binning</b> divides the range of the feature into k intervals '
        'of equal width. If the feature ranges from min to max, the bin edges are:'
    ))
    story.append(math_block(
        r'\text{width} = \frac{\max(x) - \min(x)}{k}, \quad '
        r'\text{edges} = \left[\min,\ \min+w,\ \min+2w,\ \ldots,\ \max\right]', '32.1'))
    
    story.append(P(
        'For example, with values ranging from 0 to 100 and k = 4 bins, the edges are '
        '[0, 25, 50, 75, 100], giving intervals [0–25], [25–50], [50–75], [75–100].'
    ))
    
    story.append(P(
        'Equal-width binning works well when the data is uniformly distributed. However, '
        'if the data is skewed, some bins may contain many observations while others are '
        'nearly empty — a problem that equal-frequency binning solves.'
    ))
    
    story.extend(H1('32.4 &nbsp; Equal-Frequency (Quantile) Binning'))
    
    story.append(P(
        '<b>Equal-frequency binning</b> (also called quantile binning) divides the data '
        'so that each bin contains approximately the same number of observations. The '
        'bin edges are the quantiles of the feature distribution.'
    ))
    
    story.append(P(
        'For k bins, the edges are at the quantiles 0, 1/k, 2/k, …, 1. For 4 bins, '
        'the edges are the 0th (min), 25th, 50th (median), 75th, and 100th (max) percentiles.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch32_binning_strategies.png'),
        width=440,
        caption='Figure 32.1: Three binning strategies applied to the same multimodal '
                'distribution. Top: equal-width (uniform intervals, unequal counts). '
                'Middle: equal-frequency (equal counts, unequal intervals). '
                'Bottom: K-Means-based (cluster centres determine intervals).'
    ))
    
    story.append(H2('32.4.1 &nbsp; Mathematical Definition'))
    story.append(P(
        'For k bins, the j-th bin edge (j = 0, 1, …, k) is the quantile at level j/k:'
    ))
    story.append(math_block(
        r'e_j = Q\left(\frac{j}{k}\right), \quad j = 0, 1, \ldots, k', '32.2'))
    story.append(P(
        'where Q(p) is the p-th quantile of the feature distribution. An observation '
        'xᵢ belongs to bin j if eⱼ ≤ xᵢ &lt; eⱼ₊₁ (with the last bin including the maximum).'
    ))
    
    story.append(page_break())
    story.extend(H1('32.5 &nbsp; K-Means-Based Binning'))
    
    story.append(P(
        'A third strategy uses <b>K-Means clustering</b> to determine the bin edges. '
        'Instead of using fixed-width or quantile-based intervals, K-Means finds k '
        'cluster centres that minimise the within-cluster sum of squared distances. '
        'The bin edges are then placed at the midpoints between adjacent cluster centres.'
    ))
    
    story.append(P(
        'This approach has an advantage: it adapts to the actual density of the data. '
        'Dense regions get narrower bins (more resolution), while sparse regions get '
        'wider bins. The result is often more informative than equal-width or '
        'equal-frequency binning.'
    ))
    
    story.extend(H1('32.6 &nbsp; Scikit-learn Implementation'))
    
    story.append(H2('32.6.1 &nbsp; KBinsDiscretizer'))
    
    story.extend(make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['n_bins', '5', 'Number of bins per feature'],
            ['encode', '"ordinal"', 'Encoding: "ordinal" (0,1,2...), "onehot", "onehot-dense"'],
            ['strategy', '"uniform"', 'Binning strategy: "uniform", "quantile", "kmeans"'],
        ],
        col_widths=[80, 70, 310],
        caption='Table 32.1: KBinsDiscretizer parameters.'
    ))
    
    code = """import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

# Create skewed data
np.random.seed(42)
data = np.random.lognormal(mean=2, sigma=1, size=1000).reshape(-1, 1)

# Apply three binning strategies
strategies = ['uniform', 'quantile', 'kmeans']
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

for idx, strategy in enumerate(strategies):
    disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=strategy)
    binned = disc.fit_transform(data)
    
    # Plot original
    axes[idx, 0].hist(data, bins=50, color='steelblue', edgecolor='white')
    axes[idx, 0].set_title(f'{strategy.title()} - Original')
    
    # Plot binned distribution
    unique, counts = np.unique(binned, return_counts=True)
    axes[idx, 1].bar(unique, counts, color='coral', edgecolor='white')
    axes[idx, 1].set_title(f'{strategy.title()} - After Binning')
    axes[idx, 1].set_xlabel('Bin Index')
    axes[idx, 1].set_ylabel('Count')
    
    # Print bin edges
    print(f"{strategy.title()} bin edges: {disc.bin_edges_[0]}")

plt.tight_layout()
plt.savefig('binning_strategies.png', dpi=150)
plt.show()"""
    story.extend(code_block(code, 'Listing 32.1: Comparing all three KBinsDiscretizer strategies.'))
    
    story.extend(H1('32.7 &nbsp; Binarization'))
    
    story.append(P(
        '<b>Binarization</b> converts a continuous feature into a binary feature (0 or 1) '
        'using a threshold. Values above the threshold become 1; values at or below become 0.'
    ))
    story.append(math_block(
        r'x_{\text{binary}} = \begin{cases} 1 & \text{if } x > \text{threshold} \\ 0 & \text{if } x \leq \text{threshold} \end{cases}', '32.3'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch32_binarization.png'),
        width=440,
        caption='Figure 32.2: Binarization with threshold = 55. Original continuous values '
                '(top row) are mapped to 0 or 1 (bottom row) based on whether they exceed '
                'the threshold.'
    ))
    
    story.append(H2('32.7.1 &nbsp; Binarizer in Scikit-learn'))
    
    code = """from sklearn.preprocessing import Binarizer
import numpy as np

data = np.array([[10], [25], [35], [50], [65], [80], [95]])

# Threshold at 50
binarizer = Binarizer(threshold=50)
binary = binarizer.fit_transform(data)

print("Original:", data.flatten())
print("Binary:  ", binary.flatten())
# Output: Original: [10 25 35 50 65 80 95]
#         Binary:   [ 0  0  0  0  1  1  1]"""
    story.extend(code_block(code, 'Listing 32.2: Binarizer with threshold=50.'))
    
    story.append(warning_box(
        'Important: Binarizer Threshold',
        'The Binarizer uses a <b>strictly greater-than</b> comparison: values equal to '
        'the threshold are assigned 0. If you want values ≥ threshold to be 1, set '
        'threshold to (desired_threshold - small_epsilon).'
    ))
    
    story.append(page_break())
    story.extend(H1('32.8 &nbsp; Comparison of Techniques'))
    
    story.extend(make_table(
        ['Method', 'How edges are determined', 'Best for', 'Drawback'],
        [
            ['Equal-width', 'Uniform intervals across [min, max]',
             'Uniformly distributed data', 'Skewed data → uneven bin counts'],
            ['Equal-frequency', 'Quantiles (equal count per bin)',
             'Skewed data, robust to outliers', 'May split identical values across bins'],
            ['K-Means', 'Cluster centres → midpoints as edges',
             'Multimodal / naturally clustered data', 'Computationally heavier; requires k'],
            ['Binarization', 'Single threshold → 0 or 1',
             'Yes/no decisions, simple rules', 'Extreme information loss'],
        ],
        col_widths=[85, 150, 120, 125],
        caption='Table 32.2: Comparison of binning and binarization techniques.'
    ))
    
    story.extend(H1('32.9 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'Binning converts continuous features into discrete bins to reduce noise and '
        'handle non-linearity. Equal-width uses uniform intervals, equal-frequency uses '
        'quantiles, and K-Means adapts to data density. Binarization is the extreme case '
        'with a single threshold producing 0/1 output.'
    ))
    
    story.extend(H1('32.10 &nbsp; Interview Questions'))
    story.extend(numbered([
        'What is the difference between equal-width and equal-frequency binning?',
        'When would you prefer K-Means binning over quantile binning?',
        'Does binning always improve model performance? Explain.',
        'How does Binarizer handle values exactly equal to the threshold?',
        'What does KBinsDiscretizer(encode="onehot") produce?',
    ]))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 33: HANDLING MIXED VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
def ch33_sections(story, num):
    story.extend(H1('33.1 &nbsp; What is a Mixed Variable?'))
    story.append(P(
        'A <b>mixed variable</b> (also called a mixed-type feature) is a single feature '
        'that contains both numerical and categorical information simultaneously. '
        'Examples include "5 feet 7 inches" (number + unit), "2024-01-15" (numbers + '
        'separators), or "40.7°N" (number + direction).'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch33_mixed_variable.png'),
        width=420,
        caption='Figure 33.1: Anatomy of a mixed variable. "5 feet 7 inches" contains '
                'numerical components (5, 7) and categorical components (feet, inches) '
                'within a single feature column.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch33_mixed_examples.png'),
        width=440,
        caption='Figure 33.2: Common examples of mixed variables found in real datasets.'
    ))
    
    story.extend(H1('33.2 &nbsp; Why Mixed Variables Are Problematic'))
    story.append(P(
        'Machine-learning algorithms expect each feature to have a single data type. '
        'A mixed variable cannot be directly interpreted as numeric (because it contains '
        'text) or as categorical (because it contains meaningful numbers). Feeding a '
        'mixed variable as-is into a model will either cause an error or force the model '
        'to treat every unique string as a separate category — leading to high cardinality '
        'and poor generalization.'
    ))
    
    story.extend(H1('33.3 &nbsp; Feature Extraction and Splitting'))
    
    story.append(P(
        'The primary strategy for handling mixed variables is <b>feature splitting</b>: '
        'extract the numerical parts into one feature and the categorical parts into '
        'another. This way each resulting feature has a single, clear type.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch33_feature_splitting.png'),
        width=440,
        caption='Figure 33.3: Before splitting, a single mixed column contains both '
                'numbers and text. After splitting, two clean columns result: one '
                'numeric (Height_Value) and one categorical (Height_Unit).'
    ))
    
    story.append(H2('33.3.1 &nbsp; Python Implementation'))
    
    code = """import pandas as pd
import numpy as np
import re

# Sample mixed dataset
data = {
    'height': ['5 feet 7 inches', '6 feet 2 inches', '5 feet 1 inch',
               '5 feet 9 inches', '6 feet 0 inches', '5 feet 5 inches'],
    'weight': ['70.5 kg', '82.3 kg', '55.0 kg', '91.2 kg', '68.7 kg', '73.1 kg'],
    'temperature': ['36.6 C', '37.2 C', '38.1 C', '35.9 C', '36.0 C', '37.8 F']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print(f"\\nDtypes:\\n{df.dtypes}")

# ── Split height into value and unit ──
df[['height_value', 'height_unit']] = df['height'].str.extract(
    r'(\\d+(?:\\.\\d+)?)\\s*(feet|inches?)', expand=True
)

# Convert height_value to numeric
df['height_value'] = pd.to_numeric(df['height_value'])

# ── Split weight into value and unit ──
df[['weight_value', 'weight_unit']] = df['weight'].str.extract(
    r'([\\d.]+)\\s*(kg|lbs?)', expand=True
)
df['weight_value'] = pd.to_numeric(df['weight_value'])

# ── Split temperature ──
df[['temp_value', 'temp_unit']] = df['temperature'].str.extract(
    r'([\\d.]+)\\s*([CF])', expand=True
)
df['temp_value'] = pd.to_numeric(df['temp_value'])

print("\\nAfter Feature Splitting:")
print(df[['height_value', 'height_unit',
           'weight_value', 'weight_unit',
           'temp_value', 'temp_unit']])"""
    story.extend(code_block(code, 'Listing 33.1: Splitting mixed variables using regex extraction.'))
    
    story.append(H2('33.3.2 &nbsp; Regex Pattern Explanation'))
    story.append(P(
        'The regular expression <font face="Courier" size="9">r\'(\\d+(?:\\.\\d+)?)\\s*(feet|inches?)\'</font> '
        'works as follows:'
    ))
    story.extend(bullets([
        '<font face="Courier" size="9">\\d+</font> — matches one or more digits',
        '<font face="Courier" size="9">(?:\\.\\d+)?</font> — optionally matches a decimal point '
        'followed by digits (for values like 5.5)',
        '<font face="Courier" size="9">\\s*</font> — matches zero or more whitespace characters',
        '<font face="Courier" size="9">(feet|inches?)</font> — captures either "feet" or "inch"/"inches"',
    ]))
    
    story.append(page_break())
    story.extend(H1('33.4 &nbsp; Handling Unit Conversions'))
    
    story.append(P(
        'After splitting, the unit column may contain different units for the same feature '
        '(e.g., "feet" vs "inches", "kg" vs "lbs", "C" vs "F"). You should normalise these '
        'to a single unit before using the numerical values.'
    ))
    
    code = """# Convert all heights to centimeters
def height_to_cm(row):
    value = row['height_value']
    unit = row['height_unit']
    if unit == 'feet':
        return value * 30.48  # 1 foot = 30.48 cm
    elif unit in ('inch', 'inches'):
        return value * 2.54   # 1 inch = 2.54 cm
    return np.nan

df['height_cm'] = df.apply(height_to_cm, axis=1)

# Convert all temperatures to Celsius
def temp_to_celsius(row):
    value = row['temp_value']
    unit = row['temp_unit']
    if unit == 'C':
        return value
    elif unit == 'F':
        return (value - 32) * 5 / 9
    return np.nan

df['temp_celsius'] = df.apply(temp_to_celsius, axis=1)

print(df[['height_value', 'height_unit', 'height_cm']])
print()
print(df[['temp_value', 'temp_unit', 'temp_celsius']])"""
    story.extend(code_block(code, 'Listing 33.2: Normalising units after feature splitting.'))
    
    story.extend(H1('33.5 &nbsp; Practical Considerations'))
    story.extend(bullets([
        '<b>Always split before encoding:</b> Attempting to one-hot encode a mixed '
        'variable directly will treat each unique string as a category.',
        '<b>Validate extracted values:</b> Regex extraction can fail on unexpected '
        'formats. Check for NaN values after extraction.',
        '<b>Consider the domain:</b> Sometimes the categorical part carries more '
        'information than the numerical part (e.g., unit type might correlate with target).',
        '<b>Preserve the original:</b> Keep the original mixed column until you have '
        'verified the split features are correct.',
        '<b>Automated approaches:</b> For large-scale data, consider libraries like '
        '<font face="Courier" size="9">feature-engine</font> or custom transformers '
        'that can be integrated into a sklearn Pipeline.',
    ]))
    
    story.extend(H1('33.6 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'Mixed variables combine numerical and categorical information in one feature. '
        'The primary solution is feature splitting: extract numeric parts into one column '
        'and categorical parts into another using regex or string methods, then normalise '
        'units and encode categories separately.'
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 34: DATE AND TIME VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
def ch34_sections(story, num):
    story.extend(H1('34.1 &nbsp; Understanding DateTime Data'))
    story.append(P(
        'Date and time columns are ubiquitous in real-world datasets: transaction '
        'timestamps, user activity logs, sensor readings, weather records, and more. '
        'While they may appear as simple strings, datetime values encode a wealth of '
        'information that can be extracted into predictive features.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch34_datetime_extraction.png'),
        width=420,
        caption='Figure 34.1: A single datetime value can be decomposed into year, '
                'month, day, hour, minute, second, and day-of-week — each a potential '
                'feature for machine learning.'
    ))
    
    story.extend(H1('34.2 &nbsp; Converting Strings to DateTime Objects'))
    
    code = """import pandas as pd
import numpy as np

# Various datetime string formats
dates = pd.Series([
    '2024-01-15',
    '15/01/2024',
    'Jan 15, 2024',
    '2024-01-15 14:30:00',
    '2024-01-15T14:30:00.000Z',
    '15-01-2024 14:30'
])

# Convert to datetime objects
dt = pd.to_datetime(dates, format='mixed', dayfirst=True)
print("Converted datetime objects:")
print(dt)
print(f"\\nDtype: {dt.dtype}")

# Access datetime components
print(f"\\nYear:   {dt.dt.year.tolist()}")
print(f"Month:  {dt.dt.month.tolist()}")
print(f"Day:    {dt.dt.day.tolist()}")
print(f"Hour:   {dt.dt.hour.tolist()}")"""
    story.extend(code_block(code, 'Listing 34.1: Converting various string formats to datetime objects.'))
    
    story.append(warning_box(
        'Common DateTime Error',
        'If pandas cannot infer the date format, it raises a '
        '<font face="Courier" size="9">ParserError</font>. Always specify '
        '<font face="Courier" size="9">format=</font> when the format is known, or use '
        '<font face="Courier" size="9">format=\'mixed\'</font> (pandas ≥ 2.0) for '
        'mixed-format columns. Also watch for day-first vs month-first ambiguity: '
        '"01/02/2024" could be 1 Feb or 2 Jan depending on locale.'
    ))
    
    story.append(page_break())
    story.extend(H1('34.3 &nbsp; Extracting Datetime Components'))
    
    story.extend(make_table(
        ['Attribute / Method', 'Returns', 'Example (2024-03-15 14:30:45)'],
        [
            ['.dt.year', 'Year as integer', '2024'],
            ['.dt.month', 'Month (1-12)', '3'],
            ['.dt.day', 'Day of month (1-31)', '15'],
            ['.dt.hour', 'Hour (0-23)', '14'],
            ['.dt.minute', 'Minute (0-59)', '30'],
            ['.dt.second', 'Second (0-59)', '45'],
            ['.dt.dayofweek', 'Day of week (Mon=0, Sun=6)', '4 (Friday)'],
            ['.dt.day_name()', 'Day name string', '"Friday"'],
            ['.dt.dayofyear', 'Day of year (1-366)', '75'],
            ['.dt.is_month_start', 'First day of month?', 'False'],
            ['.dt.is_month_end', 'Last day of month?', 'False'],
            ['.dt.is_weekend', 'Saturday or Sunday?', 'False'],
            ['.dt.quarter', 'Quarter (1-4)', '1'],
            ['.dt.days_in_month', 'Days in the month', '31'],
            ['.dt.to_period("M")', 'Period (month)', '2024-03'],
        ],
        col_widths=[140, 160, 170],
        caption='Table 34.1: Useful Pandas datetime accessor attributes.'
    ))
    
    code = """import pandas as pd

# Sample transaction timestamps
df = pd.DataFrame({
    'transaction_id': range(1, 11),
    'timestamp': pd.to_datetime([
        '2024-01-15 09:30:00', '2024-01-16 14:22:00',
        '2024-01-17 18:45:00', '2024-01-18 11:15:00',
        '2024-01-19 20:30:00', '2024-01-20 08:00:00',
        '2024-01-21 16:10:00', '2024-01-22 13:55:00',
        '2024-01-23 19:20:00', '2024-01-24 10:40:00'
    ]),
    'amount': [120.50, 89.99, 250.00, 45.00, 310.75,
               199.99, 75.25, 420.00, 160.00, 95.50]
})

# Extract features
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_name'] = df['timestamp'].dt.day_name()
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
df['quarter'] = df['timestamp'].dt.quarter

# Time-based bins
df['time_of_day'] = pd.cut(df['hour'],
                            bins=[0, 6, 12, 18, 24],
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])

print(df[['timestamp', 'hour', 'day_name', 'is_weekend', 'time_of_day', 'amount']])
print(f"\\nAverage amount by time of day:")
print(df.groupby('time_of_day')['amount'].agg(['mean', 'count']))"""
    story.extend(code_block(code, 'Listing 34.2: Extracting datetime features for ML.'))
    
    story.append(page_break())
    story.extend(H1('34.4 &nbsp; Cyclical Encoding of DateTime Features'))
    
    story.append(P(
        'Features like month (1–12) and hour (0–23) are <b>cyclical</b>: December (12) '
        'is close to January (1), and hour 23 is close to hour 0. If you encode these as '
        'plain integers, the model sees December and January as far apart — which is wrong.'
    ))
    
    story.append(P(
        'The solution is <b>cyclical encoding</b> using sine and cosine transforms:'
    ))
    story.append(math_block(
        r'x_{\sin} = \sin\left(\frac{2\pi \cdot x}{\text{period}}\right), \quad '
        r'x_{\cos} = \cos\left(\frac{2\pi \cdot x}{\text{period}}\right)', '34.1'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch34_cyclical_encoding.png'),
        width=440,
        caption='Figure 34.2: Left: ordinal encoding places Sunday (6) far from Monday (0), '
                'even though they are adjacent days. Right: cyclical (sin/cos) encoding '
                'maps days onto a circle where Sunday and Monday are neighbours.'
    ))
    
    code = """# Cyclical encoding for hour (period=24) and month (period=12)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Verify: hours 23 and 0 should be close
h23 = df[df['hour'] == 23][['hour_sin', 'hour_cos']].values
h0 = np.array([np.sin(0), np.cos(0)])
print(f"Hour 23 encoding: {h23}")
print(f"Hour 0 encoding:  {h0}")
print(f"Distance: {np.linalg.norm(h23 - h0):.4f}  (small = close)")"""
    story.extend(code_block(code, 'Listing 34.3: Cyclical encoding with sine and cosine.'))
    
    story.extend(H1('34.5 &nbsp; Time-Based Feature Engineering'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch34_pipeline.png'),
        width=440,
        caption='Figure 34.3: The datetime feature engineering pipeline from raw '
                'timestamp to model-ready features.'
    ))
    
    story.extend(bullets([
        '<b>Lag features:</b> Value at time t−1, t−2, … (for time-series forecasting)',
        '<b>Rolling statistics:</b> Moving average over a window (e.g., 7-day mean)',
        '<b>Elapsed time:</b> Days since a reference event (e.g., account creation)',
        '<b>Recency:</b> Hours since last transaction',
        '<b>Frequency:</b> Transactions per day/week/month',
        '<b>Holidays and events:</b> Boolean flags for public holidays',
    ]))
    
    story.extend(H1('34.6 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'DateTime columns should be parsed with pd.to_datetime, decomposed into '
        'meaningful components (year, month, day, hour, day-of-week), and cyclical '
        'features encoded with sine/cosine. Time-based features like lags, rolling '
        'means, and recency add predictive power for sequential data.'
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 35: MISSING DATA — COMPLETE CASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def ch35_sections(story, num):
    story.extend(H1('35.1 &nbsp; Understanding Missing Data'))
    story.append(P(
        'Missing data is one of the most common problems in real-world machine learning. '
        'A value is "missing" when no information is recorded for a particular observation '
        'on a particular feature. In Pandas, missing values are represented as '
        '<font face="Courier" size="9">NaN</font> (Not a Number) for numeric data and '
        '<font face="Courier" size="9">None</font> for object data. In Scikit-learn, '
        'missing values are represented as <font face="Courier" size="9">np.nan</font>.'
    ))
    
    story.append(H2('35.1.1 &nbsp; Why Do Missing Values Occur?'))
    story.extend(bullets([
        '<b>Sensor failure:</b> A physical sensor malfunctioned during recording.',
        '<b>Human error:</b> A data-entry operator skipped a field.',
        '<b>Survey non-response:</b> A respondent left a question blank.',
        '<b>Data integration:</b> Merging datasets from different sources where some '
        'fields do not overlap.',
        '<b>Privacy restrictions:</b> Sensitive fields were deliberately withheld.',
        '<b>Conditional collection:</b> A follow-up question was only asked if the '
        'previous answer was "yes".',
    ]))
    
    story.append(page_break())
    story.extend(H1('35.2 &nbsp; Missingness Mechanisms'))
    
    story.append(P(
        'Understanding <i>why</i> data is missing is crucial because it determines which '
        'handling methods are statistically valid. Rubin (1976) classified missingness into '
        'three mechanisms:'
    ))
    
    story.extend(make_table(
        ['Mechanism', 'Abbreviation', 'Definition', 'Example'],
        [
            ['Missing Completely at Random', 'MCAR',
             'Missingness is independent of both observed and unobserved data',
             'Random equipment malfunction'],
            ['Missing at Random', 'MAR',
             'Missingness depends on observed data but not on the missing value itself',
             'Younger patients skip income question more often'],
            ['Missing Not at Random', 'MNAR',
             'Missingness depends on the missing value itself',
             'High-income earners refuse to report income'],
        ],
        col_widths=[100, 55, 170, 155],
        caption='Table 35.1: Rubin\'s three missingness mechanisms.'
    ))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch35_missing_patterns.png'),
        width=460,
        caption='Figure 35.1: Visual representation of missing data patterns. Green cells '
                'are observed; red cells are missing. MCAR shows random scatter. MAR shows '
                'dependence on other observed features. MNAR shows dependence on the '
                'unobserved value itself (often harder to detect).'
    ))
    
    story.append(note_box(
        'Why This Matters',
        'If data is <b>MCAR</b>, removing incomplete rows (CCA) produces unbiased results. '
        'If data is <b>MAR</b> or <b>MNAR</b>, CCA can introduce bias because the missing '
        'pattern is informative. Understanding the mechanism guides the choice of method.'
    ))
    
    story.append(page_break())
    story.extend(H1('35.3 &nbsp; Complete Case Analysis (CCA)'))
    
    story.append(P(
        '<b>Complete Case Analysis</b> (also called listwise deletion) removes any row '
        'that contains at least one missing value. Only rows where every feature is '
        'observed are retained.'
    ))
    
    story.append(math_block(
        r'\text{CCA keeps row } i \iff \forall j:\ x_{ij} \neq \text{NaN}', '35.1'))
    
    story.append(H2('35.3.1 &nbsp; Python Implementation'))
    
    code = """import pandas as pd
import numpy as np

# Create dataset with missing values
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age': np.random.normal(35, 10, n),
    'income': np.random.lognormal(10.5, 0.7, n),
    'score': np.random.normal(70, 15, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n)
})

# Introduce missing values (MCAR - random)
for col in ['age', 'income', 'score']:
    mask = np.random.random(n) < 0.15
    df.loc[mask, col] = np.nan

print("=== MISSING DATA SUMMARY ===")
print(df.isnull().sum())
print(f"\\nTotal rows: {len(df)}")
print(f"Rows with any missing: {df.isnull().any(axis=1).sum()}")
print(f"Percentage missing rows: {df.isnull().any(axis=1).mean()*100:.1f}%")

# Apply CCA
df_complete = df.dropna()
print(f"\\n=== AFTER CCA ===")
print(f"Rows remaining: {len(df_complete)}")
print(f"Rows removed:   {len(df) - len(df_complete)}")
print(f"Data loss:      {(1 - len(df_complete)/len(df))*100:.1f}%")

# Compare statistics before and after
print(f"\\n=== STATISTICAL COMPARISON ===")
print(f"{'Feature':<10} {'Full Mean':>10} {'CCA Mean':>10} {'Full Std':>10} {'CCA Std':>10}")
for col in ['age', 'income', 'score']:
    print(f"{col:<10} {df[col].mean():>10.2f} {df_complete[col].mean():>10.2f} "
          f"{df[col].std():>10.2f} {df_complete[col].std():>10.2f}")"""
    story.extend(code_block(code, 'Listing 35.1: Complete Case Analysis implementation.'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch35_cca_impact.png'),
        width=440,
        caption='Figure 35.2: Left: CCA removes a substantial fraction of data. '
                'Right: Under MCAR, CCA preserves the distribution shape but reduces '
                'sample size, increasing the variance of parameter estimates.'
    ))
    
    story.append(page_break())
    story.extend(H1('35.4 &nbsp; When CCA is Appropriate'))
    
    story.append(P('CCA is statistically valid when:'))
    story.extend(bullets([
        'The missingness mechanism is <b>MCAR</b> (missing completely at random).',
        'The proportion of missing data is <b>small</b> (typically &lt; 5%).',
        'The dataset is <b>large enough</b> that losing some rows does not critically '
        'reduce statistical power.',
        'The missing values are not systematically related to the target variable.',
    ]))
    
    story.extend(H1('35.5 &nbsp; When CCA Should be Avoided'))
    story.extend(bullets([
        'Missingness is <b>MAR</b> or <b>MNAR</b> — deletion introduces bias.',
        'The <b>percentage of missing data is high</b> (e.g., &gt; 25%), leading to '
        'substantial information loss.',
        'The dataset is already <b>small</b> — every row matters.',
        'Missingness is <b>concentrated in important features</b> — dropping those rows '
        'removes valuable observations.',
        'The missing pattern is <b>systematic</b> — e.g., all patients who answered '
        '"yes" to a screening question have a follow-up value missing.',
    ]))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch35_cca_decision.png'),
        width=400,
        caption='Figure 35.3: Decision framework for choosing between CCA and imputation.'
    ))
    
    story.extend(H1('35.6 &nbsp; Key Takeaways'))
    story.append(takeaway(
        'Complete Case Analysis removes all rows containing any missing value. It is '
        'valid only when data is MCAR and the missing proportion is small. For MAR/MNAR '
        'or high missingness, imputation methods (Chapters 36–40) are preferred.'
    ))
    
    story.extend(H1('35.7 &nbsp; Interview Questions'))
    story.extend(numbered([
        'Explain the difference between MCAR, MAR, and MNAR with examples.',
        'Under what conditions is Complete Case Analysis statistically valid?',
        'What happens to the variance of parameter estimates when you apply CCA?',
        'How would you test whether data is MCAR?',
        'Why is CCA biased when data is MNAR?',
    ]))

# Build chapters 32-35
print('Building Chapters 32-35...')
build_chapter(32, 'Binning and Binarization',
              'Discretization, Equal-Width, Quantile, K-Means Binning, Binarizer',
              ch32_sections)
build_chapter(33, 'Handling Mixed Variables',
              'Feature Extraction, Splitting, Unit Conversion, Encoding',
              ch33_sections)
build_chapter(34, 'Handling Date and Time Variables',
              'DateTime Parsing, Feature Extraction, Cyclical Encoding, Engineering',
              ch34_sections)
build_chapter(35, 'Missing Data: Complete Case Analysis',
              'Missingness Mechanisms, MCAR/MAR/MNAR, CCA Decision Framework',
              ch35_sections)
print('Chapters 32-35 done.')
