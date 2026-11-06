"""Test the PDF framework with Chapter 31 content."""
import sys
sys.path.insert(0, r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40')
from pdf_framework import *

FIGDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40\figures'
OUTDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40'

def build_chapter_31(output_path=None):
    """Build Chapter 31: Power Transformer."""
    if output_path is None:
        output_path = os.path.join(OUTDIR, 'Chapter_31_Power_Transformer.pdf')
    
    doc = MLDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=55, rightMargin=55,
        topMargin=55, bottomMargin=55,
        chapter_title='Chapter 31 — Power Transformer',
        title='Chapter 31: Power Transformer'
    )
    
    story = []
    
    # ── Cover / Title ──────────────────────────────────────────────────────
    story.append(spacer(30))
    story.append(Paragraph('CHAPTER 31', STYLES['chapter_num']))
    story.append(Paragraph('Power Transformer', STYLES['title']))
    story.append(Paragraph(
        'Skewness, Box-Cox &amp; Yeo-Johnson Transformations, Scikit-learn Implementation',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=10, spaceAfter=10))
    story.append(Paragraph(
        'CampusX — 100 Days of Machine Learning<br/>'
        'Feature Engineering Series, Videos 31–40',
        STYLES['body_center']))
    story.append(spacer(20))
    
    # Learning objectives box
    objectives = (
        'By the end of this chapter you will understand: why skewed distributions '
        'hurt machine-learning models, how power transformations restore symmetry, '
        'the mathematics behind Box-Cox and Yeo-Johnson methods, how to implement '
        'them with Scikit-learn, and when to choose one method over the other.'
    )
    story.append(note_box('Learning Objectives', objectives))
    story.append(page_break())
    
    # ── TOC placeholder (combined PDF will have real TOC) ─────────────────
    # For standalone chapter, skip TOC or add simple one
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: Introduction and Problem Statement
    # ══════════════════════════════════════════════════════════════════════
    story.extend(H1('31.1 &nbsp; Introduction and Problem Statement'))
    
    story.append(P(
        'Most machine-learning algorithms make assumptions about the shape of the data '
        'they receive. Linear regression, for example, assumes that the relationship '
        'between features and the target is linear and that residuals are normally '
        'distributed. Logistic regression and many distance-based methods implicitly '
        'benefit from features that are roughly symmetric. When a feature is heavily '
        'skewed — meaning most values cluster on one side with a long tail on the other — '
        'these assumptions break down, and model performance can degrade significantly.'
    ))
    
    story.append(P(
        '<b>Skewness</b> is a measure of the asymmetry of a probability distribution '
        'around its mean. A perfectly symmetric distribution (like the Gaussian / normal '
        'distribution) has a skewness of zero. A distribution with a long right tail has '
        'positive skewness; one with a long left tail has negative skewness.'
    ))
    
    story.append(P(
        'In real-world datasets, skewed features are extremely common. Income data is '
        'almost always right-skewed (a few people earn very large amounts). House prices '
        'are right-skewed. Transaction amounts, response times, and many sensor readings '
        'also exhibit skewness. If we feed such features directly into a model, a small '
        'number of extreme values can dominate the learning process, pull the decision '
        'boundary, and reduce overall accuracy.'
    ))
    
    story.append(P(
        '<b>Power transformations</b> are a family of monotonic mathematical functions '
        'that map a continuous variable to a new scale in a way that makes its distribution '
        'more symmetric (closer to a normal distribution). The transformation applies the '
        'same function to every value in the feature, preserving the rank order of the '
        'observations while reshaping the overall distribution.'
    ))
    
    # ── Figure: skewness types ─────────────────────────────────────────────
    story.extend(figure(
        os.path.join(FIGDIR, 'ch31_skewness_types.png'),
        width=460,
        caption='Figure 31.1: Three fundamental distribution shapes. Left: symmetric '
                '(mean = median, skewness = 0). Centre: right-skewed / positive skew '
                '(mean > median, long right tail). Right: left-skewed / negative skew '
                '(mean < median, long left tail).'
    ))
    
    story.append(H2('31.1.1 &nbsp; Why Skewness Matters for Machine Learning'))
    
    story.append(P(
        'Consider a linear-regression model that predicts house price from square footage. '
        'If square footage is heavily right-skewed, most houses cluster around a small '
        'value while a few mansions stretch far to the right. The ordinary least-squares '
        'criterion minimises the sum of squared residuals, so extreme points contribute '
        'quadratically more to the loss. The fitted line is therefore pulled toward those '
        'extremes, and predictions for typical houses become less accurate.'
    ))
    
    story.append(P(
        'Distance-based algorithms such as k-nearest neighbours, k-means clustering, and '
        'support vector machines are even more sensitive. They compute Euclidean distances '
        'between feature values. A single feature with a range of 0–1,000,000 will dwarf '
        'a feature with a range of 0–1, effectively silencing the smaller feature entirely.'
    ))
    
    story.append(P(
        'Tree-based methods (decision trees, random forests, gradient boosting) are less '
        'affected because they split on rank order rather than absolute values. However, '
        'even tree-based models can benefit from power transformations when the transformed '
        'feature yields more informative split points.'
    ))
    
    story.append(note_box(
        'Key Insight',
        'Power transformations do not change the <b>rank order</b> of observations. '
        'If observation A had a larger value than observation B before the transformation, '
        'it will still have a larger value after. This monotonicity property is what makes '
        'power transformations safe for most machine-learning pipelines.'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: Understanding Skewness
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.2 &nbsp; Understanding Skewness'))
    
    story.append(H2('31.2.1 &nbsp; Mathematical Definition'))
    
    story.append(P(
        'The <b>skewness</b> of a random variable X with mean μ and standard deviation σ '
        'is defined as the normalised third central moment:'
    ))
    story.append(math_block(r'\text{Skewness}(X) = \frac{E\left[(X - \mu)^3\right]}{\sigma^3}', '31.1'))
    
    story.append(P(
        'where E[·] denotes the expected value (population mean). The numerator is the '
        'third central moment, and dividing by σ³ makes the measure dimensionless so it '
        'can be compared across features with different units.'
    ))
    
    story.append(H3('Interpretation'))
    story.extend(bullets([
        '<b>Skewness ≈ 0</b>: The distribution is approximately symmetric.',
        '<b>Skewness &gt; 0</b> (positive): The right tail is longer; the mass of the '
        'distribution is concentrated on the left. Mean &gt; Median.',
        '<b>Skewness &lt; 0</b> (negative): The left tail is longer; the mass is on the '
        'right. Mean &lt; Median.',
    ]))
    
    story.append(P(
        'In practice, a common rule of thumb is that absolute skewness less than 0.5 '
        'indicates an approximately symmetric distribution, values between 0.5 and 1 '
        'indicate moderate skewness, and values above 1 indicate high skewness that '
        'warrants transformation.'
    ))
    
    story.append(H2('31.2.2 &nbsp; Sample Skewness'))
    
    story.append(P(
        'For a sample of n observations, the sample skewness (Fisher-Pearson coefficient) '
        'is computed as:'
    ))
    story.append(math_block(
        r'g_1 = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^3}'
              r'{\left[\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right]^{3/2}}', '31.2'))
    
    story.append(P(
        'Pandas provides this directly via <font face="Courier" size="9">'
        'df[\'col\'].skew()</font>. Scipy provides it via '
        '<font face="Courier" size="9">scipy.stats.skew(data)</font>.'
    ))
    
    story.append(H2('31.2.3 &nbsp; Symmetric vs Asymmetric Distributions'))
    
    story.append(P(
        'A distribution is <b>symmetric</b> if it looks the same on the left and right '
        'sides of its centre. The normal (Gaussian) distribution is the classic example. '
        'For a symmetric distribution, the mean, median, and mode all coincide.'
    ))
    
    story.append(P(
        'An <b>asymmetric</b> (skewed) distribution has different behaviour on each side. '
        'In a right-skewed distribution the mean is pulled toward the long right tail, '
        'so mean &gt; median. In a left-skewed distribution the mean is pulled toward the '
        'long left tail, so mean &lt; median. This relationship between mean and median '
        'is a quick diagnostic for skewness direction.'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: Mathematical Foundations
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.3 &nbsp; Mathematical Foundations of Power Transformations'))
    
    story.append(P(
        'A <b>power transformation</b> is a function of the form:'
    ))
    story.append(math_block(r'T(x; \lambda) = \begin{cases} \dfrac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\[6pt] \ln(x) & \text{if } \lambda = 0 \end{cases}', '31.3'))
    
    story.append(P(
        'where x is the original value and λ (lambda) is a real-valued parameter that '
        'controls the shape of the transformation. Different values of λ produce different '
        'transformations from the same family.'
    ))
    
    story.append(H2('31.3.1 &nbsp; The Role of Lambda'))
    
    story.append(P(
        'Lambda is the free parameter that the algorithm estimates from the data. '
        'The goal is to choose λ so that the transformed data is as close to normally '
        'distributed as possible. Each classical choice of λ has an intuitive meaning:'
    ))
    
    story.extend(make_table(
        ['Lambda (λ)', 'Transformation', 'Formula', 'Use Case'],
        [
            ['λ = 2', 'Square', 'x²', 'Left-skewed data (mild)'],
            ['λ = 1', 'Identity (no change)', 'x', 'Data already symmetric'],
            ['λ = 0.5', 'Square root', '√x', 'Moderately right-skewed count data'],
            ['λ = 0', 'Natural logarithm', 'ln(x)', 'Heavily right-skewed data'],
            ['λ = −1', 'Reciprocal', '1/x', 'Strongly right-skewed data'],
        ],
        col_widths=[70, 100, 120, 180],
        caption='Table 31.1: Common lambda values and their corresponding transformations.'
    ))
    
    # ── Figure: Lambda effect ─────────────────────────────────────────────
    story.extend(figure(
        os.path.join(FIGDIR, 'ch31_lambda_effect.png'),
        width=400,
        caption='Figure 31.2: The Box-Cox transformation applied to x ∈ (0, 5) for '
                'different λ values. As λ decreases from 2 to 0, the curve becomes '
                'increasingly compressive on the right tail — exactly what is needed '
                'to tame right-skewed data.'
    ))
    
    story.append(P(
        'The function T(x; λ) is <b>monotonic increasing</b> for all λ. This means it '
        'preserves the ordering of data points: if x₁ &lt; x₂ then T(x₁; λ) &lt; T(x₂; λ). '
        'This property is crucial because it means no information about relative ranking '
        'is lost during the transformation.'
    ))
    
    story.append(H2('31.3.2 &nbsp; Continuity at λ = 0'))
    
    story.append(P(
        'You might notice that the formula divides by λ, which would be undefined at '
        'λ = 0. However, taking the limit as λ → 0 using L\'Hôpital\'s rule gives:'
    ))
    story.append(math_block(
        r'\lim_{\lambda \to 0} \frac{x^\lambda - 1}{\lambda} = \lim_{\lambda \to 0} '
        r'\frac{\frac{d}{d\lambda}(x^\lambda - 1)}{\frac{d}{d\lambda}\lambda} = '
        r'\lim_{\lambda \to 0} \frac{x^\lambda \ln x}{1} = \ln x', '31.4'))
    
    story.append(P(
        'This is why the log transformation (λ = 0) is defined as a separate case — '
        'it is the continuous limit of the power transformation as λ approaches zero.'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: Box-Cox
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.4 &nbsp; Box-Cox Transformation'))
    
    story.append(P(
        'The Box-Cox transformation, introduced by George Box and David Cox in 1964, '
        'is the most widely used power transformation. It finds the optimal λ by '
        'maximising the log-likelihood of the transformed data under a normal model.'
    ))
    
    story.append(H2('31.4.1 &nbsp; Definition'))
    
    story.append(P(
        'For a vector of positive observations x = (x₁, x₂, …, xₙ), all xᵢ &gt; 0, '
        'the Box-Cox transformation of each element is:'
    ))
    story.append(math_block(
        r'y_i = T(x_i; \lambda) = \begin{cases} \dfrac{x_i^\lambda - 1}{\lambda} & \lambda \neq 0 \\[6pt] \ln(x_i) & \lambda = 0 \end{cases}', '31.5'))
    
    story.append(warning_box(
        'Critical Restriction',
        'The Box-Cox transformation is <b>only defined for strictly positive values</b> '
        '(xᵢ &gt; 0 for all i). If any value in the feature is zero or negative, '
        'Box-Cox cannot be applied directly. A common workaround is to shift all values '
        'by a constant: x\' = x − min(x) + δ where δ is a small positive number. '
        'However, this shifts the optimal λ and can affect results. The Yeo-Johnson '
        'transformation (Section 31.5) solves this problem entirely.'
    ))
    
    story.append(H2('31.4.2 &nbsp; How Lambda is Estimated'))
    
    story.append(P(
        'Box and Cox proposed estimating λ by maximising the log-likelihood function '
        'under the assumption that the transformed data follow a normal distribution. '
        'The log-likelihood is:'
    ))
    
    story.append(math_block(
        r'\ell(\lambda) = -\frac{n}{2}\ln\left[\frac{1}{n}\sum_{i=1}^{n}'
        r'(y_i - \bar{y})^2\right] + (\lambda - 1)\sum_{i=1}^{n}\ln(x_i)', '31.6'))
    
    story.append(P(
        'where yᵢ = T(xᵢ; λ) and ȳ is the mean of the transformed values. The first term '
        'captures how well the transformed data fits a normal distribution (smaller '
        'residual variance is better). The second term is a Jacobian correction that '
        'accounts for the change of variable — it prevents the maximisation from simply '
        'shrinking all values toward zero.'
    ))
    
    story.append(P(
        'In practice, the optimisation is performed numerically over a grid of λ values '
        '(typically from −5 to +5), or using Newton-Raphson / gradient-based optimisation. '
        'Scipy\'s <font face="Courier" size="9">stats.boxcox()</font> and Scikit-learn\'s '
        '<font face="Courier" size="9">PowerTransformer</font> both use efficient numerical '
        'methods internally.'
    ))
    
    story.append(H2('31.4.3 &nbsp; Numerical Example'))
    
    story.append(P(
        'Suppose we have five positive observations and want to find the optimal λ by '
        'hand (simplified grid search):'
    ))
    
    code = """import numpy as np
from scipy.stats import boxcox

# Five positive observations (right-skewed)
data = np.array([1.2, 3.5, 7.8, 15.2, 45.6, 8.1, 12.3, 2.4, 20.5, 5.7])

# Find optimal lambda and transformed data
transformed, optimal_lambda = boxcox(data)

print(f"Optimal lambda: {optimal_lambda:.4f}")
print(f"Original data:  {data}")
print(f"Transformed:    {transformed}")"""
    story.extend(code_block(code, 'Listing 31.1: Estimating the optimal lambda with scipy.stats.boxcox.'))
    
    story.append(P(
        'The function returns the transformed array and the estimated λ. You can then '
        'apply the same λ to new data using the inverse operation, or simply refit the '
        'transformer on the training set and apply it to test data.'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: Yeo-Johnson
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.5 &nbsp; Yeo-Johnson Transformation'))
    
    story.append(P(
        'The Yeo-Johnson transformation, introduced by Yeo and Johnson in 2000, is a '
        'generalisation of the Box-Cox transformation that works for <b>all real numbers</b>, '
        'including zero and negative values. This is its primary advantage over Box-Cox.'
    ))
    
    story.append(H2('31.5.1 &nbsp; Definition'))
    
    story.append(P(
        'For any real value x and parameter λ, the Yeo-Johnson transformation is defined '
        'piecewise:'
    ))
    
    story.append(math_block(
        r'T_{YJ}(x; \lambda) = \begin{cases} \dfrac{(x + 1)^\lambda - 1}{\lambda} & x \geq 0,\ \lambda \neq 0 \\[6pt] \ln(x + 1) & x \geq 0,\ \lambda = 0 \\[6pt] -\dfrac{(-x + 1)^{2-\lambda} - 1}{2 - \lambda} & x < 0,\ \lambda \neq 2 \\[6pt] -\ln(-x + 1) & x < 0,\ \lambda = 2 \end{cases}', '31.7'))
    
    story.append(P(
        'Notice how the formula switches its form depending on whether x is non-negative '
        'or negative. For x ≥ 0 it behaves like a shifted Box-Cox on (x + 1). For x &lt; 0 '
        'it uses a different power that involves (2 − λ). This two-piece construction '
        'ensures the transformation is defined, continuous, and monotonically increasing '
        'for all real x.'
    ))
    
    story.append(H2('31.5.2 &nbsp; Box-Cox vs Yeo-Johnson — Direct Comparison'))
    
    story.extend(make_table(
        ['Property', 'Box-Cox', 'Yeo-Johnson'],
        [
            ['Input domain', 'x &gt; 0 only', 'All real numbers (x ∈ ℝ)'],
            ['Handles zeros?', 'No (must shift)', 'Yes'],
            ['Handles negatives?', 'No', 'Yes'],
            ['Formula complexity', 'Simpler (single expression)', 'More complex (4-piece)'],
            ['Estimation method', 'Maximise log-likelihood', 'Maximise log-likelihood'],
            ['Scikit-learn method', 'method="box-cox"', 'method="yeo-johnson"'],
            ['Default in sklearn?', 'No', 'Yes'],
            ['Introduced', '1964', '2000'],
        ],
        col_widths=[120, 170, 190],
        caption='Table 31.2: Comparison of Box-Cox and Yeo-Johnson transformations.'
    ))
    
    # ── Figure: domain comparison ─────────────────────────────────────────
    story.extend(figure(
        os.path.join(FIGDIR, 'ch31_boxcox_yeojohnson_domain.png'),
        width=420,
        caption='Figure 31.3: Domain of validity. Box-Cox (left) requires all input values '
                'to be strictly positive (x &gt; 0). Yeo-Johnson (right) is valid for all '
                'real numbers including zero and negatives.'
    ))
    
    story.append(H2('31.5.3 &nbsp; Lambda Interpretation for Yeo-Johnson'))
    
    story.append(P(
        'The interpretation of λ is similar to Box-Cox but not identical. For Yeo-Johnson:'
    ))
    story.extend(bullets([
        'λ = 1 → identity (no transformation)',
        'λ = 0 → approximately log-like transformation for positive values',
        'λ = 0.5 → approximately square-root transformation for positive values',
        'λ = 2 → approximately square transformation for positive values',
    ]))
    story.append(P(
        'The exact mapping differs slightly from Box-Cox because of the (x + 1) shift '
        'in the formula. The estimated λ should always be checked against the transformed '
        'distribution rather than interpreted in isolation.'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: Scikit-learn Implementation
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.6 &nbsp; Scikit-learn PowerTransformer'))
    
    story.append(P(
        'Scikit-learn provides the <font face="Courier" size="10">PowerTransformer</font> '
        'class in <font face="Courier" size="10">sklearn.preprocessing</font>. It supports '
        'both Box-Cox and Yeo-Johnson methods and automatically estimates the optimal λ '
        'during the <font face="Courier" size="9">fit()</font> step.'
    ))
    
    story.append(H2('31.6.1 &nbsp; Class Parameters'))
    
    story.extend(make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['method', '"yeo-johnson"', 'Transformation type: "box-cox" or "yeo-johnson"'],
            ['standardize', 'True', 'If True, center to mean=0 and scale to variance=1 after transform'],
        ],
        col_widths=[90, 100, 290],
        caption='Table 31.3: PowerTransformer parameters.'
    ))
    
    story.append(H2('31.6.2 &nbsp; Key Attributes'))
    
    story.extend(make_table(
        ['Attribute', 'Description'],
        [
            ['lambda_', 'Estimated λ for each feature (array of shape [n_features,])'],
            ['n_features_in_', 'Number of features seen during fit'],
            ['feature_names_in_', 'Feature names (if fitted with named features)'],
        ],
        col_widths=[130, 350],
        caption='Table 31.4: PowerTransformer attributes after fitting.'
    ))
    
    story.append(H2('31.6.3 &nbsp; Full Implementation Example'))
    
    code = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew

# Step 1: Create a realistically skewed dataset
np.random.seed(42)
n_samples = 1000

# Income data — typically right-skewed (log-normal)
income = np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples)

# Add a second feature with moderate skew
age_squared = np.random.uniform(18, 70, n_samples) ** 1.5 / 100

df = pd.DataFrame({
    'income': income,
    'age_transformed': age_squared
})

print("=== BEFORE TRANSFORMATION ===")
print(f"Skewness (income): {df['income'].skew():.4f}")
print(f"Skewness (age):    {df['age_transformed'].skew():.4f}")
print(f"Income range:      {df['income'].min():.0f} to {df['income'].max():.0f}")
print()

# Step 2: Apply Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson')
df_transformed = pd.DataFrame(
    pt.fit_transform(df),
    columns=df.columns
)

print("=== AFTER YEO-JOHNSON TRANSFORMATION ===")
print(f"Skewness (income): {df_transformed['income'].skew():.4f}")
print(f"Skewness (age):    {df_transformed['age_transformed'].skew():.4f}")
print(f"Estimated lambdas: {pt.lambda_}")
print()

# Step 3: Visualize before and after
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Before
axes[0, 0].hist(df['income'], bins=50, color='steelblue', edgecolor='white')
axes[0, 0].set_title(f'Income BEFORE\\nSkewness = {df["income"].skew():.2f}')
axes[0, 0].set_xlabel('Income ($)')

axes[0, 1].hist(df['age_transformed'], bins=50, color='coral', edgecolor='white')
axes[0, 1].set_title(f'Age Feature BEFORE\\nSkewness = {df["age_transformed"].skew():.2f}')

# After
axes[1, 0].hist(df_transformed['income'], bins=50, color='steelblue', edgecolor='white')
axes[1, 0].set_title(f'Income AFTER Yeo-Johnson\\nSkewness = {df_transformed["income"].skew():.2f}')
axes[1, 0].set_xlabel('Transformed Value')

axes[1, 1].hist(df_transformed['age_transformed'], bins=50, color='coral', edgecolor='white')
axes[1, 1].set_title(f'Age Feature AFTER Yeo-Johnson\\nSkewness = {df_transformed["age_transformed"].skew():.2f}')

plt.tight_layout()
plt.savefig('power_transform_result.png', dpi=150)
plt.show()"""
    
    story.extend(code_block(code, 'Listing 31.2: Complete PowerTransformer workflow with visualization.'))
    
    story.append(H2('31.6.4 &nbsp; Box-Cox Implementation'))
    
    code = """from sklearn.preprocessing import PowerTransformer

# Box-Cox requires ALL values to be strictly positive
pt_bc = PowerTransformer(method='box-cox')

# This will raise an error if any value <= 0:
# pt_bc.fit_transform(df_with_negative_values)  # ValueError!

# For positive-only data:
pt_bc.fit(df_positive)
print(f"Box-Cox lambdas: {pt_bc.lambda_}")

# Transform the data
df_bc = pd.DataFrame(
    pt_bc.transform(df_positive),
    columns=df_positive.columns
)"""
    story.extend(code_block(code, 'Listing 31.3: Box-Cox implementation with positivity check.'))
    
    story.append(H2('31.6.5 &nbsp; Standardization Option'))
    
    story.append(P(
        'When <font face="Courier" size="9">standardize=True</font> (the default), '
        'the transformer additionally scales the output so that each feature has '
        'mean = 0 and variance = 1. This is a two-step process:'
    ))
    
    story.extend(numbered([
        'Apply the power transformation T(x; λ) to each value.',
        'Standardise the result: z = (y − ȳ) / s, where ȳ is the sample mean and '
        's is the sample standard deviation of the transformed values.',
    ]))
    
    story.append(P(
        'If you only want the power transformation without standardisation, set '
        '<font face="Courier" size="9">standardize=False</font>. This is useful when '
        'you plan to apply a separate scaling method (e.g., MinMaxScaler) later in '
        'the pipeline.'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7: Before and After Visualization
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.7 &nbsp; Before-and-After Comparison'))
    
    story.extend(figure(
        os.path.join(FIGDIR, 'ch31_power_transform_effect.png'),
        width=460,
        caption='Figure 31.4: Six panels showing the effect of different power '
                'transformations on the same right-skewed dataset. The top-left panel '
                'shows the original distribution. Each other panel shows the result of '
                'applying a different λ. The Box-Cox and Yeo-Johnson panels use the '
                'automatically estimated optimal λ.'
    ))
    
    story.append(H2('31.7.1 &nbsp; Quantitative Assessment'))
    
    story.append(P(
        'After transformation, you should verify the improvement quantitatively:'
    ))
    
    code = """from scipy.stats import skew, kurtosis, shapiro

def assess_normality(data, name="Feature"):
    \"\"\"Print skewness, kurtosis, and Shapiro-Wilk test.\"\"\"
    s = skew(data)
    k = kurtosis(data)  # excess kurtosis (0 for normal)
    stat, p_value = shapiro(data[:500])  # Shapiro limited to 500 samples
    print(f"{name}:")
    print(f"  Skewness:  {s:.4f}  (ideal: 0)")
    print(f"  Kurtosis:  {k:.4f}  (ideal: 0)")
    print(f"  Shapiro-Wilk p-value: {p_value:.4f}  (>0.05 suggests normal)")
    print()

# Assess before
assess_normality(df['income'], "Income BEFORE")

# Assess after
assess_normality(df_transformed['income'], "Income AFTER PowerTransformer")"""
    story.extend(code_block(code, 'Listing 31.4: Quantitative assessment of normality before and after transformation.'))
    
    story.append(P(
        'After a successful power transformation, you should observe that the absolute '
        'skewness has decreased substantially (ideally toward zero), the kurtosis has '
        'moved closer to zero (the excess kurtosis of a normal distribution), and the '
        'Shapiro-Wilk test p-value has increased (though with n=1000 the test is very '
        'sensitive and may still reject perfect normality).'
    ))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8: Advantages, Limitations, Common Mistakes
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.8 &nbsp; Advantages, Limitations and Common Mistakes'))
    
    story.append(H2('31.8.1 &nbsp; Advantages'))
    story.extend(bullets([
        '<b>Model-agnostic:</b> Power transformations can be applied before any '
        'machine-learning algorithm, not just specific ones.',
        '<b>Monotonic:</b> Rank order is preserved, so no information about relative '
        'ordering is lost.',
        '<b>Automatic λ estimation:</b> Scikit-learn automatically finds the optimal λ '
        'during fitting — no manual tuning required.',
        '<b>Interpretable:</b> The estimated λ tells you how much skewness correction '
        'was needed.',
        '<b>Computationally cheap:</b> The transformation is a simple element-wise '
        'operation — O(n) time and space.',
    ]))
    
    story.append(H2('31.8.2 &nbsp; Limitations'))
    story.extend(bullets([
        '<b>Box-Cox requires positive data:</b> If your feature contains zeros or '
        'negatives, use Yeo-Johnson instead.',
        '<b>Does not guarantee normality:</b> Power transformations make the distribution '
        'more symmetric but cannot always achieve perfect normality, especially for '
        'multimodal distributions.',
        '<b>Data leakage risk:</b> The λ is estimated from the data. You must fit the '
        'transformer on the training set only and apply it to the test set. Fitting on '
        'the entire dataset leaks information.',
        '<b>Interpretability loss:</b> After transformation, feature values are on a '
        'new scale. Coefficients of linear models refer to the transformed scale, which '
        'is harder to explain to stakeholders.',
        '<b>Not useful for tree-based models:</b> Decision trees, random forests, and '
        'gradient boosting are invariant to monotonic transformations, so power '
        'transformations provide no benefit for these models.',
    ]))
    
    story.append(H2('31.8.3 &nbsp; Common Mistakes'))
    
    story.append(warning_box(
        'Mistake 1: Fitting on the entire dataset',
        'Always split your data into training and test sets FIRST, then fit the '
        'PowerTransformer on the training set only. If you fit on the entire dataset, '
        'the estimated λ and the standardisation statistics (mean, std) leak information '
        'from the test set into the training process.'
    ))
    
    story.append(warning_box(
        'Mistake 2: Using Box-Cox on non-positive data',
        'Box-Cox requires all values to be strictly positive. If your data contains '
        'zeros or negatives, Scikit-learn will raise a ValueError. Either switch to '
        'Yeo-Johnson or carefully shift the data (knowing this affects the result).'
    ))
    
    story.append(warning_box(
        'Mistake 3: Forgetting to inverse-transform predictions',
        'If you transform the target variable, you must inverse-transform the model '
        'predictions before evaluation. Otherwise, your metrics (MSE, MAE) will be on '
        'the transformed scale, not the original scale.'
    ))
    
    story.append(H2('31.8.4 &nbsp; Complete Pipeline Example'))
    
    code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Synthetic dataset with skewed features
np.random.seed(42)
n = 1000
X = pd.DataFrame({
    'income': np.random.lognormal(10.5, 0.8, n),
    'spending': np.random.lognormal(8.0, 1.0, n),
    'age': np.random.uniform(18, 70, n)
})
y = (2.5 * np.log(X['income']) + 1.5 * np.log(X['spending']) +
     0.1 * X['age'] + np.random.normal(0, 1, n))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit transformer on TRAINING data only
pt = PowerTransformer(method='yeo-johnson')
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)  # Apply, do not fit!

print(f"Estimated lambdas: {pt.lambda_}")

# Train model on transformed features
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_transformed)
print(f"R² Score:   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:       {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")"""
    
    story.extend(code_block(code, 'Listing 31.5: Proper train/test pipeline with PowerTransformer.'))
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 9: Key Takeaways
    # ══════════════════════════════════════════════════════════════════════
    story.append(page_break())
    story.extend(H1('31.9 &nbsp; Key Takeaways'))
    
    story.append(takeaway(
        'Power transformations reshape skewed distributions toward symmetry by applying '
        'a monotonic function controlled by a single parameter λ. Box-Cox works for '
        'positive data only; Yeo-Johnson works for all real numbers. Always fit on '
        'training data and apply to test data.'
    ))
    
    story.append(spacer(10))
    
    story.append(H2('31.9.1 &nbsp; Quick Reference'))
    
    story.extend(make_table(
        ['Concept', 'Key Point'],
        [
            ['Skewness', 'Measure of distribution asymmetry; 0 = symmetric'],
            ['Box-Cox', 'Requires x &gt; 0; optimises log-likelihood over λ'],
            ['Yeo-Johnson', 'Works for all real x; piecewise formula'],
            ['Scikit-learn', 'PowerTransformer(method="yeo-johnson"|"box-cox")'],
            ['Lambda attribute', 'pt.lambda_ after fitting'],
            ['Standardize', 'Default True: center to 0, scale to variance 1'],
            ['Data leakage', 'Fit on train, transform on test — never fit on test'],
            ['Tree models', 'No benefit — trees are invariant to monotonic transforms'],
        ],
        col_widths=[140, 340],
        caption='Table 31.5: Quick reference for Power Transformer concepts.'
    ))
    
    story.append(H2('31.9.2 &nbsp; Revision Questions'))
    
    story.extend(numbered([
        'What is skewness and how do you measure it?',
        'Why do power transformations help linear regression but not decision trees?',
        'State the Box-Cox formula and explain the role of λ.',
        'What is the main restriction of Box-Cox, and how does Yeo-Johnson overcome it?',
        'What does PowerTransformer(standardize=True) do after the power transformation?',
        'Why must you fit the transformer only on training data?',
        'If the estimated λ for a feature is 0, what transformation was effectively applied?',
        'How would you handle a feature that contains both positive and negative values '
        'when you want to apply a power transformation?',
    ]))
    
    story.append(H2('31.9.3 &nbsp; Interview Questions'))
    
    story.extend(numbered([
        '<b>Q:</b> What is the difference between Box-Cox and Yeo-Johnson?<br/>'
        '<b>A:</b> Box-Cox is defined only for positive values and uses a single formula; '
        'Yeo-Johnson is defined for all real numbers using a piecewise formula that '
        'switches form for x ≥ 0 and x &lt; 0.',
        
        '<b>Q:</b> How does Scikit-learn estimate λ in PowerTransformer?<br/>'
        '<b>A:</b> By maximising the log-likelihood of the transformed data under a '
        'normal model, using numerical optimisation.',
        
        '<b>Q:</b> Can you apply PowerTransformer to the target variable in regression?<br/>'
        '<b>A:</b> Yes, but you must inverse-transform predictions before computing '
        'metrics on the original scale.',
        
        '<b>Q:</b> When would you <i>not</i> use a power transformation?<br/>'
        '<b>A:</b> With tree-based models (no benefit), with categorical features '
        '(not continuous), or when interpretability of original-scale coefficients '
        'is required.',
    ]))
    
    # Build
    doc.multiBuild(story)
    print(f'Chapter 31 PDF built: {output_path}')
    return output_path

if __name__ == '__main__':
    build_chapter_31()
