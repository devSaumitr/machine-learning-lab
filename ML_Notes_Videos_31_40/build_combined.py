"""Build comprehensive combined PDF: Videos 31-40."""
import sys, os
sys.path.insert(0, r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40')
from pdf_framework import *
import build_ch32_35
import build_ch36_40

FIGDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40\figures'
OUTDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40'

def build_combined():
    output_path = os.path.join(OUTDIR, 'ML_Notes_Videos_31_40_Complete.pdf')
    
    doc = MLDocTemplate(
        output_path, pagesize=A4,
        leftMargin=55, rightMargin=55, topMargin=55, bottomMargin=55,
        chapter_title='Machine Learning Notes — Complete Textbook',
        book_title='CampusX 100 Days of ML — Videos 31–40'
    )
    
    story = []
    
    # ════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ════════════════════════════════════════════════════════════════════
    story.append(spacer(60))
    story.append(Paragraph('MACHINE LEARNING NOTES', STYLES['title']))
    story.append(spacer(10))
    story.append(Paragraph(
        'Feature Engineering &amp; Missing Data Handling', STYLES['subtitle']))
    story.append(HRFlowable(width="50%", thickness=3, color=PRIMARY,
                            spaceBefore=15, spaceAfter=15))
    story.append(Paragraph(
        'Videos 31–40 from<br/>'
        '<b>CampusX — 100 Days of Machine Learning</b>',
        ParagraphStyle('CoverSub', fontName=BODY_FONT, fontSize=13, leading=20,
                       alignment=TA_CENTER, textColor=DARK)))
    story.append(spacer(30))
    
    # Chapter list on cover
    chapters = [
        ('31', 'Power Transformer'),
        ('32', 'Binning and Binarization'),
        ('33', 'Handling Mixed Variables'),
        ('34', 'Handling Date and Time Variables'),
        ('35', 'Missing Data: Complete Case Analysis'),
        ('36', 'Numerical Missing Data: SimpleImputer'),
        ('37', 'Missing Categorical Data'),
        ('38', 'Missing Indicator and Random Sample Imputation'),
        ('39', 'KNN Imputer'),
        ('40', 'MICE and Iterative Imputation'),
    ]
    ch_lines = '<br/>'.join([f'Chapter {n} — {t}' for n, t in chapters])
    story.append(Paragraph(
        ch_lines,
        ParagraphStyle('CoverTOC', fontName=BODY_FONT, fontSize=10, leading=16,
                       alignment=TA_CENTER, textColor=HexColor('#555555'))))
    story.append(spacer(40))
    story.append(Paragraph(
        'A complete, self-contained textbook following the teaching standards '
        'of the ML Roadmap project.',
        ParagraphStyle('CoverNote', fontName=BODY_ITALIC, fontSize=9,
                       alignment=TA_CENTER, textColor=HexColor('#888888'))))
    
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('Table of Contents', STYLES['h1']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY,
                            spaceBefore=4, spaceAfter=10))
    story.append(build_toc())
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 31
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 31', STYLES['chapter_num']))
    story.append(Paragraph('Power Transformer', STYLES['title']))
    story.append(Paragraph(
        'Skewness, Box-Cox &amp; Yeo-Johnson Transformations',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Why skewed distributions hurt ML models; how power transformations restore '
        'symmetry; the mathematics of Box-Cox and Yeo-Johnson; Scikit-learn '
        'implementation; when to choose one method over the other.'))
    story.append(page_break())
    
    # Import ch31 content by building its sections inline
    # We need to replicate ch31's section builder - import from build_ch31
    import build_ch31
    # Temporarily redirect output
    _orig_path = None
    # Just call its section builder via a fake function
    # Actually build_ch31 has build_chapter_31 that builds the whole doc.
    # We need to extract just the story. Let's modify approach:
    # Run build_ch31's story-building logic by monkeypatching
    
    # Simpler: rebuild ch31 story by calling its internal logic
    # We'll just run the doc build but capture the story
    class StoryCapture:
        captured = None
        def multiBuild(self, story, *a, **k):
            StoryCapture.captured = story
    
    # Build ch31 story
    orig_doc_class = MLDocTemplate
    captured_ch31 = [None]
    class CaptureDoc:
        def __init__(self, *a, **k): pass
        def multiBuild(self, story, *a, **k):
            captured_ch31[0] = story
    
    # Monkeypatch
    build_ch31.MLDocTemplate = CaptureDoc
    build_ch31.build_chapter_31('_temp_ch31.pdf')
    build_ch31.MLDocTemplate = orig_doc_class
    
    if captured_ch31[0]:
        # Skip the cover page (first ~10 flowables) - find where content starts
        ch31_story = captured_ch31[0]
        # Remove cover elements: find first H1 or the learning objectives
        # Simple approach: find index of first HRFlowable after title area
        # We'll just take everything after the note_box
        start_idx = 0
        for i, item in enumerate(ch31_story):
            if isinstance(item, Table) and i > 5:
                start_idx = i + 1
                break
        story.extend(ch31_story[start_idx:])
    
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 32
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 32', STYLES['chapter_num']))
    story.append(Paragraph('Binning and Binarization', STYLES['title']))
    story.append(Paragraph(
        'Discretization, Equal-Width, Quantile, K-Means Binning',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Continuous vs discrete variables; equal-width, equal-frequency, and K-Means '
        'binning; binarization; Scikit-learn KBinsDiscretizer and Binarizer; '
        'comparison of techniques.'))
    story.append(page_break())
    build_ch32_35.ch32_sections(story, 32)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 33
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 33', STYLES['chapter_num']))
    story.append(Paragraph('Handling Mixed Variables', STYLES['title']))
    story.append(Paragraph(
        'Feature Extraction, Splitting, Unit Conversion, Encoding',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'What mixed variables are; why they are problematic; feature splitting with '
        'regex; unit normalisation; encoding strategies after splitting.'))
    story.append(page_break())
    build_ch32_35.ch33_sections(story, 33)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 34
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 34', STYLES['chapter_num']))
    story.append(Paragraph('Handling Date and Time Variables', STYLES['title']))
    story.append(Paragraph(
        'DateTime Parsing, Feature Extraction, Cyclical Encoding',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Converting strings to datetime; Pandas datetime accessor; extracting components; '
        'cyclical encoding with sin/cos; time-based feature engineering.'))
    story.append(page_break())
    build_ch32_35.ch34_sections(story, 34)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 35
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 35', STYLES['chapter_num']))
    story.append(Paragraph('Missing Data: Complete Case Analysis', STYLES['title']))
    story.append(Paragraph(
        'Missingness Mechanisms, MCAR/MAR/MNAR, CCA',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Why missing values occur; MCAR, MAR, MNAR mechanisms; when CCA is valid; '
        'when to avoid CCA; decision framework for handling strategies.'))
    story.append(page_break())
    build_ch32_35.ch35_sections(story, 35)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 36
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 36', STYLES['chapter_num']))
    story.append(Paragraph('Numerical Missing Data: SimpleImputer', STYLES['title']))
    story.append(Paragraph(
        'Mean Imputation, Median Imputation, Strategy, Statistics',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Mean vs median imputation; when median is preferred; SimpleImputer parameters '
        'and attributes; column-wise imputation; statistical implications.'))
    story.append(page_break())
    build_ch36_40.ch36_sections(story, 36)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 37
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 37', STYLES['chapter_num']))
    story.append(Paragraph('Missing Categorical Data', STYLES['title']))
    story.append(Paragraph(
        'Most Frequent, Constant, Missing Category Strategies',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Strategies for categorical missing data; most_frequent vs constant vs missing '
        'category; SimpleImputer for strings; consequences of each approach.'))
    story.append(page_break())
    build_ch36_40.ch37_sections(story, 37)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 38
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 38', STYLES['chapter_num']))
    story.append(Paragraph('Missing Indicator and Random Sample Imputation', STYLES['title']))
    story.append(Paragraph(
        'Missingness Information, Distribution Preservation',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Why missingness itself can be informative; MissingIndicator and add_indicator; '
        'random sample imputation; distribution preservation; limitations.'))
    story.append(page_break())
    build_ch36_40.ch38_sections(story, 38)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 39
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 39', STYLES['chapter_num']))
    story.append(Paragraph('KNN Imputer', STYLES['title']))
    story.append(Paragraph(
        'K-Nearest Neighbors, Euclidean Distance, nan_euclidean',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'KNN concept; Euclidean distance; nan_euclidean for missing data; '
        'KNNImputer parameters; weights; scaling requirement; computational limits.'))
    story.append(page_break())
    build_ch36_40.ch39_sections(story, 39)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CHAPTER 40
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CHAPTER 40', STYLES['chapter_num']))
    story.append(Paragraph('MICE and Iterative Imputation', STYLES['title']))
    story.append(Paragraph(
        'Chained Equations, Multivariate Imputation, Convergence',
        STYLES['subtitle']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=8, spaceAfter=8))
    story.append(note_box('Learning Objectives',
        'Why univariate imputation is insufficient; MICE algorithm; IterativeImputer; '
        'convergence; comparison with KNN; assumptions and limitations.'))
    story.append(page_break())
    build_ch36_40.ch40_sections(story, 40)
    story.append(page_break())
    
    # ════════════════════════════════════════════════════════════════════
    # CONSOLIDATED REVISION SECTION
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph('CONSOLIDATED REVISION', STYLES['chapter_num']))
    story.append(Paragraph('Quick Reference &amp; Cross-Chapter Review', STYLES['title']))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY,
                            spaceBefore=10, spaceAfter=15))
    
    # Comparison of all missing data methods
    story.append(H1('R.1 &nbsp; Comparison of All Missing-Data Methods'))
    
    story.extend(make_table(
        ['Method', 'Chapter', 'Type', 'Preserves Distribution?', 'Uses Feature Relations?', 'Best For'],
        [
            ['Complete Case Analysis', '35', 'Deletion', 'N/A (removes rows)', 'No',
             'MCAR + &lt;5% missing'],
            ['Mean Imputation', '36', 'Univariate', 'No (reduces variance)', 'No',
             'Symmetric distributions'],
            ['Median Imputation', '36', 'Univariate', 'Partial (robust to skew)', 'No',
             'Skewed distributions'],
            ['Most Frequent (categorical)', '37', 'Univariate', 'No (inflates mode)', 'No',
             'Categorical, low missingness'],
            ['Constant / Missing Category', '37', 'Univariate', 'Adds category', 'No',
             'Missingness may be predictive'],
            ['Missing Indicator', '38', 'Feature addition', 'N/A (adds feature)', 'No',
             'MAR/MNAR where missingness matters'],
            ['Random Sample', '38', 'Univariate', 'Yes (samples observed)', 'No',
             'Preserving marginal distributions'],
            ['KNN Imputation', '39', 'Multivariate', 'Yes (uses neighbours)', 'Yes (distance)',
             'Small-medium data with correlations'],
            ['MICE / Iterative', '40', 'Multivariate', 'Yes (regression-based)', 'Yes (regression)',
             'Strong feature correlations'],
        ],
        col_widths=[95, 40, 65, 95, 85, 90],
        caption='Table R.1: Comprehensive comparison of all missing-data handling methods covered in Chapters 35–40.'
    ))
    
    story.append(page_break())
    
    # Decision flowchart
    story.append(H2('R.1.1 &nbsp; Decision Framework'))
    story.append(P(
        'Use this decision process when choosing a missing-data strategy:'
    ))
    story.append(numbered([
        '<b>What percentage of data is missing?</b> If &lt;5% and MCAR → CCA may be '
        'acceptable. Otherwise proceed to step 2.',
        '<b>What is the missingness mechanism?</b> MCAR → any method works. '
        'MAR → imputation (not deletion). MNAR → advanced methods or missing indicators.',
        '<b>Is the feature numerical or categorical?</b> Numerical → mean/median/KNN/MICE. '
        'Categorical → most_frequent/constant/missing category.',
        '<b>Are features correlated?</b> Strongly correlated → KNN or MICE. Weakly '
        'correlated → simple imputation may suffice.',
        '<b>Is the dataset large?</b> Very large → prefer mean/median (fast) over KNN/MICE '
        '(slow). Small → KNN or MICE feasible.',
        '<b>Add missing indicators?</b> If missingness itself might be predictive, add '
        'indicator features alongside any imputation method.',
    ]))
    
    story.append(page_break())
    
    # Key formulas reference
    story.append(H1('R.2 &nbsp; Key Formulas Reference'))
    
    story.append(H2('R.2.1 &nbsp; Skewness'))
    story.append(math_block(
        r'\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}', 'R.1'))
    
    story.append(H2('R.2.2 &nbsp; Box-Cox Transformation'))
    story.append(math_block(
        r'T(x; \lambda) = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \ln(x) & \lambda = 0 \end{cases}', 'R.2'))
    
    story.append(H2('R.2.3 &nbsp; Yeo-Johnson Transformation (x ≥ 0)'))
    story.append(math_block(
        r'T(x; \lambda) = \begin{cases} \frac{(x+1)^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \ln(x+1) & \lambda = 0 \end{cases}', 'R.3'))
    
    story.append(H2('R.2.4 &nbsp; Euclidean Distance'))
    story.append(math_block(r'd(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{j=1}^{p}(x_j - y_j)^2}', 'R.4'))
    
    story.append(H2('R.2.5 &nbsp; Cyclical Encoding'))
    story.append(math_block(
        r'x_{\sin} = \sin\!\left(\frac{2\pi x}{\text{period}}\right), \quad '
        r'x_{\cos} = \cos\!\left(\frac{2\pi x}{\text{period}}\right)', 'R.5'))
    
    story.append(H2('R.2.6 &nbsp; KNN Weighted Imputation'))
    story.append(math_block(
        r'\hat{x} = \frac{\sum_{i \in N_k} x_i / d_i}{\sum_{i \in N_k} 1 / d_i}', 'R.6'))
    
    story.append(page_break())
    
    # Scikit-learn classes reference
    story.append(H1('R.3 &nbsp; Scikit-learn Classes Reference'))
    
    story.extend(make_table(
        ['Class', 'Module', 'Key Parameters', 'Chapter'],
        [
            ['PowerTransformer', 'sklearn.preprocessing',
             'method="yeo-johnson"|"box-cox", standardize=True', '31'],
            ['KBinsDiscretizer', 'sklearn.preprocessing',
             'n_bins, encode, strategy="uniform"|"quantile"|"kmeans"', '32'],
            ['Binarizer', 'sklearn.preprocessing', 'threshold', '32'],
            ['dropna()', 'pandas.DataFrame',
             'how="any"|"all", subset, thresh', '35'],
            ['SimpleImputer', 'sklearn.impute',
             'strategy="mean"|"median"|"most_frequent"|"constant", add_indicator', '36, 37, 38'],
            ['MissingIndicator', 'sklearn.impute',
             'features="missing-only"|"all"', '38'],
            ['KNNImputer', 'sklearn.impute',
             'n_neighbors, weights="uniform"|"distance", metric="nan_euclidean"', '39'],
            ['IterativeImputer', 'sklearn.experimental.impute',
             'estimator, max_iter, random_state, initial_strategy, imputation_order', '40'],
        ],
        col_widths=[105, 115, 210, 40],
        caption='Table R.2: Scikit-learn classes used across Chapters 31–40.'
    ))
    
    story.append(H1('R.4 &nbsp; Quick Interview Cheat Sheet'))
    
    story.extend(make_table(
        ['Question', 'Key Answer'],
        [
            ['Box-Cox vs Yeo-Johnson?', 'Box-Cox: x&gt;0 only. Yeo-Johnson: all real x.'],
            ['Mean vs median imputation?', 'Mean: symmetric data. Median: skewed/outliers.'],
            ['MCAR vs MAR vs MNAR?',
             'MCAR: random. MAR: depends on observed. MNAR: depends on missing value.'],
            ['When is CCA valid?',
             'MCAR + small missing % + large enough dataset.'],
            ['Why scale before KNN?',
             'Distance metrics are dominated by large-scale features.'],
            ['What does MICE stand for?',
             'Multivariate Imputation by Chained Equations.'],
            ['MICE assumes what about missingness?',
             'MAR (Missing at Random).'],
            ['When to use missing indicators?',
             'When the fact that a value is missing may be predictive.'],
            ['KNN vs MICE?',
             'KNN: distance-based, slower for large n. MICE: regression-based, uses all features.'],
            ['Power transform for trees?',
             'No benefit — trees are invariant to monotonic transforms.'],
        ],
        col_widths=[170, 310],
        caption='Table R.3: Quick interview answers for the most common questions.'
    ))
    
    story.append(spacer(20))
    story.append(section_divider())
    story.append(Paragraph(
        'End of Machine Learning Notes — Videos 31–40<br/>'
        'CampusX — 100 Days of Machine Learning',
        STYLES['body_center']))
    
    # Flatten any accidentally nested lists
    def flatten(story_list):
        flat = []
        for item in story_list:
            if isinstance(item, list):
                flat.extend(flatten(item))
            else:
                flat.append(item)
        return flat
    
    story = flatten(story)
    
    doc.multiBuild(story)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Combined PDF: {output_path} ({size_mb:.1f} MB)')
    return output_path

if __name__ == '__main__':
    build_combined()
