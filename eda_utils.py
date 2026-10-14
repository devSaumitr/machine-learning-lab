"""
eda_utils.py — beginner-friendly EDA workflow (no extra dependencies).

Works with the existing Python 3.14 virtual environment using only:
pandas, numpy, matplotlib, seaborn.

This is NOT a replacement for ydata-profiling. It covers the most
useful EDA steps manually so you can keep learning without changing
your Python version or virtual environment.

Typical notebook usage:
    import pandas as pd
    import eda_utils

    titanic = pd.read_csv(r"E:\\MACHINE_LEARNING\\train.csv")
    eda_utils.run_full_eda(titanic, target="Survived")
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# A. Dataset overview
# ---------------------------------------------------------------------
def overview(df):
    """Print shape, columns, dtypes, head, tail and a random sample."""
    print("=== A. DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Last 5 rows ---")
    print(df.tail())
    print("\n--- Random sample (5 rows) ---")
    # random_state makes the sample reproducible for learning
    print(df.sample(min(5, len(df)), random_state=42))
    print()
    return {"shape": df.shape, "columns": list(df.columns)}


# ---------------------------------------------------------------------
# B. Data quality
# ---------------------------------------------------------------------
def missing_report(df):
    """Return a DataFrame with missing counts and percentages, sorted."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    })
    return report.sort_values("missing_pct", ascending=False)


def data_quality(df):
    """Print missing values, duplicates, uniques, constant columns."""
    print("=== B. DATA QUALITY ===")

    report = missing_report(df)
    print("\n--- Missing values ---")
    print(report[report["missing_count"] > 0])
    if report["missing_count"].sum() == 0:
        print("No missing values.")

    n_dupes = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {n_dupes}")

    print("\n--- Unique values per column ---")
    print(df.nunique().sort_values())

    # Constant columns carry no information for ML
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(f"\nConstant columns (only 1 unique value): {constant_cols or 'None'}")

    # Possibly irrelevant ID-like columns: all values unique
    id_like = [c for c in df.columns if df[c].nunique() == len(df)]
    print(f"ID-like columns (all unique, often dropped for ML): {id_like or 'None'}")

    # Columns that look numeric but are stored as objects (or vice versa)
    print("\n--- Dtypes to double-check ---")
    print(df.dtypes)
    print("\nTip: IDs such as PassengerId/Ticket are not predictive numbers,")
    print("even if stored as int. Name/Ticket/Cabin are usually dropped or engineered.")
    print()
    return {
        "missing": report,
        "duplicates": n_dupes,
        "constant_cols": constant_cols,
        "id_like": id_like,
    }


# ---------------------------------------------------------------------
# C. Numerical analysis
# ---------------------------------------------------------------------
def numerical_summary(df):
    """Print descriptive stats and simple IQR outlier counts."""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        print("No numeric columns found.")
        return pd.DataFrame()

    print("=== C. NUMERICAL ANALYSIS ===")
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["median"] = num.median()
    print(desc.round(3))

    print("\n--- Potential outliers (IQR rule: outside Q1 - 1.5*IQR, Q3 + 1.5*IQR) ---")
    for col in num.columns:
        s = num[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            print(f"{col}: IQR=0, outlier check skipped")
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        print(f"{col}: {n_out} outliers ({n_out / len(s) * 100:.1f}%)")
    print()
    return desc


# ---------------------------------------------------------------------
# D. Categorical analysis
# ---------------------------------------------------------------------
def categorical_summary(df, high_cardinality_threshold=50):
    """Print unique counts, top values and flag high-cardinality columns."""
    cat = df.select_dtypes(include=["object", "category", "string"])
    # Also treat small-int labels like Pclass/Survived as categorical below
    if cat.empty:
        print("No object/category columns found.")
        return

    print("=== D. CATEGORICAL ANALYSIS ===")
    for col in cat.columns:
        n_unique = int(df[col].nunique(dropna=False))
        print(f"\n--- {col} ({n_unique} unique) ---")
        print(df[col].value_counts(dropna=False).head(10))
        print("\nProportions (%):")
        print((df[col].value_counts(normalize=True, dropna=False).head(10) * 100).round(2))
        if n_unique > high_cardinality_threshold:
            print(f"NOTE: high cardinality ({n_unique} > {high_cardinality_threshold}). "
                  "Consider grouping or dropping for ML.")
    print()


# ---------------------------------------------------------------------
# E + F. Visualizations and relationships
# ---------------------------------------------------------------------
def plot_eda(df, target=None, out_dir="eda_plots"):
    """Create and save a small set of useful plots. Returns saved file paths.

    - histograms for numeric columns
    - box plots for numeric columns
    - count plots for low-cardinality categoricals
    - correlation heatmap for numeric columns
    - missing-value bar chart
    - optional: target vs numeric/categorical means
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)

    # 1. Histograms (one figure, subplots)
    if num_cols:
        n = min(len(num_cols), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, col in enumerate(num_cols[:n]):
            df[col].hist(ax=axes[i], bins=30)
            axes[i].set_title(f"Histogram: {col}")
        for j in range(n, 6):
            fig.delaxes(axes[j])
        fig.tight_layout()
        p = os.path.join(out_dir, "histograms.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

        # 2. Box plots
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 1.5), 5))
        df[num_cols].boxplot(ax=ax)
        ax.set_title("Box plots (spread + outliers)")
        plt.xticks(rotation=20)
        fig.tight_layout()
        p = os.path.join(out_dir, "boxplots.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

        # 3. Correlation heatmap (only if 2+ numeric cols)
        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = df[num_cols].corr(numeric_only=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation heatmap (numeric columns)")
            fig.tight_layout()
            p = os.path.join(out_dir, "correlation.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            saved.append(p)

    # 4. Count plots for low-cardinality categoricals (avoid Name/Ticket/Cabin)
    low_card = [c for c in cat_cols if df[c].nunique() <= 10]
    for col in low_card:
        fig, ax = plt.subplots(figsize=(7, 4))
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order, ax=ax)
        ax.set_title(f"Count plot: {col}")
        fig.tight_layout()
        p = os.path.join(out_dir, f"count_{col}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    # 5. Missing-value bar chart
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        (miss / len(df) * 100).sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing values by column")
        fig.tight_layout()
        p = os.path.join(out_dir, "missing.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    # 6. Target relationships (explicit target, e.g. "Survived")
    if target is not None and target in df.columns:
        # numeric vs target: mean of each numeric grouped by target
        if num_cols:
            fig, ax = plt.subplots(figsize=(9, 4))
            df.groupby(target)[num_cols].mean().T.plot(kind="bar", ax=ax)
            ax.set_title(f"Mean of numeric columns grouped by {target}")
            plt.xticks(rotation=20)
            fig.tight_layout()
            p = os.path.join(out_dir, f"target_{target}_numeric.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            saved.append(p)
        for col in low_card:
            if col == target:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ct = pd.crosstab(df[col], df[target], normalize="index") * 100
            ct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("%")
            ax.set_title(f"{col} vs {target} (row %)")
            plt.xticks(rotation=20)
            fig.tight_layout()
            p = os.path.join(out_dir, f"target_{target}_vs_{col}.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            saved.append(p)

    print(f"Saved {len(saved)} plots to '{out_dir}':")
    for p in saved:
        print(" -", p)
    return saved


def analyze_relationships(df, target=None):
    """Print simple numeric-vs-categorical / target relationships."""
    print("=== F. RELATIONSHIPS ===")
    if target is None or target not in df.columns:
        print("No target specified. Pass target='Survived' for Titanic survival analysis.")
        print("Showing overall numeric correlations instead:")
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            print(num.corr(numeric_only=True).round(3))
        print()
        return

    print(f"--- Group means by {target} ---")
    num = df.select_dtypes(include=[np.number])
    print(df.groupby(target)[num.columns].mean().round(3))

    print(f"\n--- Survival-style rates: mean({target}) by categorical column ---")
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        if df[col].nunique() <= 15:
            print(f"\n{col}:")
            print(df.groupby(col)[target].mean().round(3).sort_values(ascending=False))
    print()


# ---------------------------------------------------------------------
# G. Simple HTML report (not equivalent to ydata-profiling)
# ---------------------------------------------------------------------
def generate_html_report(df, target=None, output_path="eda_report.html", plots_dir="eda_plots"):
    """Save a simple reusable HTML report and return its path.

    Contains: shape/dtypes, missing table, describe table, key findings,
    and linked plot images saved by plot_eda().
    """
    plots = plot_eda(df, target=target, out_dir=plots_dir)
    missing = missing_report(df)
    desc = df.describe(include="all").round(3)

    # Keep only plot filenames so the HTML works when moved with its folder
    plot_items = "".join(
        f"<li><b>{os.path.basename(p)}</b><br>"
        f"<img src='{os.path.join(os.path.basename(plots_dir), os.path.basename(p))}' "
        f"style='max-width:700px;'><br></li>"
        for p in plots
    )

    findings = [
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns.",
        f"Duplicate rows: {int(df.duplicated().sum())}.",
        f"Columns with missing values: "
        f"{', '.join(missing[missing['missing_count'] > 0].index.tolist()) or 'None'}.",
        f"Numeric columns: {', '.join(df.select_dtypes(include=[np.number]).columns) or 'None'}.",
    ]
    if target is not None and target in df.columns:
        findings.append(f"Target column analysed: {target}.")

    html = f"""<html><head><meta charset="utf-8">
<title>EDA Report (manual, not ydata-profiling)</title>
<style>body{{font-family:Arial;margin:30px;}}table{{border-collapse:collapse;}}
th,td{{border:1px solid #ccc;padding:6px;font-size:13px;}}img{{margin:10px 0;}}</style>
</head><body>
<h1>EDA Report (manual workflow)</h1>
<p><b>Note:</b> this is a lightweight alternative, not equivalent to ydata-profiling.</p>
<h2>1. Key findings</h2><ul>{''.join(f'<li>{f}</li>' for f in findings)}</ul>
<h2>2. Dtypes</h2>{df.dtypes.to_frame('dtype').to_html()}
<h2>3. Missing values</h2>{missing.to_html()}
<h2>4. Descriptive statistics</h2>{desc.to_html()}
<h2>5. Plots</h2><ul>{plot_items}</ul>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to: {output_path}")
    return output_path


def run_full_eda(df, target=None, output_html="eda_report.html", plots_dir="eda_plots"):
    """Run the full workflow in order. Convenience wrapper for beginners."""
    overview(df)
    data_quality(df)
    numerical_summary(df)
    categorical_summary(df)
    analyze_relationships(df, target=target)
    generate_html_report(df, target=target, output_path=output_html, plots_dir=plots_dir)
    print("Done. Open the HTML report and the PNG plots to review results.")
