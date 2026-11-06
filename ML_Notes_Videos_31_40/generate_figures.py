"""Generate all matplotlib figures for ML Notes chapters 31-40."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

FIGDIR = r'E:\MACHINE_LEARNING\ML_Notes_Videos_31_40\figures'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafbfc',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

np.random.seed(42)

def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)
    print(f'  saved: {name}')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 31: POWER TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 31 figures...')

# Fig 1: Skewness illustration — symmetric vs right-skewed vs left-skewed
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
x = np.linspace(-4, 8, 500)
# Symmetric
axes[0].plot(x, stats.norm.pdf(x, 2, 1), 'b-', lw=2)
axes[0].fill_between(x, stats.norm.pdf(x, 2, 1), alpha=0.2, color='blue')
axes[0].set_title('Symmetric (Skew = 0)')
axes[0].axvline(2, color='red', ls='--', lw=1, label='Mean = Median')
# Right skewed
lognorm_x = np.linspace(0.01, 8, 500)
axes[1].plot(lognorm_x, stats.lognorm.pdf(lognorm_x, 0.8), 'b-', lw=2)
axes[1].fill_between(lognorm_x, stats.lognorm.pdf(lognorm_x, 0.8), alpha=0.2, color='blue')
axes[1].set_title('Right Skewed (Skew > 0)')
mean_r = stats.lognorm.mean(0.8)
median_r = stats.lognorm.median(0.8)
axes[1].axvline(mean_r, color='red', ls='--', lw=1.5, label=f'Mean={mean_r:.1f}')
axes[1].axvline(median_r, color='green', ls=':', lw=1.5, label=f'Median={median_r:.1f}')
axes[1].legend(fontsize=7)
# Left skewed
axes[2].plot(x, stats.skewnorm.pdf(x, -5, 5, 1), 'b-', lw=2)
axes[2].fill_between(x, stats.skewnorm.pdf(x, -5, 5, 1), alpha=0.2, color='blue')
axes[2].set_title('Left Skewed (Skew < 0)')
axes[2].axvline(stats.skewnorm.mean(-5, 5, 1), color='red', ls='--', lw=1.5)
axes[2].axvline(stats.skewnorm.median(-5, 5, 1), color='green', ls=':', lw=1.5)
for ax in axes:
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
save(fig, 'ch31_skewness_types.png')

# Fig 2: Power transformation effect on skewed data
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
skewed = np.random.lognormal(mean=1.0, sigma=0.7, size=1000)
transforms = [
    ('Original (λ=1)', skewed, lambda x: x, 1),
    ('Log (λ=0)', skewed, lambda x: np.log(x + 1), 0),
    ('Square Root (λ=0.5)', skewed, lambda x: np.sqrt(x), 0.5),
    ('Square (λ=2)', skewed, lambda x: x**2, 2),
    ('Box-Cox optimized', skewed, None, None),
    ('Yeo-Johnson', skewed, None, None),
]
from scipy.stats import boxcox, yeojohnson
bc_data, bc_lam = boxcox(skewed + 1)
yj_data, yj_lam = yeojohnson(skewed)

for idx, (title, data, func, lam) in enumerate(transforms):
    ax = axes[idx // 3][idx % 3]
    if title.startswith('Box-Cox'):
        ax.hist(bc_data, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f'{title}\nλ={bc_lam:.3f}')
    elif title.startswith('Yeo'):
        ax.hist(yj_data, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f'{title}\nλ={yj_lam:.3f}')
    else:
        transformed = func(data)
        ax.hist(transformed, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f'{title}')
    ax.set_yticks([])
plt.suptitle('Effect of Power Transformations on Right-Skewed Data', fontsize=12, y=1.02)
plt.tight_layout()
save(fig, 'ch31_power_transform_effect.png')

# Fig 3: Box-Cox vs Yeo-Johnson domain
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
# Box-Cox domain
ax = axes[0]
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.axhspan(0, 8, xmin=0.2, xmax=1.0, alpha=0.2, color='green', label='Valid (x > 0)')
ax.axhspan(-2, 0, xmin=0.0, xmax=0.2, alpha=0.2, color='red', label='Invalid (x ≤ 0)')
ax.axvline(0, color='black', lw=1.5)
ax.axhline(0, color='black', lw=1.5)
ax.set_xlabel('Input value x')
ax.set_ylabel('λ parameter')
ax.set_title('Box-Cox: requires x > 0')
ax.legend(fontsize=8)
# Yeo-Johnson domain
ax = axes[1]
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
xx, yy = np.meshgrid(np.linspace(-2, 8, 200), np.linspace(-2, 8, 200))
valid = np.ones_like(xx, dtype=bool)
ax.contourf(xx, yy, valid.astype(float), levels=[0.5, 1.5], alpha=0.2, colors=['green'])
ax.axvline(0, color='black', lw=1, ls='--')
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_xlabel('Input value x')
ax.set_ylabel('λ parameter')
ax.set_title('Yeo-Johnson: valid for all x')
ax.text(3, 4, 'All values valid', fontsize=10, ha='center',
        color='green', fontweight='bold')
plt.tight_layout()
save(fig, 'ch31_boxcox_yeojohnson_domain.png')

# Fig 4: Lambda effect on Box-Cox transform shape
fig, ax = plt.subplots(figsize=(7, 4))
x = np.linspace(0.1, 5, 300)
for lam, color in [(0, 'red'), (0.5, 'blue'), (1, 'green'), (2, 'orange')]:
    if abs(lam) < 1e-10:
        y = np.log(x)
        label = f'λ=0 (log)'
    else:
        y = (x**lam - 1) / lam
        label = f'λ={lam}'
    ax.plot(x, y, lw=2, color=color, label=label)
ax.set_xlabel('x')
ax.set_ylabel('Transformed value')
ax.set_title('Box-Cox Transformation for Different λ Values')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, 'ch31_lambda_effect.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 32: BINNING AND BINARIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 32 figures...')

# Fig 1: Binning strategies comparison
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
data = np.concatenate([np.random.normal(25, 5, 200),
                       np.random.normal(55, 8, 300),
                       np.random.normal(80, 4, 150)])

# Equal width
bins_ew = np.linspace(data.min(), data.max(), 5)
colors_ew = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
for i in range(4):
    mask = (data >= bins_ew[i]) & (data <= bins_ew[i+1])
    axes[0].hist(data[mask], bins=30, color=colors_ew[i], edgecolor='white',
                 alpha=0.8, label=f'Bin {i+1}: [{bins_ew[i]:.0f}, {bins_ew[i+1]:.0f}]')
axes[0].set_title('Equal Width Binning (uniform intervals)')
axes[0].legend(fontsize=7, ncol=4)

# Equal frequency
quantiles = np.percentile(data, [0, 25, 50, 75, 100])
colors_ef = ['#9b59b6', '#1abc9c', '#e67e22', '#e74c3c']
for i in range(4):
    mask = (data >= quantiles[i]) & (data <= quantiles[i+1])
    axes[1].hist(data[mask], bins=30, color=colors_ef[i], edgecolor='white',
                 alpha=0.8, label=f'Bin {i+1}: [{quantiles[i]:.0f}, {quantiles[i+1]:.0f}]')
axes[1].set_title('Equal Frequency Binning (equal counts per bin)')
axes[1].legend(fontsize=7, ncol=4)

# K-Means based
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(data.reshape(-1, 1))
centers = sorted(km.cluster_centers_.flatten())
colors_km = ['#16a085', '#c0392b', '#2980b9', '#f1c40f']
for i in range(4):
    axes[2].hist(data[labels == i], bins=30, color=colors_km[i], edgecolor='white',
                 alpha=0.8, label=f'Cluster {i+1} (center={centers[i]:.0f})')
axes[2].set_title('K-Means Binning (cluster-based intervals)')
axes[2].legend(fontsize=7, ncol=4)
axes[2].set_xlabel('Feature Value')
plt.tight_layout()
save(fig, 'ch32_binning_strategies.png')

# Fig 2: Binarization threshold
fig, ax = plt.subplots(figsize=(8, 3))
data_bin = np.random.normal(50, 15, 200)
threshold = 55
ax.scatter(data_bin, np.zeros_like(data_bin) + 0.3, c=np.where(data_bin > threshold, 'red', 'blue'),
           s=15, alpha=0.6, label='Original values')
binary = (data_bin > threshold).astype(int)
ax.scatter(data_bin, np.zeros_like(data_bin) - 0.3 + binary * 0.6,
           c=np.where(binary == 1, 'red', 'blue'), s=15, alpha=0.6, marker='s',
           label='Binarized (0 or 1)')
ax.axvline(threshold, color='black', ls='--', lw=2, label=f'Threshold = {threshold}')
ax.set_yticks([0, 1])
ax.set_yticklabels(['0 (≤ threshold)', '1 (> threshold)'])
ax.set_xlabel('Feature Value')
ax.set_title('Binarization: Converting Continuous to Binary Using a Threshold')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(-0.5, 1.8)
plt.tight_layout()
save(fig, 'ch32_binarization.png')

# Fig 3: Binning reducing noise
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
x_noisy = np.linspace(0, 10, 100)
y_noisy = np.sin(x_noisy) + np.random.normal(0, 0.3, 100)
axes[0].scatter(x_noisy, y_noisy, s=10, alpha=0.5, color='steelblue')
axes[0].plot(x_noisy, np.sin(x_noisy), 'r--', lw=2, label='True pattern')
axes[0].set_title('Original Noisy Continuous Data')
axes[0].legend()
# Binned version
bins = np.linspace(0, 10, 11)
bin_indices = np.digitize(x_noisy, bins) - 1
bin_means_x = []
bin_means_y = []
for b in range(10):
    mask = bin_indices == b
    if mask.any():
        bin_means_x.append(x_noisy[mask].mean())
        bin_means_y.append(y_noisy[mask].mean())
axes[1].scatter(x_noisy, y_noisy, s=8, alpha=0.2, color='gray', label='Original')
axes[1].plot(bin_means_x, bin_means_y, 'bo-', lw=2, ms=8, label='Binned means')
axes[1].plot(x_noisy, np.sin(x_noisy), 'r--', lw=2, label='True pattern')
axes[1].set_title('After Equal-Width Binning (10 bins)')
axes[1].legend()
plt.tight_layout()
save(fig, 'ch32_binning_noise_reduction.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 33: HANDLING MIXED VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 33 figures...')

# Fig 1: Mixed variable concept diagram
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Main box
box = FancyBboxPatch((0.5, 1), 9, 3, boxstyle="round,pad=0.2",
                      facecolor='#eaf2f8', edgecolor='#2c3e50', lw=2)
ax.add_patch(box)
ax.text(5, 3.6, 'Mixed Variable', ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
ax.text(5, 3.2, '"5 feet 7 inches"', ha='center', fontsize=11, style='italic')

# Numerical part
box1 = FancyBboxPatch((1, 1.4), 3.5, 1.2, boxstyle="round,pad=0.1",
                       facecolor='#d5f5e3', edgecolor='#27ae60', lw=1.5)
ax.add_patch(box1)
ax.text(2.75, 2.1, 'Numerical Component', ha='center', fontsize=9, fontweight='bold', color='#1e8449')
ax.text(2.75, 1.7, '5  and  7', ha='center', fontsize=11)

# Categorical part
box2 = FancyBboxPatch((5.5, 1.4), 3.5, 1.2, boxstyle="round,pad=0.1",
                       facecolor='#fdebd0', edgecolor='#e67e22', lw=1.5)
ax.add_patch(box2)
ax.text(7.25, 2.1, 'Categorical Component', ha='center', fontsize=9, fontweight='bold', color='#b7950b')
ax.text(7.25, 1.7, '"feet" and "inches"', ha='center', fontsize=10)

# Arrows
ax.annotate('', xy=(2.75, 2.6), xytext=(4.2, 3.0),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
ax.annotate('', xy=(7.25, 2.6), xytext=(5.8, 3.0),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
plt.tight_layout()
save(fig, 'ch33_mixed_variable.png')

# Fig 2: Feature splitting approach
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={'width_ratios': [1, 1.5]})
# Before
ax = axes[0]
ax.axis('off')
ax.set_title('Before Splitting', fontweight='bold', color='#2c3e50')
table_data = [['Height_Mixed'], ['5 feet 7 inches'], ['6 feet 2 inches'], ['5 feet 1 inch']]
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
# After
ax = axes[1]
ax.axis('off')
ax.set_title('After Feature Splitting', fontweight='bold', color='#2c3e50')
table_data2 = [['Height_Value', 'Height_Unit'], ['5, 7', 'feet, inches'],
               ['6, 2', 'feet, inches'], ['5, 1', 'feet, inches']]
table2 = ax.table(cellText=table_data2[1:], colLabels=table_data2[0],
                  loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 1.5)
plt.tight_layout()
save(fig, 'ch33_feature_splitting.png')

# Fig 3: Mixed variables examples
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
examples = [
    ('Height', '5 feet 7 inches', 'Numeric + Unit category'),
    ('Weight', '70.5 kg', 'Numeric + Unit category'),
    ('Date-Time', '2024-01-15 14:30:00', 'Numeric parts + separators'),
    ('Coordinates', '40.7128° N, 74.0060° W', 'Numeric + Direction category'),
    ('File Size', '2.5 MB', 'Numeric + Unit category'),
    ('Range', '10-20', 'Two numerics + separator'),
]
y = 0.85
for name, example, analysis in examples:
    ax.text(0.02, y, name, fontsize=10, fontweight='bold', color='#2c3e50', transform=ax.transAxes)
    ax.text(0.25, y, example, fontsize=10, family='monospace', transform=ax.transAxes)
    ax.text(0.60, y, analysis, fontsize=9, color='#7f8c8d', style='italic', transform=ax.transAxes)
    y -= 0.15
ax.set_title('Common Examples of Mixed Variables', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
save(fig, 'ch33_mixed_examples.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 34: DATE AND TIME VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 34 figures...')

# Fig 1: Datetime feature extraction diagram
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Central datetime
center_box = FancyBboxPatch((3.5, 2.5), 3, 1, boxstyle="round,pad=0.15",
                             facecolor='#2c3e50', edgecolor='#2c3e50')
ax.add_patch(center_box)
ax.text(5, 3.0, '2024-03-15\n14:30:45', ha='center', va='center',
        fontsize=10, color='white', fontweight='bold', family='monospace')

# Surrounding features
features = [
    (1.5, 5.2, 'Year\n2024', '#e74c3c'),
    (5, 5.5, 'Month\n3 (March)', '#3498db'),
    (8.5, 5.2, 'Day\n15', '#2ecc71'),
    (1, 1.2, 'Hour\n14', '#9b59b6'),
    (4, 0.8, 'Minute\n30', '#e67e22'),
    (7, 0.8, 'Second\n45', '#1abc9c'),
    (9, 2.8, 'Day of Week\n(Friday)', '#e74c3c'),
]
for fx, fy, label, color in features:
    box = FancyBboxPatch((fx-0.8, fy-0.4), 1.8, 0.9, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=color, alpha=0.85)
    ax.add_patch(box)
    ax.text(fx+0.1, fy+0.05, label, ha='center', va='center', fontsize=8, color='white')
    ax.annotate('', xy=(fx, fy+0.5 if fy > 3 else fy-0.1),
                xytext=(5, 3.3 if fy > 3 else 2.5),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))
ax.set_title('Datetime Feature Extraction', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
save(fig, 'ch34_datetime_extraction.png')

# Fig 2: Cyclical encoding of day of week
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
days = np.arange(7)
# Ordinal encoding (wrong for cyclical)
axes[0].scatter(days, days, c=days, cmap='viridis', s=100, zorder=5)
axes[0].plot(days, days, 'b--', alpha=0.3)
axes[0].set_xlabel('Day Index')
axes[0].set_ylabel('Encoded Value')
axes[0].set_title('Ordinal Encoding (Mon=0...Sun=6)\nSunday(6) far from Monday(0) — WRONG')
axes[0].set_xticks(days)
axes[0].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], fontsize=7)
# Cyclical encoding (correct)
theta = 2 * np.pi * days / 7
axes[1].scatter(np.cos(theta), np.sin(theta), c=days, cmap='viridis', s=100, zorder=5)
circle = plt.Circle((0, 0), 1, fill=False, color='gray', ls='--')
axes[1].add_patch(circle)
for i, d in enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']):
    axes[1].annotate(d, (np.cos(theta[i])*1.15, np.sin(theta[i])*1.15),
                     ha='center', va='center', fontsize=7)
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_aspect('equal')
axes[1].set_title('Cyclical Encoding (sin, cos)\nSunday close to Monday — CORRECT')
axes[1].axis('off')
plt.tight_layout()
save(fig, 'ch34_cyclical_encoding.png')

# Fig 3: Time-based feature engineering workflow
fig, ax = plt.subplots(figsize=(9, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

steps = [
    (0.5, 'Raw\nTimestamp', '#3498db'),
    (2.5, 'Parse to\nDateTime', '#2ecc71'),
    (4.5, 'Extract\nComponents', '#e74c3c'),
    (6.5, 'Create New\nFeatures', '#9b59b6'),
    (8.5, 'Model\nReady', '#e67e22'),
]
for x, label, color in steps:
    box = FancyBboxPatch((x, 1), 1.4, 1.2, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', alpha=0.9)
    ax.add_patch(box)
    ax.text(x+0.7, 1.6, label, ha='center', va='center', fontsize=8.5,
            color='white', fontweight='bold')
    if x < 8:
        ax.annotate('', xy=(x+1.9, 1.6), xytext=(x+1.5, 1.6),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.set_title('Datetime Feature Engineering Pipeline', fontsize=11, fontweight='bold')
plt.tight_layout()
save(fig, 'ch34_pipeline.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 35: MISSING DATA — COMPLETE CASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 35 figures...')

# Fig 1: Missing data patterns
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
np.random.seed(42)
n, m = 20, 6

# MCAR
mcar = np.ones((n, m))
for i in range(n):
    for j in range(m):
        if np.random.random() < 0.2:
            mcar[i, j] = 0
axes[0].imshow(mcar, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[0].set_title('MCAR\n(Missing Completely\nat Random)', fontsize=9, fontweight='bold')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Samples')

# MAR
mar = np.ones((n, m))
for i in range(n):
    if mcar[i, 0] == 0:  # if feature 0 missing, feature 1 likely missing too
        if np.random.random() < 0.7:
            mar[i, 1] = 0
    if np.random.random() < 0.15:
        mar[i, 0] = 0
axes[1].imshow(mar, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[1].set_title('MAR\n(Missing at Random\n— depends on other features)', fontsize=9, fontweight='bold')
axes[1].set_xlabel('Features')

# MNAR
mnar = np.ones((n, m))
for i in range(n):
    val = np.random.random()
    if val < 0.3:  # low values more likely to be missing
        mnar[i, 2] = 0
    if np.random.random() < 0.1:
        mnar[i, 0] = 0
axes[2].imshow(mnar, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[2].set_title('MNAR\n(Missing Not at Random\n— depends on missing value)', fontsize=9, fontweight='bold')
axes[2].set_xlabel('Features')
plt.tight_layout()
save(fig, 'ch35_missing_patterns.png')

# Fig 2: CCA impact
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
full_data = np.random.normal(50, 15, 200)
missing_mask = np.random.random(200) < 0.3
cca_data = full_data[~missing_mask]
axes[0].hist(full_data, bins=30, color='steelblue', edgecolor='white', alpha=0.7, label=f'Full (n={len(full_data)})')
axes[0].hist(cca_data, bins=30, color='red', edgecolor='white', alpha=0.5, label=f'After CCA (n={len(cca_data)})')
axes[0].set_title(f'CCA Removes {missing_mask.sum()} Rows ({missing_mask.mean()*100:.0f}%)')
axes[0].legend()
axes[0].set_xlabel('Feature Value')

# Statistical consequence
means = [np.mean(np.random.normal(50, 15, 200)) for _ in range(1000)]
cca_means = []
for _ in range(1000):
    d = np.random.normal(50, 15, 200)
    mask = np.random.random(200) < 0.3
    cca_means.append(np.mean(d[~mask]))
axes[1].hist(means, bins=30, color='steelblue', edgecolor='white', alpha=0.7, label='Full data means')
axes[1].hist(cca_means, bins=30, color='red', edgecolor='white', alpha=0.5, label='CCA means')
axes[1].set_title('Distribution of Sample Means\n(CCA increases variance if MCAR)')
axes[1].legend()
axes[1].set_xlabel('Sample Mean')
plt.tight_layout()
save(fig, 'ch35_cca_impact.png')

# Fig 3: Decision flowchart for CCA
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

boxes = [
    (5, 9.2, 'Missing Data Present?', '#2c3e50', 'white'),
    (2.5, 7.5, '< 5% missing?\nAND MCAR?', '#3498db', 'white'),
    (7.5, 7.5, '> 5% missing\nOR MAR/MNAR?', '#e74c3c', 'white'),
    (2.5, 5.5, 'CCA is acceptable\n(low information loss)', '#27ae60', 'white'),
    (7.5, 5.5, 'Use imputation\nor advanced methods', '#e67e22', 'white'),
    (5, 3.5, 'Check: Is the pattern\nMCAR, MAR, or MNAR?', '#9b59b6', 'white'),
    (5, 1.5, 'Choose appropriate\nhandling strategy', '#1abc9c', 'white'),
]
for x, y, text, color, tcolor in boxes:
    box = FancyBboxPatch((x-1.8, y-0.55), 3.6, 1.1, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor='white', alpha=0.9)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, color=tcolor, fontweight='bold')

arrows = [
    (5, 8.65, 3.5, 8.05), (5, 8.65, 6.5, 8.05),
    (2.5, 6.95, 2.5, 6.05), (7.5, 6.95, 7.5, 6.05),
]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
ax.set_title('CCA Decision Framework', fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
save(fig, 'ch35_cca_decision.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 36: SIMPLEIMPUTER
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 36 figures...')

# Fig 1: Mean vs Median imputation
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
# Symmetric distribution
sym_data = np.random.normal(50, 10, 300)
sym_missing = sym_data.copy()
sym_missing[np.random.random(300) < 0.2] = np.nan
axes[0].hist(sym_data[~np.isnan(sym_missing)], bins=30, color='steelblue',
             edgecolor='white', alpha=0.7, label='Observed')
axes[0].axvline(np.nanmean(sym_missing), color='red', lw=2, label=f'Mean={np.nanmean(sym_missing):.1f}')
axes[0].axvline(np.nanmedian(sym_missing), color='green', lw=2, ls='--', label=f'Median={np.nanmedian(sym_missing):.1f}')
axes[0].set_title('Symmetric Distribution\nMean ≈ Median — both work')
axes[0].legend(fontsize=7)

# Skewed distribution
skew_data = np.random.lognormal(3, 0.8, 300)
skew_missing = skew_data.copy()
skew_missing[np.random.random(300) < 0.2] = np.nan
axes[1].hist(skew_data[~np.isnan(skew_missing)], bins=30, color='steelblue',
             edgecolor='white', alpha=0.7, label='Observed')
axes[1].axvline(np.nanmean(skew_missing), color='red', lw=2, label=f'Mean={np.nanmean(skew_missing):.0f}')
axes[1].axvline(np.nanmedian(skew_missing), color='green', lw=2, ls='--', label=f'Median={np.nanmedian(skew_missing):.0f}')
axes[1].set_title('Skewed Distribution\nMedian is more robust')
axes[1].legend(fontsize=7)
axes[1].set_xlabel('Feature Value')
plt.tight_layout()
save(fig, 'ch36_mean_vs_median.png')

# Fig 2: SimpleImputer flow
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Data with missing
ax.text(1.5, 4.5, 'Data with Missing Values', ha='center', fontsize=10, fontweight='bold')
data_before = [[1.0, np.nan, 3.0], [np.nan, 2.0, 4.0], [5.0, 6.0, np.nan]]
for i in range(3):
    for j in range(3):
        x, y = 0.5 + j*1.0, 3.5 - i*0.7
        val = data_before[i][j]
        color = '#e74c3c' if np.isnan(val) else '#3498db'
        text = 'NaN' if np.isnan(val) else f'{val:.0f}'
        rect = Rectangle((x, y), 0.85, 0.55, facecolor=color, edgecolor='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+0.42, y+0.28, text, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')

# Arrow
ax.annotate('', xy=(4.8, 2.5), xytext=(3.8, 2.5),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
ax.text(4.3, 2.8, 'SimpleImputer\n(strategy="mean")', ha='center', fontsize=8, fontweight='bold')

# Data after
ax.text(7.5, 4.5, 'Imputed Data', ha='center', fontsize=10, fontweight='bold')
col_means = [3.0, 4.0, 3.5]
for i in range(3):
    for j in range(3):
        x, y = 6.5 + j*1.0, 3.5 - i*0.7
        val = data_before[i][j]
        if np.isnan(val):
            text = f'{col_means[j]:.1f}'
            color = '#27ae60'
        else:
            text = f'{val:.0f}'
            color = '#3498db'
        rect = Rectangle((x, y), 0.85, 0.55, facecolor=color, edgecolor='white', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+0.42, y+0.28, text, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')

ax.text(5, 0.8, 'Column-wise: NaN replaced by column mean', ha='center',
        fontsize=9, style='italic', color='#2c3e50')
plt.tight_layout()
save(fig, 'ch36_simpleimputer_flow.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 37: MISSING CATEGORICAL DATA
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 37 figures...')

# Fig 1: Strategies comparison
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
categories = ['A', 'B', 'C', 'D']
counts = [45, 30, 15, 10]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

# Most frequent
axes[0, 0].bar(categories, counts, color=colors, edgecolor='white')
axes[0, 0].set_title('Original Distribution', fontsize=10)
axes[0, 0].set_ylabel('Count')

# Most frequent imputation result
axes[0, 1].bar(categories, [45+20, 30, 15, 10], color=colors, edgecolor='white')
axes[0, 1].set_title('Most Frequent Imputation\n(20 NaN → A)', fontsize=10)
axes[0, 1].bar(['A'], [20], bottom=[45], color='#e74c3c', alpha=0.5, edgecolor='white', hatch='//')

# Constant imputation
axes[1, 0].bar(categories + ['Missing'], [45, 30, 15, 10, 20],
               color=colors + ['#95a5a6'], edgecolor='white')
axes[1, 0].set_title('Constant Imputation\n(20 NaN → "Missing")', fontsize=10)

# Add missing category
axes[1, 1].bar(categories + ['(Missing)'], [45, 30, 15, 10, 20],
               color=colors + ['#95a5a6'], edgecolor='white')
axes[1, 1].set_title('Add Missing Category\n(20 NaN → new category)', fontsize=10)

for ax in axes.flat:
    ax.tick_params(labelsize=7)
plt.suptitle('Strategies for Missing Categorical Data', fontsize=12, y=1.02)
plt.tight_layout()
save(fig, 'ch37_categorical_strategies.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 38: MISSING INDICATOR + RANDOM SAMPLE
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 38 figures...')

# Fig 1: Missing indicator concept
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
np.random.seed(42)
data = np.random.normal(50, 15, 100)
missing = np.random.random(100) < 0.25
data_with_nan = data.copy()
data_with_nan[missing] = np.nan

# Original with missing
axes[0].scatter(range(100), np.where(missing, 50, data), c=np.where(missing, 'red', 'steelblue'),
                s=20, alpha=0.7)
axes[0].set_title('Original Data\n(Red = missing)', fontsize=10)
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Value')

# Indicator feature
indicator = missing.astype(int)
axes[1].scatter(range(100), indicator + np.random.normal(0, 0.05, 100),
                c=np.where(missing, 'red', 'steelblue'), s=20, alpha=0.7)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Not Missing\n(0)', 'Missing\n(1)'])
axes[1].set_title('Missing Indicator Feature\n(0 = observed, 1 = missing)', fontsize=10)
axes[1].set_xlabel('Sample Index')
plt.tight_layout()
save(fig, 'ch38_missing_indicator.png')

# Fig 2: Random sample imputation
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
np.random.seed(42)
original = np.random.lognormal(2, 0.5, 200)
missing_idx = np.random.random(200) < 0.3
observed = original[~missing_idx]

# Before
axes[0].hist(observed, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
axes[0].set_title(f'Before\n({len(observed)} observed, {missing_idx.sum()} missing)', fontsize=9)

# Random sample imputation
random_values = np.random.choice(observed, size=missing_idx.sum(), replace=True)
imputed = original.copy()
imputed[missing_idx] = random_values
axes[1].hist(imputed, bins=30, color='#2ecc71', edgecolor='white', alpha=0.7)
axes[1].set_title(f'After Random Sample\n(all values filled)', fontsize=9)

# Compare distributions
axes[2].hist(observed, bins=30, color='steelblue', edgecolor='white', alpha=0.5, label='Original observed')
axes[2].hist(imputed, bins=30, color='#2ecc71', edgecolor='white', alpha=0.5, label='After imputation')
axes[2].set_title('Distribution Preserved', fontsize=9)
axes[2].legend(fontsize=7)
axes[2].set_xlabel('Value')
plt.tight_layout()
save(fig, 'ch38_random_sample.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 39: KNN IMPUTER
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 39 figures...')

# Fig 1: KNN concept
fig, ax = plt.subplots(figsize=(7, 5))
np.random.seed(42)
# Known points
known_x = np.random.uniform(1, 9, 20)
known_y = np.random.uniform(1, 9, 20)
ax.scatter(known_x, known_y, c='steelblue', s=60, label='Observed points', zorder=5)

# Missing point
mx, my = 5, 5
ax.scatter([mx], [my], c='red', s=200, marker='X', label='Point with missing value', zorder=6)

# K nearest neighbors
distances = np.sqrt((known_x - mx)**2 + (known_y - my)**2)
k = 5
k_indices = np.argsort(distances)[:k]
for i in k_indices:
    ax.plot([mx, known_x[i]], [my, known_y[i]], 'g--', alpha=0.5, lw=1.5)
    ax.scatter([known_x[i]], [known_y[i]], c='green', s=100, edgecolors='black',
               linewidths=1.5, zorder=7)

# Estimate
est_x = known_x[k_indices].mean()
est_y = known_y[k_indices].mean()
ax.scatter([est_x], [est_y], c='orange', s=150, marker='*', label='KNN estimate', zorder=8)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('KNN Imputation: Estimate Missing Value from K Nearest Neighbors')
ax.legend(fontsize=8)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.tight_layout()
save(fig, 'ch39_knn_concept.png')

# Fig 2: Euclidean distance visualization
fig, ax = plt.subplots(figsize=(5, 5))
p1 = (2, 3)
p2 = (7, 8)
ax.scatter(*p1, c='blue', s=100, zorder=5, label=f'P1{p1}')
ax.scatter(*p2, c='red', s=100, zorder=5, label=f'P2{p2}')
# Dashed lines for components
ax.plot([p1[0], p2[0]], [p1[1], p1[1]], 'b--', lw=1.5, alpha=0.5)
ax.plot([p2[0], p2[0]], [p1[1], p2[1]], 'r--', lw=1.5, alpha=0.5)
# Distance line
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', lw=2.5, label='Euclidean distance')
# Labels
ax.text((p1[0]+p2[0])/2, p1[1]-0.3, f'Δx = {p2[0]-p1[0]}', ha='center', fontsize=10, color='blue')
ax.text(p2[0]+0.2, (p1[1]+p2[1])/2, f'Δy = {p2[1]-p1[1]}', va='center', fontsize=10, color='red')
ax.text((p1[0]+p2[0])/2+0.3, (p1[1]+p2[1])/2+0.2,
        f'd = √({p2[0]-p1[0]}² + {p2[1]-p1[1]}²)\n  = √{((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)}\n  = {np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2):.2f}',
        fontsize=9, color='green',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.set_title('Euclidean Distance Calculation')
plt.tight_layout()
save(fig, 'ch39_euclidean_distance.png')

# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 40: MICE / ITERATIVE IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
print('Chapter 40 figures...')

# Fig 1: MICE iterative process
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Columns
col_names = ['Feature A', 'Feature B', 'Feature C', 'Feature D']
col_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
for j, (name, color) in enumerate(zip(col_names, col_colors)):
    x = 1.5 + j * 2
    box = FancyBboxPatch((x-0.7, 4.8), 1.4, 0.7, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='white')
    ax.add_patch(box)
    ax.text(x, 5.15, name, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Show iteration arrows
iterations = [
    ('Iter 1: Predict A\nusing B, C, D', 1.5, 3.5),
    ('Iter 2: Predict B\nusing A, C, D', 3.5, 3.5),
    ('Iter 3: Predict C\nusing A, B, D', 5.5, 3.5),
    ('Iter 4: Predict D\nusing A, B, C', 7.5, 3.5),
]
for text, x, y in iterations:
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.08",
                          facecolor='#eaf2f8', edgecolor='#2c3e50', lw=1)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5)

# Converge arrow
ax.annotate('', xy=(5, 1.5), xytext=(5, 3.0),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
box = FancyBboxPatch((3, 0.5), 4, 1, boxstyle="round,pad=0.12",
                      facecolor='#27ae60', edgecolor='white')
ax.add_patch(box)
ax.text(5, 1.0, 'Repeat until convergence\n(all missing values imputed)', ha='center',
        va='center', fontsize=9, color='white', fontweight='bold')
ax.set_title('MICE: Multivariate Imputation by Chained Equations', fontsize=12, fontweight='bold')
plt.tight_layout()
save(fig, 'ch40_mice_process.png')

# Fig 2: MICE vs KNN comparison
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
np.random.seed(42)
# Create correlated data
n = 200
x1 = np.random.normal(0, 1, n)
x2 = 0.8 * x1 + np.random.normal(0, 0.6, n)
x3 = 0.5 * x1 - 0.3 * x2 + np.random.normal(0, 0.8, n)

# Introduce missing in x3
missing = np.random.random(n) < 0.3
x3_missing = x3.copy()
x3_missing[missing] = np.nan

# KNN-style imputation (using neighbors)
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data_knn = np.column_stack([x1, x2, x3_missing])
knn_imp = KNNImputer(n_neighbors=5)
data_knn_imputed = knn_imp.fit_transform(data_knn)

data_mice = np.column_stack([x1, x2, x3_missing])
mice_imp = IterativeImputer(max_iter=10, random_state=42)
data_mice_imputed = mice_imp.fit_transform(data_mice)

axes[0].scatter(x3[~missing], data_knn_imputed[~missing, 2], alpha=0.5, s=15,
                label='Observed', color='steelblue')
axes[0].scatter(x3[missing], data_knn_imputed[missing, 2], alpha=0.5, s=15,
                label='Imputed (KNN)', color='red')
axes[0].plot([-3, 3], [-3, 3], 'k--', lw=1, alpha=0.5)
axes[0].set_title('KNN Imputation', fontsize=10)
axes[0].set_xlabel('True Value')
axes[0].set_ylabel('Imputed Value')
axes[0].legend(fontsize=7)

axes[1].scatter(x3[~missing], data_mice_imputed[~missing, 2], alpha=0.5, s=15,
                label='Observed', color='steelblue')
axes[1].scatter(x3[missing], data_mice_imputed[missing, 2], alpha=0.5, s=15,
                label='Imputed (MICE)', color='red')
axes[1].plot([-3, 3], [-3, 3], 'k--', lw=1, alpha=0.5)
axes[1].set_title('MICE Imputation', fontsize=10)
axes[1].set_xlabel('True Value')
axes[1].set_ylabel('Imputed Value')
axes[1].legend(fontsize=7)
plt.tight_layout()
save(fig, 'ch40_mice_vs_knn.png')

print('\nAll figures generated successfully!')
print(f'Total figures: {len(os.listdir(FIGDIR))}')
