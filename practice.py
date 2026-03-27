import matplotlib
matplotlib.use('Agg')  # non-interactive backend - saves plot without opening a window
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# -- Load Data ------------------------------------------------------------------
df = pd.read_csv("house_prices.csv")
df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
df['yr_renovated'] = df['yr_renovated'].replace(0, np.nan)

print("=" * 60)
print("KING COUNTY HOUSE PRICES - EXPLORATORY ANALYSIS")
print("=" * 60)

# -- 1. Basic Overview ----------------------------------------------------------
print(f"\n[1] Dataset Shape: {df.shape[0]:,} rows - {df.shape[1]} columns")
print(f"    Date Range   : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"    Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# -- 2. Price Summary -----------------------------------------------------------
print("\n[2] Price Statistics (USD)")
print(f"    Min    : ${df['price'].min():>12,.0f}")
print(f"    Median : ${df['price'].median():>12,.0f}")
print(f"    Mean   : ${df['price'].mean():>12,.0f}")
print(f"    Max    : ${df['price'].max():>12,.0f}")
print(f"    Std Dev: ${df['price'].std():>12,.0f}")

# -- 3. Key Correlations with Price --------------------------------------------
num_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
            'floors', 'view', 'grade', 'sqft_above', 'sqft_basement',
            'yr_built', 'sqft_living15', 'sqft_lot15']
corr = df[num_cols].corr()['price'].drop('price').sort_values(ascending=False)
print("\n[3] Top Correlations with Price:")
for feat, val in corr.items():
    bar = "-" * int(abs(val) * 20)
    print(f"    {feat:<18} {val:+.3f}  {bar}")

# -- 4. Price by Bedrooms -------------------------------------------------------
print("\n[4] Median Price by Bedrooms:")
bed_price = df[df['bedrooms'] <= 8].groupby('bedrooms')['price'].median()
for bed, price in bed_price.items():
    print(f"    {bed} bed(s): ${price:>10,.0f}")

# -- 5. Waterfront Impact -------------------------------------------------------
wf = df.groupby('waterfront')['price'].median()
print(f"\n[5] Waterfront Premium:")
print(f"    Non-waterfront : ${wf.get('N', wf.iloc[0]):>10,.0f}")
print(f"    Waterfront     : ${wf.get('Y', wf.iloc[-1]):>10,.0f}")
premium_pct = (wf.iloc[-1] / wf.iloc[0] - 1) * 100
print(f"    Premium        : +{premium_pct:.1f}%")

# -- 6. Grade vs Price ----------------------------------------------------------
print("\n[6] Median Price by Grade:")
grade_price = df.groupby('grade')['price'].median().sort_index()
for grade, price in grade_price.items():
    print(f"    Grade {grade:<4}: ${price:>10,.0f}")

# -- 7. Renovation Impact ------------------------------------------------------
renovated_med   = df[df['yr_renovated'].notna()]['price'].median()
unrenovated_med = df[df['yr_renovated'].isna()]['price'].median()
print(f"\n[7] Renovation Impact on Median Price:")
print(f"    Renovated     : ${renovated_med:>10,.0f}")
print(f"    Not Renovated : ${unrenovated_med:>10,.0f}")
print(f"    Uplift        : +{(renovated_med / unrenovated_med - 1)*100:.1f}%")

# -- 8. Top 5 Zip Codes by Median Price ----------------------------------------
print("\n[8] Top 5 Zipcodes by Median Price:")
top_zip = df.groupby('zipcode')['price'].median().nlargest(5)
for zip_, price in top_zip.items():
    print(f"    {zip_}: ${price:>10,.0f}")

# -- 9. Sales Volume by Month ---------------------------------------------------
df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').size()
peak = monthly.idxmax()
print(f"\n[9] Peak Sales Month: {peak} ({monthly[peak]:,} sales)")

# ==============================================================================
# PLOTS
# ==============================================================================
dollar_fmt = FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle("King County House Prices - Key Insights", fontsize=16, fontweight='bold', y=1.01)

# Plot 1: Price Distribution
ax = axes[0, 0]
df[df['price'] < 3e6]['price'].hist(bins=60, ax=ax, color='steelblue', edgecolor='white')
ax.xaxis.set_major_formatter(dollar_fmt)
ax.set_title("Price Distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Count")

# Plot 2: Price vs sqft_living
ax = axes[0, 1]
ax.scatter(df['sqft_living'], df['price'], alpha=0.15, s=5, color='steelblue')
ax.yaxis.set_major_formatter(dollar_fmt)
ax.set_title("Price vs Living Area (sqft)")
ax.set_xlabel("sqft_living")
ax.set_ylabel("Price")

# Plot 3: Median Price by Bedrooms
ax = axes[1, 0]
bed_price.plot(kind='bar', ax=ax, color='teal', edgecolor='white')
ax.yaxis.set_major_formatter(dollar_fmt)
ax.set_title("Median Price by Bedrooms")
ax.set_xlabel("Bedrooms")
ax.set_ylabel("Median Price")
ax.tick_params(axis='x', rotation=0)

# Plot 4: Median Price by Grade
ax = axes[1, 1]
grade_price.plot(kind='bar', ax=ax, color='coral', edgecolor='white')
ax.yaxis.set_major_formatter(dollar_fmt)
ax.set_title("Median Price by Grade")
ax.set_xlabel("Grade")
ax.set_ylabel("Median Price")
ax.tick_params(axis='x', rotation=0)

# Plot 5: Correlation Heatmap
ax = axes[2, 0]
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", ax=ax,
            cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Matrix")

# Plot 6: Monthly Sales Volume
ax = axes[2, 1]
monthly.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='white')
ax.set_title("Monthly Sales Volume")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Sales")
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("house_prices_analysis.png", dpi=150, bbox_inches='tight')
print("\n[-] Plot saved to house_prices_analysis.png")
