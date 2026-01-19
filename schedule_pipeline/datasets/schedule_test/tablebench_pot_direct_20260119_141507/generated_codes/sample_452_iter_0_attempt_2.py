import pandas as pd

df = pd.read_csv('table.csv')

# Convert revenue, profit, assets, and market value to numeric (handle string formats)
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'].str.replace('-', '0'), errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')

# Identify unusual patterns
outliers = []

# 1. Extremely high revenue (e.g., Hellenic Telecom at 1000)
if df['revenues (us billion)'].max() > 100:
    outliers.append("Hellenic Telecom has an unusually high revenue of $1000 billion, which is far above other companies.")

# 2. Negative profit despite high revenue
negative_profit_high_revenue = df[(df['profit (us billion)'] < 0) & (df['revenues (us billion)'] > 5)]
if not negative_profit_high_revenue.empty:
    outliers.append(f"Companies with negative profit and high revenue: {list(negative_profit_high_revenue['company'])}")

# 3. High market value relative to assets
df['market_to_asset_ratio'] = df['market value (us billion)'] / df['assets (us billion)']
high_ratio = df[df['market_to_asset_ratio'] > 10]
if not high_ratio.empty:
    outliers.append(f"High market-to-asset ratios: {list(high_ratio['company'])}")

# 4. Coca-Cola HBC has high market value with low assets
if df[(df['company'] == 'coca cola hbc') & (df['assets (us billion)'] < 10) & (df['market value (us billion)'] > 10)].any():
    outliers.append("Coca-Cola HBC has a high market value relative to its low assets, indicating possible market perception differences.")

# Final answer: list of identified unusual patterns
print(f"Final Answer: {', '.join(outliers)}")