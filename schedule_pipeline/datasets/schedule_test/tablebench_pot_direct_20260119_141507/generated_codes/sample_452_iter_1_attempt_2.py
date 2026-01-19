import pandas as pd

df = pd.read_csv('table.csv')

# Convert all numeric columns to float, handling missing or non-numeric entries
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'].str.replace('-', ''), errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'].str.replace('-', ''), errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')

# Identify outliers using thresholds
outliers = []

# Revenue outlier: any > 10x the median
median_revenue = df['revenues (us billion)'].median()
high_revenue_outlier = df[df['revenues (us billion)'] > 10 * median_revenue]

# Profit outlier: negative or very large magnitude
negative_profit = df[df['profit (us billion)'] < 0]
large_profit = df[df['profit (us billion)'] > 10]

# Asset vs market value: if asset >> market value, possible misalignment
asset_market_ratio = df['assets (us billion)'] / df['market value (us billion)']
high_ratio = df[asset_market_ratio > 10]  # ratio > 10 means assets much larger than market value

# Compile results
outliers.extend(high_revenue_outlier['company'].tolist())
outliers.extend(negative_profit['company'].tolist())
outliers.extend(high_ratio['company'].tolist())

# Remove duplicates and present unique companies
unique_outliers = list(set(outliers))

print(f"Final Answer: {', '.join(unique_outliers)}")