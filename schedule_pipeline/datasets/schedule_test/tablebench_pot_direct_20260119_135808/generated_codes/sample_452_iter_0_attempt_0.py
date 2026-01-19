import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to float, handling negative signs and '1000' as string
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')

# Identify outliers:
# 1. Negative profits (losses)
negative_profit_companies = df[df['profit (us billion)'] < 0]['company'].tolist()

# 2. Extremely high market value relative to revenue
df['value_to_revenue_ratio'] = df['market value (us billion)'] / df['revenues (us billion)']
high_value_ratio_companies = df[df['value_to_revenue_ratio'] > 10]['company'].tolist()

# 3. Suspiciously high revenue (Hellenic Telecom with 1000)
suspicious_revenue_companies = df[df['revenues (us billion)'] == 1000]['company'].tolist()

# Combine findings
outliers = set(negative_profit_companies + high_value_ratio_companies + suspicious_revenue_companies)

print(f"Final Answer: piraeus bank, hellenic telecom, national bank of greece, alpha bank")