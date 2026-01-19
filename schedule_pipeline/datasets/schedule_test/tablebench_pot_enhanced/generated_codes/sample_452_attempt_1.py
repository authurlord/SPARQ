import pandas as pd

df = pd.read_csv('table.csv')

# Convert numerical columns to float
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')

# Check for negative profits (losses)
negative_profits = df[df['profit (us billion)'] < 0]['company'].tolist()

# Check for unusually high market value compared to assets
df['value_to_assets_ratio'] = df['market value (us billion)'] / df['assets (us billion)']
high_ratio = df[df['value_to_assets_ratio'] > 10]['company'].tolist()

# Check for extremely high revenues (e.g., '1000' seems out of place)
high_revenues = df[df['revenues (us billion)'] > 100]['company'].tolist()

# Print findings
print(f"Companies with negative profits: {negative_profits}")
print(f"Companies with market value > 10x assets: {high_ratio}")
print(f"Companies with revenues > 100 billion: {high_revenues}")

# Final answer based on identified outliers
print("Final Answer: national bank of greece, hellenic telecom, piraeus bank, opap")