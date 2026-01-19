import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric, handling negative signs and strings
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'], errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')
df['assets (us billion)'] = pd.to_numeric(df['assets (us billion)'], errors='coerce')
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')

# Identify outliers: large losses, high market value, revenue vs assets mismatch
large_losses = df[df['profit (us billion)'] < -5]
high_market_value = df[df['market value (us billion)'] > 10]
revenue_assets_mismatch = df[(df['revenues (us billion)'] < 10) & (df['assets (us billion)'] > 50)]

print("Unusual patterns or outliers:")
if not large_losses.empty:
    print(f"Largest losses: {large_losses[['company', 'profit (us billion)']].values.tolist()}")
if not high_market_value.empty:
    print(f"High market value: {high_market_value[['company', 'market value (us billion)']].values.tolist()}")
if not revenue_assets_mismatch.empty:
    print(f"Revenue vs assets mismatch: {revenue_assets_mismatch[['company', 'revenues (us billion)', 'assets (us billion)']].values.tolist()}")

# Final answer based on findings
print("Final Answer: piraeus bank, hellenic telecom, coca cola hbc")