import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric, coercing errors to NaN
df['revenue (million)'] = pd.to_numeric(df['revenue (million)'], errors='coerce')
df['profit / (loss) before tax (m)'] = pd.to_numeric(df['profit / (loss) before tax (m)'], errors='coerce')
df['net profit (m)'] = pd.to_numeric(df['net profit (m)'], errors='coerce')
df['earnings per share (p)'] = pd.to_numeric(df['earnings per share (p)'], errors='coerce')

# Drop rows with missing values for correlation
df_clean = df.dropna()

# Calculate correlation with earnings per share
correlations = df_clean[['revenue (million)', 'profit / (loss) before tax (m)', 'net profit (m)', 'earnings per share (p)']].corr()['earnings per share (p)'].abs().sort_values(ascending=False)

# Get the most influential factor (excluding earnings per share itself)
influential_factor = correlations.index[1]  # Index 0 is 'earnings per share (p)' itself
print(f"Final Answer: {influential_factor}")