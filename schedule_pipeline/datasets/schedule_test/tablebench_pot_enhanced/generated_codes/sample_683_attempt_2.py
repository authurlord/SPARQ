import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total fertility rate' and 'natural growth' to numeric, coercing errors to NaN
df['total fertility rate'] = pd.to_numeric(df['total fertility rate'], errors='coerce')
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')

# Drop rows with missing values in either column
df_clean = df.dropna(subset=['total fertility rate', 'natural growth'])

# Calculate correlation coefficient
correlation = df_clean['total fertility rate'].corr(df_clean['natural growth'])
print(f"Final Answer: {correlation:.4f}")