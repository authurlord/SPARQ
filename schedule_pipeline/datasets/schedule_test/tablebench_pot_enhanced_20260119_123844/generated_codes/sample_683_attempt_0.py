import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total fertility rate' and 'natural growth' to numeric, coercing errors to NaN
df['total fertility rate'] = pd.to_numeric(df['total fertility rate'], errors='coerce')
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['total fertility rate', 'natural growth'], inplace=True)

# Calculate correlation coefficient
correlation = df['total fertility rate'].corr(df['natural growth'])

print(f"Final Answer: {correlation:.3f}")