import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, coercing errors to NaN
df['median household income'] = pd.to_numeric(df['median household income'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Drop rows with missing values in the relevant columns
df.dropna(subset=['median household income', 'population'], inplace=True)

# Calculate correlation coefficient
correlation = df['median household income'].corr(df['population'])

print(f"Final Answer: {correlation:.4f}")