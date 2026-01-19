import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'official itv rating (millions)' and 'share (%)' to numeric, coercing errors to NaN
df['official itv rating (millions)'] = pd.to_numeric(df['official itv rating (millions)'], errors='coerce')
df['share (%)'] = pd.to_numeric(df['share (%)'], errors='coerce')

# Drop rows with missing values in either column
df.dropna(subset=['official itv rating (millions)', 'share (%)'], inplace=True)

# Calculate the correlation coefficient
correlation = df['official itv rating (millions)'].corr(df['share (%)'])

print(f"Final Answer: {correlation:.3f}")