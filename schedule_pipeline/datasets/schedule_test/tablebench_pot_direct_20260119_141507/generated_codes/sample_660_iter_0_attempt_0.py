import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'share (%)' to numeric (some values are 'n / a', which we treat as NaN)
df['share (%)'] = pd.to_numeric(df['share (%)'].str.replace('n / a', '', regex=False), errors='coerce')

# Calculate the correlation between 'official itv rating (millions)' and 'share (%)'
correlation = df['official itv rating (millions)'].corr(df['share (%)'])

print(f"Final Answer: {correlation:.3f}")