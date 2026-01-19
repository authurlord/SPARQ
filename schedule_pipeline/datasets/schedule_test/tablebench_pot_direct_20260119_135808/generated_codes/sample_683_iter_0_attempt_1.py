import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, replacing 'na' with NaN and dropping rows with NaN
df['total fertility rate'] = pd.to_numeric(df['total fertility rate'], errors='coerce')
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')
df.dropna(subset=['total fertility rate', 'natural growth'], inplace=True)

# Calculate correlation coefficient
correlation = df['total fertility rate'].corr(df['natural growth'])
print(f"Final Answer: {correlation:.3f}")