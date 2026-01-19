import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'revenue (millions)' and 'profit (millions)'
correlation = df['revenue (millions)'].corr(df['profit (millions)'])
print(f"Final Answer: {correlation:.2f}")