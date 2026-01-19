import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation coefficient between revenue and profit
correlation = df['revenue (millions)'].corr(df['profit (millions)'])
print(f"Final Answer: {correlation:.2f}")