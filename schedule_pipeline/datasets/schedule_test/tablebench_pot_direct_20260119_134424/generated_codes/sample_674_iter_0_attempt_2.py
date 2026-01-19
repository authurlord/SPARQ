import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'poles' and 'wins'
correlation = df['poles'].corr(df['wins'])
print(f"Final Answer: {correlation:.4f}")