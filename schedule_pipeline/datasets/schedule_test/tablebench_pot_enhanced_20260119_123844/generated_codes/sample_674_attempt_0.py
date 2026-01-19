import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'poles' and 'wins'
correlation = df['poles'].astype(int).corr(df['wins'].astype(int))
print(f"Final Answer: {correlation:.3f}")