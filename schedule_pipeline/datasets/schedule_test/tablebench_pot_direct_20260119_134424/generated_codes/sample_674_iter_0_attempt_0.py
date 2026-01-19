import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'poles' and 'wins'
correlation = df['poles'].astype(float).corr(df['wins'].astype(float))
print(f"Final Answer: {correlation:.3f}")