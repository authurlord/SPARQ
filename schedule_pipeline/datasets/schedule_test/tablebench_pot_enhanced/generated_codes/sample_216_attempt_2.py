import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'her age' and 'of children'
correlation = df['her age'].astype(int).corr(df['of children'].astype(int))
print(f"Final Answer: {correlation:.3f}")