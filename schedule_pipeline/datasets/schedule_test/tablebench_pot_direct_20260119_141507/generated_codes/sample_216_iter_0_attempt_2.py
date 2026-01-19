import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between wife's age and number of children
correlation = df['her age'].corr(df['of children'])
print(f"Final Answer: {correlation:.2f}")