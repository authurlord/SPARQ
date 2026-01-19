import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'population' and 'density'
correlation = df['population'].corr(df['density'])
print(f"Final Answer: {correlation:.2f}")