import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'mccain %' and 'total'
correlation = df['mccain %'].corr(df['total'])
print(f"Final Answer: {correlation:.3f}")