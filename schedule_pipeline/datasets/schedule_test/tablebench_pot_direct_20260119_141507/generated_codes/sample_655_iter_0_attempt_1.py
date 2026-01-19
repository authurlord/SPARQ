import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'p1 diameter (mm)' and 'p max (bar)'
correlation = df['p1 diameter (mm)'].corr(df['p max ( bar )'])
print(f"Final Answer: {correlation:.3f}")