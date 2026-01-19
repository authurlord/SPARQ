import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'area (km square)' and 'pop'
correlation = df['area (km square)'].corr(df['pop'])
print(f"Final Answer: {correlation:.3f}")