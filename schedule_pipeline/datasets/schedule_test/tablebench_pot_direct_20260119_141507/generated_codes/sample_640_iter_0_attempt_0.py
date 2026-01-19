import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'size (cents)' and 'error'
correlation = df['size (cents)'].corr(df['error'])
print(f"Final Answer: {correlation:.3f}")