import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'km from kingston' and 'length meters'
correlation = df['km from kingston'].corr(df['length meters'])
print(f"Final Answer: {correlation:.3f}")