import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'IP' and 'SO'
correlation = df['IP'].corr(df['SO'])
print(f"Final Answer: {correlation:.3f}")