import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row and keep only years from 2004 to 2010
filtered_df = df[df['team'] == 'castleford tigers'] & (df['year'] != 'total')
# Extract tries column and compute variance
tries_values = filtered_df['tries'].dropna()
variance = tries_values.var()
print(f"Final Answer: {variance:.1f}")