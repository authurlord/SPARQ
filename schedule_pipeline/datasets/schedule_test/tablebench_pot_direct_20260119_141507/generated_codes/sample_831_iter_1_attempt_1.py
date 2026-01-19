import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row (where team is 'castleford tigers' and the row is 'total')
filtered_df = df[df['team'] != 'castleford tigers' | df['year'] != 'total']
# Extract the 'tries' column for years 2004 to 2010
tries_values = df[df['team'] == 'castleford tigers']['tries'].dropna()
# Compute variance of tries
variance = tries_values.var()
print(f"Final Answer: {variance:.1f}")