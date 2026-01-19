import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row and keep only years from 2004 to 2010
filtered_df = df[(df['team'] == 'castleford tigers') & (df['year'] != 'total')]
# Extract the 'tries' column and compute variance
tries_var = filtered_df['tries'].var()
print(f"Final Answer: {tries_var:.1f}")