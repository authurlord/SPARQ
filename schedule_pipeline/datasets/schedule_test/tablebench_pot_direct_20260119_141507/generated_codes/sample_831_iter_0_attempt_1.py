import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row and extract only the years from 2004 to 2010
filtered_df = df[df['year'].str.contains(r'^200[4-9]$') & (df['team'] == 'castleford tigers')]
# Extract the 'tries' column and calculate variance
tries_var = filtered_df['tries'].var()
print(f"Final Answer: {tries_var:.1f}")