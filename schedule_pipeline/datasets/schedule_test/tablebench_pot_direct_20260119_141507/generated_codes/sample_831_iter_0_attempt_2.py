import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Castleford Tigers excluding the total row
filtered_df = df[df['team'] == 'castleford tigers'] & (df['year'] != 'total')
# Extract the 'tries' column and compute variance
tries_var = filtered_df['tries'].var()
print(f"Final Answer: {tries_var:.1f}")