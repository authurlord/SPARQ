import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row
df_filtered = df[df['year'] != 'total']
# Convert 'tries' to numeric
df_filtered['tries'] = pd.to_numeric(df_filtered['tries'])
# Calculate variance
variance_tries = df_filtered['tries'].var()
print(f"Final Answer: {variance_tries:.2f}")