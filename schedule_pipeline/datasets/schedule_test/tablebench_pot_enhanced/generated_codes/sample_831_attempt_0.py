import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'total' row
df_years = df[df['year'] != 'total']
# Convert 'tries' column to numeric
df_years['tries'] = pd.to_numeric(df_years['tries'])
# Calculate variance
variance_tries = df_years['tries'].var()
print(f"Final Answer: {variance_tries:.2f}")