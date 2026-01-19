import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the 'total' row
df_yearly = df[df['year'] != 'total']
# Convert 'tries' to numeric
df_yearly['tries'] = pd.to_numeric(df_yearly['tries'])
# Calculate variance
variance_tries = df_yearly['tries'].var()
print(f"Final Answer: {variance_tries:.2f}")