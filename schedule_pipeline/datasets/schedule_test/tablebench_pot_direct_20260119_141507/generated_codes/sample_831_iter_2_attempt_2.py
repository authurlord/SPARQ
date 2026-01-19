import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the "total" row to only include years from 2004 to 2010
filtered_df = df[df['year'] != 'total']
# Calculate the variance of the 'tries' column
variance_tries = filtered_df['tries'].var()
print(f"Final Answer: {variance_tries:.1f}")