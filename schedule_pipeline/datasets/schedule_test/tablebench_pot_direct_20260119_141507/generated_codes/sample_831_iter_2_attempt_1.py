import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2004 and 2010, excluding the "total" row
filtered_df = df[df['year'].str.isdigit() & (df['year'].astype(int) >= 2004) & (df['year'].astype(int) <= 2010)]
# Extract the 'tries' column and calculate variance
tries_variance = filtered_df['tries'].var()
print(f"Final Answer: {tries_variance:.1f}")