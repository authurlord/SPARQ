import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for comparison
df['year'] = pd.to_numeric(df['year'])
# Filter rows where year is between 1960 and 1980 (inclusive)
filtered_df = df[(df['year'] >= 1960) & (df['year'] <= 1980)]
# Calculate average floors
avg_floors = filtered_df['floors'].mean()
print(f"Final Answer: {avg_floors:.1f}")