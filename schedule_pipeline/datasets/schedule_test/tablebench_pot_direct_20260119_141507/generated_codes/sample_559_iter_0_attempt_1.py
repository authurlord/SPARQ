import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 1895 and 1980 inclusive
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]
# Drop rows where 'floors' is missing (i.e., '-')
filtered_df = filtered_df[filtered_df['floors'].notna()]
# Convert 'floors' to numeric and calculate mean
mean_floors = filtered_df['floors'].astype(float).mean()
print(f"Final Answer: {mean_floors:.1f}")