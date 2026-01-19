import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 1895 and 1980 (inclusive)
filtered_df = df[(df['year'].astype(int) >= 1895) & (df['year'].astype(int) <= 1980)]
# Convert 'floors' to integer and calculate mean
avg_floors = filtered_df['floors'].astype(int).mean()
print(f"Final Answer: {avg_floors:.1f}")