import pandas as pd

df = pd.read_csv('table.csv')
# Filter buildings constructed from 1895 to 1980
filtered_df = df[(df['year'].astype(int) >= 1895) & (df['year'].astype(int) <= 1980)]
# Calculate average floors
avg_floors = filtered_df['floors'].astype(int).mean()
print(f"Final Answer: {avg_floors:.1f}")