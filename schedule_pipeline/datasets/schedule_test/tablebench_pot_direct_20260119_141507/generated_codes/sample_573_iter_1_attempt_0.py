import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer
df['year'] = df['year'].astype(int)
# Filter buildings constructed between 1960 and 1980 (inclusive)
filtered_buildings = df[(df['year'] >= 1960) & (df['year'] <= 1980)]
# Calculate average number of floors
avg_floors = filtered_buildings['floors'].mean()
print(f"Final Answer: {avg_floors:.1f}")