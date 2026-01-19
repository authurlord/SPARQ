import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer, handling any parsing errors
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter buildings constructed from 1895 to 1980 (inclusive)
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]

# Calculate average number of floors in the filtered buildings
average_floors = filtered_df['floors'].mean()

print(f"Final Answer: {average_floors:.1f}")