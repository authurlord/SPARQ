import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for comparison
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter buildings constructed from 1895 to 1980
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]

# Convert 'floors' to integer and calculate mean
average_floors = filtered_df['floors'].astype(int).mean()

print(f"Final Answer: {average_floors:.1f}")