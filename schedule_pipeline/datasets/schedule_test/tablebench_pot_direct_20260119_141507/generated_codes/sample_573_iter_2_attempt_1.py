import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'floors' to numeric, coercing errors to NaN
df['floors'] = pd.to_numeric(df['floors'], errors='coerce')

# Filter buildings constructed between 1960 and 1980 (inclusive)
filtered_df = df[(df['year'].astype(int) >= 1960) & (df['year'].astype(int) <= 1980)]

# Calculate average floors for the filtered rows
average_floors = filtered_df['floors'].mean()

print(f"Final Answer: {average_floors:.1f}")