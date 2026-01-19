import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer and 'floors' to numeric, coercing errors to NaN
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['floors'] = pd.to_numeric(df['floors'], errors='coerce')

# Filter buildings constructed from 1895 to 1980
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]

# Calculate average floors
avg_floors = filtered_df['floors'].mean()

print(f"Final Answer: {avg_floors:.1f}")