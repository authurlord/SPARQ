import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'floors' to numeric, coercing errors to NaN
df['floors'] = pd.to_numeric(df['floors'], errors='coerce')

# Filter rows where year is between 1895 and 1980 inclusive
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]

# Calculate the average number of floors in the filtered data
average_floors = filtered_df['floors'].mean()

print(f"Final Answer: {average_floors:.1f}")