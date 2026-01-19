import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of tropical storms' to numeric, coercing errors to NaN
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'], errors='coerce')

# Filter years from 1850 to 1855 inclusive
filtered_df = df[(df['year'] >= '1850') & (df['year'] <= '1855')]

# Calculate the average number of tropical storms
average_storms = filtered_df['number of tropical storms'].mean()

print(f"Final Answer: {average_storms:.1f}")