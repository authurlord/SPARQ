import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for years between 2004 and 2013 (inclusive)
filtered_df = df[(df['Year'] >= '2004') & (df['Year'] <= '2013')]

# Convert 'No. of performances' to numeric, coercing errors to NaN
filtered_df['No. of performances'] = pd.to_numeric(filtered_df['No. of performances'], errors='coerce')

# Drop rows with NaN in 'No. of performances'
filtered_df = filtered_df.dropna(subset=['No. of performances'])

# Calculate average number of performances
average_performances = filtered_df['No. of performances'].mean()

print(f"Final Answer: {average_performances:.1f}")