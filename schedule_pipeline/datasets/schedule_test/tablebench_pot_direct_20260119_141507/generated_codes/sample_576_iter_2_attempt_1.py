import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 1850 and 1855 inclusive
filtered_df = df[(df['year'] >= '1850') & (df['year'] <= '1855')]
# Convert 'number of tropical storms' to numeric (handles string values)
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'], errors='coerce')
# Calculate the average
average_storms = filtered_df['number of tropical storms'].mean()
print(f"Final Answer: {average_storms:.1f}")