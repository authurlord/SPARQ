import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer type to enable numeric comparison
df['year'] = df['year'].astype(int)
# Filter rows where year is between 1850 and 1855 inclusive
filtered_df = df[(df['year'] >= 1850) & (df['year'] <= 1855)]
# Calculate the average number of tropical storms in the filtered rows
average_storms = filtered_df['number of tropical storms'].mean()
print(f"Final Answer: {average_storms:.1f}")