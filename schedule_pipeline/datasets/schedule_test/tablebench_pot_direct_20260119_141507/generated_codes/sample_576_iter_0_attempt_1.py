import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 1850 to 1855 inclusive
filtered_df = df[df['year'].between(1850, 1855)]
# Calculate the mean of 'number of tropical storms'
mean_storms = filtered_df['number of tropical storms'].mean()
print(f"Final Answer: {mean_storms:.1f}")