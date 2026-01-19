import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between '1850' and '1855' inclusive
filtered_df = df[(df['year'] >= '1850') & (df['year'] <= '1855')]
# Calculate the average number of tropical storms in the filtered rows
avg_storms = filtered_df['number of tropical storms'].mean()
print(f"Final Answer: {avg_storms:.1f}")