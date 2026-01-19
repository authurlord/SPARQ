import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years from 1850 to 1855 inclusive
filtered_df = df[df['year'].str.startswith('185') & (df['year'] <= '1855')]
# Calculate the average number of tropical storms
avg_storms = filtered_df['number of tropical storms'].mean()
print(f"Final Answer: {avg_storms:.1f}")