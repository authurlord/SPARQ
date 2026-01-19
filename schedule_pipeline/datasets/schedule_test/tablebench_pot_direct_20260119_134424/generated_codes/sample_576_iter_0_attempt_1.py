import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1850 to 1855
filtered_df = df[(df['year'] >= '1850') & (df['year'] <= '1855')]
# Calculate average number of tropical storms
avg_storms = filtered_df['number of tropical storms'].astype(int).mean()
print(f"Final Answer: {avg_storms:.1f}")