import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 1850 to 1855
filtered_df = df[df['year'].between(1850, 1855)]
# Calculate the mean of 'number of tropical storms'
average_storms = filtered_df['number of tropical storms'].mean()
print(f"Final Answer: {average_storms:.1f}")