import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1935 to 1943
filtered_df = df[(df['Year'].astype(int) >= 1935) & (df['Year'].astype(int) <= 1943)]
# Convert 'Quantity withdrawn' to numeric and calculate mean
avg_withdrawn = filtered_df['Quantity withdrawn'].astype(int).mean()
print(f"Final Answer: {avg_withdrawn:.1f}")