import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2005 to 2009
filtered_df = df[(df['year'] >= '2005') & (df['year'] <= '2009')]
# Calculate average of 'apps' column
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")