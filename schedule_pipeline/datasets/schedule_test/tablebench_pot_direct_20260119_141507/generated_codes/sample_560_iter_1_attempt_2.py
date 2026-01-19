import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2005 and 2009 (exclusive of 'total' row)
filtered_df = df[(df['year'].str.isdigit()) & (df['year'].astype(int) >= 2005) & (df['year'].astype(int) <= 2009)]
# Calculate average appearances (apps)
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")