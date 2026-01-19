import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Castleford Tigers from 2005 to 2009 (excluding 'total' row)
filtered_df = df[(df['year'].str.isdigit()) & (df['year'].astype(int) >= 2005) & (df['year'].astype(int) <= 2009) & (df['team'] == 'castleford tigers')]
# Calculate average appearances (apps)
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")