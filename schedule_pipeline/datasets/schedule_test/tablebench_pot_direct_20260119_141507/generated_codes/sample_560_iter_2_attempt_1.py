import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Castleford Tigers from 2005 to 2009 (excluding the total row)
filtered_df = df[(df['team'] == 'castleford tigers') & (df['year'].astype(int) >= 2005) & (df['year'].astype(int) <= 2009)]
# Calculate the average number of appearances
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")