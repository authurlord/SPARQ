import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2005 and 2009 (excluding the 'total' row)
filtered_df = df[(df['year'].astype(str).str.isdigit()) & (df['year'].between(2005, 2009))]
# Calculate average appearances (apps) for the filtered years
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")