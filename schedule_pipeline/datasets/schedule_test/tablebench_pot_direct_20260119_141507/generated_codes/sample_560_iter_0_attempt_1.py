import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2005 to 2009 (excluding the total row)
filtered_df = df[df['year'].between(2005, 2009)]
# Calculate the mean of 'apps' for these years
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")