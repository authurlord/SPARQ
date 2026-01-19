import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2005 to 2009
filtered_df = df[(df['year'] >= '2005') & (df['year'] <= '2009')]
# Convert 'apps' column to numeric
filtered_df['apps'] = pd.to_numeric(filtered_df['apps'])
# Calculate average apps
avg_apps = filtered_df['apps'].mean()
print(f"Final Answer: {avg_apps:.1f}")