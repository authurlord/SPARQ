import pandas as pd

df = pd.read_csv('table.csv')
# Convert population density to numeric
df['pop density (per km square)'] = pd.to_numeric(df['pop density (per km square)'])
# Count cities with population density over 3000
count_high_density = df[df['pop density (per km square)'] > 3000].shape[0]
print(f"Final Answer: {count_high_density}")