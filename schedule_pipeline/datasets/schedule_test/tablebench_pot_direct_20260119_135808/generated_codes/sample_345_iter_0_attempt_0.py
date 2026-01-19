import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop density (per km square)' to numeric
df['pop density (per km square)'] = pd.to_numeric(df['pop density (per km square)'])
# Count cities with population density > 3000
count = df[df['pop density (per km square)'] > 3000].shape[0]
print(f"Final Answer: {count}")