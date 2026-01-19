import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop density (per km square)' to numeric for calculation
df['pop density (per km square)'] = pd.to_numeric(df['pop density (per km square)'])
# Calculate median population density
median_density = df['pop density (per km square)'].median()
print(f"Final Answer: {median_density:.2f}")