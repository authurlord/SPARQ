import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density' column to float
df['density'] = pd.to_numeric(df['density'])
# Find the province with the highest density
max_density_province = df.loc[df['density'].idxmax(), 'province']
# Calculate average density
avg_density = df['density'].mean()
# Calculate difference
difference = df['density'].max() - avg_density
print(f"Final Answer: {max_density_province}, {difference:.2f}")