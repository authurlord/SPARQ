import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density' column to float for numerical operations
df['density'] = pd.to_numeric(df['density'])

# Find the province with the highest population density
max_density_province = df.loc[df['density'].idxmax(), 'province']
max_density = df['density'].max()

# Calculate average population density
avg_density = df['density'].mean()

# Compute the difference
difference = max_density - avg_density

print(f"Final Answer: {max_density_province}, {difference:.2f}")