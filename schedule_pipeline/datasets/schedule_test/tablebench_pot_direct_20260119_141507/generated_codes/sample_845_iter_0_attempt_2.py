import pandas as pd

df = pd.read_csv('table.csv')
# Find the province with the highest population density
max_density_province = df.loc[df['density'].idxmax(), 'province']
max_density = df['density'].max()

# Calculate average population density
avg_density = df['density'].mean()

# Compute the difference
density_difference = max_density - avg_density

print(f"Final Answer: {max_density_province}, {density_difference:.2f}")