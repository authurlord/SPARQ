import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density' column to numeric, handling any potential parsing issues
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Find the province with the highest density
max_density_row = df.loc[df['density'].idxmax()]
highest_density_province = max_density_row['province']
max_density = df['density'].max()

# Compute average density
avg_density = df['density'].mean()

# Calculate the difference
density_difference = max_density - avg_density

print(f"Final Answer: {highest_density_province}, {density_difference:.2f}")