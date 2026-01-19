import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'density' column to numeric, handling any potential parsing issues
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Find the province with the highest population density
max_density_row = df.loc[df['density'].idxmax()]
highest_density_province = max_density_row['province']
max_density = df['density'].max()

# Calculate average population density
average_density = df['density'].mean()

# Compute the difference
density_difference = max_density - average_density

print(f"Final Answer: {highest_density_province}, {density_difference:.2f}")