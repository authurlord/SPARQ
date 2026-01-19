import pandas as pd

df = pd.read_csv('table.csv')
# Find the province with the highest density
max_density_row = df.loc[df['density'].idxmax()]
highest_density_province = max_density_row['province']
max_density = max_density_row['density']

# Calculate average density
average_density = df['density'].mean()

# Compute the difference
density_difference = max_density - average_density

print(f"Final Answer: {highest_density_province}, {density_difference:.2f}")