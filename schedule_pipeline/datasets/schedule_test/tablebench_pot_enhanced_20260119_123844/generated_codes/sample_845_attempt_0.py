import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density' column to float
df['density'] = pd.to_numeric(df['density'])
# Find the province with the highest population density
max_density_province = df.loc[df['density'].idxmax(), 'province']
# Calculate average population density
avg_density = df['density'].mean()
# Calculate the difference
difference = df['density'].max() - avg_density
print(f"Final Answer: {max_density_province}, {difference:.2f}")