import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density' and 'area' to float for numerical operations
df['density'] = pd.to_numeric(df['density'])
df['area'] = pd.to_numeric(df['area'])
df['population'] = pd.to_numeric(df['population'])

# Sort by density in descending order and get top 5
top_5_density = df.nlargest(5, 'density')

# Calculate total population and average area
total_population = top_5_density['population'].sum()
average_area = top_5_density['area'].mean()

print(f"Final Answer: {total_population}, {average_area:.2f}")