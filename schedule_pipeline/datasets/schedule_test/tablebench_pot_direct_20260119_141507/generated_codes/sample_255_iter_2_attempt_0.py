import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'population', 'area', and 'density' to numeric types
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Sort by density in descending order and get top 5
top_5 = df.sort_values(by='density', ascending=False).head(5)

# Calculate total population and average area
total_population = top_5['population'].sum()
average_area = top_5['area'].mean()

print(f"Final Answer: {total_population}, {average_area:.2f}")