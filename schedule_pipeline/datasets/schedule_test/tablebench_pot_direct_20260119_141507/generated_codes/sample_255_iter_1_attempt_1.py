import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'density' to numeric, coercing errors to NaN
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Remove any rows with NaN density (invalid entries)
df = df.dropna(subset=['density'])

# Sort by density in descending order and take top 5
top_5 = df.nlargest(5, 'density')

# Calculate total population and average area
total_population = top_5['population'].sum()
avg_area = top_5['area'].mean()

print(f"Final Answer: {total_population}, {avg_area:.1f}")