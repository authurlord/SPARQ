import pandas as pd

df = pd.read_csv('table.csv')
# Sort by density in descending order and take top 5
top_5_density = df.nlargest(5, 'density')

# Calculate total population and average area
total_population = top_5_density['population'].sum()
avg_area = top_5_density['area'].mean()

print(f"Final Answer: {total_population}, {avg_area:.1f}")