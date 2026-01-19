import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'density' to numeric, coercing errors to NaN if any
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Sort by density in descending order and take top 5
top_5 = df.sort_values(by='density', ascending=False).head(5)

# Calculate total population and average area
total_population = top_5['population'].sum()
average_area = top_5['area'].mean()

print(f"Final Answer: {total_population}, {average_area:.2f}")