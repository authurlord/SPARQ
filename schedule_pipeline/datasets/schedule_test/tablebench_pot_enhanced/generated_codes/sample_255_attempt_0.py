import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'density' to float for sorting
df['density'] = pd.to_numeric(df['density'])
# Sort by density in descending order and get top 5
top_5 = df.nlargest(5, 'density')
# Calculate total population and average area
total_population = top_5['population'].sum()
average_area = top_5['area'].mean()
print(f"Final Answer: {total_population}, {average_area:.2f}")