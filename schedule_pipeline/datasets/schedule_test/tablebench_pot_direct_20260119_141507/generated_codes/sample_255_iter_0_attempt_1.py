import pandas as pd

df = pd.read_csv('table.csv')
# Sort by density in descending order and take top 5
top_5_density = df.sort_values(by='density', ascending=False).head(5)

# Total population of top 5 provinces
total_population = top_5_density['population'].sum()

# Average area of these provinces
avg_area = top_5_density['area'].mean()

print(f"Final Answer: {total_population}, {avg_area:.1f}")