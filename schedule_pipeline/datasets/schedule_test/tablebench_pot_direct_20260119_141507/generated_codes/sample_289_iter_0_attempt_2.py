import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for "la haute - côte - nord"
filtered_df = df[df['regional county municipality'] == 'la haute - côte - nord']

# Calculate average population of the region
average_population = filtered_df['population'].mean()

# Find the minimum population in the region
min_population = filtered_df['population'].min()

# Number of people to add
people_to_add = average_population - min_population
print(f"Final Answer: {people_to_add:.0f}")