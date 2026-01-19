import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where regional county municipality is "la haute - côte - nord"
filtered_df = df[df['regional county municipality'] == 'la haute - côte - nord']

# Convert population column to numeric
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Calculate average population of all municipalities in the region
avg_population = filtered_df['population'].mean()

# Find the minimum population in the region
min_population = filtered_df['population'].min()

# Number of people to add
people_to_add = avg_population - min_population

print(f"Final Answer: {people_to_add:.0f}")