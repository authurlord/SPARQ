import pandas as pd

df = pd.read_csv('table.csv')

# Filter for "la haute - côte - nord"
filtered_df = df[df['regional county municipality'] == 'la haute - côte - nord']

# Convert population to numeric (some might be strings with non-numeric issues)
filtered_df['population'] = pd.to_numeric(filtered_df['population'], errors='coerce')

# Remove any invalid entries (if any)
filtered_df = filtered_df.dropna(subset=['population'])

# Calculate average population of the region
avg_population = filtered_df['population'].mean()

# Find minimum population in the region
min_population = filtered_df['population'].min()

# People needed to add to reach average
people_needed = avg_population - min_population

print(f"Final Answer: {people_needed:.0f}")