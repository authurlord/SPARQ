import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter for 'la haute - côte - nord' regional county municipality
filtered_df = df[df['regional county municipality'] == 'la haute - côte - nord']

# Convert population to numeric (in case of any string issues)
filtered_df['population'] = pd.to_numeric(filtered_df['population'], errors='coerce')

# Remove any NaN (invalid entries)
filtered_df = filtered_df.dropna(subset=['population'])

# Calculate average population in the region
average_population = filtered_df['population'].mean()

# Find the minimum population
min_population = filtered_df['population'].min()

# Number of people to add = average - min (if min < average)
people_to_add = average_population - min_population

print(f"Final Answer: {people_to_add:.0f}")