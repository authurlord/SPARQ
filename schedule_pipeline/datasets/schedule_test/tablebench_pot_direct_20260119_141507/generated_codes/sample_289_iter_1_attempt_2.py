import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter for 'la haute - côte - nord' region
filtered_df = df[df['regional county municipality'] == 'la haute - côte - nord']

# Convert population to numeric (in case of any string issues)
filtered_df['population'] = pd.to_numeric(filtered_df['population'], errors='coerce')

# Drop any rows with invalid population values
filtered_df = filtered_df.dropna(subset=['population'])

# Calculate average population of the region
average_population = filtered_df['population'].mean()

# Find the minimum population
min_population = filtered_df['population'].min()

# People needed to add to the smallest municipality to reach average
people_to_add = average_population - min_population

print(f"Final Answer: {people_to_add:.0f}")