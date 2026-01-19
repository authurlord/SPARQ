import pandas as pd

df = pd.read_csv('table.csv')
# Convert the 'Year' column to integer for proper sorting and comparison
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Sort by Year to ensure chronological order
df = df.sort_values(by='Year')

# Extract Conservative councillors and compute year-over-year differences
conservative_councillors = df['Conservative councillors'].values
differences = []
for i in range(1, len(conservative_councillors)):
    diff = conservative_councillors[i] - conservative_councillors[i-1]
    differences.append(diff)

# Find the index of the maximum increase
max_increase_index = differences.index(max(differences)) + 1  # +1 because it's 0-indexed in differences
year_with_max_increase = df['Year'].iloc[max_increase_index]

print(f"Final Answer: {year_with_max_increase}")