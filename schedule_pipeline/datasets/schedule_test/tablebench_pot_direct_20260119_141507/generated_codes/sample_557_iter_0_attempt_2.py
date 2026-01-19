import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper sorting
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Sort by Year to ensure chronological order
df = df.sort_values(by='Year')

# Extract Conservative councillors and compute year-on-year differences
conservative = df['Conservative councillors'].values
differences = []
for i in range(1, len(conservative)):
    diff = conservative[i] - conservative[i-1]
    differences.append(diff)

# Find the index of maximum increase
max_increase_index = differences.index(max(differences)) + 1  # +1 because we start from 0 in the original list
year_with_max_increase = df['Year'].iloc[max_increase_index]

print(f"Final Answer: {year_with_max_increase}")