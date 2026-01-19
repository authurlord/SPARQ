import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year to integer and sort by Year
df['Year'] = df['Year'].astype(int)
df = df.sort_values(by='Year')

# Extract Conservative councillors values
conservative_councillors = df['Conservative councillors']

# Calculate year-over-year differences
differences = conservative_councillors.diff().dropna()

# Find the index of the maximum increase
max_increase_index = differences.idxmax()
# Get the corresponding year
year_with_max_increase = df.loc[max_increase_index, 'Year']

print(f"Final Answer: {year_with_max_increase}")