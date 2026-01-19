import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper indexing
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Extract Conservative councillors values
conservative_councillors = df['Conservative councillors']

# Calculate the difference between consecutive years
differences = conservative_councillors.diff()

# Find the index of the maximum positive difference
max_increase_index = differences.idxmax()

# Since differences is indexed by row, we need to find the corresponding year
# The maximum increase occurs at the next year after the previous one
# So the year of maximum increase is the year at index +1
year_of_max_increase = df.loc[differences.idxmax() + 1, 'Year']

print(f"Final Answer: {year_of_max_increase}")