import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper numerical comparison
df['Year'] = df['Year'].astype(int)

# Extract the Conservative councillors column
conservative_councillors = df['Conservative councillors']

# Calculate the difference between consecutive years
differences = conservative_councillors.diff()

# Find the index of the maximum increase (largest positive difference)
max_increase_index = differences.idxmax()

# Get the corresponding year
year_with_max_increase = df.loc[max_increase_index, 'Year']

print(f"Final Answer: {year_with_max_increase}")