import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper comparison
df['Year'] = df['Year'].astype(int)

# Extract the Conservative councillors column
conservative_councillors = df['Conservative councillors']

# Compute the difference between consecutive years
diff = conservative_councillors.diff()

# Find the index of the maximum increase (excluding the first row since diff starts at index 1)
max_increase_index = diff.idxmax()

# Get the corresponding year
year_with_max_increase = df.loc[max_increase_index, 'Year']

print(f"Final Answer: {year_with_max_increase}")