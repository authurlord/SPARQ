import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the values for years 1850 to 1870
# The first column is 'Year'
years_of_interest = [1850, 1855, 1860, 1865, 1870]

# Find the corresponding values — the value in the first column after 'Year'?
# Actually, the first column is Year, and the second column is the value for that year.
# So for each row, the value is in the second column (index 1)

values = []
for year in years_of_interest:
    # Find the row where Year == year
    row = df[df['Year'] == year]
    if not row.empty:
        value = row.iloc[0][1]  # second column (index 1)
        values.append(value)

# Compute year-on-year differences
differences = []
for i in range(1, len(values)):
    diff = values[i] - values[i-1]
    differences.append(diff)

# Find the index of maximum difference
max_diff_index = differences.index(max(differences))

# The year with the highest increase is the one at max_diff_index + 1850
year_with_max_increase = years_of_interest[max_diff_index + 1]

# The value in that year
value_in_max_increase_year = values[max_diff_index + 1]

print(f"Final Answer: {value_in_max_increase_year}")