import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the values for the years between 1850 and 1870 from the second column (index 1)
# The second column is labeled as '-' and contains the values
values = df[df['Year'].between(1850, 1870)]['-'].astype(int)

# Convert to list for easier processing
values_list = values.tolist()

# Compute differences between consecutive years
differences = []
for i in range(1, len(values_list)):
    diff = values_list[i] - values_list[i-1]
    differences.append(diff)

# Find the index of the maximum difference
max_diff_index = differences.index(max(differences))

# The value in the year with the highest increase is the value at max_diff_index + 1 (since we start from 1850)
year_with_max_increase_value = values_list[max_diff_index + 1]

print(f"Final Answer: {year_with_max_increase_value}")