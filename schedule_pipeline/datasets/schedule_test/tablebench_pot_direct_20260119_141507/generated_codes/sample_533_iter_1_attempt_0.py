import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Identify the row for total population by age group
# We are interested in the row where SPECIFICATION is 'I.' and the first value is 'TOTAL'
# The row with values: ['I.', 'TOTAL', 'TOTAL', 'TOTAL', 'TOTAL', 'person', '287', '41', '45', '47', '27', '38', '31', '20', '24', '14']
# This corresponds to the row where the first column is 'I.' and the second is 'TOTAL'

# Extract the row for total population
total_row = df[df['SPECIFICATION'] == 'I.'].iloc[1]  # Skip header row

# Extract the population values by age group
# Columns: 'POPULATION (by age group in 2002)' onwards
pop_values = total_row[7:]  # From index 7 to end (since index 6 is 'TOTAL', index 7 is '0-9')

# Age groups:
# 0-9 → index 0 in pop_values
# 10-19 → index 1
# 20-29 → index 2
# 30-39 → index 3
# 40-49 → index 4
# 50-59 → index 5
# 60-69 → index 6
# 70-79 → index 7
# 80+ → index 8

# Numerator: 0-9 + 60-69 + 70-79 + 80+
numerator = pop_values[0] + pop_values[6] + pop_values[7] + pop_values[8]

# Denominator: 10-19 + 20-29 + 30-39 + 40-49 + 50-59
denominator = pop_values[1] + pop_values[2] + pop_values[3] + pop_values[4] + pop_values[5]

# Calculate dependency ratio
dependency_ratio = numerator / denominator

print(f"Final Answer: {dependency_ratio:.2f}")