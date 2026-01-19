import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# We are interested in the row where SPECIFICATION is 'I. TOTAL'
# This row has the population by age group
total_row = df[df['SPECIFICATION'] == 'I. TOTAL']

# Extract the population values for age groups
# The values are in columns: POPULATION (by age group in 2002) onwards
# Columns indices: 
#   0: SPECIFICATION
#   1: SPECIFICATION_1 → 0-9
#   2: 10-19
#   3: 20-29
#   4: 30-39
#   5: 40-49
#   6: 50-59
#   7: 60-69
#   8: 70-79
#   9: 80 +

# Extract the values from the total row (only one row)
pop_data = total_row.iloc[0][1:].astype(float)  # Convert to float to avoid string division

# Age groups:
# 0-9 → index 0
# 10-19 → index 1
# 20-29 → index 2
# 30-39 → index 3
# 40-49 → index 4
# 50-59 → index 5
# 60-69 → index 6
# 70-79 → index 7
# 80+ → index 8

# Numerator: 0-9 + 80+
numerator = pop_data[0] + pop_data[8]

# Denominator: 10-19 to 50-59 → indices 1 to 5
denominator = pop_data[1] + pop_data[2] + pop_data[3] + pop_data[4] + pop_data[5]

# Calculate ratio
dependency_ratio = numerator / denominator

# Output the approximate value
print(f"Final Answer: {dependency_ratio:.2f}")