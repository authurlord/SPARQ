import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the population values for age groups
# The relevant row is the one with "POPULATION (by age group in 2002)"
# We pick the row where SPECIFICATION is "I." and the value for "POPULATION (by age group in 2002)" is present

# Since the data is structured, we extract the values directly from the row after the header
# Find the row that contains the actual population values (after the first row with headers)

# The first row is header, then the next rows have data
# We extract the values from the columns: 'POPULATION (by age group in 2002)_1' to '_9'
# But the values are in order: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+

# Extract the values from the row where SPECIFICATION is 'I.' and the row after the header
# Actually, the row with index 1 is: ['I.', 'TOTAL', 'TOTAL', 'TOTAL', 'TOTAL', 'person', '287', '41', '45', '47', '27', '38', '31', '20', '24', '14']
# So we take the values from index 7 to 15 (which are the age group populations)

# Get the population values
pop_values = df.iloc[1][7:]  # From index 7 to end (age groups)

# 0-9 is at index 7 → 41
# 10-19 → index 8 → 45
# 20-29 → index 9 → 47
# 30-39 → index 10 → 27
# 40-49 → index 11 → 38
# 50-59 → index 12 → 31
# 60-69 → index 13 → 20
# 70-79 → index 14 → 24
# 80+ → index 15 → 14

# Dependent: 0-10 (use 0-9) and 60+
dependent_population = pop_values.iloc[0] + pop_values.iloc[8]  # 0-9 and 80+

# Working age: 11-59 → sum of 10-19, 20-29, 30-39, 40-49, 50-59
working_population = pop_values.iloc[1] + pop_values.iloc[2] + pop_values.iloc[3] + pop_values.iloc[4] + pop_values.iloc[5]

# Calculate ratio
dependency_ratio = dependent_population / working_population

print(f"Final Answer: {dependency_ratio:.2f}")