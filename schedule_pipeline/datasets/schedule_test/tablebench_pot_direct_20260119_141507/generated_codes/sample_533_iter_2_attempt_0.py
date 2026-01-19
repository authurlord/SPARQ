import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant values from the row where SPECIFICATION is 'TOTAL' and POPULATION is given
# We look for the row where the first value is 'I.' and the next is 'TOTAL' for each age group
# The row with 'TOTAL' for age groups is the one with values: ['I.', 'TOTAL', 'TOTAL', 'TOTAL', 'TOTAL', 'person', '287', '41', '45', '47', '27', '38', '31', '20', '24', '14']

# Find the row where the first column is 'I.' and the second is 'TOTAL'
# The actual row of interest is the one after the header
# We can locate it by checking if the first column is 'I.' and the second is 'TOTAL'

# Extract the age group population values
age_groups = df.iloc[1]  # This is the row with 'I., TOTAL, TOTAL, ...'
# The age group columns start from 'POPULATION (by age group in 2002)' onwards
# The values for age groups are in positions 6 to 15

# Values:
# index 6: TOTAL
# index 7: 0-9
# index 8: 10-19
# index 9: 20-29
# index 10: 30-39
# index 11: 40-49
# index 12: 50-59
# index 13: 60-69
# index 14: 70-79
# index 15: 80+

# We extract the values from index 7 to 15 (excluding the first "TOTAL")
pop_0_to_9 = df.iloc[1][7]  # 0-9
pop_80_plus = df.iloc[1][15]  # 80+

# Denominator: sum of 10-19 to 50-59
pop_11_to_59 = (
    df.iloc[1][8] +  # 10-19
    df.iloc[1][9] +  # 20-29
    df.iloc[1][10] + # 30-39
    df.iloc[1][11] + # 40-49
    df.iloc[1][12]   # 50-59
)

# Numerator: 0-10 + 60+
numerator = pop_0_to_9 + pop_80_plus
denominator = pop_11_to_59

# Calculate dependency ratio
dependency_ratio = numerator / denominator

print(f"Final Answer: {dependency_ratio:.2f}")