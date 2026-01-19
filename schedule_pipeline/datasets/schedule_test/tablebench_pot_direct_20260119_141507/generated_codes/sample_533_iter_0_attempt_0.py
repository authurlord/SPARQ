import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant population values from the "POPULATION (by age group in 2002)" column
# We are interested in:
# - 0-9 (index 2 in the column list)
# - 80+ (index 14 in the column list)
# - 10-19, 20-29, 30-39, 40-49, 50-59 (indices 3, 4, 5, 6, 7)

# Get the total population values (excluding header rows)
# Filter rows where the first column is not a specification
data_rows = df[df['SPECIFICATION'] != 'SPECIFICATION']
population_data = data_rows[['POPULATION (by age group in 2002)', 
                             'POPULATION (by age group in 2002)_1', 
                             'POPULATION (by age group in 2002)_2', 
                             'POPULATION (by age group in 2002)_3', 
                             'POPULATION (by age group in 2002)_4', 
                             'POPULATION (by age group in 2002)_5', 
                             'POPULATION (by age group in 2002)_6', 
                             'POPULATION (by age group in 2002)_7', 
                             'POPULATION (by age group in 2002)_8', 
                             'POPULATION (by age group in 2002)_9']]

# The first row (after header) contains the total population
# We pick the row where SPECIFICATION is "I." and then take the values
# We'll extract the values directly from the row where the first column is "I."
# Find the row with "I." and "TOTAL" in the first two columns
total_row = df[(df['SPECIFICATION'] == 'I.') & (df['SPECIFICATION_1'] == 'TOTAL')].iloc[0]
# Extract the values for the age groups
age_groups = {
    '0-9': total_row['POPULATION (by age group in 2002)'],  # index 1
    '10-19': total_row['POPULATION (by age group in 2002)_1'],
    '20-29': total_row['POPULATION (by age group in 2002)_2'],
    '30-39': total_row['POPULATION (by age group in 2002)_3'],
    '40-49': total_row['POPULATION (by age group in 2002)_4'],
    '50-59': total_row['POPULATION (by age group in 2002)_5'],
    '60-69': total_row['POPULATION (by age group in 2002)_6'],
    '70-79': total_row['POPULATION (by age group in 2002)_7'],
    '80 +': total_row['POPULATION (by age group in 2002)_8']
}

# Extract values
pop_0_to_9 = age_groups['0-9']
pop_80_plus = age_groups['80 +']
pop_11_to_59 = (age_groups['10-19'] + age_groups['20-29'] + age_groups['30-39'] + age_groups['40-49'] + age_groups['50-59'])

# Calculate dependency ratio
numerator = pop_0_to_9 + pop_80_plus
denominator = pop_11_to_59
dependency_ratio = numerator / denominator

print(f"Final Answer: {dependency_ratio:.2f}")