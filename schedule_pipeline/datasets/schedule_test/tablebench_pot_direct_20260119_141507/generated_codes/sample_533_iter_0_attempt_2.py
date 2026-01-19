import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant population values from the 'POPULATION (by age group in 2002)' columns
# Age groups:
# 0-9: index 8 (value at column index 8)
# 10-19: index 9
# 20-29: index 10
# 30-39: index 11
# 40-49: index 12
# 50-59: index 13
# 60-69: index 14
# 70-79: index 15
# 80+: index 16

# Get the row where SPECIFICATION is 'I.' and 'POPULATION (by age group in 2002)' has values
# We are interested in the row with "TOTAL" under "POPULATION (by age group in 2002)"
# The values are in the last 9 columns starting from index 8

# Extract the population values for each age group
pop_0_to_9 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[1]  # 0-9
pop_10_to_19 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[2]  # 10-19
pop_20_to_29 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[3]  # 20-29
pop_30_to_39 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[4]  # 30-39
pop_40_to_49 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[5]  # 40-49
pop_50_to_59 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[6]  # 50-59
pop_60_to_69 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[7]  # 60-69
pop_70_to_79 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[8]  # 70-79
pop_80_plus = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)'].iloc[9]   # 80+

# Calculate total for 0-10 and 60+
population_0_to_10 = pop_0_to_9
population_60_plus = pop_60_to_69 + pop_70_to_79 + pop_80_plus

# Population aged 11-59 (sum of 10-19 to 50-59)
population_11_to_59 = (pop_10_to_19 + pop_20_to_29 + pop_30_to_39 + pop_40_to_49 + pop_50_to_59)

# Calculate dependency ratio
dependency_ratio = (population_0_to_10 + population_60_plus) / population_11_to_59

print(f"Final Answer: {dependency_ratio:.2f}")