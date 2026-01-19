import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Find the row where SPECIFICATION is 'I.' and the second column is 'TOTAL'
# The row with "I. TOTAL" is at index 1
# Extract the population values from that row
row = df.iloc[1]

# Extract the relevant population values
age_0_to_9 = row['POPULATION (by age group in 2002)_1']  # 0-9
age_80_plus = row['POPULATION (by age group in 2002)_9']  # 80+

# Age groups 10-19 to 50-59
age_10_to_19 = row['POPULATION (by age group in 2002)_2']
age_20_to_29 = row['POPULATION (by age group in 2002)_3']
age_30_to_39 = row['POPULATION (by age group in 2002)_4']
age_40_to_49 = row['POPULATION (by age group in 2002)_5']
age_50_to_59 = row['POPULATION (by age group in 2002)_6']

# Calculate numerator: 0-10 + 60+
numerator = age_0_to_9 + age_80_plus

# Calculate denominator: 11-59 (sum of 10-19 to 50-59)
denominator = age_10_to_19 + age_20_to_29 + age_30_to_39 + age_40_to_49 + age_50_to_59

# Compute dependency ratio
dependency_ratio = numerator / denominator

print(f"Final Answer: {dependency_ratio:.2f}")