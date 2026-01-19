import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where the SPECIFICATION_1 is 'Males' (or similar) and extract relevant values
# We know the male data row is: ['I.', '1.', '1.', 'A.', 'Males', ...]
# Look for the row that starts with 'I.' and has 'Males' in the second or third column

# Find the row where 'SPECIFICATION_1' is 'Males' (after filtering)
# The actual data row is: ['I.', '1.', '1.', 'A.', 'Males', 'person', '74', '13', '16', '10', '8', '10', '9', '4', '3', '1']
# So we want the 20–29 (index 8) and 30–39 (index 9) values

# Locate the row where the value is 'Males'
male_row = df[df['SPECIFICATION_1'] == 'Males']

# Extract the values for 20–29 and 30–39 age groups
# These are at indices 8 and 9 in the population columns
if not male_row.empty:
    males_20_29 = male_row.iloc[0]['POPULATION (by age group in 2002)_3']  # 20–29
    males_30_39 = male_row.iloc[0]['POPULATION (by age group in 2002)_4']  # 30–39
    total_males = males_20_29 + males_30_39
    print(f"Final Answer: {total_males}")
else:
    print("Final Answer: 0")