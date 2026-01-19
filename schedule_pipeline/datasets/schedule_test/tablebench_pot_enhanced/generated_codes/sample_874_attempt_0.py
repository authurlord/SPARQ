import pandas as pd

df = pd.read_csv('table.csv')

# Filter for males (row with SPECIFICATION_1 = 'Males')
males_row = df[df['SPECIFICATION_1'] == 'Males']

# Extract values for 10–19 and 20–29 age groups (10–29)
males_10_to_29 = males_row.iloc[0]['POPULATION (by age group in 2002)_2'] + males_row.iloc[0]['POPULATION (by age group in 2002)_3']

# Extract values for 60–69 and 80+ age groups (60+)
males_60_plus = males_row.iloc[0]['POPULATION (by age group in 2002)_6'] + males_row.iloc[0]['POPULATION (by age group in 2002)_7']

# Calculate the difference
difference = males_10_to_29 - males_60_plus

print(f"Final Answer: {difference}")