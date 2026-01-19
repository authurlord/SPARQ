import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where 'SPECIFICATION' is 'I. 1. A. Males'
male_row = df[df['SPECIFICATION'] == 'I. 1. A. Males']

# Extract values for age groups
males_10_19 = male_row['POPULATION (by age group in 2002)_1'].values[0]
males_20_29 = male_row['POPULATION (by age group in 2002)_2'].values[0]
males_60_plus = male_row['POPULATION (by age group in 2002)_7'].values[0]

# Total males in 10–29 age group
males_10_29 = males_10_19 + males_20_29

# Calculate difference
difference = males_10_29 - males_60_plus

print(f"Final Answer: {difference}")