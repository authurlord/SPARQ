import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant row for males
males_data = df[df['SPECIFICATION'] == 'I. 1. A. Males']

# Get values for 10–19 and 20–29 (10–29 age group)
males_10_19 = males_data['POPULATION (by age group in 2002)_1']
males_20_29 = males_data['POPULATION (by age group in 2002)_2']

# Get value for 60+ (80 +)
males_80_plus = males_data['POPULATION (by age group in 2002)_9']

# Total males in 10–29
males_10_29 = males_10_19 + males_20_29

# Difference
difference = males_10_29 - males_80_plus

print(f"Final Answer: {difference}")