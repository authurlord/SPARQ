import pandas as pd

df = pd.read_csv('table.csv')

# Extract male population data
# Rows with 'A.' under 'SPECIFICATION_1' are for males
males_data = df[df['SPECIFICATION'] == 'I.'].iloc[4:6]  # Rows for males

# Get values for 10–19 and 20–29 age groups (10–29)
males_10_29 = males_data.iloc[0]['POPULATION (by age group in 2002)_2'] + males_data.iloc[0]['POPULATION (by age group in 2002)_3']

# Get values for 60–69 and 80+ age groups (60+)
males_60_plus = males_data.iloc[0]['POPULATION (by age group in 2002)_7'] + males_data.iloc[0]['POPULATION (by age group in 2002)_9']

# Calculate difference
difference = males_10_29 - males_60_plus

print(f"Final Answer: {difference}")