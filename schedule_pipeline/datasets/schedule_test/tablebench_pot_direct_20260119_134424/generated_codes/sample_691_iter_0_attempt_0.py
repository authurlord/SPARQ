import pandas as pd

df = pd.read_csv('table.csv')

# Exclude 'All Ages' and 'TOTAL' rows for analysis
age_groups = df[~df['Age Group'].isin(['All Ages', 'TOTAL'])]

# Find the age group with the highest percentage
max_percentage_row = age_groups.loc[age_groups['%'].idxmax()]
highest_age_group = max_percentage_row['Age Group']
highest_percentage = float(max_percentage_row['%'])

# Find the age group with the lowest percentage
min_percentage_row = age_groups.loc[age_groups['%'].idxmin()]
lowest_percentage = float(min_percentage_row['%'])

# Calculate the difference
difference = highest_percentage - lowest_percentage

print(f"Final Answer: {highest_age_group}, {difference:.2f}")