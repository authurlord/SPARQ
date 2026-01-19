import pandas as pd

df = pd.read_csv('table.csv')
# Exclude 'All Ages' and 'TOTAL' rows for analysis
age_groups = df[df['Age Group'] != 'All Ages']
age_groups = age_groups[age_groups['Age Group'] != 'TOTAL']

# Find the age group with the highest percentage
max_percentage_row = age_groups.loc[age_groups['%'].idxmax()]
max_percentage = float(max_percentage_row['%'])

# Find the age group with the lowest percentage
min_percentage_row = age_groups.loc[age_groups['%'].idxmin()]
min_percentage = float(min_percentage_row['%'])

# Calculate the difference
difference = max_percentage - min_percentage

# Print final answer
print(f"Final Answer: {max_percentage_row['Age Group']}, {difference:.2f}")