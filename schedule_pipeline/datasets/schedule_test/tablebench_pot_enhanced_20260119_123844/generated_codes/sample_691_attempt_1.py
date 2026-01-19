import pandas as pd

df = pd.read_csv('table.csv')

# Filter out 'All Ages' and 'TOTAL' rows
data_filtered = df[(df['Age Group'] != 'All Ages') & (df['Age Group'] != 'TOTAL')]

# Find the age group with the highest percentage
max_percentage_row = data_filtered.loc[data_filtered['%'].idxmax()]
highest_age_group = max_percentage_row['Age Group']
highest_percentage = float(max_percentage_row['%'])

# Find the age group with the lowest percentage
min_percentage_row = data_filtered.loc[data_filtered['%'].idxmin()]
lowest_percentage = float(min_percentage_row['%'])

# Calculate the difference
difference = highest_percentage - lowest_percentage

print(f"Final Answer: {highest_age_group}, {difference:.2f}")