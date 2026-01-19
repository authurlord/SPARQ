import pandas as pd

df = pd.read_csv('table.csv')

# Filter out 'All Ages' and 'TOTAL' rows
data_filtered = df[~df['Age Group'].isin(['All Ages', 'TOTAL'])]

# Find the row with the highest percentage in the 'Both Gender' column
max_percentage_row = data_filtered.loc[data_filtered['%'].idxmax()]
age_group_highest = max_percentage_row['Age Group']
highest_percentage = float(max_percentage_row['%'])

# Find the row with the lowest percentage in the 'Both Gender' column
min_percentage_row = data_filtered.loc[data_filtered['%'].idxmin()]
lowest_percentage = float(min_percentage_row['%'])

# Calculate the difference
difference = highest_percentage - lowest_percentage

print(f"Final Answer: {age_group_highest}, {difference:.2f}")