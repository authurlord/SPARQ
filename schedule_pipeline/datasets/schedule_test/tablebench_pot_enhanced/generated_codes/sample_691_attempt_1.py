import pandas as pd

df = pd.read_csv('table.csv')

# Exclude 'All Ages' and 'TOTAL' rows for analysis
data = df[~df['Age Group'].isin(['All Ages', 'TOTAL'])]

# Find the row with the highest percentage in 'Both Gender'
max_row = data.loc[data['%'].idxmax()]
highest_age_group = max_row['Age Group']
highest_percentage = float(max_row['%'])

# Find the row with the lowest percentage in 'Both Gender'
min_row = data.loc[data['%'].idxmin()]
lowest_percentage = float(min_row['%'])

# Calculate the difference
difference = highest_percentage - lowest_percentage

print(f"Final Answer: {highest_age_group}, {difference:.2f}")