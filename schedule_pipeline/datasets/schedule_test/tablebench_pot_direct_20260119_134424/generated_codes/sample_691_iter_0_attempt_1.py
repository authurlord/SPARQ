import pandas as pd

df = pd.read_csv('table.csv')

# Exclude 'All Ages' and 'TOTAL' rows
age_groups = df[~df['Age Group'].isin(['All Ages', 'TOTAL'])]

# Extract the '%' column for comparison
percentages = age_groups['%'].astype(float)

# Find the max and min percentages
max_percentage = percentages.max()
min_percentage = percentages.min()

# Calculate the difference
difference = max_percentage - min_percentage

print(f"Final Answer: {max_percentage:.2f}, {difference:.2f}")