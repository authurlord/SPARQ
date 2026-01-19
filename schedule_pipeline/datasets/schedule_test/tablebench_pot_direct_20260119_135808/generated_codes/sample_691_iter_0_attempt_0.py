import pandas as pd

df = pd.read_csv('table.csv')

# Drop the 'All Ages' and 'TOTAL' rows for analysis
df_filtered = df[df['Age Group'] != 'All Ages']
df_filtered = df_filtered[df_filtered['Age Group'] != 'TOTAL']

# Find the age group with the highest percentage
max_percentage_row = df_filtered.loc[df_filtered['%'].idxmax()]
highest_age_group = max_percentage_row['Age Group']
highest_percentage = float(max_percentage_row['%'])

# Find the age group with the lowest percentage
min_percentage_row = df_filtered.loc[df_filtered['%'].idxmin()]
lowest_percentage = float(min_percentage_row['%'])

# Calculate the difference
difference = highest_percentage - lowest_percentage

print(f"Final Answer: {highest_age_group}, {difference:.2f}")