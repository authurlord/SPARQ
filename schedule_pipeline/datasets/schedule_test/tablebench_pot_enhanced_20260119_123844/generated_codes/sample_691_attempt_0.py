import pandas as pd

df = pd.read_csv('table.csv')
# Exclude 'All Ages' and 'TOTAL' rows for analysis
age_groups = df.iloc[1:-1]  # Skip first and last rows
# Extract percentages from the '%', '%_1', and '%_2' columns
percentages = pd.concat([age_groups['%'], age_groups['%_1'], age_groups['%_2']], ignore_index=True)
max_percentage = percentages.max()
min_percentage = percentages.min()
difference = max_percentage - min_percentage
# Find the corresponding age group with the highest percentage
max_row = age_groups[age_groups['%'] == max_percentage]
highest_age_group = max_row['Age Group'].values[0]
print(f"Final Answer: {highest_age_group}, {difference:.2f}")