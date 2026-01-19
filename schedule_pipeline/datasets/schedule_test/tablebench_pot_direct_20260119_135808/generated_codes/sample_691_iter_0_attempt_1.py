import pandas as pd

df = pd.read_csv('table.csv')

# Exclude 'All Ages' and 'TOTAL' rows
age_groups = df[df['Age Group'] != 'All Ages']
age_groups = age_groups[age_groups['Age Group'] != 'TOTAL']

# Extract percentages from the '%', '%_1', and '%_2' columns
percentages = pd.concat([age_groups['%'], age_groups['%_1'], age_groups['%_2']], ignore_index=True).astype(float)

# Find max and min percentages
max_percent = percentages.max()
min_percent = percentages.min()

# Find the corresponding age group with max percentage
max_age_group = age_groups.loc[age_groups['%'].idxmax()]['Age Group']
difference = max_percent - min_percent

print(f"Final Answer: {max_age_group}, {difference:.2f}")