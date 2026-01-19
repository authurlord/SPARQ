import pandas as pd

df = pd.read_csv('table.csv')
# Remove the last row (TOTAL) since it's not an age group
df_filtered = df[df['Age Group'] != 'TOTAL']

# Extract the percentage column (%)
percentages = df_filtered['%'].astype(float)

# Find the maximum and minimum percentages
max_percentage = percentages.max()
min_percentage = percentages.min()
difference = max_percentage - min_percentage

# Find the corresponding age group with max and min
max_age_group = df_filtered.loc[percentages.idxmax(), 'Age Group']
min_age_group = df_filtered.loc[percentages.idxmin(), 'Age Group']

print(f"Final Answer: {max_age_group}, {min_age_group}, {difference:.2f}")