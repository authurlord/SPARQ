import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the TOTAL row
df_filtered = df[df['Age Group'] != 'TOTAL']

# Extract the percentage column (the third column in the data)
percentages = df_filtered['%'].astype(float)

# Find the maximum and minimum percentages
max_percentage = percentages.max()
min_percentage = percentages.min()

# Get the corresponding age groups
max_age_group = df_filtered.loc[percentages.idxmax(), 'Age Group']
min_age_group = df_filtered.loc[percentages.idxmin(), 'Age Group']

# Calculate the difference
difference = max_percentage - min_percentage

print(f"Final Answer: {max_age_group}, {min_age_group}, {difference:.2f}")