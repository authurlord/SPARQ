import pandas as pd

df = pd.read_csv('table.csv')
# Extract the percentage column for 'Both Gender'
percentages = df['%'].astype(float)
# Find the maximum and minimum percentages
max_percentage = percentages.max()
min_percentage = percentages.min()
# Find the corresponding age group
max_age_group = df.loc[percentages.idxmax(), 'Age Group']
min_age_group = df.loc[percentages.idxmin(), 'Age Group']
difference = max_percentage - min_percentage
print(f"Final Answer: {max_age_group}, {difference:.2f}")