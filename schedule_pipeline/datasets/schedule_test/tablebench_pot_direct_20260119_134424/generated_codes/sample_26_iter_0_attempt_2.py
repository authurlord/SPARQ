import pandas as pd

df = pd.read_csv('table.csv')
# Define the age groups of interest
age_groups = ['10 - 14', '15 - 17', '18 - 21', '22 - 35']
# Filter the data for these age groups
filtered_df = df[df['Age Group'].isin(age_groups)]
# Sum the 'Both Gender' column
total_individuals = filtered_df['Both Gender'].sum()
print(f"Final Answer: {total_individuals}")