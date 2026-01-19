import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for age groups between 10 and 35 inclusive
age_groups = ['10 - 14', '15 - 17', '18 - 21', '22 - 35']
filtered_df = df[df['Age Group'].isin(age_groups)]
# Sum the 'Both Gender' values
total_individuals = filtered_df['Both Gender'].sum()
print(f"Final Answer: {total_individuals}")