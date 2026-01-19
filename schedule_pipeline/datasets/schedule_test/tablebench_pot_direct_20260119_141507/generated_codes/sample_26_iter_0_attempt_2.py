import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Age Group is between 10 and 35
age_groups_of_interest = ['10 - 14', '15 - 17', '18 - 21', '22 - 35']
filtered_data = df[df['Age Group'].isin(age_groups_of_interest)]
# Sum the 'Both Gender' column for these rows
total_individuals = filtered_data['Both Gender'].sum()
print(f"Final Answer: {total_individuals}")