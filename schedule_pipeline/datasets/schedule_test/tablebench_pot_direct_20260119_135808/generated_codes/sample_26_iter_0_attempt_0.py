import pandas as pd

df = pd.read_csv('table.csv')
# Filter the rows for age groups between 10 and 35
age_groups = ['10 - 14', '15 - 17', '18 - 21', '22 - 35']
filtered_df = df[df['Age Group'].isin(age_groups)]
# Sum the 'Both Gender' column
total_individuals = filtered_df['Both Gender'].sum()
print(f"Final Answer: {total_individuals}")