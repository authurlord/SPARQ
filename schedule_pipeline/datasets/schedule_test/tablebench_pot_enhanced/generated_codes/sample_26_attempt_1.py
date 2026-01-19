import pandas as pd

df = pd.read_csv('table.csv')
# Filter age groups between 10 and 35
age_groups_10_to_35 = df[df['Age Group'].isin(['10 - 14', '15 - 17', '18 - 21', '22 - 35'])]
# Sum the 'Both Gender' column
total_individuals = age_groups_10_to_35['Both Gender'].sum()
print(f"Final Answer: {total_individuals}")