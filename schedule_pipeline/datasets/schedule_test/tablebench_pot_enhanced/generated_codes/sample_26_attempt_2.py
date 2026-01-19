import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for age groups 10-14, 15-17, 18-21, and 22-35
age_groups = ['10 - 14', '15 - 17', '18 - 21', '22 - 35']
filtered_data = df[df['Age Group'].isin(age_groups)]
# Convert 'Both Gender' to numeric (removing commas if necessary)
total_individuals = filtered_data['Both Gender'].str.replace(',', '').astype(int).sum()
print(f"Final Answer: {total_individuals}")