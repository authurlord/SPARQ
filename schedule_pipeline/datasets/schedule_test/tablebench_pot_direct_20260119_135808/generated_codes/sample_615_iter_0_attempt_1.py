import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where location is 'london'
london_unis = df[df['location'] == 'london']
# Sum the total number of students for these universities
total_students_london = london_unis['total number of students'].sum()
print(f"Final Answer: {total_students_london}")