import pandas as pd

df = pd.read_csv('table.csv')
# Filter institutions located in London
london_institutions = df[df['location'] == 'london']
# Sum the total number of students for London institutions
total_students_london = london_institutions['total number of students'].sum()
print(f"Final Answer: {total_students_london}")