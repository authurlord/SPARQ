import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where subject is 'mathematics' and match Level 1 or Level 2
math_rows = df[df['subject'].str.contains('mathematics', case=False)]
# Extract number of students for math level 1 and level 2
total_students = math_rows['number of students'].sum()
print(f"Final Answer: {total_students}")