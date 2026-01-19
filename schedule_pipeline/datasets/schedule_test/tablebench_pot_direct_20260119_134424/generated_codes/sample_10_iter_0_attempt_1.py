import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where subject is 'mathematics'
math_students = df[df['subject'] == 'mathematics']
# Sum the number of students
total_math_students = math_students['number of students'].sum()
print(f"Final Answer: {total_math_students}")