import pandas as pd

df = pd.read_csv('table.csv')
# Extract the numerical columns for correlation
students = df['total number of students']
funding = df['research funding (000)']

# Calculate Pearson correlation coefficient
correlation = students.corr(funding)
print(f"Final Answer: research funding (000)")