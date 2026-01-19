import pandas as pd

df = pd.read_csv('table.csv')
# Filter schools located in Belfast
believe_schools = df[df['Location'] == 'Belfast']
# Find the school with the maximum 'Outright Titles'
top_school = believe_schools.loc[believe_schools['Outright Titles'].idxmax(), 'School']
print(f"Final Answer: {top_school}")