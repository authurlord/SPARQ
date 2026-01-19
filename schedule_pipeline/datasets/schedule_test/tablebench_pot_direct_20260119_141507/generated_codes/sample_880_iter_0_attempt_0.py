import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for "Literates" and extract Male and Female values
literates_row = df[df['Total'] == 'Literates']
male_literates = int(literates_row['Male'].values[0])
female_literates = int(literates_row['Female'].values[0])

if male_literates > female_literates:
    print("Final Answer: Male")
else:
    print("Final Answer: Female")