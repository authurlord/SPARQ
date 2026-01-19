import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where the first column is 'Literates'
literates_row = df[df['-'] == 'Literates']
male_literates = int(literates_row['Male'].values[0])
female_literates = int(literates_row['Female'].values[0])

# Compare and determine the higher gender
if male_literates > female_literates:
    result = "Male"
else:
    result = "Female"

print(f"Final Answer: {result}")