import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where the first column is 'Literates'
literates_row = df[df['-'] == 'Literates']
# Extract male and female literate counts
male_literates = int(literates_row['Male'].values[0])
female_literates = int(literates_row['Female'].values[0])

# Compare and determine which gender has more literates
if male_literates > female_literates:
    result = "Male"
else:
    result = "Female"

print(f"Final Answer: {result}")