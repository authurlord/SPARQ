import pandas as pd

df = pd.read_csv('table.csv')
# Find the literate values for male and female
literate_male = df.loc[df['Total'] == 'Literates', 'Male'].values[0]
literate_female = df.loc[df['Total'] == 'Literates', 'Female'].values[0]

# Compare and determine which gender has more literates
if literate_male > literate_female:
    result = "Male"
else:
    result = "Female"

print(f"Final Answer: {result}")