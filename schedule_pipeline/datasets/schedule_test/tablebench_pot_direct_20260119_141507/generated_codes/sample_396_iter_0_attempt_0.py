import pandas as pd

df = pd.read_csv('table.csv')

# Descriptive explanation of the table
explanation = (
    "The table lists skyscrapers by rank, name, height (feet and meters), number of floors, and construction year. "
    "The tallest building is One America Plaza at 500 feet (152 m). "
    "Multiple buildings share the same height or rank, indicated by '5 =' or '8 ='. "
    "Buildings range from 310 feet (94 m) to 500 feet (152 m), with construction years spanning from 1927 to 2012. "
    "There is a general trend of taller buildings having more floors, though exceptions exist."
)

print(explanation)