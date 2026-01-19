import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to parse goal strings like "2 + 11" or "1 + 9"
def parse_goals(g):
    if isinstance(g, str):
        # Extract numbers and sum them
        numbers = [int(x) for x in re.findall(r'\d+', g)]
        return sum(numbers)
    return g

# Apply parsing to the goal columns
df['league goals'] = df['league goals'].apply(lambda x: parse_goals(x) if isinstance(x, str) else x)
df['fa cup goals'] = df['fa cup goals'].apply(lambda x: parse_goals(x) if isinstance(x, str) else x)
df['league cup goals'] = df['league cup goals'].apply(lambda x: parse_goals(x) if isinstance(x, str) else x)

# Compute sum of the three goal types
df['sum_of_goals'] = df['league goals'] + df['fa cup goals'] + df['league cup goals']

# Check if sum matches total (within small tolerance due to possible rounding)
matches = df['sum_of_goals'] == df['total']

# If all rows match, the factors contribute significantly
if matches.all():
    print("Final Answer: league goals, fa cup goals, league cup goals")
else:
    print("Final Answer: no clear impact")