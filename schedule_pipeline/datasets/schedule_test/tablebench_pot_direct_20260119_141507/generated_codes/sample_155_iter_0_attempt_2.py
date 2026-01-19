import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to parse expressions like "2 + 11"
def parse_goals(g):
    if isinstance(g, str):
        # Extract numbers from string using regex
        numbers = re.findall(r'\d+', g)
        if len(numbers) == 2:
            return int(numbers[0]) + int(numbers[1])
        elif len(numbers) == 1:
            return int(numbers[0])
        else:
            return 0
    return g

# Apply parsing to the goal columns
df['league goals'] = df['league goals'].apply(lambda x: parse_goals(x) if isinstance(x, str) else x)
df['fa cup goals'] = df['fa cup goals'].apply(lambda x: parse_goals(x) if isinstance(x, str) else x)
df['league cup goals'] = df['league cup goals'].apply(lambda x: parse_goals(x) if isinstance(x, str) else x)

# Calculate sum of the three goal columns
df['sum_of_goals'] = df['league goals'] + df['fa cup goals'] + df['league cup goals']

# Check if sum matches total
match_count = (df['sum_of_goals'] == df['total']).sum()
total_rows = len(df)

# If all rows match, all factors contribute
if match_count == total_rows:
    final_answer = "league goals, fa cup goals, league cup goals"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")