import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract melting point (first number in "X / Y" format)
def parse_melting_point(point_str):
    match = re.search(r'^(-?\d+)', point_str)
    if match:
        return int(match.group(1))
    return None

# Apply parsing to the melting point column
df['melting_point'] = df['melting / boiling point'].apply(parse_melting_point)

# Count agents with melting point below 0
count_below_zero = df[df['melting_point'] < 0].shape[0]

print(f"Final Answer: {count_below_zero}")