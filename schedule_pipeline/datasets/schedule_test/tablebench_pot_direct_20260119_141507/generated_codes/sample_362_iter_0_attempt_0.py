import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract melting point from the string (first number before "/")
def parse_melting_point(point_str):
    # Match the first number (negative or positive) before "/"
    match = re.search(r'^(-?\d+)', point_str)
    if match:
        return float(match.group(1))
    return None

# Apply parsing to the 'melting / boiling point' column
df['melting_point'] = df['melting / boiling point'].apply(parse_melting_point)

# Count how many have melting point below 0
count_below_zero = df[df['melting_point'] < 0].shape[0]

print(f"Final Answer: {count_below_zero}")