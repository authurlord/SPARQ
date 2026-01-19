import pandas as pd

df = pd.read_csv('table.csv')

# Extract numeric part from 'total goals' column (before parentheses if present)
def extract_goals(goal_str):
    # Remove any text in parentheses and convert to int
    if isinstance(goal_str, str):
        return int(goal_str.split()[0])  # Take the first number before any space or parentheses
    return int(goal_str)

# Apply the function to convert 'total goals' to integers
df['total goals'] = df['total goals'].apply(extract_goals)

# Count players with more than 3 total goals
count = (df['total goals'] > 3).sum()

print(f"Final Answer: {count}")