import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract numeric value from 'total goals' column (e.g., '30 (12)' -> 30)
def extract_goals(goals_str):
    try:
        return int(goals_str.split()[0])
    except:
        return 0

# Apply the function to extract numeric goals
df['total_goals_numeric'] = df['total goals'].apply(extract_goals)

# Count players with more than 3 goals
count = (df['total_goals_numeric'] > 3).sum()

print(f"Final Answer: {count}")