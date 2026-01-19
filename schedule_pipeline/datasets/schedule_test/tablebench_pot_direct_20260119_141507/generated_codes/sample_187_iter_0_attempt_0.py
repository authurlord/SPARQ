import pandas as pd

df = pd.read_csv('table.csv')

# Calculate expected points from wins and draws
df['expected_points'] = 3 * df['wins'] + df['draws']

# Check if expected points match actual points
match_count = (df['expected_points'] == df['points']).sum()
total_teams = len(df)

# If all teams match, wins and draws significantly contribute
if match_count == total_teams:
    final_answer = "wins, draws"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")