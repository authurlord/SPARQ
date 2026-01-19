import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Check if the total is the sum of the individual goals
# Convert 'league goals', 'fa cup goals', 'league cup goals' to numeric (handling strings like '2 + 11')
df['league goals'] = pd.to_numeric(df['league goals'], errors='coerce')
df['fa cup goals'] = pd.to_numeric(df['fa cup goals'], errors='coerce')
df['league cup goals'] = pd.to_numeric(df['league cup goals'], errors='coerce')

# Calculate the sum of individual goals
sum_goals = df['league goals'] + df['fa cup goals'] + df['league cup goals']

# Compare with total column
# If the sum matches total (within small tolerance due to rounding), then the factors contribute
match_count = (abs(sum_goals - df['total']) < 0.1).sum()

# If all rows match, the factors contribute significantly
if match_count == len(df):
    final_answer = "league goals, fa cup goals, league cup goals"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")