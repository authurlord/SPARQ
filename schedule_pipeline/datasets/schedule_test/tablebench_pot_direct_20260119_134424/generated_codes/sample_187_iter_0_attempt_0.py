import pandas as pd

df = pd.read_csv('table.csv')

# Calculate expected points from wins and draws
df['expected_points'] = df['wins'].astype(int) * 3 + df['draws'].astype(int) * 1

# Check if expected points match actual points
matches = (df['expected_points'] == df['points']).all()

if matches:
    print("Final Answer: wins, draws")
else:
    print("Final Answer: no clear impact")