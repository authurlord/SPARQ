import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to integers
df['wins'] = pd.to_numeric(df['wins'])
df['draws'] = pd.to_numeric(df['draws'])
df['losses'] = pd.to_numeric(df['losses'])
df['points'] = pd.to_numeric(df['points'])

# Calculate expected points using standard formula
df['expected_points'] = df['wins'] * 3 + df['draws'] * 1

# Check if expected points match actual points
match = (df['expected_points'] == df['points']).all()

if match:
    print("Final Answer: wins, draws")
else:
    print("Final Answer: no clear impact")