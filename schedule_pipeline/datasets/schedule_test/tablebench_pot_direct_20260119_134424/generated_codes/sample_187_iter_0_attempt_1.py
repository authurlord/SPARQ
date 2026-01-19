import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to integers
df['wins'] = pd.to_numeric(df['wins'])
df['draws'] = pd.to_numeric(df['draws'])
df['points'] = pd.to_numeric(df['points'])

# Calculate expected points
df['expected_points'] = df['wins'] * 3 + df['draws'] * 1

# Check if expected points match actual points
matches = (df['expected_points'] == df['points']).all()

if matches:
    print("Final Answer: wins, draws")
else:
    print("Final Answer: no clear impact")