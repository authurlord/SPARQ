import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where club is 'llanhilleth rfc' and get the 'points' value
points_llanhilleth = df[df['club'] == 'llanhilleth rfc']['points'].values[0]
print(f"Final Answer: {points_llanhilleth}")