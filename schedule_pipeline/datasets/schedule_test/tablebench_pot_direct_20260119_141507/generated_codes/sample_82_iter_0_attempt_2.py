import pandas as pd

df = pd.read_csv('table.csv')
# Filter team with 13 points and goal difference of 17
target_team = df[(df['points'] == 13) & (df['difference'] == 17)]['team'].values[0]
print(f"Final Answer: {target_team}")