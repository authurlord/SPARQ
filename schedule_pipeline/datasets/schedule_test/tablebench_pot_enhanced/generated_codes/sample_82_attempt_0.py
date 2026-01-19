import pandas as pd

df = pd.read_csv('table.csv')
# Filter the team with 13 points and goal difference of 17
team = df[(df['points'] == '13') & (df['difference'] == '17')]['team'].iloc[0]
print(f"Final Answer: {team}")