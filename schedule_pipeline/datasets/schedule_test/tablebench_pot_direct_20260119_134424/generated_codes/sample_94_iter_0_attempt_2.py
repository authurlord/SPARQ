import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for George McLean
george_mclean = df[df['name'] == 'george mclean']
# Extract league goals
league_goals = george_mclean['league goals'].values[0]
print(f"Final Answer: {league_goals}")